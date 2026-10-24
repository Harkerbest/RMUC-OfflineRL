"""Masked multi-discrete PPO with a centralised critic (the 2027 decision layer).

* **Actor** `MultiHeadPolicy`: one MLP, one categorical head per action head
  (`arena27.spec.HEADS`), illegal choices masked out.  Observation normalisation
  lives inside the module (running mean/std, clipped to +-5), so an exported
  policy is self-contained.  The output layer starts near zero, i.e. near-uniform
  over the legal choices.
* **Critic** `ValueNet`: reads the privileged global state (full-information
  observation + `arena27.obs.global_extras`); it exists only during training.
* **Loss**: clipped surrogate, clipped value regression to normalised returns
  (MAPPO), entropy bonus.  `imitation=True` swaps the surrogate for the
  log-likelihood of the recorded actions — used to start the actor from a
  behaviour tree before PPO takes over.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

from .mappo import RunningMeanStd, gae  # noqa: F401  (gae re-exported for the trainer)
from .networks import mlp

NEG = -1e9


class MultiHeadPolicy(nn.Module):
    def __init__(self, obs_dim: int, head_sizes, hidden=256, depth=3):
        super().__init__()
        self.obs_dim, self.sizes = int(obs_dim), [int(n) for n in head_sizes]
        self.config = dict(obs_dim=self.obs_dim, head_sizes=self.sizes, hidden=hidden, depth=depth)
        self.obs_norm = RunningMeanStd((obs_dim,))
        self.body = nn.Sequential(mlp([obs_dim] + [hidden] * depth), nn.LayerNorm(hidden), nn.ReLU())
        self.head = nn.Linear(hidden, sum(self.sizes))
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def logits(self, obs, mask):
        z = self.obs_norm.normalize(obs).clamp(-5.0, 5.0)
        return self.head(self.body(z)).masked_fill(~mask, NEG)

    def dists(self, obs, mask):
        return [Categorical(logits=l) for l in torch.split(self.logits(obs, mask), self.sizes, dim=-1)]

    @torch.no_grad()
    def act(self, obs, mask, greedy=False):
        ds = self.dists(obs, mask)
        a = torch.stack([d.probs.argmax(-1) if greedy else d.sample() for d in ds], -1)
        logp = sum(d.log_prob(a[:, i]) for i, d in enumerate(ds))
        return a, logp

    def evaluate(self, obs, mask, act):
        ds = self.dists(obs, mask)
        logp = sum(d.log_prob(act[:, i]) for i, d in enumerate(ds))
        ent = sum(d.entropy() for d in ds)
        return logp, ent


class ValueNet(nn.Module):
    def __init__(self, gdim: int, hidden=256, depth=3):
        super().__init__()
        self.gnorm = RunningMeanStd((gdim,))
        self.v = mlp([gdim] + [hidden] * depth + [1])

    def forward(self, g):
        return self.v(self.gnorm.normalize(g).clamp(-5.0, 5.0)).squeeze(-1)


class PPO(nn.Module):
    def __init__(self, policy: MultiHeadPolicy, gdim: int, hidden=256, depth=3, clip=0.2,
                 value_clip=0.2, vf_coef=0.5, ent_coef=0.01):
        super().__init__()
        self.policy = policy
        self.critic = ValueNet(gdim, hidden, depth)
        self.value_norm = RunningMeanStd(())
        self.clip, self.value_clip, self.vf_coef, self.ent_coef = clip, value_clip, vf_coef, ent_coef

    @torch.no_grad()
    def values(self, g):
        """Critic values in return units."""
        return self.value_norm.denormalize(self.critic(g))

    def forward(self, b, imitation=False, prior=None, prior_kl=0.0):
        """Loss on a minibatch.  Keys: obs, mask, act, logp, adv, ret, v_old, g.
        `prior` (a frozen MultiHeadPolicy) adds prior_kl * KL(pi || prior) — the anchor
        to the imitated tree that keeps early PPO from drifting (AlphaStar's KL to its
        supervised policy)."""
        v = self.critic(b["g"])
        ret_n, v_old_n = self.value_norm.normalize(b["ret"]), self.value_norm.normalize(b["v_old"])
        v_clip = v_old_n + (v - v_old_n).clamp(-self.value_clip, self.value_clip)
        v_loss = torch.max((v - ret_n).pow(2), (v_clip - ret_n).pow(2)).mean()
        ds = self.policy.dists(b["obs"], b["mask"])
        logp = sum(d.log_prob(b["act"][:, i]) for i, d in enumerate(ds))
        ent = sum(d.entropy() for d in ds).mean()
        if imitation:
            pi_loss = -logp.mean()
            with torch.no_grad():
                parts = torch.split(self.policy.logits(b["obs"], b["mask"]), self.policy.sizes, dim=-1)
                pred = torch.stack([p.argmax(-1) for p in parts], -1)
                imit_acc = (pred == b["act"]).all(-1).float().mean()
            metrics = dict(v_loss=v_loss.detach(), pi_loss=pi_loss.detach(), entropy=ent.detach(),
                           imit_acc=imit_acc)
            return pi_loss + self.vf_coef * v_loss, metrics
        ratio = (logp - b["logp"]).exp()
        adv = b["adv"]
        pi_loss = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (logp - b["logp"])).mean()
            clipfrac = ((ratio - 1).abs() > self.clip).float().mean()
        metrics = dict(v_loss=v_loss.detach(), pi_loss=pi_loss.detach(), entropy=ent.detach(),
                       approx_kl=approx_kl, clipfrac=clipfrac)
        loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
        if prior is not None and prior_kl > 0:
            with torch.no_grad():
                pd = prior.dists(b["obs"], b["mask"])
            kl = sum(kl_divergence(d, q) for d, q in zip(ds, pd)).mean()
            loss = loss + prior_kl * kl
            metrics["kl_prior"] = kl.detach()
        return loss, metrics


def save_policy(policy: MultiHeadPolicy, path: str, extra=None):
    with open(path, "wb") as f:
        torch.save(dict(config=policy.config, state=policy.state_dict(), **(extra or {})), f)


def load_policy(path: str, device="cpu") -> MultiHeadPolicy:
    import os
    if os.path.isdir(path):
        path = os.path.join(path, "policy.pt")
    with open(path, "rb") as f:
        ck = torch.load(f, map_location=device, weights_only=False)
    c = ck["config"]
    pol = MultiHeadPolicy(c["obs_dim"], c["head_sizes"], c["hidden"], c["depth"])
    pol.load_state_dict(ck["state"])
    return pol.eval()
