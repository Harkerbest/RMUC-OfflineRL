"""MAPPO (Yu et al., 2022) fine-tuning of an offline tactical policy.

* **Shared, decentralised actor.**  One `HybridPolicy` drives every controlled
  infantry (parameter sharing across same-role agents) from its own 161-D
  observation only — the deployable input.
* **Centralised critic.**  V(s) reads privileged engine state the actor never
  sees: the full-information 161-D view of that agent plus every robot's ammo,
  heat and level and both teams' coins and damage (`sim.episode.global_extras`).
  It exists only during training.
* **Offline prior.**  The actor starts from an offline run (BC / IQL policy
  head) and is anchored to a frozen copy of it:
      loss += kl_coef * KL(pi || pi_offline)
  so fine-tuning leaves the human/scripted prior only where the advantage
  estimate pays for it — the same role the human-data KL plays in AlphaStar.
* **Value normalisation** (running mean/std of the return targets), as in the
  MAPPO paper; the critic predicts normalised returns.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .networks import HybridPolicy, mlp


class RunningMeanStd(nn.Module):
    """Streaming mean/variance (Chan et al.), kept as buffers so it checkpoints."""

    def __init__(self, shape=(), eps=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, *self.mean.shape).double()
        bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean.double(), self.count.double() + bn
        m2 = self.var.double() * self.count + bv * bn + delta.pow(2) * self.count * bn / tot
        self.mean.copy_((self.mean.double() + delta * bn / tot).float())
        self.var.copy_((m2 / tot).float())
        self.count.copy_(tot.float())

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-6)

    def denormalize(self, x):
        return x * (self.var.sqrt() + 1e-6) + self.mean


class CentralCritic(nn.Module):
    def __init__(self, gdim, hidden=256, depth=2):
        super().__init__()
        self.gnorm = RunningMeanStd((gdim,))
        self.v = mlp([gdim] + [hidden] * depth + [1])

    def forward(self, g):
        return self.v(self.gnorm.normalize(g).clamp(-10, 10)).squeeze(-1)


class MAPPO(nn.Module):
    def __init__(self, actor: HybridPolicy, gdim: int, prior: HybridPolicy | None = None,
                 hidden=256, depth=2, clip=0.2, value_clip=0.2, vf_coef=1.0,
                 ent_coef=0.0, kl_coef=0.05):
        super().__init__()
        self.actor = actor
        self.critic = CentralCritic(gdim, hidden, depth)
        self.value_norm = RunningMeanStd(())
        self.prior = copy.deepcopy(prior if prior is not None else actor)
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.clip, self.value_clip = clip, value_clip
        self.vf_coef, self.ent_coef, self.kl_coef = vf_coef, ent_coef, kl_coef

    @torch.no_grad()
    def values(self, g):
        """Critic values in return units."""
        return self.value_norm.denormalize(self.critic(g))

    def forward(self, b, train_actor=True):
        """PPO loss on a minibatch.  Keys: obs (normalised), act (raw sample),
        logp, adv, ret, v_old (return units), g (global state), valid."""
        valid = b["valid"]
        n = valid.sum().clamp(min=1.0)
        # ---- critic: clipped regression to normalised returns ----
        v = self.critic(b["g"])
        ret_n = self.value_norm.normalize(b["ret"])
        v_old_n = self.value_norm.normalize(b["v_old"])
        v_clip = v_old_n + (v - v_old_n).clamp(-self.value_clip, self.value_clip)
        v_loss = torch.max((v - ret_n).pow(2), (v_clip - ret_n).pow(2)).mean()
        metrics = dict(v_loss=v_loss.detach())
        if not train_actor:
            return self.vf_coef * v_loss, metrics
        # ---- actor: clipped surrogate on the steps where the robot was alive ----
        logp = self.actor.log_prob(b["obs"], b["act"], clamp_nav=False)
        ratio = (logp - b["logp"]).exp()
        adv = b["adv"]
        surr = torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv)
        pi_loss = -(surr * valid).sum() / n
        ent = (self.actor.entropy(b["obs"]) * valid).sum() / n
        kl_prior = (self.actor.kl(b["obs"], self.prior) * valid).sum() / n
        loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent + self.kl_coef * kl_prior
        with torch.no_grad():
            approx_kl = (((ratio - 1) - (logp - b["logp"])) * valid).sum() / n
            clipfrac = (((ratio - 1).abs() > self.clip).float() * valid).sum() / n
        metrics.update(pi_loss=pi_loss.detach(), entropy=ent.detach(), kl_prior=kl_prior.detach(),
                       approx_kl=approx_kl, clipfrac=clipfrac)
        return loss, metrics


def gae(rew: np.ndarray, val: np.ndarray, gamma=0.99, lam=0.95):
    """Advantages and returns for one finished episode (terminal after the last step)."""
    adv = np.zeros_like(rew, dtype=np.float64)
    last = 0.0
    for t in range(len(rew) - 1, -1, -1):
        nxt = val[t + 1] if t + 1 < len(rew) else 0.0
        delta = rew[t] + gamma * nxt - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv.astype(np.float32), (adv + val).astype(np.float32)
