"""Tactical policies for closed-loop play: trained BC / IQL / DT runs, and controls.

Every policy maps the 161-D raw observation of one agent to the normalised
tactical action vector (nav in [-1, 1], fire gate, 7 target probabilities);
`deploy.decode_action` turns that into a world-frame command.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..algos.action_spec import N_TARGET_CLASSES
from ..deploy import load_policy


class RunPolicy:
    """A trained run directory (best.pt + norm_stats.npz + meta.json)."""

    def __init__(self, run_dir, device="cpu", deterministic=True, target_pct=90.0):
        self.model, self.norm, self.info = load_policy(run_dir, device)
        self.algo, self.device, self.deterministic = self.info["algo"], device, deterministic
        if self.info.get("action_mode") != "tactical":
            raise ValueError(f"{run_dir}: closed-loop play needs a tactical-action run")
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.act_scale = self.info["act_scale"]
        if self.algo == "dt":
            self.ctx = self.model.ctx
            self.rtg_scale, self.target_return = _dt_targets(run_dir, target_pct)
        self.reset()

    def reset(self):
        self.hist = {}

    @torch.no_grad()
    def act(self, key, obs_vec: np.ndarray, t_index: int, last_reward: float = 0.0) -> np.ndarray:
        o = self.norm.normalize(torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device))
        if self.algo != "dt":
            return self.model.act(o.unsqueeze(0), deterministic=self.deterministic)[0].cpu().numpy()
        h = self.hist.setdefault(key, dict(obs=[], act=[], rtg=[], ts=[], ret=self.target_return))
        if h["obs"]:
            h["ret"] -= last_reward              # return-to-go, decremented by realised reward
        h["obs"].append(o)
        h["act"].append(torch.zeros(self.model.act_dim, device=self.device))
        h["rtg"].append(h["ret"] / self.rtg_scale)
        h["ts"].append(min(t_index, self.model.max_timestep))
        k = min(len(h["obs"]), self.ctx)
        obs = torch.zeros(1, self.ctx, o.shape[-1], device=self.device)
        act = torch.zeros(1, self.ctx, self.model.act_dim, device=self.device)
        rtg = torch.zeros(1, self.ctx, 1, device=self.device)
        ts = torch.zeros(1, self.ctx, dtype=torch.long, device=self.device)
        mask = torch.zeros(1, self.ctx, device=self.device)
        obs[0, -k:] = torch.stack(h["obs"][-k:])
        act[0, -k:] = torch.stack(h["act"][-k:])
        rtg[0, -k:, 0] = torch.tensor(h["rtg"][-k:], device=self.device)
        ts[0, -k:] = torch.tensor(h["ts"][-k:], device=self.device)
        mask[0, -k:] = 1.0
        a = self.model.act(obs, act, rtg, ts, mask)[0]
        # feed back the action as the dataset encodes it: hard gate, target distribution
        fed = a.clone()
        fed[2] = (fed[2] > 0.5).float()
        h["act"][-1] = fed
        for key_ in ("obs", "act", "rtg", "ts"):
            del h[key_][:-self.ctx]
        return a.cpu().numpy()


def _dt_targets(run_dir, pct):
    """Return (rtg_scale, target return) from the run's rtg_targets.txt."""
    path = os.path.join(run_dir, "rtg_targets.txt")
    with open(path) as f:
        lines = f.read().splitlines()
    quantiles = json.loads(lines[1])                 # percentiles [50, 75, 90, 99]
    scale = float(lines[2].split("=")[1])
    table = dict(zip([50.0, 75.0, 90.0, 99.0], quantiles))
    if pct not in table:
        raise ValueError(f"target percentile must be one of {sorted(table)}")
    return scale, float(table[pct])


class RandomPolicy:
    """Uniform nav, coin-flip weapons-free, uniform target: the floor any policy must clear."""

    algo = "random"

    def __init__(self, seed=0, fire_rate=0.5):
        self.rng, self.fire_rate = np.random.default_rng(seed), fire_rate
        self.act_scale = np.array([7.5, 7.5] + [1.0] * (1 + N_TARGET_CLASSES), np.float32)

    def reset(self):
        pass

    def act(self, key, obs_vec, t_index, last_reward=0.0):
        p = np.zeros(N_TARGET_CLASSES, np.float32)
        p[self.rng.integers(N_TARGET_CLASSES)] = 1.0
        return np.concatenate([self.rng.uniform(-1, 1, 2),
                               [float(self.rng.random() < self.fire_rate)], p]).astype(np.float32)
