"""Run a trained 2027 decision layer on the robot.

    rt = SentryRuntime("rm_runs/sentry27/latest", team="red", slot=7)
    cmd = rt.decide(snapshot)       # once per spec["decision_dt"] seconds
    # cmd = dict(goal="own_central", goal_xy=(10.8, 7.5), target="nearest",
    #            posture="keep", enhance="off", supply="none", source="policy")

`snapshot` follows `inputs.py`'s schema, filled from the referee system and the
robot's own perception (docs/sentry27/README.md lists each field's source).  The
same ObsBuilder that trained the policy turns it into features, so the robot
computes exactly what training computed — the parity property the closed-loop
work relied on.

The policy runs in numpy (`export_numpy` writes policy.npz next to policy.pt), so
the robot needs no torch.  Any failure — a malformed snapshot, a non-finite
feature, an exception — falls back to the behaviour tree for that decision and
is counted in `rt.fallbacks`; the navigation stack and auto-aim keep their own
safety limits regardless of what the decision layer asks for.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .bt import TREES, to_indices
from .obs import ObsBuilder
from .spec import HEADS, HEAD_SIZES


def export_numpy(run_dir: str) -> str:
    """policy.pt -> policy.npz (weights + observation normaliser)."""
    from ..algos.macro_ppo import load_policy
    pol = load_policy(run_dir)
    arrays = {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}
    arrays["_config"] = np.frombuffer(json.dumps(pol.config).encode(), np.uint8)
    out = os.path.join(run_dir, "policy.npz")
    np.savez(out, **arrays)
    return out


class NumpyPolicy:
    """Greedy forward pass of `macro_ppo.MultiHeadPolicy` in numpy."""

    def __init__(self, path: str):
        z = np.load(path)
        self.cfg = json.loads(bytes(z["_config"]).decode())
        self.w = {k: z[k] for k in z.files if k != "_config"}
        self.mean, self.std = self.w["obs_norm.mean"], np.sqrt(self.w["obs_norm.var"]) + 1e-6
        # body = mlp(...) + LayerNorm + ReLU: Linear/LayerNorm pairs in index order
        idx = sorted({int(k.split(".")[2]) for k in self.w if k.startswith("body.0.")})
        self.layers = [(f"body.0.{i}", "linear" if f"body.0.{i}.weight" in self.w and
                        self.w[f"body.0.{i}.weight"].ndim == 2 else "norm") for i in idx]

    @staticmethod
    def _ln(x, g, b):
        m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * g + b

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> list[int]:
        x = np.clip((obs - self.mean) / self.std, -5.0, 5.0)
        n = len(self.layers)
        for j, (name, kind) in enumerate(self.layers):
            if kind == "linear":
                x = x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]
            else:
                x = np.maximum(self._ln(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"]), 0.0)
        x = np.maximum(self._ln(x, self.w["body.1.weight"], self.w["body.1.bias"]), 0.0)
        logits = x @ self.w["head.weight"].T + self.w["head.bias"]
        logits = np.where(mask, logits, -1e9)
        out, off = [], 0
        for size in HEAD_SIZES:
            out.append(int(np.argmax(logits[off:off + size])))
            off += size
        return out


class SentryRuntime:
    def __init__(self, run_dir: str, team: str, slot: int = 7, tree: str = "balanced"):
        with open(os.path.join(run_dir, "spec.json"), encoding="utf-8") as f:
            self.spec = json.load(f)
        npz = os.path.join(run_dir, "policy.npz")
        self.policy = NumpyPolicy(npz if os.path.exists(npz) else export_numpy(run_dir))
        self.wps = {k: tuple(v) for k, v in self.spec["waypoints"][team].items()}
        self.builder = ObsBuilder(team, slot, self.wps)
        self.tree, self.fallbacks, self.decisions = tree, 0, 0

    def decide(self, snapshot: dict) -> dict:
        self.decisions += 1
        source = "policy"
        try:
            obs = self.builder(snapshot)
            if self.builder.names != self.spec["obs_names"]:
                raise ValueError("feature layout differs from the trained run")
            mask = self.builder.mask(snapshot)
            if not np.all(np.isfinite(obs)):
                raise ValueError("non-finite feature")
            idx = self.policy(obs, mask)
        except Exception:
            self.fallbacks += 1
            source = "tree"
            mask = self.builder.mask(snapshot)
            idx = to_indices(TREES[self.tree](snapshot, self.wps, last_supply=self.builder.last_supply), mask)
        cmd = {h: choices[i] for (h, choices), i in zip(HEADS, idx)}
        cmd["goal_xy"] = self.wps.get(cmd["goal"])
        cmd["source"] = source
        return cmd
