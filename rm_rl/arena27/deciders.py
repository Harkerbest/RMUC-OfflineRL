"""Decision makers for a seat: policy, behaviour tree, uniform random.

A decider is called once per decision with a batch of requests (one per seat it
plays; a mirror self-play policy plays both teams' seats in one call):
    request = dict(key, team, slot, obs, mask, inp, wps, builder)  (builder: its ObsBuilder)
and returns, per request, (head indices as an int array, extra) where extra holds
the behaviour log-probability for PPO.
"""
from __future__ import annotations

import numpy as np
import torch

from .bt import TREES, to_indices
from .spec import HEAD_SIZES


class PolicyDecider:
    def __init__(self, policy, greedy=False, label="policy", goal_dt=1.0, forbid=()):
        self.policy, self.greedy, self.label, self.goal_dt = policy, greedy, label, goal_dt
        self.forbid = tuple(forbid)

    def __call__(self, reqs):
        obs = torch.as_tensor(np.stack([r["obs"] for r in reqs]))
        mask = torch.as_tensor(np.stack([r["mask"] for r in reqs]))
        a, logp = self.policy.act(obs, mask, greedy=self.greedy)
        return [(a[i].numpy(), {"logp": float(logp[i])}) for i in range(len(reqs))]


class TreeDecider:
    goal_dt, forbid = 1.0, ()     # trees decide freely every second (their natural mode)

    def __init__(self, tactic="balanced"):
        if tactic not in TREES:
            raise ValueError(f"unknown tree {tactic!r} (one of {tuple(TREES)})")
        self.tree, self.label = TREES[tactic], f"bt:{tactic}"

    def __call__(self, reqs):
        return [(np.asarray(to_indices(self.tree(r["inp"], r["wps"], last_supply=r["builder"].last_supply),
                                       r["mask"])), {"logp": 0.0}) for r in reqs]


class RandomDecider:
    label, goal_dt, forbid = "random", 1.0, ()

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, reqs):
        out = []
        for r in reqs:
            idx, off = [], 0
            for n in HEAD_SIZES:
                allowed = np.flatnonzero(r["mask"][off:off + n])
                idx.append(int(self.rng.choice(allowed)))
                off += n
            out.append((np.asarray(idx), {"logp": 0.0}))
        return out


def make_decider(spec: str, cache: dict | None = None, greedy=True, seed=0):
    """'bt:<tactic>' | 'random' | path to a run dir / policy.pt (':sample' to sample)."""
    from ..algos.macro_ppo import load_policy
    if spec.startswith("bt:"):
        return TreeDecider(spec[3:])
    if spec == "random":
        return RandomDecider(seed)
    path, sample = (spec[:-7], True) if spec.endswith(":sample") else (spec, False)
    cache = cache if cache is not None else {}
    if path not in cache:
        cache[path] = (load_policy(path), run_spec(path))
    pol, rs = cache[path]
    return PolicyDecider(pol, greedy=greedy and not sample, label=spec, goal_dt=rs.get("goal_dt", 1.0),
                         forbid=rs.get("forbid_goals", ()))


def run_spec(path: str) -> dict:
    """spec.json of a run directory, checked against this code's action/feature spec."""
    import json
    import os
    from .spec import SPEC_VERSION
    d = path if os.path.isdir(path) else os.path.dirname(path)
    f = os.path.join(d, "spec.json")
    if not os.path.exists(f):
        return {}
    with open(f, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("spec_version", 1) != SPEC_VERSION:
        raise SystemExit(f"{path} was trained with action/feature spec v{spec.get('spec_version', 1)}; this code "
                         f"is v{SPEC_VERSION}.  Evaluate it from the commit it was trained with "
                         f"(v1: a3ced70).")
    return spec
