"""Pre-flight checks before a long run: python -m rm_rl.arena27.selftest [--run RUN_DIR]

1. Arena-1 2027 loads; observation / critic sizes.
2. A short match, tree vs random sentries: trajectories finite, every recorded
   action legal under its mask, seat statistics present.
3. With --run: the numpy runtime (robot side) reproduces the torch policy's greedy
   actions on that match's observations, and SentryRuntime decides on a live
   snapshot without falling back to the tree.
Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default=None)
    ap.add_argument("--run", default=None, help="trained run dir (policy.pt + spec.json)")
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()
    from ..sim.arena import load_arena
    from .deciders import RandomDecider, TreeDecider
    from .match import MatchConfig, Seat, probe_spec, run_match
    from .spec import HEAD_SIZES

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        ok &= bool(cond)

    arena = load_arena(args.arena)
    spec = probe_spec(arena)
    check(spec["obs_dim"] > 0 and spec["gdim"] > spec["obs_dim"], f"probe: obs={spec['obs_dim']} g={spec['gdim']}")
    seats = {("red", 7): Seat(TreeDecider("balanced"), record=True, label="bt"),
             ("blue", 7): Seat(RandomDecider(0), record=True, label="random")}
    out = run_match(arena, MatchConfig(red="claude_opus55_xhigh_2027_team", blue="gpt61_2027_control", seed=1,
                                       duration=args.duration, global_state=True), seats)
    for k, tr in out["trajs"].items():
        check(all(np.isfinite(tr[n]).all() for n in ("obs", "g", "rew")), f"{k}: finite obs/g/rew ({len(tr['obs'])} steps)")
        legal, off = np.ones(len(tr["act"]), bool), 0
        for h, n in enumerate(HEAD_SIZES):
            legal &= tr["mask"][np.arange(len(tr["act"])), off + tr["act"][:, h]]
            off += n
        check(legal.all(), f"{k}: every action legal under its mask")
    check(set(out["result"]["seats"]) == {"red-7", "blue-7"}, "seat statistics reported")

    if args.run:
        import torch
        from ..algos.macro_ppo import load_policy
        from .deploy import NumpyPolicy, SentryRuntime, export_numpy
        from .inputs import snapshot_from_arena
        from ..sim.arena import ArenaMatch, ScriptedSide
        pol = load_policy(args.run)
        npol = NumpyPolicy(export_numpy(args.run))
        tr = out["trajs"][("red", 7)]
        a, _ = pol.act(torch.as_tensor(tr["obs"]), torch.as_tensor(tr["mask"]), greedy=True)
        same = np.mean([list(a[i].numpy()) == npol(tr["obs"][i], tr["mask"][i]) for i in range(len(tr["obs"]))])
        check(same == 1.0, f"numpy runtime == torch greedy on {len(tr['obs'])} decisions ({same:.3f})")
        side = lambda t: ScriptedSide(arena, "examples/claude_opus55_xhigh_2027_team", t)
        m = ArenaMatch(arena, side("red"), side("blue"), seed=2, duration=5.0, map_name="rmuc2027")
        try:
            obs = m.decide()
            rt = SentryRuntime(args.run, "blue", 7)
            cmd = rt.decide(snapshot_from_arena(obs["blue"], 7, "own"))
        finally:
            m.close()
        check(rt.fallbacks == 0 and cmd["source"] == "policy", f"SentryRuntime decides from a live snapshot: {cmd}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
