"""Closed-loop evaluation in the Arena-1 simulator.

Offline estimators cannot rank policies whose behaviour differs from the log
(README "评估的局限"): FQE only ever penalises deviation.  Here every policy
actually plays.  Each *arm* takes over the three infantry (slots 3/4/7) of a
team whose other robots run a fixed script, against a pool of scripted
opponents, over several seeds and both sides; the same (opponent, seed, side)
grid is used for every arm, so arms are compared on paired matches.

Arms:
    script                 the team script drives the infantry too (reference)
    random                 uniform tactical commands (floor)
    NAME=RUN_DIR[:FLAG...] a trained tactical run (BC / IQL / DT / MAPPO export);
                           ':vision' gates enemy positions by Arena's camera model,
                           ':sample' draws the weapons-free gate from p_fire instead
                           of thresholding it at 0.5 (see sim.policies.RunPolicy)

    python -m rm_rl.eval.closed_loop \\
        --arm script --arm random \\
        --arm bc=rm_runs/infantry_bc_tactical --arm iql=rm_runs/infantry_iql_tactical \\
        --arm dt=rm_runs/infantry_dt_tactical \\
        --opponents claude_opus55_xhigh_hold,claude_opus55_xhigh_team \\
        --seeds 3 --vis-map data/vis_map.npz --workers 4 --out rm_runs/closed_loop

Results stream to OUT/results.jsonl (a re-run resumes); OUT/summary.md holds the
table.  `--summarize-only` rebuilds the table from the JSONL.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import numpy as np

DEFAULT_OPPONENTS = "claude_opus55_xhigh_hold,claude_opus55_xhigh_team"


# --------------------------------------------------------------------------- arms
def parse_arm(spec: str, default_info: str):
    if "=" not in spec:
        if spec not in ("script", "random"):
            raise SystemExit(f"unknown arm {spec!r}")
        return dict(name=spec, kind=spec, run=None, enemy_info=default_info,
                    gate="sample" if spec == "random" else "script")
    name, rest = spec.split("=", 1)
    arm = dict(name=name, kind="run", enemy_info=default_info, gate="threshold")
    while rest.rsplit(":", 1)[-1] in ("vision", "full", "sample", "threshold"):
        rest, flag = rest.rsplit(":", 1)
        arm["gate" if flag in ("sample", "threshold") else "enemy_info"] = flag
    arm["run"] = rest
    return arm


class PolicyAgent:
    def __init__(self, policy):
        self.policy, self.act_scale = policy, policy.act_scale

    def __call__(self, reqs):
        return {r["key"]: (self.policy.act(r["key"], r["obs"], r["t"], r["last_reward"]), None)
                for r in reqs}


_CACHE = {}


def _setup(arena_path, vis_map_path):
    if "arena" not in _CACHE:
        import torch
        torch.set_num_threads(1)
        from ..sim.arena import load_arena
        from ..data.vis_map import VisibilityMap
        _CACHE["arena"] = load_arena(arena_path)
        _CACHE["vmap"] = VisibilityMap.load(vis_map_path)
    return _CACHE["arena"], _CACHE["vmap"]


def play(job):
    """One (arm, opponent, seed, side) match in a worker process."""
    from ..sim.episode import EpisodeConfig, run_episode
    from ..data.reward import RewardConfig
    from ..sim.policies import RandomPolicy, RunPolicy
    arena, vmap = _setup(job["arena"], job["vis_map"])
    arm = job["arm"]
    cfg = EpisodeConfig(team=job["side"], own=f"examples/{job['own']}",
                        opponent=f"examples/{job['opponent']}", seed=job["seed"],
                        duration=job["duration"], enemy_info=arm["enemy_info"])
    if arm["kind"] == "script":
        cfg.controlled, agent = {}, None
    elif arm["kind"] == "random":
        agent = PolicyAgent(RandomPolicy(job["seed"]))
    else:
        pol = RunPolicy(arm["run"], gate=arm["gate"], seed=job["seed"])
        if pol.meta.get("vis_map") and vmap is None:
            raise SystemExit(f"{arm['run']} was trained with a vis_map; pass --vis-map")
        cfg.use_team_prior = bool(pol.meta.get("team_prior"))
        # the run's own reward weights: DT conditions on returns measured in them
        cfg.reward = RewardConfig.from_dict(pol.meta.get("reward", {}))
        agent = PolicyAgent(pol)
    try:
        out = run_episode(arena, cfg, agent, vis_map=vmap)
    except Exception as exc:
        return dict(job_key=job["key"], error=repr(exc))
    res = out["result"]
    res.update(job_key=job["key"], arm=arm["name"], side=job["side"], enemy_info=arm["enemy_info"],
               gate=arm["gate"], run=arm["run"])
    return res


# --------------------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bootstrap_ci(x, n_boot=4000, seed=0):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows, ref_arms=("script", "bc")):
    by_arm = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by_arm[r["arm"]].append(r)
    lines = ["| arm | n | W-D-L | win rate (draw=½) | 95% CI | Δdamage | Δbase HP | Δoutpost HP | Δkills | ms/decision |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for arm, rs in by_arm.items():
        n = len(rs)
        w = sum(r["win"] == 1.0 for r in rs)
        d = sum(r["win"] == 0.5 for r in rs)
        score = sum(r["win"] for r in rs)
        lo, hi = wilson(score, n)
        f = lambda key, other: float(np.mean([r[key] - r[other] for r in rs]))
        table[arm] = dict(n=n, wins=w, draws=d, losses=n - w - d, win_rate=score / n, ci=(lo, hi),
                          d_damage=f("damage", "damage_enemy"), d_base=f("base", "base_enemy"),
                          d_outpost=f("outpost", "outpost_enemy"), d_kills=f("kills", "kills_enemy"),
                          decide_ms=float(np.mean([r["decide_ms"] for r in rs])))
        t = table[arm]
        lines.append(f"| {arm} | {n} | {w}-{d}-{n - w - d} | {t['win_rate']:.3f} | [{lo:.2f}, {hi:.2f}] | "
                     f"{t['d_damage']:+.0f} | {t['d_base']:+.0f} | {t['d_outpost']:+.0f} | "
                     f"{t['d_kills']:+.2f} | {t['decide_ms']:.1f} |")
    # paired comparisons on identical (opponent, seed, side) matches
    key = lambda r: (r["opponent"], r["seed"], r["side"])
    idx = {arm: {key(r): r for r in rs} for arm, rs in by_arm.items()}
    pair_lines = ["", "Paired differences (same opponent, seed and side; 95% bootstrap CI):", "",
                  "| arm − reference | pairs | Δwin | CI | Δ(damage dealt − taken) | CI |",
                  "|---|---|---|---|---|---|"]
    pairs = {}
    for ref in ref_arms:
        if ref not in idx:
            continue
        for arm in idx:
            if arm == ref:
                continue
            common = sorted(set(idx[arm]) & set(idx[ref]))
            if not common:
                continue
            dw = [idx[arm][k]["win"] - idx[ref][k]["win"] for k in common]
            dd = [(idx[arm][k]["damage"] - idx[arm][k]["damage_enemy"])
                  - (idx[ref][k]["damage"] - idx[ref][k]["damage_enemy"]) for k in common]
            cw, cd = bootstrap_ci(dw), bootstrap_ci(dd)
            pairs[f"{arm}-{ref}"] = dict(n=len(common), d_win=float(np.mean(dw)), ci_win=cw,
                                         d_damage=float(np.mean(dd)), ci_damage=cd)
            pair_lines.append(f"| {arm} − {ref} | {len(common)} | {np.mean(dw):+.3f} | "
                              f"[{cw[0]:+.2f}, {cw[1]:+.2f}] | {np.mean(dd):+.0f} | "
                              f"[{cd[0]:+.0f}, {cd[1]:+.0f}] |")
    return table, pairs, "\n".join(lines + pair_lines)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default=None)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--own", default="claude_opus55_xhigh_team",
                    help="team script driving the non-infantry robots (and the 'script' arm)")
    ap.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--sides", default="red,blue")
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--enemy-info", default="full", choices=["full", "vision"])
    ap.add_argument("--vis-map", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    if not args.summarize_only:
        arms = [parse_arm(a, args.enemy_info) for a in (args.arm or ["script", "random"])]
        done = {r["job_key"] for r in rows if "error" not in r}
        jobs = []
        for arm in arms:
            for opp in args.opponents.split(","):
                for s in range(args.seed0, args.seed0 + args.seeds):
                    for side in args.sides.split(","):
                        k = f"{arm['name']}|{opp}|{s}|{side}"
                        if k not in done:
                            jobs.append(dict(key=k, arm=arm, opponent=opp, own=args.own, seed=s,
                                             side=side, duration=args.duration,
                                             arena=args.arena, vis_map=args.vis_map))
        print(f"[closed-loop] {len(done)} matches done, {len(jobs)} to play on {args.workers} workers",
              flush=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool, open(path, "a", encoding="utf-8") as f:
            for n, res in enumerate(pool.imap_unordered(play, jobs), 1):
                if "error" in res:
                    print(f"[closed-loop] ERROR {res['job_key']}: {res['error']}", file=sys.stderr, flush=True)
                    continue
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(res)
                print(f"[closed-loop] {n}/{len(jobs)} {res['job_key']}: win {res['win']} "
                      f"dmg {res['damage']:.0f}/{res['damage_enemy']:.0f} base {res['base']:.0f}/"
                      f"{res['base_enemy']:.0f} ({res['wall']}s, {time.time() - t0:.0f}s elapsed)",
                      flush=True)
    table, pairs, md = summarize(rows)
    print(md)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(arms=table, paired=pairs), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
