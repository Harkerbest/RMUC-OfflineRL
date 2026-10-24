"""Paired evaluation of 2027 sentry decision makers in Arena-1.

Every *arm* plays the own team's slot-7 sentry; everything else is identical
across arms: the same grid of (own team script, opposing team script, seed,
side), the same sentry hardware (`--sentry-model`, fielded for every arm), the
same opposing sentry (`--opp-sentry`, default: the opposing script's own).  Arms
are therefore compared match by match.

Arms (NAME=SPEC, or a bare SPEC used as its own name):
    script          the own team script's own sentry logic
    bt:<tactic>     behaviour tree in the policy's action space (balanced/guard/assault)
    random          uniform legal macro actions (floor)
    RUN[:sample]    a trained run (rm_runs/sentry27/latest), greedy unless :sample

    python -m rm_rl.eval.eval_sentry27 --out rm_runs/sentry27_eval --workers 136 \\
        --arm script --arm bt:balanced --arm rl=rm_runs/sentry27/latest --seeds 2

Defaults use the held-out team scripts (train_sentry27.EVAL_TEAMS) as both
teammates and opponents.  Results stream to OUT/results.jsonl (re-runs resume);
OUT/summary.md compares every arm to `--ref` on paired matches.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
from collections import defaultdict

import numpy as np

from ..arena27.spec import OTHER
from .closed_loop import bootstrap_ci, wilson

_W = {}


def parse_arm(text):
    name, spec = text.split("=", 1) if "=" in text else (text, text)
    return dict(name=name, spec=spec)


def _init(arena_path):
    import torch
    torch.set_num_threads(1)
    from ..sim.arena import load_arena
    _W.update(arena=load_arena(arena_path), cache={})


def play(job):
    from ..arena27.deciders import make_decider
    from ..arena27.match import MatchConfig, Seat, run_match
    side, other = job["side"], OTHER[job["side"]]
    seats = {}
    if job["spec"] != "script":
        seats[(side, 7)] = Seat(make_decider(job["spec"], _W["cache"], seed=job["seed"]), label=job["arm"])
    if job["opp_spec"] != "script":
        seats[(other, 7)] = Seat(make_decider(job["opp_spec"], _W["cache"], seed=job["seed"] + 1), label="opp")
    over = {side: {"7": job["sentry_model"]}}
    if job["opp_spec"] != "script":
        over[other] = {"7": job["sentry_model"]}
    cfg = MatchConfig(red=job["own"] if side == "red" else job["opp"], blue=job["opp"] if side == "red" else job["own"],
                      seed=job["seed"], duration=job["duration"], decision_dt=job["decision_dt"],
                      enemy_info=job["enemy_info"], lineup_overrides=over)
    res = run_match(_W["arena"], cfg, seats)["result"]
    w = res["winner"]
    seat = res["seats"].get(f"{side}-7", {})
    return dict(key=job["key"], arm=job["arm"], spec=job["spec"], own=job["own"], opp=job["opp"],
                seed=job["seed"], side=side, winner=w, reason=res["reason"], time=res["time"],
                score=1.0 if w == side else 0.0 if w == other else 0.5,
                control_lead=res["control"][side] - res["control"][other],
                damage_lead=res["damage"][side] - res["damage"][other],
                base_lead=res["base"][side] - res["base"][other],
                **{f"seat_{k}": seat.get(k) for k in ("deaths", "damage", "no_ammo", "own_central")},
                wall=res["wall"])


def summarize(rows, ref):
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    pair = lambda r: (r["own"], r["opp"], r["seed"], r["side"])
    by_ref = {pair(r): r for r in rows if r["arm"] == ref}
    lines = [f"# 2027 sentry evaluation (paired against `{ref}`)", "",
             "| arm | n | score | 95% CI | W/D/L | control lead | damage lead | Δscore vs ref [95% CI] "
             "| Δcontrol vs ref [95% CI] | Δdamage vs ref [95% CI] | deaths | no-ammo | on own central |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        n = len(rs)
        sc = sum(r["score"] for r in rs)
        lo, hi = wilson(sc, n)
        w = sum(r["score"] == 1 for r in rs); d = sum(r["score"] == 0.5 for r in rs)
        diff = {k: [r[k] - by_ref[pair(r)][k] for r in rs if pair(r) in by_ref]
                for k in ("score", "control_lead", "damage_lead")}
        ci = {k: bootstrap_ci(v) for k, v in diff.items()}
        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else float("nan")
        seat = lambda k: "—" if np.isnan(mean(k)) else f"{mean(k):.2f}"
        fmt = lambda k, p: (f"{np.mean(diff[k]):+.{p}f} [{ci[k][0]:+.{p}f}, {ci[k][1]:+.{p}f}]"
                            if len(diff[k]) > 1 and a != ref else "—")
        lines.append(f"| {a} | {n} | {sc / n:.3f} | [{lo:.2f}, {hi:.2f}] | {w}/{d}/{n - w - d} "
                     f"| {mean('control_lead'):+.1f} | {mean('damage_lead'):+.0f} | {fmt('score', 3)} "
                     f"| {fmt('control_lead', 1)} | {fmt('damage_lead', 0)} | {seat('seat_deaths')} "
                     f"| {seat('seat_no_ammo')} | {seat('seat_own_central')} |")
        table[a] = dict(n=n, score=sc / n, score_ci=[lo, hi],
                        d_score=float(np.mean(diff["score"])) if diff["score"] else None,
                        d_score_ci=list(ci["score"]), d_control=float(np.mean(diff["control_lead"]))
                        if diff["control_lead"] else None, d_control_ci=list(ci["control_lead"]))
    lines += ["", "score: win 1, draw 0.5, loss 0.  Leads are own minus opponent at the end of the match.",
              "Δ columns: mean paired difference to the reference arm on the same (own, opp, seed, side), "
              "bootstrap 95% CI."]
    return "\n".join(lines) + "\n", table


def main():
    from ..train.train_sentry27 import EVAL_TEAMS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arena", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--ref", default="bt:balanced")
    ap.add_argument("--own-teams", default=EVAL_TEAMS)
    ap.add_argument("--opp-teams", default=EVAL_TEAMS)
    ap.add_argument("--opp-sentry", default="script")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--decision-dt", type=float, default=1.0)
    ap.add_argument("--enemy-info", default="own", choices=["own", "team", "full"])
    ap.add_argument("--sentry-model", type=int, default=2771)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # SIGTERM -> SystemExit, so the worker pool is torn down instead of orphaned
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    path = os.path.join(args.out, "results.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
    done = {r["key"] for r in rows}
    arms = [parse_arm(a) for a in args.arm] or [parse_arm("script"), parse_arm("bt:balanced")]
    jobs = []
    for own in args.own_teams.split(","):
        for opp in args.opp_teams.split(","):
            for k in range(args.seeds):
                for side in ("red", "blue"):
                    for arm in arms:
                        key = f"{arm['name']}|{own}|{opp}|{args.seed0 + k}|{side}|{args.opp_sentry}|{args.duration}"
                        if key not in done:
                            jobs.append(dict(key=key, arm=arm["name"], spec=arm["spec"], own=own, opp=opp,
                                             seed=args.seed0 + k, side=side, opp_spec=args.opp_sentry,
                                             duration=args.duration, decision_dt=args.decision_dt,
                                             enemy_info=args.enemy_info, sentry_model=args.sentry_model))
    if jobs and not args.summarize_only:
        print(f"[eval27] {len(jobs)} matches ({len(rows)} already done), {args.workers} workers", flush=True)
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.arena,)) as pool, \
                open(path, "a", encoding="utf-8") as f:
            for i, row in enumerate(pool.imap_unordered(play, jobs), 1):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)
                if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                    print(f"[eval27] {i}/{len(jobs)}", flush=True)
    md, table = summarize(rows, args.ref)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print(md)


if __name__ == "__main__":
    main()
