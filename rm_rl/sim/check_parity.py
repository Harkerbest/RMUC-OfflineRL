"""Check that the live observation and reward equal the offline ones, row by row.

Plays a short Arena-1 match with a random tactical policy on the infantry and
keeps the 161-D observation each agent was actually given.  The match is then
written to SQLite with the exporter's schema, read back with
`build_dataset.load_game_arrays`, and every row is rebuilt with
`features.build_obs` / `reward.compute_reward`.  Both must agree.

    python -m rm_rl.sim.check_parity --duration 60 [--vis-map data/arena_vis_map.npz]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile

import numpy as np

from ..data import schema as S
from ..data.build_dataset import load_game_arrays
from ..data.features import build_obs, obs_feature_names
from ..data.team_prior import DEFAULTS
from ..data.vis_map import VisibilityMap
from .arena import TEAM_CAMP, load_arena
from .episode import EpisodeConfig, run_episode
from .export_sqlite import _create, _insert
from .policies import RandomPolicy
from .referee_log import EV_COLUMNS, MATCH_COLUMNS, TS_COLUMNS


class _Random:
    def __init__(self, seed):
        self.p = RandomPolicy(seed)
        self.act_scale = self.p.act_scale

    def __call__(self, reqs):
        return {r["key"]: (self.p.act(r["key"], r["obs"], r["t"]), None) for r in reqs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--team", default="blue", help="blue also exercises the mirror")
    ap.add_argument("--vis-map", default=None)
    args = ap.parse_args()

    arena = load_arena(args.arena)
    vmap = VisibilityMap.load(args.vis_map)
    cfg = EpisodeConfig(team=args.team, seed=args.seed, duration=args.duration)
    out = run_episode(arena, cfg, _Random(args.seed), vis_map=vmap, keep_log=True)
    log, res = out["log"], out["result"]
    print(f"[parity] played {res['time']:.0f}s as {args.team}: winner {res['winner']}, "
          f"{res['decisions']} decisions, {res['decide_ms']} ms/decision")

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "parity.sqlite")
        con = sqlite3.connect(db)
        _create(con)
        _insert(con, "matches", MATCH_COLUMNS, [log.match_row(res["winner"], res["time"], 1)])
        _insert(con, "timeseries", TS_COLUMNS, log.ts_rows)
        _insert(con, "events", EV_COLUMNS, log.ev_rows)
        con.commit()
        game = load_game_arrays(con, log.game_id)
        con.close()

    from ..data.reward import RewardConfig, compute_reward
    camp = TEAM_CAMP[args.team]
    winner = TEAM_CAMP.get(res["winner"]) if res["winner"] else None
    names = obs_feature_names()
    worst_obs, worst_rew, n = 0.0, 0.0, 0
    team_feat = np.asarray(list(DEFAULTS) * 2, np.float32)
    for slot, rtype in cfg.controlled.items():
        steps = out["steps"][f"{args.team}-{slot}"]
        off = build_obs(game, camp, rtype, vis_map=vmap, team_feat=team_feat)
        rew, _ = compute_reward(game, camp, rtype, winner, cfg.reward)
        for s in steps:
            d = np.abs(off[s["t"]] - s["obs"])
            if d.max() > worst_obs:
                worst_obs = float(d.max())
                print(f"  obs mismatch {rtype} t={s['t']}: {names[int(d.argmax())]} "
                      f"offline {off[s['t']][d.argmax()]:.6f} online {s['obs'][d.argmax()]:.6f}")
            worst_rew = max(worst_rew, abs(float(rew[s["t"]]) - s["rew"]))
            n += 1
    print(f"[parity] {n} agent-seconds compared: max |obs diff| = {worst_obs:.2e}, "
          f"max |reward diff| = {worst_rew:.2e}")
    ok = worst_obs < 1e-5 and worst_rew < 1e-4
    print("[parity] OK" if ok else "[parity] MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
