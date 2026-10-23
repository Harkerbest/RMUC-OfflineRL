"""Play scripted Arena-1 matches and save them as an official-schema SQLite log.

The output is a drop-in replacement for `rmuc_2026_region_dataset.sqlite`, so
the unchanged offline pipeline trains on it:

    python -m rm_rl.sim.export_sqlite --out data/arena_sim.sqlite --games 64 --workers 3
    python -m rm_rl.data.vis_map    --db data/arena_sim.sqlite --out data/arena_vis_map.npz
    python -m rm_rl.data.team_prior --db data/arena_sim.sqlite --out data/arena_team_prior.json
    python -m rm_rl.data.build_dataset --db data/arena_sim.sqlite --out data/arena_infantry_tactical \\
        --agent infantry --action-mode tactical --goal-horizon 5 \\
        --vis-map data/arena_vis_map.npz --team-prior data/arena_team_prior.json

Every Arena-1 strategy plays the role of a "school".  Matches are scheduled as
a round robin over ordered pairs (so each strategy plays both sides), cycling
through seeds; the file is appended to after every match, and re-running with
the same arguments resumes where it stopped.
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import sqlite3
import sys
import time

from .arena import ArenaMatch, ScriptedSide, load_arena
from .referee_log import EV_COLUMNS, MATCH_COLUMNS, TS_COLUMNS, RefereeLog

# Arena-1 RMUC 2026 full-team examples that play a whole match under the current
# rules API (siege_* field their own registered infantry hardware; the rest the
# default).  rush / vanguard / bastion / mixed-squad issue retired operations.
DEFAULT_STRATEGIES = [
    "claude_opus55_xhigh_team", "claude_opus55_xhigh_hold", "claude_opus55_xhigh_squad",
    "claude_opus55_xhigh_distributed", "siege_team", "siege_assault_team",
    "siege_blitz", "siege_economy", "siege_flank", "siege_guard",
]


def schedule(strategies, games, seed0=0):
    pairs = [(a, b) for a, b in itertools.permutations(strategies, 2)]
    out = []
    for i in range(games):
        red, blue = pairs[i % len(pairs)]
        out.append(dict(game_id=i + 1, red=red, blue=blue, seed=seed0 + i))
    return out


def play(job, arena_path=None, duration=420.0, flip_y=False):
    """One match -> (match_row, ts_rows, ev_rows, summary)."""
    arena = load_arena(arena_path)
    red = ScriptedSide(arena, f"examples/{job['red']}", "red")
    blue = ScriptedSide(arena, f"examples/{job['blue']}", "blue")
    m = ArenaMatch(arena, red, blue, seed=job["seed"], duration=duration)
    log = RefereeLog(job["game_id"], {"red": job["red"], "blue": job["blue"]}, flip_y=flip_y)
    t0 = time.time()
    try:
        while not m.finished:
            m.decide()
            for f in m.advance():
                log.on_frame(f)
        f = m.frame
    finally:
        m.close()
    summary = dict(game_id=job["game_id"], red=job["red"], blue=job["blue"], seed=job["seed"],
                   winner=f["winner"], reason=f["reason"], time=f["time"],
                   wall=round(time.time() - t0, 1),
                   base={t: f["bases"][t]["hp"] for t in ("red", "blue")},
                   damage={t: f["teams"][t].get("attack_damage", 0.0) for t in ("red", "blue")})
    return log.match_row(f["winner"], f["time"], job["game_id"]), log.ts_rows, log.ev_rows, summary


def _worker(args):
    job, arena_path, duration, flip_y = args
    try:
        return play(job, arena_path, duration, flip_y), None
    except Exception as exc:                      # keep the pool alive; report and skip
        return None, f"game {job['game_id']} ({job['red']} vs {job['blue']}): {exc!r}"


def _create(con):
    q = lambda cols: ", ".join(f'"{c}"' for c in cols)
    con.execute(f"CREATE TABLE IF NOT EXISTS matches ({q(MATCH_COLUMNS)})")
    con.execute(f"CREATE TABLE IF NOT EXISTS timeseries ({q(TS_COLUMNS)})")
    con.execute(f"CREATE TABLE IF NOT EXISTS events ({q(EV_COLUMNS)})")
    con.execute('CREATE INDEX IF NOT EXISTS ts_game ON timeseries(game_id)')
    con.execute('CREATE INDEX IF NOT EXISTS ev_game ON events(game_id)')


def _insert(con, table, cols, rows):
    ph = ", ".join("?" * len(cols))
    con.executemany(f"INSERT INTO {table} VALUES ({ph})", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default=None, help="Arena-1 checkout (default: $ARENA1_ROOT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=56)
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--flip-y", action="store_true",
                    help="write y as 15 - y (only if the target dataset's y axis is mirrored)")
    args = ap.parse_args()
    load_arena(args.arena)                         # fail fast if the engine is missing

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    con = sqlite3.connect(args.out)
    _create(con)
    done = {r[0] for r in con.execute("SELECT game_id FROM matches")}
    jobs = [j for j in schedule(args.strategies.split(","), args.games, args.seed0)
            if j["game_id"] not in done]
    print(f"[export] {len(done)} games already in {args.out}; playing {len(jobs)} "
          f"with {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        it = pool.imap_unordered(_worker, [(j, args.arena, args.duration, args.flip_y) for j in jobs])
        for n, (res, err) in enumerate(it, 1):
            if err:
                print(f"[export] FAILED {err}", file=sys.stderr, flush=True)
                continue
            match_row, ts, ev, s = res
            _insert(con, "matches", MATCH_COLUMNS, [match_row])
            _insert(con, "timeseries", TS_COLUMNS, ts)
            _insert(con, "events", EV_COLUMNS, ev)
            con.commit()
            print(f"[export] {n}/{len(jobs)} game {s['game_id']}: {s['red']} vs {s['blue']} "
                  f"-> {s['winner']} ({s['reason']}) base {s['base']} dmg {s['damage']} "
                  f"wall {s['wall']}s  elapsed {time.time() - t0:.0f}s", flush=True)
    con.close()


if __name__ == "__main__":
    main()
