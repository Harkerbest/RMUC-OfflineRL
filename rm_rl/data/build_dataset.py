"""Build an offline-RL dataset from the RMUC 2026 referee SQLite file.

Pipeline
--------
1. read `matches` -> winner per game + per-school win rate (a "team strength"
   signal used for optional quality weighting).
2. for every game, load `timeseries`, align every robot onto a 1-second grid,
   and assemble `GameArrays`.
3. for each side (red / blue, blue mirrored into the red frame) build the ego
   agent's (obs, action, reward, done, valid) transition trajectory.
4. fit normalisation stats (obs mean/std, per-dim action scale) and write
   compressed train / val shards + `norm_stats.npz` + `meta.json`.

Run:
    python -m rm_rl.data.build_dataset \
        --db /path/rmuc_2026_region_dataset.sqlite \
        --out data/sentry --agent 哨兵 --config configs/sentry_iql.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(x, **k):
        return x

from . import schema as S
from .features import GameArrays, Entity, build_obs, build_action_raw, obs_dim, action_dim
from .reward import RewardConfig, compute_reward
from .team_prior import TeamPrior
from .vis_map import VisibilityMap
from ..algos.action_spec import get_spec

# columns we actually read from timeseries, mapped to ascii
_TS_FIELDS = ["x", "y", "z", "hp", "maxhp", "yaw", "power", "heat17",
              "heat17_max", "heat42_max", "ammo17", "coin_left", "coin_total",
              "vuln"]


# ---------------------------------------------------------------------------
def load_matches(con) -> pd.DataFrame:
    inv = {v: k for k, v in S.MATCH_COLS.items()}
    cols = ", ".join(f'"{inv[a]}" AS {a}' for a in S.MATCH_COLS.values())
    return pd.read_sql(f"SELECT {cols} FROM {S.T_MATCHES}", con)


def team_strength(matches: pd.DataFrame) -> Dict[str, float]:
    """Per-school win rate, normalised to a weight around 1.0."""
    wins, games = {}, {}
    for _, m in matches.iterrows():
        red, blue, win = m["red_school"], m["blue_school"], m["winner"]
        for s in (red, blue):
            games[s] = games.get(s, 0) + 1
        if win == S.CAMP_RED:
            wins[red] = wins.get(red, 0) + 1
        elif win == S.CAMP_BLUE:
            wins[blue] = wins.get(blue, 0) + 1
    rate = {s: wins.get(s, 0) / max(games[s], 1) for s in games}
    mean = np.mean(list(rate.values())) if rate else 0.5
    mean = max(mean, 1e-3)
    return {s: float(np.clip(r / mean, 0.5, 2.0)) for s, r in rate.items()}


def load_game_arrays(con, game_id: int) -> GameArrays:
    inv = {v: k for k, v in S.TS_COLS.items()}
    sel = ", ".join(f'"{inv[a]}" AS {a}' for a in S.TS_COLS.values())
    df = pd.read_sql(
        f"SELECT {sel} FROM {S.T_TIMESERIES} WHERE game_id = ?",
        con, params=[int(game_id)])
    return game_arrays_from_df(df)


def game_arrays_from_df(df: pd.DataFrame) -> GameArrays:
    """Timeseries rows (ASCII column aliases) -> GameArrays on the 1 s grid.

    Shared by the SQLite loader and the simulator bridge (rm_rl.sim), so an
    observation built online from a live match goes through exactly the same
    alignment, clipping and forward-fill as one built from the official log.
    """
    if df.empty:
        return GameArrays(0, {})
    df["t"] = df["t"].round().astype(int)
    T = int(df["t"].max())
    tidx = np.arange(1, T + 1)

    # clip radar-noise positions into a slightly padded field box
    df["x"] = df["x"].clip(-2.0, S.FIELD_X + 2.0)
    df["y"] = df["y"].clip(-2.0, S.FIELD_Y + 2.0)

    ent: Dict[int, Entity] = {}
    for rid, sub in df.groupby("robot_id"):
        sub = sub.sort_values("t").drop_duplicates("t", keep="last")
        sub = sub.set_index("t").reindex(tidx)
        # persist last-known value forward only (leading NaN => absent => 0)
        sub[_TS_FIELDS] = sub[_TS_FIELDS].ffill()
        sub = sub.fillna(0.0)
        hp = sub["hp"].to_numpy(np.float32)
        maxhp = sub["maxhp"].to_numpy(np.float32)
        maxhp = np.where(maxhp <= 0, np.nanmax(maxhp) if np.any(maxhp > 0) else 1.0, maxhp)
        ent[int(rid)] = Entity(
            x=sub["x"].to_numpy(np.float32), y=sub["y"].to_numpy(np.float32),
            z=sub["z"].to_numpy(np.float32), hp=hp, maxhp=maxhp,
            yaw=sub["yaw"].to_numpy(np.float32),
            power=sub["power"].to_numpy(np.float32),
            heat17=sub["heat17"].to_numpy(np.float32),
            heat17_max=sub["heat17_max"].to_numpy(np.float32),
            heat42_max=sub["heat42_max"].to_numpy(np.float32),
            ammo17=sub["ammo17"].to_numpy(np.float32),
            coin_left=sub["coin_left"].to_numpy(np.float32),
            coin_total=sub["coin_total"].to_numpy(np.float32),
            vuln=sub["vuln"].to_numpy(np.float32),
            alive=(hp > 0).astype(np.float32),
        )
    return GameArrays(T, ent)


def build_game_trajectories(game: GameArrays, agent_types, winner: str,
                            rcfg: RewardConfig, min_len: int,
                            strength: Dict[str, float], schools: Dict[str, str],
                            vis_radius: float = 0.0, vis_dropout: float = 0.0,
                            rng=None, action_mode: str = "velocity",
                            goal_horizon: int = 5, vis_map=None,
                            team_prior=None, game_id: int = 0):
    """Yield one trajectory dict per (agent type, side) that fields the agent."""
    out = []
    if game.T < min_len + 1:
        return out
    if isinstance(agent_types, str):
        agent_types = [agent_types]
    for agent_type in agent_types:
        out.extend(_traj_for_type(
            game, agent_type, winner, rcfg, min_len, strength, schools,
            vis_radius, vis_dropout, rng, action_mode, goal_horizon,
            vis_map, team_prior, game_id))
    return out


def _traj_for_type(game, agent_type, winner, rcfg, min_len, strength, schools,
                   vis_radius, vis_dropout, rng, action_mode, goal_horizon,
                   vis_map, team_prior, game_id):
    out = []
    for camp in S.CAMPS:
        ego = game.get(agent_type, camp)
        if ego.alive.sum() < min_len * 0.3:   # agent barely present -> skip
            continue
        tfeat = team_prior.feats(game_id, camp) if team_prior is not None else None
        obs = build_obs(game, camp, agent_type, vis_radius=vis_radius,
                        vis_dropout=vis_dropout, rng=rng,
                        vis_map=vis_map, team_feat=tfeat)  # [T, od]
        act = build_action_raw(game, camp, agent_type,
                               action_mode=action_mode,
                               goal_horizon=goal_horizon)   # [T-1, 4]
        rew, comp = compute_reward(game, camp, agent_type, winner, rcfg)

        cur = obs[:-1]
        nxt = obs[1:]
        valid = (ego.alive[:-1] > 0).astype(np.float32)
        done = np.zeros(len(rew), dtype=np.float32)
        done[-1] = 1.0
        w = strength.get(schools.get(camp, ""), 1.0)
        out.append(dict(
            obs=cur.astype(np.float32), next_obs=nxt.astype(np.float32),
            act=act.astype(np.float32), rew=rew.astype(np.float32),
            done=done, valid=valid,
            ret=float(rew.sum()), win=1.0 if winner == camp else 0.0,
            weight=float(w), camp=0 if camp == S.CAMP_RED else 1,
            agent_type=agent_type,
        ))
    return out


# ---------------------------------------------------------------------------
def main():
    # UTF-8 console so printing Chinese (agent name / meta) never crashes on Windows
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="infantry",
                    help="ASCII alias (sentry/hero/engineer/infantry3/infantry4/"
                         "aerial), the Chinese 机器人类型, the group 'infantry' "
                         "(= 步兵3+步兵4, pooled), or a comma-separated list. "
                         "Default 'infantry': those are the human-driven slots "
                         "and are structurally identical to the sentry, so they "
                         "are the right demonstrations for a sentry policy.")
    ap.add_argument("--config", default=None, help="yaml with a 'reward:' block")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--limit-games", type=int, default=0,
                    help="0 = all; >0 for a quick smoke test")
    ap.add_argument("--vis-radius", type=float, default=0.0,
                    help="metres; >0 hides enemy positions beyond this range "
                         "(realistic sentry perception). 0 = full referee vision")
    ap.add_argument("--vis-dropout", type=float, default=0.0,
                    help="[0,1) random enemy-detection dropout per second")
    ap.add_argument("--action-mode", default="velocity",
                    choices=["velocity", "goal", "tactical"],
                    help="'goal' outputs a navigation sub-goal instead of "
                         "velocity; 'tactical' additionally replaces the turret "
                         "rate with a target choice and the fire rate with a "
                         "weapons-free gate (the deployable layout)")
    ap.add_argument("--goal-horizon", type=int, default=5,
                    help="seconds ahead for the nav sub-goal (goal/tactical)")
    ap.add_argument("--vis-map", default=None,
                    help="npz from rm_rl.data.vis_map; adds a per-enemy "
                         "empirical engagement (line-of-sight) prior")
    ap.add_argument("--team-prior", default=None,
                    help="json from rm_rl.data.team_prior; adds leak-safe "
                         "opponent-strength features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    agents = S.resolve_agents(args.agent)     # alias / group -> [机器人类型...]
    if not agents:
        raise SystemExit(f"could not resolve --agent {args.agent!r}")
    args.agent = ",".join(agents)

    rcfg = RewardConfig()
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rcfg = RewardConfig.from_dict(cfg.get("reward", {}))

    os.makedirs(args.out, exist_ok=True)
    vmap = VisibilityMap.load(args.vis_map)
    tprior = TeamPrior.load(args.team_prior)
    print(f"[build] agents={agents} vis_map={'on' if vmap else 'off'} "
          f"team_prior={'on' if tprior else 'off'} action_mode={args.action_mode}")
    con = sqlite3.connect(args.db)
    matches = load_matches(con)
    strength = team_strength(matches)
    m_by_id = {int(r.game_id): r for r in matches.itertuples(index=False)}

    game_ids = list(m_by_id.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(game_ids)
    if args.limit_games > 0:
        game_ids = game_ids[: args.limit_games]
    n_val = max(1, int(len(game_ids) * args.val_frac))
    val_ids = set(game_ids[:n_val])

    trajs = {"train": [], "val": []}
    for gid in tqdm(game_ids, desc="games"):
        m = m_by_id[gid]
        game = load_game_arrays(con, gid)
        if game.T == 0:
            continue
        schools = {S.CAMP_RED: m.red_school, S.CAMP_BLUE: m.blue_school}
        g_rng = np.random.default_rng(args.seed + int(gid) % 2_000_000_000)
        ts = build_game_trajectories(game, agents, m.winner, rcfg,
                                     args.min_len, strength, schools,
                                     vis_radius=args.vis_radius,
                                     vis_dropout=args.vis_dropout, rng=g_rng,
                                     action_mode=args.action_mode,
                                     goal_horizon=args.goal_horizon,
                                     vis_map=vmap, team_prior=tprior,
                                     game_id=int(gid))
        split = "val" if gid in val_ids else "train"
        for tr in ts:
            tr["game_id"] = int(gid)
            trajs[split].append(tr)
    con.close()

    if not trajs["train"]:
        raise SystemExit("No trajectories built — check --agent / --db.")

    # ---- normalisation stats from TRAIN only ----
    all_obs = np.concatenate([t["obs"] for t in trajs["train"]], axis=0)
    obs_mean = all_obs.mean(0)
    obs_std = np.maximum(all_obs.std(0), 1e-3)

    all_act = np.concatenate([t["act"] for t in trajs["train"]], axis=0)
    all_valid = np.concatenate([t["valid"] for t in trajs["train"]], axis=0) > 0
    va = np.abs(all_act[all_valid]) if all_valid.any() else np.abs(all_act)
    act_scale = np.maximum(np.percentile(va, 99, axis=0), 1e-3).astype(np.float32)
    spec = get_spec(args.action_mode, all_act.shape[1])
    if spec.is_tactical:
        # only the navigation channels are physical quantities needing a scale.
        # The fire gate is already {0,1} and the target label is a probability
        # distribution — rescaling either would destroy its meaning.
        act_scale[spec.n_cont:] = 1.0

    np.savez(os.path.join(args.out, "norm_stats.npz"),
             obs_mean=obs_mean.astype(np.float32),
             obs_std=obs_std.astype(np.float32),
             act_scale=act_scale)

    def dump(split):
        T = trajs[split]
        if not T:
            return 0, 0
        obs = np.concatenate([t["obs"] for t in T])
        nxt = np.concatenate([t["next_obs"] for t in T])
        act = np.concatenate([t["act"] for t in T])
        act = np.clip(act / act_scale, -1.0, 1.0).astype(np.float32)  # -> [-1,1]
        if spec.is_tactical:                     # keep probabilities in [0,1]
            act[:, spec.n_cont:] = np.clip(act[:, spec.n_cont:], 0.0, 1.0)
        rew = np.concatenate([t["rew"] for t in T])
        done = np.concatenate([t["done"] for t in T])
        valid = np.concatenate([t["valid"] for t in T])
        lens = np.array([len(t["rew"]) for t in T], np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        np.savez_compressed(
            os.path.join(args.out, f"{split}.npz"),
            obs=obs, next_obs=nxt, act=act, rew=rew, done=done, valid=valid,
            ep_start=starts, ep_len=lens,
            ep_return=np.array([t["ret"] for t in T], np.float32),
            ep_win=np.array([t["win"] for t in T], np.float32),
            ep_weight=np.array([t["weight"] for t in T], np.float32),
            ep_camp=np.array([t["camp"] for t in T], np.int64),
            ep_game_id=np.array([t["game_id"] for t in T], np.int64),
        )
        return len(T), len(rew)

    n_tr_ep, n_tr = dump("train")
    n_va_ep, n_va = dump("val")

    meta = dict(
        agent_type=args.agent,
        agent_types=agents,
        obs_dim=int(all_obs.shape[1]),
        action_dim=int(all_act.shape[1]),
        expected_obs_dim=obs_dim(agents[0]),
        expected_action_dim=action_dim(args.action_mode),
        n_train_episodes=n_tr_ep, n_train_transitions=int(n_tr),
        n_val_episodes=n_va_ep, n_val_transitions=int(n_va),
        act_scale=act_scale.tolist(),
        reward=rcfg.__dict__,
        field=dict(x=S.FIELD_X, y=S.FIELD_Y),
        vis_radius=args.vis_radius, vis_dropout=args.vis_dropout,
        action_mode=args.action_mode, goal_horizon=args.goal_horizon,
        vis_map=bool(vmap), team_prior=bool(tprior),
    )
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
