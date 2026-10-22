# -*- coding: utf-8 -*-
"""评估公共骨架：复现 val 划分、重建 obs/act、构造 Jev state、闭式 target oracle。

依赖 rm_rl 包本身，所以 obs 和标签与训练时逐位一致（不重新实现任何特征）。
"""
from __future__ import annotations

import io, json, math, os, sqlite3, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# 数据集不在仓库里：默认读 README「获取数据」一节约定的位置，放在别处时用 RMUC_DB 指定
DB = os.environ.get("RMUC_DB", os.path.join(REPO, "dataset", "rmuc_2026_region_dataset.sqlite"))
RUN = os.path.join(REPO, "rm_runs", "infantry_iql_tactical")
sys.path.insert(0, REPO)

from rm_rl.data import schema as S                                    # noqa: E402
from rm_rl.data.build_dataset import load_game_arrays, load_matches   # noqa: E402
from rm_rl.data import features as F                                  # noqa: E402
from rm_rl.data.vis_map import VisibilityMap                          # noqa: E402
from rm_rl.data.team_prior import TeamPrior                           # noqa: E402
from rm_rl.algos.action_spec import NO_TARGET, TACTICAL               # noqa: E402

ROLE_EN = {S.TYPE_HERO: "hero", S.TYPE_ENGINEER: "engineer", S.TYPE_INFANTRY3: "infantry_3",
           S.TYPE_INFANTRY4: "infantry_4", S.TYPE_AERIAL: "aerial", S.TYPE_SENTRY: "sentry"}
TARGET_NAMES = [ROLE_EN[t] for t in S.MOBILE_TYPES] + ["none"]
AGENTS = [S.TYPE_INFANTRY3, S.TYPE_INFANTRY4]


def utf8():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_env(path=os.path.join(HERE, ".env")):
    if os.environ.get("TYPESAFE_API_KEY") or not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def val_split(con, seed=0, val_frac=0.1):
    """逐位复现 build_dataset.py 的确定性划分（seed 0, val_frac 0.1）。"""
    matches = load_matches(con)
    m_by_id = {int(r.game_id): r for r in matches.itertuples(index=False)}
    game_ids = list(m_by_id.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    n_val = max(1, int(len(game_ids) * val_frac))
    return set(game_ids[:n_val]), m_by_id, game_ids


_VMAP = _TPRIOR = None


def aux():
    global _VMAP, _TPRIOR
    if _VMAP is None:
        _VMAP = VisibilityMap.load(os.path.join(REPO, "data", "vis_map.npz"))
        _TPRIOR = TeamPrior.load(os.path.join(REPO, "data", "team_prior.json"))
    return _VMAP, _TPRIOR


def run_meta():
    return json.load(open(os.path.join(RUN, "meta.json"), encoding="utf-8"))


def build_pair(game, camp, atype, gid):
    """(obs[T-1,161] 未归一化, act[T-1,10] 已按 act_scale 归一, valid[T-1])。"""
    vmap, tprior = aux()
    meta = run_meta()
    tfeat = tprior.feats(gid, camp) if tprior is not None else None
    obs = F.build_obs(game, camp, atype, vis_radius=meta["vis_radius"],
                      vis_dropout=meta["vis_dropout"],
                      rng=np.random.default_rng(0), vis_map=vmap, team_feat=tfeat)
    act = F.build_action_raw(game, camp, atype, action_mode="tactical",
                             goal_horizon=meta["goal_horizon"])
    scale = np.asarray(meta["act_scale"], np.float32)
    a = np.clip(act / scale, -1.0, 1.0).astype(np.float32)
    a[:, TACTICAL.n_cont:] = np.clip(a[:, TACTICAL.n_cont:], 0.0, 1.0)
    ego = game.get(atype, camp)
    valid = (ego.alive[:-1] > 0).astype(np.float32)
    return obs[:-1], a, valid


def target_oracle(game, camp, atype):
    """闭式几何 oracle：只用 obs 里已有的量（枪口朝向 + 敌人相对位置）复现标签。

    这不是模型，是三行几何。它的分数就是 target head 这个指标的泄漏上界。
    """
    soft = F.target_soft(game, camp, atype)
    return soft[:-1].argmax(-1)


# ---------------------------------------------------------------------------
# Jev state
# ---------------------------------------------------------------------------
def _cap(e, rtype, i):
    return {"hp": round(float(e.hp[i])), "max_hp": round(float(e.maxhp[i])),
            "heat17_max": round(float(e.heat17_max[i])),
            "heat42_max": round(float(e.heat42_max[i])),
            "chassis_power_w": round(float(e.power[i]), 1)}


def jev_state(game, camp, atype, i, gid, with_turret=True, team_block=True):
    """构造与 obs 信息量对齐的 Jev state（同样镜像到红方规范帧，t = i+1）。"""
    vmap, tprior = aux()
    mirror = camp == S.CAMP_BLUE
    ecamp = S.enemy_camp(camp)
    ego = game.get(atype, camp)
    ex, ey = F._mirror_xy(ego.x[i], ego.y[i], mirror)
    eyaw = F._mirror_yaw(ego.yaw[i], mirror)

    e_block = {"role": ROLE_EN[atype], "x": round(float(ex), 1), "y": round(float(ey), 1),
               "alive": bool(ego.alive[i]), "radar_marked_vulnerable": bool(ego.vuln[i]),
               "heat_17mm": round(float(ego.heat17[i])),
               "rounds_fired_so_far": round(float(ego.ammo17[i])), **_cap(ego, atype, i)}
    if with_turret:
        e_block["turret_bearing_deg"] = round(float(eyaw), 1)
        e_block["turret_reading_valid"] = bool(F.yaw_valid(ego.yaw[i]))

    allies = []
    for at in F._ally_types(atype):
        a = game.get(at, camp)
        ax, ay = F._mirror_xy(a.x[i], a.y[i], mirror)
        allies.append({"role": ROLE_EN[at], "is_ego": at == atype,
                       "dx": round(float(ax - ex), 1), "dy": round(float(ay - ey), 1),
                       "dist_m": round(float(math.hypot(ax - ex, ay - ey)), 1),
                       "alive": bool(a.alive[i]), **_cap(a, at, i)})

    enemies = []
    for et in S.MOBILE_TYPES:
        e = game.get(et, ecamp)
        px, py = F._mirror_xy(e.x[i], e.y[i], mirror)
        known = bool(F._pos_known(np.array([e.x[i]]), np.array([e.y[i]]))[0] * e.alive[i])
        row = {"role": ROLE_EN[et], "alive": bool(e.alive[i]),
               "radar_marked_vulnerable": bool(e.vuln[i]), "position_known": known,
               **_cap(e, et, i)}
        if known:
            row.update(dx=round(float(px - ex), 1), dy=round(float(py - ey), 1),
                       dist_m=round(float(math.hypot(px - ex, py - ey)), 1),
                       bearing_deg=round(math.degrees(math.atan2(py - ey, px - ex)), 1))
            if vmap is not None:
                row["engagement_prior"] = round(float(
                    vmap.prior(np.array([ex]), np.array([ey]),
                               np.array([px]), np.array([py]), np.array([1.0]))[0]), 3)
        enemies.append(row)

    b = {}
    for owner, tag in ((camp, "my"), (ecamp, "enemy")):
        base, out = game.get(S.TYPE_BASE, owner), game.get(S.TYPE_OUTPOST, owner)
        b[f"{tag}_base_hp"] = round(float(base.hp[i]))
        b[f"{tag}_outpost_hp"] = round(float(out.hp[i]))
    own, opp = game.get(S.TYPE_BASE, camp), game.get(S.TYPE_BASE, ecamp)
    st = {"t_sec": i + 1, "seconds_remaining": max(0, 420 - (i + 1)), "my_team": "red",
          "ego": e_block, "team_slots": allies, "enemies": enemies, "buildings": b,
          "economy": {"my_gold_left": round(float(own.coin_left[i])),
                      "my_gold_total": round(float(own.coin_total[i])),
                      "enemy_gold_total": round(float(opp.coin_total[i]))}}
    if team_block and tprior is not None:
        tf = tprior.feats(gid, camp)
        if tf is not None:
            st["opponent_strength_prior"] = [round(float(v), 3) for v in np.asarray(tf).ravel()]
    return st


def team_state(game, i, gid):
    """队级状态（完整裁判系统视角，红方视角），用于胜率问题。"""
    out = {"t_sec": i + 1, "seconds_remaining": max(0, 420 - (i + 1))}
    for camp, tag in ((S.CAMP_RED, "red"), (S.CAMP_BLUE, "blue")):
        base, op = game.get(S.TYPE_BASE, camp), game.get(S.TYPE_OUTPOST, camp)
        robots = []
        for rt in S.MOBILE_TYPES:
            e = game.get(rt, camp)
            robots.append({"role": ROLE_EN[rt], "hp": round(float(e.hp[i])),
                           "max_hp": round(float(e.maxhp[i])), "alive": bool(e.alive[i]),
                           "radar_marked_vulnerable": bool(e.vuln[i]),
                           "heat17_max": round(float(e.heat17_max[i])),
                           "rounds_fired_so_far": round(float(e.ammo17[i]))})
        out[tag] = {"base_hp": round(float(base.hp[i])), "outpost_hp": round(float(op.hp[i])),
                    "gold_total": round(float(base.coin_total[i])),
                    "gold_left": round(float(base.coin_left[i])),
                    "robots": robots,
                    "team_hp_sum": round(sum(float(game.get(rt, camp).hp[i])
                                             for rt in S.MOBILE_TYPES))}
    return out
