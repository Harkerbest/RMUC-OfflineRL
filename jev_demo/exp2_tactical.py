# -*- coding: utf-8 -*-
"""实验 2：在同一批 val 状态上三方对比 —— 人类标签 / 你们的 IQL / Jev / 闭式 oracle。

  python jev_demo/exp2_tactical.py --games 6  --n-jev 0     # 自检：复现 0.7455
  python jev_demo/exp2_tactical.py --games 61 --n-jev 1200  # 全量
"""
from __future__ import annotations

import argparse, json, os, random, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from harness import (DB, HERE, RUN, AGENTS, F, TARGET_NAMES, S, build_pair, jev_state,
                     load_env, load_game_arrays, run_meta, target_oracle, utf8, val_split)

utf8()
load_env()
from rm_rl.deploy import load_policy                                   # noqa: E402
from rm_rl.algos.action_spec import TACTICAL                           # noqa: E402
from typesafe_sdk import Choice, Noul, TypeSafeClient                  # noqa: E402
from briefing import BRIEF, SEMANTICS, TARGETS_FULL                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=6)
ap.add_argument("--n-jev", type=int, default=0)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--out", default=os.path.join(HERE, "exp2_results.json"))
A = ap.parse_args()


# ---------------------------------------------------------------------------
# 1. 重建 val 集（与训练时逐位一致）
# ---------------------------------------------------------------------------
con = sqlite3.connect(DB)
val_ids, m_by_id, _ = val_split(con)
val_list = sorted(val_ids)[: A.games]
print(f"[val] {len(val_ids)} val games total, using {len(val_list)}")

OBS, ACT, VALID, ORACLE, KEYS, GAMES = [], [], [], [], [], {}
for gid in val_list:
    game = load_game_arrays(con, gid)
    if game.T == 0:
        continue
    GAMES[gid] = game
    for camp in S.CAMPS:
        for at in AGENTS:
            ego = game.get(at, camp)
            if ego.alive.sum() < 30 * 0.3:
                continue
            o, a, v = build_pair(game, camp, at, gid)
            OBS.append(o); ACT.append(a); VALID.append(v)
            ORACLE.append(target_oracle(game, camp, at))
            KEYS.extend((gid, camp, at, i) for i in range(len(v)))
con.close()

obs = np.concatenate(OBS); act = np.concatenate(ACT)
valid = np.concatenate(VALID) > 0; oracle = np.concatenate(ORACLE)
print(f"[val] {len(obs)} transitions, obs_dim={obs.shape[1]}, act_dim={act.shape[1]}, "
      f"valid={valid.sum()}")

y_t = act[:, TACTICAL.sl_target].argmax(-1)
y_fire = act[:, TACTICAL.sl_fire].squeeze(-1).round()
named = y_t != TACTICAL.n_target - 1


# ---------------------------------------------------------------------------
# 2. 你们的 IQL
# ---------------------------------------------------------------------------
model, norm, info = load_policy(RUN, device="cpu")
with torch.no_grad():
    o = norm.normalize(torch.as_tensor(obs, dtype=torch.float32))
    pred = torch.cat([model.act(o[k:k + 8192], deterministic=True)
                      for k in range(0, len(o), 8192)]).numpy()
iql_t = pred[:, TACTICAL.sl_target].argmax(-1)
iql_fire = pred[:, TACTICAL.sl_fire].squeeze(-1).round()
iql_nav = pred[:, TACTICAL.sl_cont]


def f1(p, y, m):
    tp = float(((p == 1) & (y == 1) & m).sum()); fp = float(((p == 1) & (y == 0) & m).sum())
    fn = float(((p == 0) & (y == 1) & m).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


def tgt_named(p, m):
    mm = m & named
    return float((p[mm] == y_t[mm]).mean()) if mm.sum() else float("nan")


R = {"n_transitions": int(len(obs)), "n_valid": int(valid.sum()),
     "n_named": int((valid & named).sum()), "games": len(GAMES),
     "label_target_dist": {TARGET_NAMES[k]: int((y_t[valid] == k).sum()) for k in range(7)},
     "fire_pos_rate": float(y_fire[valid].mean())}
R["iql"] = {"target_top1_named": tgt_named(iql_t, valid),
            "target_top1_all": float((iql_t[valid] == y_t[valid]).mean()),
            "fire_f1": f1(iql_fire, y_fire, valid),
            "fire_acc": float((iql_fire[valid] == y_fire[valid]).mean()),
            "nav_mse": float(((iql_nav[valid] - act[valid][:, TACTICAL.sl_cont]) ** 2).mean())}
R["oracle_closed_form"] = {"target_top1_named": tgt_named(oracle, valid)}
# 无泄漏的 target 启发式基线：选最近的活敌（不看枪口朝向）
NEAREST = np.full(len(obs), TACTICAL.n_target - 1, np.int64)
_off = 0
for gid in val_list:
    if gid not in GAMES:
        continue
    g = GAMES[gid]
    for camp in S.CAMPS:
        for at in AGENTS:
            ego = g.get(at, camp)
            if ego.alive.sum() < 30 * 0.3:
                continue
            n = len(ego.alive) - 1
            mir = camp == S.CAMP_BLUE
            ex_, ey_ = F._mirror_xy(ego.x[:n], ego.y[:n], mir)
            best = np.full(n, np.inf); who = np.full(n, TACTICAL.n_target - 1, np.int64)
            for j, et in enumerate(S.MOBILE_TYPES):
                e = g.get(et, S.enemy_camp(camp))
                px, py = F._mirror_xy(e.x[:n], e.y[:n], mir)
                ok = (F._pos_known(e.x[:n], e.y[:n]) * e.alive[:n]) > 0
                d = np.where(ok, np.hypot(px - ex_, py - ey_), np.inf)
                who = np.where(d < best, j, who); best = np.minimum(best, d)
            NEAREST[_off:_off + n] = who
            _off += n

R["baselines"] = {
    "target_random_over_6": 1 / 6,
    "target_majority_named": float(max(
        (y_t[valid & named] == k).mean() for k in range(6))),
    "target_nearest_live_enemy": tgt_named(NEAREST, valid),
    "fire_always_off_acc": float((y_fire[valid] == 0).mean()),
    "nav_predict_zero_err_m": float(np.linalg.norm(
        act[valid][:, TACTICAL.sl_cont] * np.asarray(run_meta()["act_scale"], np.float32)[:2],
        axis=1).mean()),
}
print(json.dumps({k: R[k] for k in ("iql", "oracle_closed_form", "baselines")},
                 ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# 3. Jev（两个条件：给枪口朝向 = 与 IQL 同信息；不给 = 无泄漏）
# ---------------------------------------------------------------------------
if A.n_jev <= 0:
    json.dump(R, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    sys.exit(0)

CELL, NX, NY = 2.0, 14, 8
GRID = {}
for cx in range(NX):
    for cy in range(NY):
        GRID[f"c{cx:02d}_{cy:02d}"] = (
            f"x {cx * CELL:.0f}-{(cx + 1) * CELL:.0f} m, y {cy * CELL:.0f}-{(cy + 1) * CELL:.0f} m; "
            f"{'own half' if cx < NX / 2 else 'enemy half'}, "
            f"{'bottom lane' if cy < NY / 3 else ('top lane' if cy > 2 * NY / 3 else 'middle lane')}")
GRID["stay"] = "Hold the current position, do not relocate."


def cell_center(name):
    if name == "stay":
        return None
    cx, cy = int(name[1:3]), int(name[4:6])
    return (cx + 0.5) * CELL, (cy + 0.5) * CELL


scale = np.asarray(run_meta()["act_scale"], np.float32)
rows = np.where(valid & named)[0]
random.Random(0).shuffle(rows := list(rows))
rows = rows[: A.n_jev]
print(f"[jev] querying {len(rows)} states x 2 conditions, {A.workers} workers")

QS = {
    "target": Choice(instructions="Which enemy should the acting robot put its turret on for the "
                                  "next second?", criteria=TARGETS_FULL),
    # 标签语义：fire_gate[i] = 累计17mm发弹[i] - [i-1] > 0，即"刚结束的这一秒里是否出弹"。
    # 所以这里必须问同一件事（回溯），否则和标签比就是在比两个不同的量。
    "weapons_free": Noul(instructions="In the one-second interval that has just ended, did this "
                                      "robot fire any 17mm rounds? Roughly 19% of robot-seconds "
                                      "are firing seconds, so most seconds the answer is no."),
    "nav_cell": Choice(instructions="Which arena grid cell should the acting robot move toward over "
                                    "the next 5 seconds?", criteria=GRID),
}
client = TypeSafeClient(timeout=60.0)


def ask(args):
    k, with_turret = args
    gid, camp, at, i = KEYS[k]
    st = jev_state(GAMES[gid], camp, at, i, gid, with_turret=with_turret)
    state = {"rmuc_rules": BRIEF, "field_semantics": SEMANTICS, "match_state": st}
    for attempt in range(4):
        try:
            r = client.system_one(state=state, questions=QS)
            a = r.answers
            return k, with_turret, dict(
                target=TARGET_NAMES.index(a["target"].choice),
                target_conf=float(a["target"].confidence),
                fire_p=float(a["weapons_free"].noul),
                nav=a["nav_cell"].choice, nav_conf=float(a["nav_cell"].confidence),
                tok=int(r.usage.input_tokens or 0))
        except Exception as e:
            if attempt == 3:
                return k, with_turret, {"error": f"{type(e).__name__}: {e}"}
            time.sleep(1.5 * (attempt + 1))


jobs = [(k, w) for w in (True, False) for k in rows]
t0 = time.perf_counter()
out = {True: {}, False: {}}
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    for n, (k, w, res) in enumerate(ex.map(ask, jobs), 1):
        out[w][k] = res
        if n % 200 == 0:
            print(f"  {n}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
client.close()
wall = time.perf_counter() - t0
print(f"[jev] done in {wall:.0f}s")


def auc(score, label):
    """midrank AUC —— Jev 的概率只有 2 位小数，并列极多，序数排名会偏。"""
    score = np.asarray(score, float); y = np.asarray(label).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); s = score[order]
    ranks = np.empty(len(s), float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), float); r[order] = ranks
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sweep_f1(p, y):
    """最优阈值下的 F1，以及按正类率匹配的阈值下的 F1。"""
    p = np.asarray(p, float); y = np.asarray(y).astype(int)
    best, best_th = 0.0, 0.5
    for th in np.unique(np.round(p, 3)):
        v = f1((p > th).astype(int), y, np.ones(len(y), bool))
        if v > best:
            best, best_th = v, float(th)
    q = float(np.quantile(p, 1.0 - y.mean()))
    return {"best_f1": best, "best_threshold": best_th,
            "f1_at_base_rate_threshold": f1((p > q).astype(int), y, np.ones(len(y), bool)),
            "base_rate_threshold": q}


for w in (True, False):
    ok = [k for k, v in out[w].items() if "error" not in v]
    if not ok:
        R[f"jev_turret_{w}"] = {"errors": len(out[w])}
        continue
    jt = np.array([out[w][k]["target"] for k in ok])
    jf = np.array([out[w][k]["fire_p"] for k in ok])
    yt_, yf_ = y_t[ok], y_fire[ok]
    navs, iqls, hums = [], [], []
    for k in ok:
        gid, camp, at, i = KEYS[k]
        g = GAMES[gid]; ego = g.get(at, camp)
        mir = camp == S.CAMP_BLUE
        ex_, ey_ = S.FIELD_X - ego.x[i] if mir else ego.x[i], S.FIELD_Y - ego.y[i] if mir else ego.y[i]
        h = act[k, TACTICAL.sl_cont] * scale[:2]
        c = cell_center(out[w][k]["nav"])
        j = np.zeros(2) if c is None else np.array([c[0] - ex_, c[1] - ey_])
        navs.append(np.linalg.norm(j - h)); hums.append(np.linalg.norm(h))
        iqls.append(np.linalg.norm(iql_nav[k] * scale[:2] - h))
    R[f"jev_turret_{w}"] = {
        "n": len(ok), "errors": len(out[w]) - len(ok),
        "target_top1_named": float((jt == yt_).mean()),
        "target_mean_conf": float(np.mean([out[w][k]["target_conf"] for k in ok])),
        "fire_f1@0.5": f1((jf > 0.5).astype(int), yf_, np.ones(len(ok), bool)),
        "fire_acc@0.5": float(((jf > 0.5).astype(int) == yf_).mean()),
        "fire_auc": auc(jf, yf_), "fire_mean_p": float(jf.mean()),
        "fire_sweep": sweep_f1(jf, yf_),
        "nav_err_m_mean": float(np.mean(navs)), "nav_err_m_median": float(np.median(navs)),
        "nav_stay_rate": float(np.mean([out[w][k]["nav"] == "stay" for k in ok])),
        "mean_input_tokens": float(np.mean([out[w][k]["tok"] for k in ok])),
    }
    np.savez(os.path.join(HERE, f"exp2_raw_turret{int(w)}.npz"),
             rows=np.array(ok), jev_target=jt, jev_fire_p=jf, y_target=yt_, y_fire=yf_,
             nav_err=np.array(navs), iql_nav_err=np.array(iqls), human_norm=np.array(hums),
             conf=np.array([out[w][k]["target_conf"] for k in ok]))

# ---- 同一批 1200 行上的其它决策者（无需再调 API，行选择是确定性的）----
sub = np.array(rows)
m1 = np.ones(len(sub), bool)
iql_f_raw = pred[sub][:, TACTICAL.sl_fire].squeeze(-1)
R["matched_subsample"] = {
    "n": int(len(sub)), "fire_pos_rate": float(y_fire[sub].mean()),
    "human_nav_goal_norm_m": float(np.linalg.norm(
        act[sub][:, TACTICAL.sl_cont] * scale[:2], axis=1).mean()),
    "iql": {"target_top1_named": float((iql_t[sub] == y_t[sub]).mean()),
            "fire_f1@0.5": f1(iql_fire[sub].astype(int), y_fire[sub], m1),
            "fire_acc@0.5": float((iql_fire[sub] == y_fire[sub]).mean()),
            "fire_auc": auc(iql_f_raw, y_fire[sub]),
            "fire_sweep": sweep_f1(iql_f_raw, y_fire[sub]),
            "nav_err_m_mean": float(np.mean([
                np.linalg.norm(iql_nav[k] * scale[:2] - act[k, TACTICAL.sl_cont] * scale[:2])
                for k in sub]))},
    "oracle_closed_form": {"target_top1_named": float((oracle[sub] == y_t[sub]).mean())},
    "nearest_live_enemy": {"target_top1_named": float((NEAREST[sub] == y_t[sub]).mean())},
    "random_over_6": 1 / 6,
}
R["jev_wall_seconds"] = wall
R["jev_n_calls"] = len(jobs)
print(json.dumps(R["matched_subsample"], ensure_ascii=False, indent=2))
json.dump(R, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in R.items() if k.startswith("jev_turret")},
                 ensure_ascii=False, indent=2))
