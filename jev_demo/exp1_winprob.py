# -*- coding: utf-8 -*-
"""实验 1：Jev 的胜率判断 vs 裁判系统统计基线，613 场 x 3 个时刻。

回答的是"Jev 到底有没有 RM 先验"——如果它相对一个只看前哨站/基地 HP 差的
逻辑回归没有 excess AUC，那它在这个域里就没有超出表面统计的知识。

  python jev_demo/exp1_winprob.py --slices 100,200,300 --workers 14
"""
from __future__ import annotations

import argparse, json, os, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from harness import DB, HERE, ROLE_EN, S, load_env, utf8

utf8(); load_env()
from typesafe_sdk import Noul, TypeSafeClient          # noqa: E402
from briefing import BRIEF, SEMANTICS                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slices", default="100,200,300")
ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(HERE, "exp1_results.json"))
A = ap.parse_args()
SLICES = [int(s) for s in A.slices.split(",")]

con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("select game_id, 胜方, 时长秒 from matches where 胜方 in ('红','蓝')")
M = {int(g): (w, int(d)) for g, w, d in cur.fetchall()}
if A.limit:
    M = {k: M[k] for k in sorted(M)[: A.limit]}
print(f"[exp1] {len(M)} decided matches, slices={SLICES}")

ph = ",".join("?" * len(SLICES))
cur.execute(f"""select game_id, 时刻秒, 机器人类型, 阵营, 当前血量, 最大血量, 小热量上限,
                       累计17mm发弹, 队伍总金币, 队伍剩余金币, 是否易伤
                from timeseries where 时刻秒 in ({ph})""", SLICES)
ROWS = {}
for gid, t, rt, camp, hp, mhp, h17, ammo, ct, cl, vu in cur.fetchall():
    if int(gid) in M:
        ROWS.setdefault((int(gid), int(t)), []).append(
            (rt, camp, hp or 0.0, mhp or 0.0, h17 or 0.0, ammo or 0.0, ct or 0.0, cl or 0.0, vu or 0))
con.close()
print(f"[exp1] {len(ROWS)} (match, t) states loaded")


def build(gid, t):
    rows = ROWS[(gid, t)]
    st = {"t_sec": t, "seconds_remaining": max(0, 420 - t)}
    feat = {}
    for camp, tag in ((S.CAMP_RED, "red"), (S.CAMP_BLUE, "blue")):
        side = {"robots": []}
        hp_sum = 0.0
        for rt, c, hp, mhp, h17, ammo, ct, cl, vu in rows:
            if c != camp:
                continue
            if rt == S.TYPE_BASE:
                side["base_hp"] = round(hp); side["gold_total"] = round(ct)
                side["gold_left"] = round(cl)
            elif rt == S.TYPE_OUTPOST:
                side["outpost_hp"] = round(hp)
            elif rt in ROLE_EN:
                hp_sum += hp
                side["robots"].append({"role": ROLE_EN[rt], "hp": round(hp),
                                       "max_hp": round(mhp), "alive": hp > 0,
                                       "heat17_max": round(h17),
                                       "rounds_fired_so_far": round(ammo),
                                       "radar_marked_vulnerable": bool(vu)})
        side.setdefault("base_hp", 0); side.setdefault("outpost_hp", 0)
        side.setdefault("gold_total", 0); side.setdefault("gold_left", 0)
        side["team_hp_sum"] = round(hp_sum)
        side["n_alive"] = sum(1 for r in side["robots"] if r["alive"])
        st[tag] = side
        feat[tag] = side
    x = np.array([
        feat["red"]["outpost_hp"] - feat["blue"]["outpost_hp"],
        feat["red"]["base_hp"] - feat["blue"]["base_hp"],
        feat["red"]["team_hp_sum"] - feat["blue"]["team_hp_sum"],
        feat["red"]["gold_total"] - feat["blue"]["gold_total"],
        feat["red"]["n_alive"] - feat["blue"]["n_alive"],
    ], np.float64)
    return st, x


KEYS = [(g, t) for g in sorted(M) for t in SLICES
        if (g, t) in ROWS and t < M[g][1] - 1]
STATES, X, Y = [], [], []
for g, t in KEYS:
    st, x = build(g, t)
    STATES.append(st); X.append(x); Y.append(1 if M[g][0] == "红" else 0)
X = np.array(X); Y = np.array(Y)
print(f"[exp1] {len(KEYS)} usable states, red win rate {Y.mean():.3f}")

Q = {"red_wins": Noul(instructions="Will the RED team win this round? Answer from the match state "
                                   "as it stands at this instant.")}
client = TypeSafeClient(timeout=60.0)


def ask(n):
    state = {"rmuc_rules": BRIEF, "field_semantics": SEMANTICS, "match_state": STATES[n]}
    for k in range(4):
        try:
            r = client.system_one(state=state, questions=Q)
            return n, float(r.answers["red_wins"].noul), int(r.usage.input_tokens or 0)
        except Exception as e:
            if k == 3:
                return n, None, f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (k + 1))


t0 = time.perf_counter()
P = np.full(len(KEYS), np.nan)
TOK = []
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    for i, (n, p, tk) in enumerate(ex.map(ask, range(len(KEYS))), 1):
        if p is not None:
            P[n] = p
            TOK.append(tk)
        if i % 300 == 0:
            print(f"  {i}/{len(KEYS)}  {time.perf_counter() - t0:.0f}s", flush=True)
client.close()
wall = time.perf_counter() - t0
print(f"[exp1] jev done in {wall:.0f}s, {np.isnan(P).sum()} failures")


# ---------------------------------------------------------------------------
# 指标：AUC（带并列修正的 midrank）/ Brier / ECE
# ---------------------------------------------------------------------------
def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); s = score[order]
    ranks = np.empty(len(s), float); i = 0
    while i < len(s):                      # midrank，Jev 的概率只有 2 位小数，并列很多
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), float); r[order] = ranks
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, float); e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins + (1e-9 if b == bins - 1 else 0))
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_logreg(Xt, yt, iters=4000, lr=1.0, l2=1e-3):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Z = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - yt) / len(yt) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return mu, sd, w


def pred_logreg(m, Xe):
    mu, sd, w = m
    Z = np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1))])
    return 1.0 / (1.0 + np.exp(-Z @ w))


ok = ~np.isnan(P)
games = np.array([g for g, _ in KEYS])
uniq = np.array(sorted(set(games.tolist())))
fold_of = {g: i % 5 for i, g in enumerate(uniq)}         # 按比赛分折，避免同场泄漏
folds = np.array([fold_of[g] for g in games])

# 5 折 CV 的逻辑回归（完整特征）与单特征基线
cv = np.full(len(KEYS), np.nan)
for f in range(5):
    tr, te = (folds != f), (folds == f)
    cv[te] = pred_logreg(fit_logreg(X[tr], Y[tr].astype(float)), X[te])

FEATS = ["outpost_diff", "base_diff", "team_hp_diff", "gold_diff", "n_alive_diff"]
R = {"n_states": int(len(KEYS)), "n_matches": len(uniq), "slices": SLICES,
     "red_win_rate": float(Y.mean()), "jev_failures": int((~ok).sum()),
     "jev_wall_seconds": wall, "mean_input_tokens": float(np.mean(TOK)) if TOK else None,
     "single_feature_auc": {f: auc(X[:, k], Y) for k, f in enumerate(FEATS)},
     "baseline_logreg_cv": {"auc": auc(cv[ok], Y[ok]), "brier": brier(cv[ok], Y[ok]),
                            "ece": ece(cv[ok], Y[ok])},
     "jev": {"auc": auc(P[ok], Y[ok]), "brier": brier(P[ok], Y[ok]), "ece": ece(P[ok], Y[ok]),
             "mean_p": float(P[ok].mean()), "n_distinct_values": int(len(set(P[ok].tolist())))},
     "always_predict_base_rate_brier": brier(np.full(ok.sum(), Y[ok].mean()), Y[ok])}
R["excess_auc_over_logreg"] = R["jev"]["auc"] - R["baseline_logreg_cv"]["auc"]
R["excess_auc_over_best_single_feature"] = R["jev"]["auc"] - max(
    R["single_feature_auc"].values())

R["per_slice"] = {}
for t in SLICES:
    m = ok & (np.array([tt for _, tt in KEYS]) == t)
    if m.sum() < 20:
        continue
    R["per_slice"][str(t)] = {
        "n": int(m.sum()), "jev_auc": auc(P[m], Y[m]), "logreg_auc": auc(cv[m], Y[m]),
        "jev_brier": brier(P[m], Y[m]), "logreg_brier": brier(cv[m], Y[m]),
        "outpost_diff_auc": auc(X[m][:, 0], Y[m])}

np.savez(os.path.join(HERE, "exp1_raw.npz"), P=P, Y=Y, X=X, cv=cv,
         games=games, t=np.array([tt for _, tt in KEYS]))
json.dump(R, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
