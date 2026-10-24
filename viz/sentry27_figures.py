"""Figures for the 2027 sentry decision layer.

    python -m viz.sentry27_figures --train rm_runs/sentry27 --eval rm_runs/sentry27_eval --out viz/figures

sentry27_train.png  per update: the learner's score against each opponent kind
                    (rolling mean), imitation accuracy then policy entropy, and the
                    learner sentry's behaviour (time on the own central card, time
                    without ammo)
sentry27_eval.png   per arm: score and the paired score difference to the reference
                    arm with bootstrap 95% CIs (eval_sentry27 summary.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .sim_figures import REF, SERIES, TH, _axes, _style

KINDS = (("script", "对方脚本哨兵", "#3987e5"), ("bt", "行为树", "#e5a139"),
         ("snapshot", "历史快照", "#9b6bdf"), ("mirror", "镜像自博弈", "#7d8590"))


def _roll(v, k=5):
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = w.mean()
    return out


def train(run, out, plt):
    rows = list(csv.DictReader(open(os.path.join(run, "train_log.csv"), encoding="utf-8")))
    it = np.array([int(r["iteration"]) for r in rows])
    get = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])
    imitate = np.array([r.get("phase") == "imitate" for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8), facecolor=TH["bg"], sharex=True)
    for ax in axes:
        _axes(ax)
        ax.grid(axis="y", color=TH["grid"], lw=0.8)
        if imitate.any():
            ax.axvspan(it[0] - 0.5, it[imitate].max() + 0.5, color=TH["grid"], alpha=0.6, lw=0)
    ax = axes[0]
    for key, label, col in KINDS:
        v = get(f"score_{key}")
        if np.isfinite(v).any():
            ax.plot(it, _roll(v), color=col, lw=2, label=label)
    ax.axhline(0.5, color=REF, lw=1, ls=(0, (3, 3)))
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("学习者得分（胜 1 / 平 ½ / 负 0），按对手类型，5 次更新滑动平均", fontsize=10, color=TH["fg"], loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TH["fg"], ncol=4, loc="lower right")
    ax = axes[1]
    acc, ent = get("imit_acc"), get("entropy")
    ax.plot(it[imitate], acc[imitate], color=SERIES, lw=2, label="模仿准确率（五个头全对）")
    ax.plot(it[~imitate], ent[~imitate] / max(np.nanmax(ent), 1e-6), color="#e5a139", lw=2, label="策略熵（相对最大值）")
    ax.set_title("模仿阶段：与行为树一致的比例；PPO 阶段：策略熵", fontsize=10, color=TH["fg"], loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TH["fg"], loc="lower right")
    ax = axes[2]
    ax.plot(it, _roll(get("seat_own_central")), color=SERIES, lw=2, label="在己方中央高地卡上")
    ax.plot(it, _roll(get("seat_no_ammo")), color="#e5a139", lw=2, label="无弹")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("学习者哨兵：存活时间占比", fontsize=10, color=TH["fg"], loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TH["fg"], loc="upper right")
    axes[-1].set_xlabel("更新次数（灰底为模仿阶段）")
    fig.savefig(out, dpi=150, facecolor=TH["bg"], bbox_inches="tight")
    print("wrote", out)


def evaluation(path, out, plt, ref):
    table = json.load(open(path, encoding="utf-8"))
    arms = list(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(arms) + 1.4), facecolor=TH["bg"], sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(arms))[::-1]
    for ax, key, ci_key, title in ((axes[0], "score", "score_ci", "得分 · Wilson 95% CI"),
                                   (axes[1], "d_score", "d_score_ci", f"相对 {ref} 的配对得分差 · bootstrap 95% CI")):
        _axes(ax)
        for yy, a in zip(y, arms):
            v, (lo, hi) = table[a].get(key), table[a].get(ci_key) or (np.nan, np.nan)
            if v is None or (a == ref and key == "d_score"):
                continue
            col = REF if a in ("script", "random") or a.startswith("bt:") else SERIES
            ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(v, yy, "o", ms=8, color=col, mec=TH["bg"], mew=2, zorder=3)
            ax.annotate(f"{v:+.3f}" if key == "d_score" else f"{v:.3f}", (hi if np.isfinite(hi) else v, yy),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TH["fg"],
                        bbox=dict(boxstyle="round,pad=0.15", fc=TH["bg"], ec="none"))
        ax.axvline(0.0 if key == "d_score" else table.get(ref, {}).get("score", 0.5), color=REF, lw=1,
                   ls=(0, (3, 3)))
        ax.set_title(title, fontsize=10, color=TH["fg"], loc="left")
    axes[0].set_yticks(y, [f"{a}  n={table[a]['n']}" for a in arms], color=TH["fg"])
    fig.savefig(out, dpi=150, facecolor=TH["bg"], bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="training run dir (train_log.csv)")
    ap.add_argument("--eval", default=None, help="eval_sentry27 output dir (summary.json)")
    ap.add_argument("--ref", default="bt:balanced")
    ap.add_argument("--out", default="viz/figures")
    ap.add_argument("--font", default="WenQuanYi Zen Hei")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt, args.font)
    os.makedirs(args.out, exist_ok=True)
    if args.train:
        train(args.train, os.path.join(args.out, "sentry27_train.png"), plt)
    if args.eval:
        evaluation(os.path.join(args.eval, "summary.json"), os.path.join(args.out, "sentry27_eval.png"), plt, args.ref)


if __name__ == "__main__":
    main()
