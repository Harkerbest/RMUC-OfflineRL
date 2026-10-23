"""Figures for the Arena-1 closed-loop experiments.

    python -m viz.sim_figures --closed-loop rm_runs/arena/closed_loop \\
        --mappo rm_runs/arena/mappo_from_bc --out viz/figures

closed_loop.png  per arm: win rate (Wilson 95% CI) and mean damage difference
                 (dealt - taken, bootstrap 95% CI), on the shared match grid
mappo.png        MAPPO fine-tuning: rollout win rate, damage difference and
                 KL to the offline prior per iteration
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

TH = dict(bg="#0d1117", fg="#e6edf3", dim="#7d8590", grid="#21262d")
SERIES = "#3987e5"          # one series per panel: the categorical slot-1 dark step
REF = "#7d8590"
ORDER = ["script", "bc", "iql", "dt", "bc_vision", "mappo", "random"]
LABEL = {"script": "脚本(参照)", "bc": "BC", "iql": "IQL", "dt": "DT", "bc_vision": "BC·视野受限",
         "mappo": "MAPPO(BC 初始化)", "random": "随机"}


def _style(plt, font):
    for fam in ("font.sans-serif",):
        plt.rcParams[fam] = [font, "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False,
                         "text.color": TH["fg"], "axes.labelcolor": TH["dim"],
                         "xtick.color": TH["dim"], "ytick.color": TH["dim"],
                         "axes.edgecolor": TH["grid"]})


def _axes(ax):
    ax.set_facecolor(TH["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=TH["grid"], lw=0.8)
    ax.set_axisbelow(True)


def closed_loop(path, out, plt):
    from rm_rl.eval.closed_loop import bootstrap_ci, wilson
    rows = [json.loads(l) for l in open(os.path.join(path, "results.jsonl"), encoding="utf-8") if l.strip()]
    arms = [a for a in ORDER if any(r["arm"] == a for r in rows)]
    arms += sorted({r["arm"] for r in rows} - set(arms))
    stats = []
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        score = sum(r["win"] for r in rs)
        dd = [r["damage"] - r["damage_enemy"] for r in rs]
        stats.append((a, len(rs), score / len(rs), wilson(score, len(rs)), float(np.mean(dd)), bootstrap_ci(dd)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(arms) + 1.4), facecolor=TH["bg"],
                             sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(stats))[::-1]
    panels = [(axes[0], 2, 3, "胜率（平局计 ½）· Wilson 95% CI", "{:.2f}"),
              (axes[1], 4, 5, "伤害差（造成 - 承受）· bootstrap 95% CI", "{:+.0f}")]
    for ax, vi, ci, title, fmt in panels:
        _axes(ax)
        for yy, s in zip(y, stats):
            lo, hi = s[ci]
            col = REF if s[0] in ("script", "random") else SERIES
            ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.plot(s[vi], yy, "o", ms=8, color=col, mec=TH["bg"], mew=2, zorder=3)
            ax.annotate(fmt.format(s[vi]), (hi, yy), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TH["fg"])
        ref = next((s for s in stats if s[0] == "script"), None)
        if ref is not None:
            ax.axvline(ref[vi], color=REF, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.set_title(title, fontsize=10, color=TH["fg"], loc="left")
    axes[0].set_yticks(y, [f"{LABEL.get(s[0], s[0])}  n={s[1]}" for s in stats], color=TH["fg"])
    axes[0].set_xlim(-0.02, 1.08)
    fig.savefig(out, dpi=150, facecolor=TH["bg"], bbox_inches="tight")
    print("wrote", out)


def mappo(path, out, plt):
    import csv
    rows = list(csv.DictReader(open(os.path.join(path, "train_log.csv"), encoding="utf-8")))
    it = np.array([int(r["iteration"]) for r in rows])
    get = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
    panels = [("win_rate", "采样对局胜率（每轮 {} 局）".format(rows[0]["episodes"])),
              ("d_damage", "伤害差（造成 - 承受）"), ("kl_prior", "KL(π || π_BC)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 6.4), facecolor=TH["bg"], sharex=True)
    for ax, (key, title) in zip(axes, panels):
        _axes(ax)
        ax.grid(axis="y", color=TH["grid"], lw=0.8)
        v = get(key)
        ax.plot(it, v, color=SERIES, lw=1, alpha=0.45)
        k = 5
        if len(v) >= k:
            sm = np.convolve(np.nan_to_num(v), np.ones(k) / k, mode="valid")
            ax.plot(it[k - 1:], sm, color=SERIES, lw=2, label=f"{k} 轮滑动平均")
        warm = it[get("actor_trained") < 0.5]
        if len(warm):
            ax.axvspan(it[0] - 0.5, warm.max() + 0.5, color=TH["grid"], alpha=0.6, lw=0)
        ax.set_title(title, fontsize=10, color=TH["fg"], loc="left")
    axes[0].text(it[0], axes[0].get_ylim()[1], " critic 预热（actor 冻结）", va="top", fontsize=8,
                 color=TH["dim"])
    axes[-1].set_xlabel("迭代")
    fig.savefig(out, dpi=150, facecolor=TH["bg"], bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-loop", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--out", default="viz/figures")
    ap.add_argument("--font", default="WenQuanYi Zen Hei")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt, args.font)
    os.makedirs(args.out, exist_ok=True)
    if args.closed_loop:
        closed_loop(args.closed_loop, os.path.join(args.out, "closed_loop.png"), plt)
    if args.mappo:
        mappo(args.mappo, os.path.join(args.out, "mappo.png"), plt)


if __name__ == "__main__":
    main()
