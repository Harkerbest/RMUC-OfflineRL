"""Online MAPPO fine-tuning in Arena-1, starting from an offline tactical run.

    python -m rm_rl.train.train_mappo --init rm_runs/infantry_bc_tactical \\
        --vis-map data/vis_map.npz --out rm_runs/mappo_from_bc --iters 40 --workers 4

Each iteration plays `--episodes` full matches in parallel worker processes —
the shared actor (sampling) on infantry 3/4/7, the team script on the other
robots, an opponent drawn from `--opponents` on a random side — then runs PPO
epochs in this process.  Rewards are `reward.compute_reward` on the match's
per-second referee log, i.e. the offline reward with the offline weights.

The actor is exported after every `--export-every` iterations to
OUT/iter_XXX and OUT/latest as an ordinary tactical run directory (best.pt,
norm_stats.npz, meta.json), so `rm_rl.eval.closed_loop --arm mappo=OUT/latest`
and `deploy.load_policy` read it like any BC run.  `--resume` continues from
OUT/mappo_state.pt.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import torch

from ..algos.mappo import MAPPO, gae
from ..data.reward import RewardConfig
from ..deploy import load_policy
from . import common as C

METRICS = ("v_loss", "pi_loss", "entropy", "kl_prior", "approx_kl", "clipfrac")
# Training pool.  siege_team is held out: closed-loop evaluation against it
# measures generalisation to an opponent never met during fine-tuning.
TRAIN_OPPONENTS = ("claude_opus55_xhigh_hold,claude_opus55_xhigh_team,claude_opus55_xhigh_squad,"
                   "claude_opus55_xhigh_distributed,siege_assault_team,siege_blitz,"
                   "siege_economy,siege_flank,siege_guard")


def load_init(run_dir):
    model, norm, info = load_policy(run_dir)
    if info.get("action_mode") != "tactical":
        raise SystemExit(f"{run_dir}: MAPPO fine-tuning needs a tactical-action run")
    if info["algo"] == "dt":
        raise SystemExit("initialise from a Markov run (BC / IQL); the DT has no policy head to share")
    ckpt = os.path.join(run_dir, "best.pt") if os.path.isdir(run_dir) else run_dir
    with open(ckpt, "rb") as f:
        cfg = (torch.load(f, map_location="cpu").get("config") or {})
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return model.policy, norm, info, meta, dict(hidden=m.get("hidden", 256), depth=m.get("depth", 2))


# --------------------------------------------------------------------------- workers
_W = {}


def _init_worker(arena_path, vis_map_path, init_run):
    torch.set_num_threads(1)
    from ..data.vis_map import VisibilityMap
    from ..sim.arena import load_arena
    actor, norm, info, meta, _ = load_init(init_run)
    _W.update(arena=load_arena(arena_path), vmap=VisibilityMap.load(vis_map_path), actor=actor,
              mean=norm.mean.numpy(), std=norm.std.numpy(), info=info, meta=meta)


class SamplingAgent:
    def __init__(self, actor, mean, std, act_scale):
        self.actor, self.mean, self.std, self.act_scale = actor, mean, std, act_scale

    def __call__(self, reqs):
        o = torch.as_tensor((np.stack([r["obs"] for r in reqs]) - self.mean) / self.std,
                            dtype=torch.float32)
        a, logp = self.actor.sample(o)
        out = {}
        for i, r in enumerate(reqs):
            raw = a[i].numpy()
            exe = raw.copy()
            exe[:2] = np.clip(exe[:2], -1.0, 1.0)
            out[r["key"]] = (exe, dict(raw=raw, logp=float(logp[i])))
        return out


def rollout(job):
    from ..sim.episode import EpisodeConfig, run_episode
    with open(job["actor_path"], "rb") as f:
        _W["actor"].load_state_dict(torch.load(f, map_location="cpu"))
    _W["actor"].eval()
    agent = SamplingAgent(_W["actor"], _W["mean"], _W["std"], _W["info"]["act_scale"])
    cfg = EpisodeConfig(team=job["side"], own=f"examples/{job['own']}",
                        opponent=f"examples/{job['opponent']}", seed=job["seed"],
                        duration=job["duration"], enemy_info=job["enemy_info"],
                        use_team_prior=bool(_W["meta"].get("team_prior")), record_global=True,
                        reward=RewardConfig.from_dict(_W["meta"].get("reward", {})))
    out = run_episode(_W["arena"], cfg, agent, vis_map=_W["vmap"])
    trajs = []
    for key, steps in out["steps"].items():
        if not steps:
            continue
        trajs.append(dict(
            key=key,
            obs=np.stack([s["obs"] for s in steps]).astype(np.float32),
            g=np.stack([s["gstate"] for s in steps]).astype(np.float32),
            act=np.stack([s["extra"]["raw"] for s in steps]).astype(np.float32),
            logp=np.array([s["extra"]["logp"] for s in steps], np.float32),
            alive=np.array([s["alive"] for s in steps], np.float32),
            rew=np.array([s["rew"] for s in steps], np.float32)))
    return dict(result=out["result"], trajs=trajs)


# --------------------------------------------------------------------------- export
def export_run(algo, init_run, out_dir, it, info, model_cfg, args):
    os.makedirs(out_dir, exist_ok=True)
    state = dict(model={f"policy.{k}": v for k, v in algo.actor.state_dict().items()},
                 algo="bc", obs_dim=int(info["obs_dim"]), act_dim=int(info["act_dim"]),
                 action_mode="tactical", act_scale=np.asarray(info["act_scale"]).tolist(),
                 config=dict(model=model_cfg), step=it, source="mappo")
    with open(os.path.join(out_dir, "best.pt"), "wb") as f:
        torch.save(state, f)
    for name in ("norm_stats.npz", "meta.json"):
        shutil.copy(os.path.join(init_run, name), os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "mappo.json"), "w", encoding="utf-8") as f:
        json.dump(dict(init=init_run, iteration=it, args=vars(args)), f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="offline tactical run (BC or IQL) to start from")
    ap.add_argument("--prior", default=None, help="run to anchor the KL to (default: --init)")
    ap.add_argument("--arena", default=None)
    ap.add_argument("--vis-map", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--own", default="claude_opus55_xhigh_team")
    ap.add_argument("--opponents", default=TRAIN_OPPONENTS)
    ap.add_argument("--enemy-info", default="full", choices=["full", "vision"])
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--episodes", type=int, default=None, help="matches per iteration (default: workers)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr-actor", type=float, default=5e-5)
    ap.add_argument("--lr-critic", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--critic-warmup", type=int, default=2,
                    help="iterations that fit the critic only, actor frozen")
    ap.add_argument("--critic-hidden", type=int, default=256)
    ap.add_argument("--export-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    C.ensure_utf8_stdout()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    episodes = args.episodes or args.workers
    os.makedirs(args.out, exist_ok=True)

    actor, _, info, meta, model_cfg = load_init(args.init)
    prior = load_init(args.prior)[0] if args.prior else None
    if meta.get("vis_map") and not args.vis_map:
        raise SystemExit(f"{args.init} was trained with a vis_map; pass --vis-map")
    gdim = int(info["obs_dim"]) + 40           # full-information view + global_extras
    algo = MAPPO(actor, gdim, prior, hidden=args.critic_hidden, clip=args.clip,
                 kl_coef=args.kl_coef, ent_coef=args.ent_coef)
    opt_a = torch.optim.Adam(algo.actor.parameters(), lr=args.lr_actor)
    opt_c = torch.optim.Adam(algo.critic.parameters(), lr=args.lr_critic)
    state_path = os.path.join(args.out, "mappo_state.pt")
    start = 1
    if args.resume and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st = torch.load(f, map_location="cpu")
        algo.load_state_dict(st["algo"])
        opt_a.load_state_dict(st["opt_a"])
        opt_c.load_state_dict(st["opt_c"])
        rng = np.random.default_rng(st["rng_seed"])
        start = st["iteration"] + 1
        print(f"[mappo] resumed at iteration {start}")
    log = C.CsvLogger(os.path.join(args.out, "train_log.csv"))
    actor_path = os.path.join(args.out, "actor_rollout.pt")
    opponents = args.opponents.split(",")
    print(f"[mappo] init={args.init} gdim={gdim} episodes/iter={episodes} workers={args.workers} "
          f"opponents={opponents}", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.arena, args.vis_map, args.init)) as pool:
        for it in range(start, args.iters + 1):
            t0 = time.time()
            with open(actor_path, "wb") as f:
                torch.save(algo.actor.state_dict(), f)
            jobs = [dict(actor_path=actor_path, own=args.own, duration=args.duration,
                         enemy_info=args.enemy_info, seed=int(10_000 * it + j + 100_000 * args.seed),
                         side=str(rng.choice(["red", "blue"])), opponent=str(rng.choice(opponents)))
                    for j in range(episodes)]
            rolls = pool.map(rollout, jobs)
            t_roll = time.time() - t0

            # ---- batch + GAE (values from the centralised critic) ----
            trajs = [tr for r in rolls for tr in r["trajs"]]
            algo.critic.gnorm.update(torch.as_tensor(np.concatenate([t["g"] for t in trajs])))
            adv_l, ret_l, vold_l = [], [], []
            for tr in trajs:
                v = algo.values(torch.as_tensor(tr["g"])).numpy()
                adv, ret = gae(tr["rew"], v, args.gamma, args.lam)
                adv_l.append(adv); ret_l.append(ret); vold_l.append(v)
            cat = lambda k: torch.as_tensor(np.concatenate([t[k] for t in trajs]))
            batch = dict(obs=None, act=cat("act"), logp=cat("logp"), g=cat("g"), valid=cat("alive"),
                         adv=torch.as_tensor(np.concatenate(adv_l)), ret=torch.as_tensor(np.concatenate(ret_l)),
                         v_old=torch.as_tensor(np.concatenate(vold_l)))
            mean, std = (torch.as_tensor(x) for x in (_norm_stats(args.init)))
            batch["obs"] = (cat("obs") - mean) / std
            algo.value_norm.update(batch["ret"])
            m = batch["valid"] > 0
            a = batch["adv"]
            batch["adv"] = (a - a[m].mean()) / (a[m].std() + 1e-8)
            ev = 1.0 - float(torch.var(batch["ret"] - batch["v_old"]) / (torch.var(batch["ret"]) + 1e-8))

            # ---- PPO epochs ----
            train_actor = it > args.critic_warmup
            N, agg, nmb = batch["valid"].shape[0], {}, 0
            algo.train()
            for _ in range(args.epochs):
                perm = torch.randperm(N)
                for s in range(0, N, args.minibatch):
                    idx = perm[s:s + args.minibatch]
                    mb = {k: v[idx] for k, v in batch.items()}
                    loss, met = algo(mb, train_actor=train_actor)
                    opt_a.zero_grad(set_to_none=True); opt_c.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(algo.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(algo.critic.parameters(), 0.5)
                    if train_actor:
                        opt_a.step()
                    opt_c.step()
                    for k, v in met.items():
                        agg[k] = agg.get(k, 0.0) + float(v)
                    nmb += 1
            algo.eval()

            res = [r["result"] for r in rolls]
            # every row carries every metric (CsvLogger fixes its header on the first row,
            # which is a critic-warmup iteration without the actor's metrics)
            mets = {k: agg[k] / max(nmb, 1) if k in agg else float("nan") for k in METRICS}
            row = dict(iteration=it, episodes=len(res), samples=int(N),
                       win_rate=float(np.mean([r["win"] for r in res])),
                       d_damage=float(np.mean([r["damage"] - r["damage_enemy"] for r in res])),
                       d_base=float(np.mean([r["base"] - r["base_enemy"] for r in res])),
                       ret_per_agent=float(np.mean([np.mean(list(r["returns"].values())) for r in res])),
                       explained_var=ev, actor_trained=int(train_actor),
                       rollout_s=round(t_roll, 1), update_s=round(time.time() - t0 - t_roll, 1),
                       **mets)
            log.log(row)
            print(f"[mappo] it {it}: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                 for k, v in row.items()), flush=True)
            with open(state_path, "wb") as f:
                torch.save(dict(algo=algo.state_dict(), opt_a=opt_a.state_dict(), opt_c=opt_c.state_dict(),
                                iteration=it, rng_seed=int(rng.integers(2**31))), f)
            if it % args.export_every == 0 or it == args.iters:
                export_run(algo, args.init, os.path.join(args.out, f"iter_{it:03d}"), it, info, model_cfg, args)
                export_run(algo, args.init, os.path.join(args.out, "latest"), it, info, model_cfg, args)


def _norm_stats(run_dir):
    z = np.load(os.path.join(run_dir, "norm_stats.npz"))
    return z["obs_mean"], z["obs_std"]


if __name__ == "__main__":
    main()
