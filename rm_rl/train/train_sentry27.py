"""Train the 2027 sentry's decision layer in Arena-1: tree imitation, then PPO self-play.

    python -m rm_rl.train.train_sentry27 --out rm_runs/sentry27 --workers 136 --iters 400

Phases
  1. Imitation (`--imitate-iters`, default 6): the learner's seats run the
     `--imitate` behaviour tree and the actor is fitted to its choices (critic
     too).  PPO then starts *at* the tree, so "beat the tree" is measured from
     the tree's own level instead of from a random policy.
  2. PPO with an opponent league.  Every match: a random own team script and a
     random opposing team script from `--train-teams` (teammates and opponents
     vary — the policy must not lean on one script's habits), the learner on a
     random side, and the opposing sentry drawn from `--opp-mix`:
        mirror     the current policy (both sides' trajectories are trained on)
        snapshot   a frozen past policy, prioritised by how often it still beats
                   the learner (PFSP, AlphaStar)
        bt         a behaviour tree (balanced / guard / assault)
        script     the opposing team script's own sentry
     A snapshot joins the league every `--snapshot-every` iterations.

Rollouts are asynchronous: every worker plays full matches back to back with the
newest policy available when its match starts; the learner trains on matches
whose policy is at most `--max-lag` versions old (PPO's ratio uses the recorded
behaviour log-probabilities, so a one-version lag is harmless).

Outputs (OUT/): train_log.csv, latest/ and iter_XXXX/ (policy.pt + spec.json,
read by rm_rl.eval.eval_sentry27 and by the robot), league/, state.pt (--resume).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from ..algos.macro_ppo import PPO, MultiHeadPolicy, gae, load_policy, save_policy
from ..arena27.spec import HEAD_SIZES, OTHER, export_spec
from . import common as C

# 2027 team scripts bundled with Arena-1 whose lineup has a slot-7 sentry.  The
# held-out ones (EVAL_TEAMS) never appear in training; rm_rl.eval.eval_sentry27
# uses them as unseen teammates and opponents.
TRAIN_TEAMS = ("claude_opus55_xhigh_2027_team,claude_opus55_xhigh_2027_hold,claude_opus55_xhigh_2027_squad,"
               "claude_opus55_xhigh_2027_distributed,dsv41_1017_heavy,dsv41_1017_pair,dsv41_1017_pair_omni,"
               "dsv41_1017_sentry,dsv41_2027_control,gemini38_2027_blitz,gemini38_2027_tech_overload,"
               "gpt61_2027_control,gpt61_2027_lineup_assault,gpt61_2027_lineup_omni,gpt61_2027_pressure")
EVAL_TEAMS = ("gemini38_2027_hunter,gpt61_2027_trident,gpt61_2027_lineup_serial,gpt61_2027_lineup_dual_arm,"
              "dsv41_1017_pair_s2772,dsv41_1017_sentry_dual_arm,step5_2027_echo")
TACTICS = ("balanced", "guard", "assault")
METRICS = ("v_loss", "pi_loss", "entropy", "approx_kl", "clipfrac", "imit_acc")


# --------------------------------------------------------------------------- workers
_W = {}


def _init_worker(arena_path):
    torch.set_num_threads(1)
    from ..sim.arena import load_arena
    _W.update(arena=load_arena(arena_path), cache={})


def _policy(path):
    cache = _W["cache"]
    if path not in cache:
        if len(cache) > 48:
            cache.clear()
        cache[path] = load_policy(path)
    return cache[path]


def rollout(job):
    from ..arena27.deciders import PolicyDecider, TreeDecider
    from ..arena27.match import MatchConfig, RewardConfig27, Seat, run_match
    L = job["learner_team"]
    O = OTHER[L]
    learner = TreeDecider(job["imitate"]) if job["imitate"] else PolicyDecider(_policy(job["policy"]))
    seats = {(L, s): Seat(learner, record=True, label="learner") for s in job["slots"]}
    opp = job["opponent"]
    if opp["kind"] == "mirror":
        seats.update({(O, s): Seat(learner, record=True, label="learner") for s in job["slots"]})
    elif opp["kind"] == "snapshot":
        dec = PolicyDecider(_policy(opp["path"]))
        seats.update({(O, s): Seat(dec, label=opp["name"]) for s in job["slots"]})
    elif opp["kind"] == "bt":
        dec = TreeDecider(opp["tactic"])
        seats.update({(O, s): Seat(dec, label=opp["name"]) for s in job["slots"]})
    cfg = MatchConfig(red=job["teams"]["red"], blue=job["teams"]["blue"], seed=job["seed"],
                      duration=job["duration"], decision_dt=job["decision_dt"], enemy_info=job["enemy_info"],
                      sentry_model=job["sentry_model"], global_state=True, reward=RewardConfig27(**job["reward"]))
    out = run_match(_W["arena"], cfg, seats)
    trajs = [dict(team=k[0], **tr) for k, tr in out["trajs"].items()]
    return dict(job=job, result=out["result"], trajs=trajs)


# --------------------------------------------------------------------------- league
class League:
    """Frozen past policies and the learner's running score against each (PFSP)."""

    def __init__(self, root, max_size=40):
        self.root, self.max_size = root, max_size
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "league.json")
        self.members = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def add(self, src, version):
        name = f"v{version:05d}"
        dst = os.path.join(self.root, f"{name}.pt")
        shutil.copy(src, dst)
        self.members[name] = dict(path=dst, version=version, games=0, score=0.0)
        if len(self.members) > self.max_size:          # drop the snapshot the learner beats most
            old = max((n for n in self.members if n != name),
                      key=lambda n: self._score(n) if self.members[n]["games"] >= 10 else -1)
            os.remove(self.members.pop(old)["path"])
        self.save()

    def _score(self, name):
        m = self.members[name]
        return (m["score"] + 1.0) / (m["games"] + 2.0)       # learner's score, prior 0.5

    def sample(self, rng):
        names = list(self.members)
        w = np.array([(1.0 - self._score(n)) ** 2 + 0.02 for n in names])
        return names[int(rng.choice(len(names), p=w / w.sum()))]

    def record(self, name, score):
        if name in self.members:
            self.members[name]["games"] += 1
            self.members[name]["score"] += score

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.members, f, indent=1)


def _parse_mix(text):
    mix = {}
    for part in text.split(","):
        k, v = part.split(":")
        mix[k.strip()] = float(v)
    bad = set(mix) - {"mirror", "snapshot", "bt", "script"}
    if bad:
        raise SystemExit(f"--opp-mix: unknown kinds {bad}")
    return mix


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arena", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--episodes", type=int, default=None, help="matches per update (default: workers)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--decision-dt", type=float, default=1.0)
    ap.add_argument("--slots", default="7", help="controlled slots, e.g. 7 or 3,4,7")
    ap.add_argument("--enemy-info", default="own", choices=["own", "team", "full"])
    ap.add_argument("--sentry-model", type=int, default=2771)
    ap.add_argument("--train-teams", default=TRAIN_TEAMS)
    ap.add_argument("--opp-mix", default="mirror:0.35,snapshot:0.3,bt:0.15,script:0.2")
    ap.add_argument("--imitate", default="balanced", help="behaviour tree to start from ('' = none)")
    ap.add_argument("--imitate-iters", type=int, default=6)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--league-size", type=int, default=40)
    ap.add_argument("--max-lag", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--imitate-epochs", type=int, default=10)
    ap.add_argument("--target-kl", type=float, default=0.03, help="stop an update's epochs beyond this KL")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-coef-final", type=float, default=0.002)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--reward", default="", help="overrides, e.g. control=0.02,win=10")
    ap.add_argument("--shaping-final", type=float, default=0.3, help="dense-reward scale at --iters")
    ap.add_argument("--export-every", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    C.ensure_utf8_stdout()
    # SIGTERM -> SystemExit, so the worker pool is torn down instead of orphaned
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    from ..arena27.match import RewardConfig27, probe_spec
    from ..sim.arena import load_arena

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    episodes = args.episodes or args.workers
    slots = [int(s) for s in args.slots.split(",")]
    teams = args.train_teams.split(",")
    mix = _parse_mix(args.opp_mix)
    reward0 = asdict(RewardConfig27())
    for kv in filter(None, args.reward.split(",")):
        k, v = kv.split("=")
        if k not in reward0:
            raise SystemExit(f"--reward: unknown term {k}")
        reward0[k] = float(v)

    arena = load_arena(args.arena)
    spec = probe_spec(arena)
    policy = MultiHeadPolicy(spec["obs_dim"], HEAD_SIZES, args.hidden, args.depth)
    algo = PPO(policy, spec["gdim"], args.hidden, args.depth, clip=args.clip, vf_coef=args.vf_coef,
               ent_coef=args.ent_coef).to(device)
    opt = torch.optim.Adam(algo.parameters(), lr=args.lr, eps=1e-5)
    league = League(os.path.join(args.out, "league"), args.league_size)
    vdir = os.path.join(args.out, "versions")
    os.makedirs(vdir, exist_ok=True)
    state_path = os.path.join(args.out, "state.pt")
    it, version, seed_ctr = 0, 0, 1_000_000 * args.seed
    if args.resume and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st = torch.load(f, map_location=device, weights_only=False)
        algo.load_state_dict(st["algo"])
        opt.load_state_dict(st["opt"])
        it, version, seed_ctr = st["iteration"], st["version"], st["seed_ctr"]
        rng = np.random.default_rng(st["rng"])
        print(f"[sentry27] resumed at iteration {it}, version {version}", flush=True)
    spec_json = export_spec(spec["arena"], args.decision_dt, spec["obs_names"],
                            extra=dict(enemy_info=args.enemy_info, sentry_model=args.sentry_model,
                                       slots=slots, args=vars(args)))

    def publish():
        path = os.path.join(vdir, f"v{version:05d}.pt")
        save_policy(algo.policy, path)
        for name in os.listdir(vdir):                  # keep the versions jobs may still load
            if name.endswith(".pt") and int(name[1:6]) < version - args.max_lag - 2:
                os.remove(os.path.join(vdir, name))
        return path

    def export(dirname):
        d = os.path.join(args.out, dirname)
        os.makedirs(d, exist_ok=True)
        save_policy(algo.policy, os.path.join(d, "policy.pt"), extra=dict(version=version, iteration=it))
        with open(os.path.join(d, "spec.json"), "w", encoding="utf-8") as f:
            json.dump(spec_json, f, ensure_ascii=False, indent=1)

    policy_path = publish()
    log = C.CsvLogger(os.path.join(args.out, "train_log.csv"))

    def make_job():
        nonlocal seed_ctr
        seed_ctr += 1
        imitating = it < args.imitate_iters and bool(args.imitate)
        kinds = {k: v for k, v in mix.items() if k != "snapshot" or league.members}
        names = list(kinds)
        kind = names[int(rng.choice(len(names), p=np.array([kinds[k] for k in names]) / sum(kinds.values())))]
        opp = dict(kind=kind, name=kind)
        if kind == "snapshot":
            n = league.sample(rng)
            opp.update(name=n, path=league.members[n]["path"])
        elif kind == "bt":
            tac = str(rng.choice(TACTICS))
            opp.update(tactic=tac, name=f"bt:{tac}")
        side = str(rng.choice(["red", "blue"]))
        shaping = reward0["shaping"] + (args.shaping_final - reward0["shaping"]) * min(it / max(args.iters, 1), 1.0)
        return dict(version=version, policy=policy_path, imitate=args.imitate if imitating else "",
                    learner_team=side, slots=slots, opponent=opp, seed=int(seed_ctr),
                    teams={side: str(rng.choice(teams)), OTHER[side]: str(rng.choice(teams))},
                    duration=args.duration, decision_dt=args.decision_dt, enemy_info=args.enemy_info,
                    sentry_model=args.sentry_model, reward=dict(reward0, shaping=shaping))

    results = queue.Queue()
    print(f"[sentry27] obs={spec['obs_dim']} g={spec['gdim']} heads={HEAD_SIZES} workers={args.workers} "
          f"episodes/update={episodes} device={device} slots={slots}", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.arena,)) as pool:
        def submit():
            pool.apply_async(rollout, (make_job(),), callback=results.put, error_callback=results.put)

        for _ in range(args.workers):
            submit()
        errors = 0
        while it < args.iters:
            t0 = time.time()
            rolls, stale = [], 0
            while len(rolls) < episodes:
                r = results.get()
                submit()
                if isinstance(r, BaseException):
                    errors += 1
                    print(f"[sentry27] rollout error ({errors}): {r!r}", flush=True)
                    if errors > max(10, args.iters):
                        raise r
                    continue
                opp = r["job"]["opponent"]
                if opp["kind"] == "snapshot":
                    w = r["result"]["winner"]
                    L = r["job"]["learner_team"]
                    league.record(opp["name"], 1.0 if w == L else 0.0 if w == OTHER[L] else 0.5)
                if version - r["job"]["version"] > args.max_lag or bool(r["job"]["imitate"]) != (
                        it < args.imitate_iters and bool(args.imitate)):
                    stale += 1
                    continue
                rolls.append(r)
            t_roll = time.time() - t0
            imitation = it < args.imitate_iters and bool(args.imitate)
            row = update(algo, opt, rolls, args, device, imitation, it)
            it += 1
            version += 1
            policy_path = publish()
            if not imitation and it % args.snapshot_every == 0:
                league.add(policy_path, version)
            league.save()
            row.update(_rollout_stats(rolls), iteration=it, version=version, phase="imitate" if imitation else "ppo",
                       stale=stale, errors=errors, rollout_s=round(t_roll, 1),
                       update_s=round(time.time() - t0 - t_roll, 1), league=len(league.members),
                       ent_coef=round(algo.ent_coef, 5), shaping=rolls[0]["job"]["reward"]["shaping"])
            log.log(row)
            print("[sentry27] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                           for k, v in row.items()), flush=True)
            with open(state_path, "wb") as f:
                torch.save(dict(algo=algo.state_dict(), opt=opt.state_dict(), iteration=it, version=version,
                                seed_ctr=seed_ctr, rng=int(rng.integers(2**31))), f)
            export("latest")
            if it % args.export_every == 0 or it == args.iters:
                export(f"iter_{it:04d}")
        pool.terminate()


def update(algo, opt, rolls, args, device, imitation, it):
    trajs = [tr for r in rolls for tr in r["trajs"]]
    cat = lambda k, dt=None: torch.as_tensor(np.concatenate([t[k] for t in trajs]), dtype=dt, device=device)
    obs, g = cat("obs", torch.float32), cat("g", torch.float32)
    algo.policy.obs_norm.update(obs)
    algo.critic.gnorm.update(g)
    adv_l, ret_l, v_l = [], [], []
    for tr in trajs:
        v = algo.values(torch.as_tensor(tr["g"], device=device)).cpu().numpy()
        adv, ret = gae(tr["rew"], v, args.gamma, args.lam)
        adv_l.append(adv); ret_l.append(ret); v_l.append(v)
    ret = torch.as_tensor(np.concatenate(ret_l), device=device)
    algo.value_norm.update(ret)
    adv = torch.as_tensor(np.concatenate(adv_l), device=device)
    batch = dict(obs=obs, g=g, mask=cat("mask", torch.bool), act=cat("act", torch.long),
                 logp=cat("logp", torch.float32), ret=ret, v_old=torch.as_tensor(np.concatenate(v_l), device=device),
                 adv=(adv - adv.mean()) / (adv.std() + 1e-8))
    ev = 1.0 - float(torch.var(batch["ret"] - batch["v_old"]) / (torch.var(batch["ret"]) + 1e-8))
    frac = min(max(it - args.imitate_iters, 0) / max(args.iters - args.imitate_iters, 1), 1.0)
    algo.ent_coef = args.ent_coef + (args.ent_coef_final - args.ent_coef) * frac
    n, agg, nmb, epochs = obs.shape[0], {}, 0, 0
    algo.train()
    for _ in range(args.imitate_epochs if imitation else args.epochs):
        perm = torch.randperm(n, device=device)
        kl = []
        for s in range(0, n, args.minibatch):
            idx = perm[s:s + args.minibatch]
            loss, met = algo({k: v[idx] for k, v in batch.items()}, imitation=imitation)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(algo.parameters(), 0.5)
            opt.step()
            for k, v in met.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            kl.append(float(met.get("approx_kl", 0.0)))
            nmb += 1
        epochs += 1
        if not imitation and np.mean(kl) > args.target_kl:
            break
    algo.eval()
    row = {k: agg[k] / nmb if k in agg else float("nan") for k in METRICS}
    row.update(samples=int(n), epochs=epochs, explained_var=round(ev, 4),
               ret_mean=float(np.mean([t["rew"].sum() for t in trajs])))
    return row


def _rollout_stats(rolls):
    """Learner score per opponent kind, and the learner seats' behaviour."""
    out, by = {}, {}
    for r in rolls:
        L, w = r["job"]["learner_team"], r["result"]["winner"]
        by.setdefault(r["job"]["opponent"]["kind"], []).append(1.0 if w == L else 0.0 if w == OTHER[L] else 0.5)
    for k in ("mirror", "snapshot", "bt", "script"):      # every row has every column (CsvLogger)
        v = by.get(k, [])
        out[f"score_{k}"] = round(float(np.mean(v)), 4) if v else float("nan")
        out[f"n_{k}"] = len(v)
    seats = [s for r in rolls for key, s in r["result"]["seats"].items()
             if s["label"] == "learner" and key.startswith(r["job"]["learner_team"])]
    for k in ("deaths", "damage", "no_ammo", "own_central", "alive"):
        out[f"seat_{k}"] = round(float(np.mean([s[k] for s in seats])), 4) if seats else float("nan")
    lead = [r["result"]["control"][r["job"]["learner_team"]] - r["result"]["control"][OTHER[r["job"]["learner_team"]]]
            for r in rolls]
    out["control_lead"] = round(float(np.mean(lead)), 2)
    return out


if __name__ == "__main__":
    main()
