"""One closed-loop Arena-1 match with a tactical policy on some of a team's slots.

Shared by closed-loop evaluation (rm_rl.eval.closed_loop) and online
fine-tuning (rm_rl.train.train_mappo):

* the team script (`own`) plays every slot of `team`; the opponent script plays
  the other side;
* on the controlled slots (default: the three infantry 3/4/7 — 7 is Arena's
  2026 stand-in for the sentry and fills the log's 哨兵 column) a tactical
  policy decides once per second from the 161-D observation, and
  `TacticalController` executes it at 10 Hz;
* the per-second referee log of the match is kept (`RefereeLog`), so rewards are
  `reward.compute_reward` on exactly the rows a dataset build would see.

`agent_fn(requests) -> {key: (a_norm, extra)}` is called once per decision with
one request per controlled slot:
    dict(key, slot, agent_type, obs, gstate|None, t, last_reward, alive)
`a_norm` is the normalised tactical vector; `extra` is stored with the step
(e.g. log-probabilities for PPO).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..data.reward import RewardConfig
from ..deploy import decode_action
from .arena import ArenaMatch, ScriptedSide, TEAM_CAMP
from .control import TacticalController
from .obs_adapter import ObsAdapter
from .referee_log import RefereeLog
from .reward import step_rewards

CONTROLLED = {3: "步兵3", 4: "步兵4", 7: "哨兵"}
OTHER = {"red": "blue", "blue": "red"}


@dataclass
class EpisodeConfig:
    team: str = "red"
    own: str = "examples/claude_opus55_xhigh_team"
    opponent: str = "examples/claude_opus55_xhigh_hold"
    seed: int = 0
    duration: float = 420.0
    enemy_info: str = "full"            # policy observation: "full" | "vision"
    controlled: dict = field(default_factory=lambda: dict(CONTROLLED))
    use_team_prior: bool = True
    record_global: bool = False         # also build privileged critic states
    reward: RewardConfig = field(default_factory=RewardConfig)


def global_extras(frame: dict, team: str) -> np.ndarray:
    """Privileged engine state the actor never sees: every robot's ammo, heat and
    level, both teams' coins and damage dealt (team-relative order)."""
    feats = []
    for side in (team, OTHER[team]):
        for r in sorted((r for r in frame["robots"] if r["team"] == side), key=lambda r: r["slot"]):
            feats += [min(r["ammo"], 400) / 400.0, r["heat"] / max(r["max_heat"], 1.0),
                      r.get("level", 1) / 10.0]
        t = frame["teams"][side]
        feats += [t.get("resources", 0.0) / 1000.0, t.get("attack_damage", 0.0) / 5000.0]
    return np.asarray(feats, np.float32)


def run_episode(arena, cfg: EpisodeConfig, agent_fn, vis_map=None, sides=None,
                keep_log=False) -> dict:
    """Play one match. Returns dict(result=..., steps={key: [step dicts]}[, log=RefereeLog])."""
    team, enemy = cfg.team, OTHER[cfg.team]
    sides = sides or {}
    own = sides.get(team) or ScriptedSide(arena, cfg.own, team)
    opp = sides.get(enemy) or ScriptedSide(arena, cfg.opponent, enemy)
    m = ArenaMatch(arena, *((own, opp) if team == "red" else (opp, own)),
                   seed=cfg.seed, duration=cfg.duration)
    names = {team: own.name, enemy: opp.name}
    log = RefereeLog(cfg.seed, names)
    adapter = ObsAdapter(log, vis_map=vis_map, use_team_prior=cfg.use_team_prior,
                         enemy_info=cfg.enemy_info)
    ctrl = TacticalController(arena.sdk)
    camp = TEAM_CAMP[team]
    keys = {slot: f"{team}-{slot}" for slot in cfg.controlled}
    steps = {k: [] for k in keys.values()}
    returns = {k: 0.0 for k in keys.values()}
    log.on_frame(m.frame)
    next_decision, decide_s, n_dec = 1, 0.0, 0
    t_wall = time.time()
    try:
        while not m.finished:
            team_obs = m.decide()
            if cfg.controlled and log.last_second >= next_decision:
                t0 = time.time()
                next_decision = log.last_second + 1
                game = adapter.refresh()
                tsec = game.T                      # rows 1..T recorded; decide at second T
                visible = {e["slot"] for e in team_obs[team]["enemies"]}
                robots = {r["slot"]: r for r in team_obs[team]["robots"]}
                reqs = []
                for slot, rtype in cfg.controlled.items():
                    key = keys[slot]
                    last_r = float(step_rewards(game, camp, rtype, cfg.reward)[-1]) if tsec >= 2 else 0.0
                    returns[key] += last_r
                    reqs.append(dict(
                        key=key, slot=slot, agent_type=rtype, t=tsec - 1, last_reward=last_r,
                        alive=bool(robots[slot]["alive"]),
                        obs=adapter.observe(team, rtype, visible),
                        gstate=(np.concatenate([adapter.full_obs(team, rtype),
                                                global_extras(m.frame, team)])
                                if cfg.record_global else None)))
                out = agent_fn(reqs)
                for req in reqs:
                    a, extra = out[req["key"]]
                    dec = decode_action(np.asarray(a, np.float32), _act_scale(agent_fn),
                                        camp, "tactical")
                    ctrl.command(str(req["slot"]), dec, robots[req["slot"]])
                    steps[req["key"]].append(dict(t=req["t"], obs=req["obs"], gstate=req["gstate"],
                                                  act=np.asarray(a, np.float32), alive=req["alive"],
                                                  extra=extra))
                decide_s += time.time() - t0
                n_dec += 1
            for slot in cfg.controlled:
                m.override(team, slot, ctrl.action(str(slot), team_obs[team],
                                                   m.commands[team].get(str(slot))))
            for f in m.advance():
                log.on_frame(f)
        frame = m.frame
    finally:
        m.close()

    # rewards for every logged transition, terminal terms included
    game = adapter.refresh()
    winner = frame["winner"]
    winner_camp = TEAM_CAMP.get(winner) if winner else None
    for slot, rtype in cfg.controlled.items():
        r = step_rewards(game, camp, rtype, cfg.reward, winner_camp, terminal=True)
        for s in steps[keys[slot]]:
            s["rew"] = float(r[s["t"]]) if s["t"] < len(r) else 0.0
        returns[keys[slot]] = float(r.sum())
    tm = frame["teams"]
    result = dict(
        team=team, opponent=opp.name, own=own.name, seed=cfg.seed, winner=winner,
        win=1.0 if winner == team else 0.0 if winner == enemy else 0.5,
        reason=frame["reason"], time=frame["time"],
        base=frame["bases"][team]["hp"], base_enemy=frame["bases"][enemy]["hp"],
        outpost=frame["outposts"][team]["hp"], outpost_enemy=frame["outposts"][enemy]["hp"],
        damage=tm[team].get("attack_damage", 0.0), damage_enemy=tm[enemy].get("attack_damage", 0.0),
        kills=tm[team].get("kills", 0), kills_enemy=tm[enemy].get("kills", 0),
        returns=returns, decisions=n_dec,
        decide_ms=round(1000 * decide_s / max(n_dec, 1), 2), wall=round(time.time() - t_wall, 1))
    out = dict(result=result, steps=steps)
    if keep_log:
        out["log"] = log
    return out


def _act_scale(agent_fn):
    scale = getattr(agent_fn, "act_scale", None)
    if scale is None:
        raise AttributeError("agent_fn must expose act_scale (the run's action scale)")
    return np.asarray(scale, np.float32)
