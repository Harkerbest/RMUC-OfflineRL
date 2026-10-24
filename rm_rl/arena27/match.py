"""One RMUC 2027 match in Arena-1 with any mix of scripts, trees and policies.

Each team is a full-team script (`ScriptedSide`); any (team, slot) can be taken
over by a *seat*: a decider (policy, tree, random) that chooses a macro action
every `decision_dt` seconds, executed at 10 Hz by that seat's `Executor`.  Seats
on both teams make self-play; a seat with `record=True` returns its trajectory.

Reward (per decision, from the team's point of view; `shaping` scales the dense
terms so they can be annealed toward the sparse result):
    control * d(control-degree lead)       the first 2027 timeout criterion
  + damage  * d(team attack-damage lead)   the second
  + own_damage * d(damage this robot dealt)
  - death   * (this robot died)
  + win     * (+1 win, -1 loss, 0 draw)    at the end
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..sim.arena import ArenaMatch, ScriptedSide
from .executor import Executor
from .inputs import snapshot_from_arena
from .obs import ObsBuilder, global_extras
from .spec import HEADS, MAP_2027, OTHER, waypoints


@dataclass
class RewardConfig27:
    control: float = 0.02
    damage: float = 1 / 2000
    own_damage: float = 1 / 1000
    death: float = 0.5
    win: float = 10.0
    shaping: float = 1.0


@dataclass
class Seat:
    decider: object
    record: bool = False
    label: str = ""


@dataclass
class MatchConfig:
    red: str
    blue: str
    seed: int = 0
    duration: float = 420.0
    map_name: str = MAP_2027
    decision_dt: float = 1.0
    enemy_info: str = "own"
    sentry_model: int | None = 2771       # slot-7 model fielded by teams that have seats
    lineup_overrides: dict | None = None  # {team: {slot: model}}; replaces the rule above
    global_state: bool = False            # build critic states for recorded seats
    reward: RewardConfig27 = field(default_factory=RewardConfig27)


def script_path(name: str) -> str:
    return name if "/" in name else f"examples/{name}"


class _Tracker:
    def __init__(self, team, slot, cfg: RewardConfig27):
        self.team, self.enemy, self.slot, self.cfg = team, OTHER[team], slot, cfg
        self.prev, self.deaths, self.was_alive = None, 0, True

    def _state(self, frame):
        tm = frame["teams"]
        me = next(r for r in frame["robots"] if r["team"] == self.team and int(r["slot"]) == self.slot)
        died = self.was_alive and not me["alive"]
        self.was_alive = me["alive"]
        self.deaths += died
        return (tm[self.team].get("control", 0.0) - tm[self.enemy].get("control", 0.0),
                tm[self.team].get("attack_damage", 0.0) - tm[self.enemy].get("attack_damage", 0.0),
                me.get("damage", 0.0), died)

    def step(self, frame) -> float:
        cur = self._state(frame)
        if self.prev is None:
            self.prev = cur
            return 0.0
        c = self.cfg
        dense = (c.control * (cur[0] - self.prev[0]) + c.damage * (cur[1] - self.prev[1])
                 + c.own_damage * (cur[2] - self.prev[2]) - c.death * cur[3])
        self.prev = cur
        return c.shaping * dense

    def terminal(self, winner) -> float:
        return self.cfg.win * (1.0 if winner == self.team else -1.0 if winner == self.enemy else 0.0)


def run_match(arena, cfg: MatchConfig, seats: dict) -> dict:
    """seats: {(team, slot): Seat}.  Returns dict(result, trajs={key: arrays})."""
    sides = {t: ScriptedSide(arena, script_path(getattr(cfg, t)), t) for t in ("red", "blue")}
    over = cfg.lineup_overrides
    if over is None:
        over = {t: {"7": cfg.sentry_model} for t, s in seats if s == 7 and cfg.sentry_model}
    m = ArenaMatch(arena, sides["red"], sides["blue"], seed=cfg.seed, duration=cfg.duration,
                   map_name=cfg.map_name, lineup_overrides=over or None)
    wps = {t: waypoints(m.frame["arena"], t) for t in ("red", "blue")}
    builders = {k: ObsBuilder(k[0], k[1], wps[k[0]]) for k in seats}
    full = {k: ObsBuilder(k[0], k[1], wps[k[0]]) for k, s in seats.items() if s.record and cfg.global_state}
    execs = {k: Executor(arena.sdk, k[0], k[1], wps[k[0]]) for k in seats}
    trackers = {k: _Tracker(k[0], k[1], cfg.reward) for k in seats}
    steps = {k: [] for k, s in seats.items() if s.record}
    stats = {k: defaultdict(float) for k in seats}
    returns = {k: 0.0 for k in seats}
    groups = defaultdict(list)
    for k, s in seats.items():
        groups[id(s.decider)].append(k)
    next_dec, n_dec, t_dec, t0 = 0.0, 0, 0.0, time.time()
    try:
        while not m.finished:
            team_obs = m.decide()
            if seats and m.time + 1e-6 >= next_dec:
                ts = time.time()
                next_dec += cfg.decision_dt
                reqs = {}
                for k, seat in seats.items():
                    team, slot = k
                    inp = snapshot_from_arena(team_obs[team], slot, cfg.enemy_info,
                                              frame=m.frame if cfg.enemy_info == "full" else None)
                    ob = builders[k]
                    o = ob(inp)
                    r = trackers[k].step(m.frame)
                    returns[k] += r
                    if k in steps and steps[k]:
                        steps[k][-1]["rew"] += r
                    g = None
                    if k in full:
                        fi = snapshot_from_arena(team_obs[team], slot, "full", frame=m.frame)
                        g = np.concatenate([full[k](fi), global_extras(m.frame, team)])
                    reqs[k] = dict(key=k, team=team, slot=slot, obs=o, mask=ob.mask(inp), inp=inp,
                                   wps=wps[team], g=g, builder=ob)
                    st, s = stats[k], inp["self"]
                    st["decisions"] += 1
                    st["alive"] += s["alive"]
                    st["no_ammo"] += s["alive"] and s["ammo"] <= 0
                    st["own_central"] += s["alive"] and team_obs[team]["zones"].get(f"{team}_central", {}) \
                        .get("occupants", []).count(f"{team}-{slot}") > 0
                for keys in groups.values():
                    batch = [reqs[k] for k in keys]
                    for req, (idx, extra) in zip(batch, seats[keys[0]].decider(batch)):
                        k = req["key"]
                        names = {h: choices[int(i)] for (h, choices), i in zip(HEADS, idx)}
                        execs[k].latch(names, req["inp"], builders[k].last_seen)
                        stats[k]["choice_" + names["goal"]] += 1
                        if k in steps:
                            steps[k].append(dict(obs=req["obs"], mask=req["mask"], g=req["g"],
                                                 act=np.asarray(idx, np.int64), logp=extra.get("logp", 0.0),
                                                 rew=0.0))
                n_dec += 1
                t_dec += time.time() - ts
            for (team, slot), ex in execs.items():
                m.override(team, slot, ex.action(team_obs[team], m.commands[team].get(str(slot))))
            m.advance()
        frame = m.frame
    finally:
        m.close()

    winner = frame.get("winner")
    for k in seats:
        r = trackers[k].step(frame) + trackers[k].terminal(winner)
        returns[k] += r
        if k in steps and steps[k]:
            steps[k][-1]["rew"] += r
    trajs = {}
    for k, st in steps.items():
        if not st:
            continue
        trajs[k] = dict(obs=np.stack([s["obs"] for s in st]).astype(np.float32),
                        mask=np.stack([s["mask"] for s in st]),
                        act=np.stack([s["act"] for s in st]),
                        logp=np.asarray([s["logp"] for s in st], np.float32),
                        rew=np.asarray([s["rew"] for s in st], np.float32),
                        g=np.stack([s["g"] for s in st]).astype(np.float32) if st[0]["g"] is not None else None)
    tm = frame["teams"]
    seat_info = {}
    for k, s in seats.items():
        st, n = stats[k], max(stats[k]["decisions"], 1)
        me = next(r for r in frame["robots"] if r["team"] == k[0] and int(r["slot"]) == k[1])
        seat_info[f"{k[0]}-{k[1]}"] = dict(
            label=s.label, ret=round(returns[k], 4), deaths=trackers[k].deaths, damage=me.get("damage", 0.0),
            alive=round(st["alive"] / n, 4), no_ammo=round(st["no_ammo"] / max(st["alive"], 1), 4),
            own_central=round(st["own_central"] / max(st["alive"], 1), 4),
            goals={g[7:]: int(v) for g, v in st.items() if g.startswith("choice_")})
    result = dict(
        red=cfg.red, blue=cfg.blue, seed=cfg.seed, winner=winner, reason=frame.get("reason"),
        time=frame["time"], control={t: tm[t].get("control", 0.0) for t in ("red", "blue")},
        damage={t: tm[t].get("attack_damage", 0.0) for t in ("red", "blue")},
        base={t: frame["bases"][t]["hp"] for t in ("red", "blue")},
        seats=seat_info, decisions=n_dec, decide_ms=round(1000 * t_dec / max(n_dec, 1), 2),
        wall=round(time.time() - t0, 1))
    return dict(result=result, trajs=trajs)


def probe_spec(arena, map_name=MAP_2027, team_script="claude_opus55_xhigh_2027_team"):
    """Observation / critic-state sizes and feature names, from the first frame of a match."""
    side = lambda t: ScriptedSide(arena, script_path(team_script), t)
    m = ArenaMatch(arena, side("red"), side("blue"), seed=0, duration=5.0, map_name=map_name)
    try:
        obs = m.decide()
        wps = waypoints(m.frame["arena"], "red")
        ob, fb = ObsBuilder("red", 7, wps), ObsBuilder("red", 7, wps)
        o = ob(snapshot_from_arena(obs["red"], 7, "own"))
        g = np.concatenate([fb(snapshot_from_arena(obs["red"], 7, "full", frame=m.frame)),
                            global_extras(m.frame, "red")])
        return dict(obs_dim=len(o), gdim=len(g), obs_names=ob.names, arena=m.frame["arena"])
    finally:
        m.close()
