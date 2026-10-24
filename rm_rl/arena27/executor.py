"""Macro action -> 10 Hz Arena-1 commands for one robot.

This is the simulator's stand-in for the robot's own navigation stack and
auto-aim, and the same executor runs under every decision maker (policy, tree,
random), so evaluations compare decisions only.

    goal      `sdk.navigate` to the waypoint; a robot that makes no progress for
              1.5 s backs off at an angle for 0.7 s; the supercap boost (`power`)
              is drawn only while stalled.  `chase` closes to `chase_standoff`
              metres of the chosen enemy's last sensed position.
    target    auto-aim over enemies the robot's *own* camera sees, ordered by the
              priority; structures (a hostile outpost, an enemy base that can be
              damaged) when the priority asks for them or no robot is in reach.
              Fires when aligned, `sdk.can_fire`, and 20 heat of margin (none
              needed while attack posture is enhanced).  `hold_fire` still tracks.
    posture / enhance  sent when they differ from the robot's state and the 5 s
              switch cooldown has passed.
    supply    sent once per decision: ammo is bought locally (100 rounds) on an
              exchange card, otherwise remotely (100 rounds, 150 coins).
"""
from __future__ import annotations

import math

from .spec import OTHER, to_team, wrap

_GROUND = ("heavy", "infantry", "sentry")


def _dist(a, b):
    return math.hypot(a["x"] - b[0], a["y"] - b[1])


class Executor:
    def __init__(self, sdk, team: str, slot: int, wps: dict, reach=10.0, aim_tol=0.14, arrive=0.35,
                 chase_standoff=3.5):
        self.sdk, self.team, self.slot, self.wps = sdk, team, int(slot), wps
        self.reach, self.aim_tol, self.arrive, self.standoff = reach, aim_tol, arrive, chase_standoff
        self.act, self.oneshot, self.on_exchange, self.chase = None, False, False, None
        self.trail, self.tries, self.escape_until, self.escape = [], 0, -1.0, None

    # ------------------------------------------------------------------ decisions
    def latch(self, act: dict, inp: dict, memory: dict):
        """act: head -> choice name.  memory: ObsBuilder.last_seen (team frame)."""
        self.act, self.oneshot, self.on_exchange = act, act["supply"] != "none", inp["self"]["on_exchange"]
        self.chase = None
        if act["goal"] == "chase" and memory:
            t, enemies = inp["time"], inp["enemies"]
            me = to_team(inp["self"]["x"], inp["self"]["y"], self.team)
            fresh = {s: m for s, m in memory.items() if t - m[0] <= 3.0}
            if fresh:
                key = {"weakest": lambda s: (enemies.get(s, {}).get("hp", 1e9), _d(me, fresh[s])),
                       "heavy_first": lambda s: (enemies.get(s, {}).get("role") != "heavy", _d(me, fresh[s]))
                       }.get(act["target"], lambda s: _d(me, fresh[s]))
                slot = min(fresh, key=key)
                self.chase = to_team(fresh[slot][1], fresh[slot][2], self.team)

    # ------------------------------------------------------------------ 10 Hz
    def action(self, obs: dict, scripted: dict | None = None) -> dict:
        sdk, a = self.sdk, self.act
        robot = next(r for r in obs["robots"] if int(r["slot"]) == self.slot)
        out = {}
        if scripted and "profile" in scripted:
            out["profile"] = scripted["profile"]      # infantry hardware choice at tick 0
        if a is None:
            return {**out, "move": [0.0, 0.0], "fire": False}
        t = obs["time"]
        if not robot["alive"]:
            if self.oneshot and a["supply"] == "revive":
                out["revive"] = True
            self.oneshot = False
            return out
        target = self._target(robot, obs)
        out.update(self._drive(robot, self._goal(robot, obs), obs["arena"], t))
        out["fire"] = False
        if target is not None:
            aim = sdk.aim_at(robot, target)
            out["aim"], out["pitch"] = aim, sdk.aim_pitch(robot, target)
            free = robot.get("posture") == "attack" and robot.get("posture_enhanced")
            out["fire"] = bool(a["target"] != "hold_fire" and abs(wrap(aim - robot["aim"])) < self.aim_tol
                               and sdk.can_fire(robot)
                               and (free or robot["heat"] + 20 <= robot["max_heat"]))
        if robot.get("role") == "sentry" and t >= (robot.get("posture_ready_at") or 0.0):
            if a["posture"] != "keep" and robot.get("posture") != a["posture"]:
                out["posture"] = a["posture"]
            elif bool(robot.get("posture_enhanced")) != (a["enhance"] == "on"):
                out["posture_enhanced"] = a["enhance"] == "on"
        if self.oneshot:
            if a["supply"] == "buy_ammo":
                out["buy_ammo"] = {"amount": 100, "remote": not self.on_exchange}
            elif a["supply"] == "remote_heal":
                out["remote_heal"] = True
            self.oneshot = False
        return out

    # ------------------------------------------------------------------ helpers
    def _goal(self, robot, obs):
        g = self.act["goal"]
        if g == "hold":
            return None
        if g == "chase":
            if self.chase is None:
                return None
            ex, ey = self.chase
            d = _dist(robot, (ex, ey))
            if d <= self.standoff:
                return None
            k = (d - self.standoff) / d
            return (robot["x"] + (ex - robot["x"]) * k, robot["y"] + (ey - robot["y"]) * k)
        return self.wps[g]

    def _drive(self, robot, goal, arena, t):
        if goal is None or _dist(robot, goal) <= self.arrive:
            self.trail.clear()
            return {"move": [0.0, 0.0]}
        if t < self.escape_until:
            return {"move": self.escape, "heading": robot["heading"]}
        mv = self.sdk.navigate(robot, goal, arena)
        out = {"move": [float(mv[0]), float(mv[1])]}
        if getattr(mv, "heading", None) is not None:
            out["heading"] = mv.heading
        self.trail.append((t, robot["x"], robot["y"]))
        while t - self.trail[0][0] > 1.5:
            self.trail.pop(0)
        t0, x0, y0 = self.trail[0]
        if t - t0 >= 1.4 and math.hypot(robot["x"] - x0, robot["y"] - y0) < 0.1:
            self.tries += 1
            ang = math.atan2(out["move"][1], out["move"][0]) + math.pi + (0.9 if self.tries % 2 else -0.9)
            self.escape = [0.7 * math.cos(ang), 0.7 * math.sin(ang)]
            self.escape_until = t + 0.7
            self.trail.clear()
        if math.hypot(robot.get("vx", 0.0), robot.get("vy", 0.0)) < 0.3 and math.hypot(*out["move"]) > 0.2:
            out["power"] = 200
        return out

    def _structure(self, robot, obs):
        sdk, team = self.sdk, self.team
        enemy = OTHER[team]
        for site in (enemy, team):
            o = obs["outposts"][site]
            if o["hp"] > 0 and o.get("owner", site) != team:
                o = dict(o, kind="outpost")
                if sdk.ballistic_reachable(robot, o):
                    return o
        p = obs.get("positions", {}).get(f"{enemy}_outpost", {})
        protected = p.get("controller") == enemy and p.get("control") == "full"
        base = obs["bases"][enemy]
        if not protected and base["hp"] > 0:
            base = dict(base, kind="base")
            if sdk.ballistic_reachable(robot, base):
                return base
        return None

    def _target(self, robot, obs):
        sdk, mode = self.sdk, self.act["target"]
        if mode == "structure_first":
            s = self._structure(robot, obs)
            if s is not None:
                return s
        ids = set((obs.get("vision", {}).get(str(self.slot)) or {}).get("enemies", []))
        foes = [e for e in obs["enemies"]
                if e["id"] in ids and e["alive"] and e.get("targetable", True) and not e.get("invulnerable")
                and e.get("role") in _GROUND and _dist(robot, (e["x"], e["y"])) <= self.reach
                and sdk.ballistic_reachable(robot, e)]
        if foes:
            d = lambda e: _dist(robot, (e["x"], e["y"]))
            key = {"weakest": lambda e: (e["hp"], d(e)),
                   "heavy_first": lambda e: (e.get("role") != "heavy", d(e))}.get(mode, d)
            return sdk.intercept(robot, min(foes, key=key))
        return self._structure(robot, obs) if mode != "structure_first" else None


def _d(a, b):
    return math.hypot(a[0] - b[1], a[1] - b[2])
