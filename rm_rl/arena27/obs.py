"""Snapshot -> feature vector, action masks, and the critic's privileged state.

`ObsBuilder` is stateful only in what a robot can remember on its own: where and
when each enemy was last sensed, and when it last visited the supply zone (the
sentry's ammo ration accrues there every 60 s).  Features are in the team frame,
scaled to O(1) and clipped to +-5 — the closed-loop experiments showed what an
unclipped constant column does to a policy on the robot.
"""
from __future__ import annotations

import math

import numpy as np

from .spec import (GROUND_SLOTS, HEAD_SIZES, OTHER, POINT_GOALS, POSTURES, ROLES, SUPPLY,
                   TARGETS, GOAL_NAMES, UNREACHABLE, ang_to_team, to_team, vec_to_team)

CLIP = 5.0
AMMO_CAP_17 = 1000.0          # 17 mm a team may buy per match
CHASE_MEMORY_S = 3.0
_POSTURE_IDX = {"attack": 0, "defense": 1, "move": 2}     # order of posture_time


def revive_cost(level: float, t: float) -> float:
    return math.ceil(t / 60) * 80 + level * 20


def heal_cost(t: float) -> float:
    return 50 + math.ceil(t / 60 * 20)


def disengaged(s: dict, t: float) -> bool:
    return t - max(s.get("last_shot_time") or -99.0, s.get("last_damage_time") or -99.0) >= 6.2


class _F:
    __slots__ = ("v", "n")

    def __init__(self):
        self.v, self.n = [], []

    def add(self, name, value):
        self.v.append(float(value))
        self.n.append(name)


class ObsBuilder:
    def __init__(self, team: str, slot: int, wps: dict):
        self.team, self.enemy, self.slot = team, OTHER[team], int(slot)
        self.wp = {k: to_team(*v, team) for k, v in wps.items()}
        self.last_seen = {}                  # enemy slot -> (t, x, y) in the team frame
        self.last_supply = 0.0
        self.names = None

    # ------------------------------------------------------------------ features
    def __call__(self, inp: dict) -> np.ndarray:
        T, E, t = self.team, self.enemy, inp["time"]
        f = _F()
        # match clock and economy
        f.add("t", t / 420); f.add("t_left", max(0.0, 420 - t) / 420)
        for mark in (180, 240, 300, 361):
            f.add(f"after_{mark}", t >= mark)
        eco = inp["economy"]
        f.add("coins", eco["coins"] / 1000); f.add("logistics", eco["logistics"])
        f.add("ammo_cap_left", (AMMO_CAP_17 - eco["purchased_17"]) / AMMO_CAP_17)
        f.add("control_own", eco["control_own"] / 200); f.add("control_enemy", eco["control_enemy"] / 200)
        f.add("control_lead", (eco["control_own"] - eco["control_enemy"]) / 100)
        f.add("damage_own", eco["damage_own"] / 10000); f.add("damage_enemy", eco["damage_enemy"] / 10000)
        f.add("damage_lead", (eco["damage_own"] - eco["damage_enemy"]) / 5000)
        # control positions
        for tag, name in (("own_outpost", f"{T}_outpost"), ("enemy_outpost", f"{E}_outpost"),
                          ("own_base", f"{T}_base"), ("enemy_base", f"{E}_base")):
            p = inp["positions"].get(name, {})
            f.add(f"{tag}_ours", p.get("controller") == T); f.add(f"{tag}_theirs", p.get("controller") == E)
            f.add(f"{tag}_full", p.get("control") == "full"); f.add(f"{tag}_progress", p.get("progress", 0.0) / 5)
            f.add(f"{tag}_we_on", T in p.get("active", [])); f.add(f"{tag}_they_on", E in p.get("active", []))
        # structures
        for tag, side in (("own", T), ("enemy", E)):
            b = inp["bases"][side]
            f.add(f"{tag}_base_hp", b["hp"] / max(b["max_hp"], 1)); f.add(f"{tag}_base_open", b["armor_open"])
            o = inp["outposts"][side]
            f.add(f"{tag}_site_outpost_hp", o["hp"] / 1000); f.add(f"{tag}_site_outpost_ours", o["owner"] == T)
            f.add(f"{tag}_site_outpost_still", o["rotation_stopped"])
        # self
        s = inp["self"]
        x, y = to_team(s["x"], s["y"], T)
        vx, vy = vec_to_team(s.get("vx") or 0.0, s.get("vy") or 0.0, T)
        h = ang_to_team(s.get("heading") or 0.0, T)
        f.add("x", x / 28); f.add("y", y / 15); f.add("vx", vx / 4); f.add("vy", vy / 4)
        f.add("sin_h", math.sin(h)); f.add("cos_h", math.cos(h))
        f.add("hp", s["hp"] / max(s["max_hp"], 1)); f.add("max_hp", s["max_hp"] / 600)
        f.add("heat", s["heat"] / max(s["max_heat"], 1)); f.add("ammo", s["ammo"] / 500)
        f.add("no_ammo", s["ammo"] <= 0)
        f.add("alive", s["alive"]); f.add("weak", s["weak"]); f.add("invulnerable", s["invulnerable"])
        for p in ("attack", "defense", "move"):
            f.add(f"posture_{p}", s.get("posture") == p)
        f.add("enhanced", s["posture_enhanced"])
        f.add("posture_ready_in", min(max(s["posture_ready_at"] - t, 0.0), 5.0) / 5)
        for i, p in enumerate(("attack", "defense", "move")):
            f.add(f"posture_time_{p}", s["posture_time"][i] / 180)
            f.add(f"enhanced_time_{p}", s["posture_enhanced_time"][i] / 15)
        f.add("since_damage", min(max(t - (s["last_damage_time"] or -99), 0.0), 10.0) / 10)
        f.add("since_shot", min(max(t - (s["last_shot_time"] or -99), 0.0), 10.0) / 10)
        f.add("disengaged", disengaged(s, t))
        req = s.get("respawn_required") or 0.0
        f.add("respawn", (s.get("respawn_progress") or 0.0) / req if req > 0 else 0.0)
        f.add("pending_ammo", s["pending_ammo"]); f.add("pending_heal", s["pending_heal"])
        f.add("defense_buff", s.get("defense_buff") or 0.0); f.add("vulnerability", s.get("vulnerability") or 0.0)
        f.add("cooling_mult", (s.get("cooling_multiplier") or 1.0) / 5)
        f.add("on_exchange", s["on_exchange"]); f.add("in_supply", s["in_supply"])
        for r in ROLES:
            f.add(f"role_{r}", s["role"] == r)
        f.add("level", (s.get("level") or 1) / 10); f.add("experience", (s.get("experience") or 0.0) / 5000)
        if s["in_supply"]:
            self.last_supply = t
        f.add("rations", min(math.floor((t - self.last_supply) / 60), 3) / 3)
        # waypoints
        for g in POINT_GOALS:
            gx, gy = self.wp[g]
            f.add(f"{g}_dx", (gx - x) / 28); f.add(f"{g}_dy", (gy - y) / 15)
            f.add(f"{g}_dist", math.hypot(gx - x, gy - y) / 30)
        # teammates (ground slots other than self, slot order)
        for k, slot in enumerate(sl for sl in GROUND_SLOTS if sl != self.slot):
            m = inp["mates"].get(slot)
            f.add(f"mate{k}_present", m is not None)
            mx, my = to_team(m["x"], m["y"], T) if m else (x, y)
            f.add(f"mate{k}_dx", (mx - x) / 28); f.add(f"mate{k}_dy", (my - y) / 15)
            f.add(f"mate{k}_dist", math.hypot(mx - x, my - y) / 30)
            f.add(f"mate{k}_hp", m["hp"] / max(m["max_hp"], 1) if m else 0.0)
            f.add(f"mate{k}_alive", bool(m and m["alive"])); f.add(f"mate{k}_weak", bool(m and m["weak"]))
            for r in ROLES:
                f.add(f"mate{k}_{r}", bool(m and m["role"] == r))
        # enemies: referee HP for all, positions only where sensed
        for slot in GROUND_SLOTS:
            e, seen = inp["enemies"].get(slot), inp["seen"].get(slot)
            alive = bool(e and e["alive"])
            if not alive:
                self.last_seen.pop(slot, None)
            elif seen is not None:
                self.last_seen[slot] = (t, *to_team(seen["x"], seen["y"], T))
            mem = self.last_seen.get(slot)
            f.add(f"e{slot}_present", e is not None); f.add(f"e{slot}_alive", alive)
            f.add(f"e{slot}_hp", e["hp"] / max(e["max_hp"], 1) if e else 0.0)
            for r in ROLES:
                f.add(f"e{slot}_{r}", bool(e and e["role"] == r))
            f.add(f"e{slot}_seen", seen is not None)
            f.add(f"e{slot}_known", mem is not None)
            f.add(f"e{slot}_age", min(t - mem[0], 10.0) / 10 if mem else 1.0)
            ex, ey = (mem[1], mem[2]) if mem else (x, y)
            f.add(f"e{slot}_dx", (ex - x) / 28); f.add(f"e{slot}_dy", (ey - y) / 15)
            f.add(f"e{slot}_dist", math.hypot(ex - x, ey - y) / 30 if mem else 1.0)
            f.add(f"e{slot}_x", ex / 28 if mem else 0.0); f.add(f"e{slot}_y", ey / 15 if mem else 0.0)
            evx, evy = vec_to_team(seen["vx"], seen["vy"], T) if seen else (0.0, 0.0)
            f.add(f"e{slot}_vx", evx / 4); f.add(f"e{slot}_vy", evy / 4)
            f.add(f"e{slot}_invulnerable", bool(seen and seen["invulnerable"]))
        if self.names is None:
            self.names = f.n
        return np.clip(np.asarray(f.v, np.float32), -CLIP, CLIP)

    # ------------------------------------------------------------------ masks
    def chase_target_known(self, t: float) -> bool:
        return any(t - m[0] <= CHASE_MEMORY_S for m in self.last_seen.values())

    def mask(self, inp: dict) -> np.ndarray:
        """Concatenated per-head masks (True = allowed).  Call after __call__ (memory)."""
        s, t = inp["self"], inp["time"]
        heads = [np.ones(n, bool) for n in HEAD_SIZES]
        goal, target, posture, enhance, supply = heads
        eco = inp["economy"]
        current_enh = int(bool(s["posture_enhanced"]))
        if not s["alive"]:
            goal[:] = False; goal[GOAL_NAMES.index("hold")] = True
            target[:] = False; target[TARGETS.index("hold_fire")] = True
            posture[:] = False; posture[0] = True
            enhance[:] = False; enhance[current_enh] = True
            supply[:] = False; supply[0] = True
            supply[SUPPLY.index("revive")] = eco["coins"] >= revive_cost(s.get("level") or 1, t)
            return np.concatenate(heads)
        for g in UNREACHABLE.get(s["role"], ()):
            goal[GOAL_NAMES.index(g)] = False
        goal[GOAL_NAMES.index("chase")] = self.chase_target_known(t)
        ready = t >= s["posture_ready_at"]
        if s["role"] != "sentry":
            posture[1:] = False
            enhance[:] = False; enhance[0] = True
        else:
            for i, p in enumerate(POSTURES[1:], 1):
                posture[i] = ready and s.get("posture") != p
            if not ready:
                enhance[:] = False; enhance[current_enh] = True
            elif s.get("posture") in _POSTURE_IDX and s["posture_enhanced_time"][_POSTURE_IDX[s["posture"]]] >= 15.0:
                enhance[1] = False
        dis = disengaged(s, t)
        supply[SUPPLY.index("revive")] = False
        supply[SUPPLY.index("buy_ammo")] = (
            not s["weak"] and not s["pending_ammo"] and t >= 1.05 and s["ammo"] < 400
            and eco["purchased_17"] + 100 <= AMMO_CAP_17
            and ((s["on_exchange"] and eco["coins"] >= 100) or (dis and eco["coins"] >= 150)))
        supply[SUPPLY.index("remote_heal")] = (
            not s["weak"] and not s["pending_heal"] and dis and s["hp"] < s["max_hp"]
            and eco["coins"] >= heal_cost(t))
        return np.concatenate(heads)


# ---------------------------------------------------------------------- critic state
def global_extras(frame: dict, team: str) -> np.ndarray:
    """Privileged engine state for the centralised critic (never shown to the actor):
    every ground robot's position, HP, ammo, heat and level on both sides, and both
    teams' coins, tech progress and launcher state."""
    enemy = OTHER[team]
    by = {(r["team"], int(r["slot"])): r for r in frame["robots"]}
    v = []
    for side in (team, enemy):
        for slot in GROUND_SLOTS:
            r = by.get((side, slot))
            if r is None:
                v += [0.0] * 9
                continue
            x, y = to_team(r["x"], r["y"], team)
            v += [1.0, x / 28, y / 15, r["hp"] / max(r["max_hp"], 1), float(r["alive"]),
                  min(r.get("ammo", 0), 1000) / 500, r.get("heat", 0) / max(r.get("max_heat") or 1, 1),
                  (r.get("level") or 1) / 10, float(bool(r.get("weak")))]
        tm = frame["teams"].get(side, {})
        heavy = by.get((side, 1)) or {}
        v += [tm.get("resources", 0.0) / 1000, len(tm.get("completed_difficulties") or []) / 4,
              tm.get("tech_defense", 0.0), float(heavy.get("launcher") in ("fitted", "installed")),
              tm.get("control", 0.0) / 200, tm.get("attack_damage", 0.0) / 10000]
    return np.clip(np.asarray(v, np.float32), -CLIP, CLIP)
