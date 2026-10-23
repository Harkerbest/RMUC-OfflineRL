"""Tactical command (1 Hz) -> Arena-1 robot actions (10 Hz).

The tactical layer decides *where to go*, *whom to engage* and *whether weapons
are free*; on a real robot the navigation stack, auto-aim and trigger logic
execute that.  This controller plays those roles with Arena-1's public SDK:

  nav     goal = position at decision time + (goal_dx, goal_dy); `sdk.navigate`
          drives there with the full command the scripts use, stopping inside
          `arrive` metres.  (Arena chassis are power-limited: measured 0.6-1.3 m/s
          against 3.8-4.6 m/s reference speeds, and half throttle is as fast as
          full, so scaling the command by the sub-goal's implied speed would
          only make policy-driven robots slower than scripted ones.)
  target  the chosen enemy, if a team camera sees it: lead with `sdk.aim_at`,
          ballistic pitch from `sdk.aim_pitch`
  fire    weapons free AND `sdk.can_fire` (heat, ammo, status) AND aligned AND
          ballistically reachable

Auto-aim fallback while weapons are free: if the chosen enemy is not in view,
engage the nearest one that is; if no robot is in view, hand aiming to the team
script, which shoots outpost/base armour.  (The official log labels structure
fire as "no target", so the policy cannot ask for it explicitly.)

Everything that is not motion or weapons — buying ammo at the supply zone,
revives, chassis/weapon profiles — is taken from the team script's own command
for that slot, so every arm in an evaluation shares the same mechanisms.

Resupply.  The observation's `ego.ammo` is rounds fired so far: the official log
never records the rounds *left*, so a policy trained on it cannot know it is
empty.  Measured in Arena, a BC-driven infantry spent 56% of its live time with
no ammunition.  Like auto-aim, resupply is therefore an onboard rule rather
than a tactical decision: while a robot has no ammo the team script drives it
(home to buy, exactly what the scripted infantry do), and the policy's latched
command resumes as soon as it is rearmed.  `resupply_ticks` counts that time.
"""
from __future__ import annotations

import math

from ..algos.action_spec import NO_TARGET
from ..data import schema as S
from .arena import TYPE_SLOT_2026

FIELD = (S.FIELD_X, S.FIELD_Y)
_MOTION = ("move", "aim", "pitch", "fire", "heading", "yaw_rate", "brake")


def _wrap(a):
    return (a + math.pi) % math.tau - math.pi


class TacticalController:
    def __init__(self, sdk, aim_tol=0.12, arrive=0.3, fallback=True, resupply=True):
        self.sdk, self.aim_tol, self.arrive, self.fallback = sdk, aim_tol, arrive, fallback
        self.resupply = resupply
        self.cmd = {}
        self.live_ticks = self.resupply_ticks = 0

    def command(self, slot: str, decoded: dict, robot: dict):
        """Latch a decoded tactical command (deploy.decode_action output, world frame)."""
        gx = min(max(robot["x"] + decoded["goal_dx"], 0.3), FIELD[0] - 0.3)
        gy = min(max(robot["y"] + decoded["goal_dy"], 0.3), FIELD[1] - 0.3)
        tgt = decoded.get("target")
        self.cmd[slot] = dict(
            goal=(gx, gy), fire=bool(decoded["fire"]),
            target=None if tgt is None or tgt == NO_TARGET else TYPE_SLOT_2026[S.MOBILE_TYPES[tgt]])

    def action(self, slot: str, obs: dict, scripted: dict | None) -> dict:
        """10 Hz command for `slot` given the team observation and the script's command."""
        sdk = self.sdk
        robot = next(r for r in obs["robots"] if str(r["slot"]) == str(slot))
        scripted = scripted or {}
        out = {k: v for k, v in scripted.items() if k not in _MOTION}
        cmd = self.cmd.get(slot)
        if cmd is None or not robot["alive"]:
            return {**out, "move": [0.0, 0.0], "fire": False}
        self.live_ticks += 1
        if self.resupply and robot["ammo"] <= 0 and scripted:
            self.resupply_ticks += 1
            return dict(scripted)

        gx, gy = cmd["goal"]
        if math.hypot(gx - robot["x"], gy - robot["y"]) > self.arrive:
            move = sdk.navigate(robot, (gx, gy), obs["arena"])
            out["move"] = [move[0], move[1]]
            if getattr(move, "heading", None) is not None:
                out["heading"] = move.heading
        else:
            out["move"] = [0.0, 0.0]

        foes = [e for e in obs["enemies"] if e["alive"] and e.get("targetable", True)]
        target = next((e for e in foes if e["slot"] == cmd["target"]), None)
        if target is None and cmd["fire"] and self.fallback and foes:
            target = min(foes, key=lambda e: math.hypot(e["x"] - robot["x"], e["y"] - robot["y"]))
        out["fire"] = False
        if target is not None:
            aim = sdk.aim_at(robot, target)
            out["aim"], out["pitch"] = aim, sdk.aim_pitch(robot, target)
            out["fire"] = (cmd["fire"] and sdk.can_fire(robot)
                           and abs(_wrap(robot["aim"] - aim)) < self.aim_tol
                           and sdk.ballistic_reachable(robot, target))
        elif cmd["fire"] and self.fallback and scripted.get("fire"):
            out.update(aim=scripted.get("aim"), pitch=scripted.get("pitch", 0.04), fire=True)
        elif "aim" in scripted:
            out.update(aim=scripted.get("aim"), pitch=scripted.get("pitch", 0.04))
        return out
