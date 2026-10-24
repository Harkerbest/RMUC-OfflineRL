"""Frames, named waypoints and action heads of the 2027 decision layer.

Team frame.  The 2027 field is point-symmetric about (14, 7.5); every feature
is expressed as if the robot played red: blue's (x, y) maps to (28 - x, 15 - y)
and its angles turn by pi.  One policy therefore plays both sides.

Action.  Five categorical heads, chosen once per decision (default 1 s):

    goal      where to be: a named waypoint, `hold` (stay) or `chase` (close in on
              the chosen target)
    target    fire-control priority for auto-aim: nearest / weakest / heavy first /
              structures first / hold fire
    posture   sentry posture: keep / attack / defense / move
    enhance   sentry posture enhancement: off / on
    supply    none / buy ammo / remote heal / revive

That is the interface a sentry behaviour tree drives on the real robot (a goal for
the navigation stack, a priority for auto-aim, referee commands), so a policy and a
tree are interchangeable on it.  `export_spec` writes the heads and the world
coordinates of every waypoint for both sides into each trained run.
"""
from __future__ import annotations

import math

FIELD_X, FIELD_Y = 28.0, 15.0
CENTER = (FIELD_X / 2, FIELD_Y / 2)
OTHER = {"red": "blue", "blue": "red"}
GROUND_SLOTS = (1, 3, 4, 7)
ROLES = ("heavy", "infantry", "sentry")
MAP_2027 = "rmuc2027"

# goal head: (name, side, zone kind).  Points are the zone's reference point except
# where a structure stands on the card (see `waypoints`).
GOALS = (
    ("hold", None, None),
    ("own_central", "own", "central"),
    ("own_outpost", "own", "outpost"),
    ("own_highland", "own", "highland_low"),
    ("own_fortress", "own", "fortress"),
    ("own_base", "own", "base"),
    ("own_supply", "own", "supply"),
    ("own_road", "own", "road_0"),
    ("own_high_point", "own", "high_point"),
    ("enemy_central", "enemy", "central"),
    ("enemy_outpost", "enemy", "outpost"),
    ("enemy_highland", "enemy", "highland_low"),
    ("enemy_fortress", "enemy", "fortress"),
    ("enemy_road", "enemy", "road_0"),
    ("chase", None, None),
)
GOAL_NAMES = tuple(g[0] for g in GOALS)
POINT_GOALS = tuple(g[0] for g in GOALS if g[1] is not None)
TARGETS = ("nearest", "weakest", "heavy_first", "structure_first", "hold_fire")
POSTURES = ("keep", "attack", "defense", "move")
ENHANCE = ("off", "on")
SUPPLY = ("none", "buy_ammo", "remote_heal", "revive")
HEADS = (("goal", GOAL_NAMES), ("target", TARGETS), ("posture", POSTURES),
         ("enhance", ENHANCE), ("supply", SUPPLY))
HEAD_SIZES = tuple(len(v) for _, v in HEADS)
HEAD_NAMES = tuple(k for k, _ in HEADS)
# goals a robot cannot reach, by role (the omni sentry cannot climb the high-point steps)
UNREACHABLE = {"sentry": {"own_high_point"}, "heavy": {"own_high_point"}}

# standing offsets on cards whose reference point is a structure (team frame, metres)
_BASE_STANDOFF = 1.45          # east of the base body, still on the base card
_OUTPOST_STANDOFF = 0.62       # from the outpost toward the field centre


def wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi


def to_team(x: float, y: float, team: str) -> tuple[float, float]:
    """World <-> team frame (an involution)."""
    return (x, y) if team == "red" else (FIELD_X - x, FIELD_Y - y)


def vec_to_team(vx: float, vy: float, team: str) -> tuple[float, float]:
    return (vx, vy) if team == "red" else (-vx, -vy)


def ang_to_team(a: float, team: str) -> float:
    return a if team == "red" else wrap(a + math.pi)


def _inside(x, y, poly):
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def waypoints(arena: dict, team: str) -> dict[str, tuple[float, float]]:
    """World coordinates of every point goal for `team`."""
    zones = {z["id"]: z for z in arena["zones"]}
    out = {}
    for name, side, kind in GOALS:
        if side is None:
            continue
        owner = team if side == "own" else OTHER[team]
        z = zones[f"{owner}_{kind}"]
        x, y = z["x"], z["y"]
        if kind == "base":
            tx, ty = to_team(x, y, owner)
            x, y = to_team(tx + _BASE_STANDOFF, ty, owner)
        elif kind == "outpost":
            dx, dy = CENTER[0] - x, CENTER[1] - y
            d = math.hypot(dx, dy) or 1.0
            x, y = x + dx / d * _OUTPOST_STANDOFF, y + dy / d * _OUTPOST_STANDOFF
        poly = z.get("polygon")
        if poly and not _inside(x, y, [tuple(p[:2]) for p in poly]):
            x, y = z["x"], z["y"]
        out[name] = (float(x), float(y))
    return out


def export_spec(arena: dict, decision_dt: float, obs_names: list[str], extra=None) -> dict:
    return dict(
        decision_dt=decision_dt, team_frame="red; blue (x, y) -> (28 - x, 15 - y), angles + pi",
        heads=[dict(name=k, choices=list(v)) for k, v in HEADS],
        waypoints={t: {k: list(v) for k, v in waypoints(arena, t).items()} for t in ("red", "blue")},
        obs_names=list(obs_names), **(extra or {}))
