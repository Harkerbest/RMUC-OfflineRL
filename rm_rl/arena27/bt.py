"""Hand-written sentry behaviour trees in the policy's action space (baselines).

`balanced` is the standard RMUC sentry tree: hold the own central-highland card
(it keeps the own outpost position, and so the base's invulnerability), go home
to heal when weak or low on HP while the position is safe without it, rearm when
dry (buy on the own outpost card while the team fully controls it and is under
its 1000-round cap, else collect the supply-zone rations), pick the posture the fight calls for, buy ammo when
low, revive when it can afford to.
`guard` defaults to defense posture; `assault` pushes to the enemy central card
once the team holds the enemy outpost position.

These run through the same `Executor` as the policy, so a policy-vs-tree
comparison measures decisions only — the comparison a team makes on the real
robot when it swaps its tree for a policy.  Port your own tree here (a function
from the snapshot to head choices) to measure the policy against it.
"""
from __future__ import annotations

import math
from functools import partial

from .obs import AMMO_CAP_17, revive_cost
from .spec import OTHER, HEADS, to_team

REACH = 10.0


def _xy(inp, team):
    return to_team(inp["self"]["x"], inp["self"]["y"], team)


def sentry_tree(inp: dict, wps: dict, tactic: str = "balanced", last_supply: float = 0.0) -> dict:
    s, t, T = inp["self"], inp["time"], inp["team"]
    E, eco, pos = OTHER[T], inp["economy"], inp["positions"]
    act = dict(goal="own_central", target="nearest", posture="keep",
               enhance="on" if s["posture_enhanced"] else "off", supply="none")
    if not s["alive"]:
        cost = revive_cost(s.get("level") or 1, t)
        act.update(goal="hold", target="hold_fire",
                   supply="revive" if t < 330 and eco["coins"] >= cost + 200 else "none")
        return act
    central = wps["own_central"]
    own = pos.get(f"{T}_outpost", {})
    own_full = own.get("controller") == T and own.get("control") == "full"
    covered = any(m["alive"] and not m["weak"] and math.hypot(m["x"] - central[0], m["y"] - central[1]) < 0.6
                  for m in inp["mates"].values())
    safe = covered or own_full                   # the card holds without us for a while
    dry = s["ammo"] <= 0 and not s["pending_ammo"]
    if s["weak"] or (s["hp"] < 0.25 * s["max_hp"] and safe):
        act["goal"] = "own_supply"
    elif dry and own_full and eco["coins"] >= 100 and eco["purchased_17"] + 100 <= AMMO_CAP_17:
        act["goal"] = "own_outpost"              # nearest local exchange: the own outpost card
    elif dry and safe and t - last_supply >= 60:
        act["goal"] = "own_supply"               # collect the accrued 100-round rations
    elif tactic == "assault" and pos.get(f"{E}_outpost", {}).get("controller") == T:
        act["goal"] = "enemy_central"
    goal = wps[act["goal"]]
    far = math.hypot(s["x"] - goal[0], s["y"] - goal[1])
    moving = far > 0.6
    threatened = t - (s["last_damage_time"] or -99) < 2.0
    engaged = any(math.hypot(e["x"] - s["x"], e["y"] - s["y"]) <= REACH for e in inp["seen"].values())
    if s["hp"] < 0.5 * s["max_hp"] and threatened:
        wanted = "defense"
    elif engaged and not moving:
        wanted = "attack"
    elif moving and far > 4:
        wanted = "move"
    else:
        wanted = "defense" if tactic == "guard" else s.get("posture") or "move"
    act["posture"] = "keep" if wanted == s.get("posture") else wanted
    act["enhance"] = "on" if (wanted == "defense" and s["hp"] < 0.3 * s["max_hp"] and threatened) else "off"
    if s["ammo"] < 40:
        act["supply"] = "buy_ammo"
    return act


# name -> tree(inp, wps, last_supply=...) -> head choices.  Register your own team's tree
# here to evaluate against it (eval_sentry27 --arm bt:<name> --ref bt:<name>).
TREES = {t: partial(sentry_tree, tactic=t) for t in ("balanced", "guard", "assault")}
TACTICS = tuple(TREES)


def to_indices(act: dict, mask) -> list[int]:
    """Head choices -> indices, replacing a masked choice by the head's first allowed one
    (a tree asks; the referee and the robot's state decide what is possible)."""
    out, off = [], 0
    for name, choices in HEADS:
        m = mask[off:off + len(choices)]
        i = choices.index(act[name])
        if not m[i]:
            i = int(m.argmax())
        out.append(i)
        off += len(choices)
    return out
