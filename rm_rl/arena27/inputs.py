"""The decision layer's input: one snapshot per decision, in plain fields.

`snapshot_from_arena` fills it from an Arena-1 team observation.  On the robot the
same dict is filled from the referee system and onboard perception (field-by-field
sources in docs/sentry27/DEPLOY.md), so the policy reads nothing the real robot
cannot know.  Enemy *positions* are the one place where that needs a choice;
`enemy_info` says whose sensors produced them:

    own    the controlled robot's own camera (its auto-aim detections)   deployable
    team   any own robot's camera (needs inter-robot messages or a radar link)
    full   every enemy, always (privileged: critic and ablations only; needs `frame`)

Enemy HP, deaths and structures need no sensor: the referee broadcasts them.

Schema (world frame; ObsBuilder converts to the team frame):
    time, team, slot
    self      x y vx vy heading hp max_hp heat max_heat ammo alive weak invulnerable
              role level experience posture posture_enhanced posture_ready_at
              posture_time[3] posture_enhanced_time[3] last_shot_time last_damage_time
              respawn_progress respawn_required pending_ammo pending_heal
              defense_buff vulnerability cooling_multiplier on_exchange in_supply
    mates     {slot: x y hp max_hp alive weak role}        ground robots only
    enemies   {slot: hp max_hp alive role}                 referee HP broadcast
    seen      {slot: x y vx vy invulnerable targetable}    sensed this decision
    bases     {team: hp max_hp armor_open}
    outposts  {site: hp max_hp owner rotation_stopped}     `site` = side it stands on
    positions {name: controller control progress active}   the four control positions
    economy   coins logistics control_own control_enemy damage_own damage_enemy
              purchased_17 (team 17 mm bought so far; the team cap is 1000 a match)
"""
from __future__ import annotations

from .spec import OTHER

_SELF = ("x", "y", "vx", "vy", "heading", "hp", "max_hp", "heat", "max_heat", "ammo", "alive",
         "weak", "invulnerable", "role", "level", "experience", "last_shot_time",
         "last_damage_time", "respawn_progress", "respawn_required", "defense_buff",
         "vulnerability", "cooling_multiplier")


def _ground(r):
    return r.get("role") in ("heavy", "infantry", "sentry")


def snapshot_from_arena(obs: dict, slot, enemy_info: str = "own", frame: dict | None = None) -> dict:
    team = obs["team"]
    enemy = OTHER[team]
    me = next(r for r in obs["robots"] if int(r["slot"]) == int(slot))
    s = {k: me.get(k) for k in _SELF}
    s.update(
        posture=me.get("posture"), posture_enhanced=bool(me.get("posture_enhanced")),
        posture_ready_at=me.get("posture_ready_at") or 0.0,
        posture_time=list(me.get("posture_time") or [0.0, 0.0, 0.0]),
        posture_enhanced_time=list(me.get("posture_enhanced_time") or [0.0, 0.0, 0.0]),
        pending_ammo=bool(me.get("pending_ammo")), pending_heal=me.get("pending_heal") is not None)
    zones = obs.get("zones", {})
    on = lambda zid: me["id"] in (zones.get(zid) or {}).get("occupants", [])
    positions = obs.get("positions", {})
    full = lambda owner: (positions.get(f"{owner}_outpost", {}).get("controller") == team
                          and positions.get(f"{owner}_outpost", {}).get("control") == "full")
    s["in_supply"] = on(f"{team}_supply")
    # local exchange: own base / supply / high-point card, or the outpost card (not the
    # central card) of any outpost position the team fully controls
    s["on_exchange"] = (s["in_supply"] or on(f"{team}_base") or on(f"{team}_high_point")
                        or any(full(o) and on(f"{o}_outpost") for o in (team, enemy)))

    mates = {int(r["slot"]): dict(x=r["x"], y=r["y"], hp=r["hp"], max_hp=r["max_hp"], alive=r["alive"],
                                  weak=r.get("weak", False), role=r["role"])
             for r in obs["robots"] if _ground(r) and int(r["slot"]) != int(slot)}
    enemies = {int(r["slot"]): dict(hp=r["hp"], max_hp=r["max_hp"], alive=r["alive"], role=r["role"])
               for r in obs["referee"]["robots"] if r["team"] == enemy and _ground(r)}

    if enemy_info == "full":
        if frame is None:
            raise ValueError("enemy_info='full' needs the engine frame")
        pool = [r for r in frame["robots"] if r["team"] == enemy]
    else:
        pool = obs["enemies"]
        if enemy_info == "own":
            ids = set((obs.get("vision", {}).get(str(slot)) or {}).get("enemies", []))
            pool = [e for e in pool if e["id"] in ids]
        elif enemy_info != "team":
            raise ValueError(f"enemy_info {enemy_info!r}")
    seen = {int(e["slot"]): dict(x=e["x"], y=e["y"], vx=e.get("vx", 0.0), vy=e.get("vy", 0.0),
                                 invulnerable=bool(e.get("invulnerable")),
                                 targetable=bool(e.get("targetable", True)))
            for e in pool if e.get("alive") and _ground(e)}

    teams = obs["teams"]
    return dict(
        time=float(obs["time"]), team=team, slot=int(slot), self=s, mates=mates, enemies=enemies, seen=seen,
        bases={t: dict(hp=b["hp"], max_hp=b["max_hp"], armor_open=bool(b.get("armor_open")))
               for t, b in obs["bases"].items()},
        outposts={t: dict(hp=o["hp"], max_hp=o["max_hp"], owner=o.get("owner", t),
                          rotation_stopped=bool(o.get("rotation_stopped")))
                  for t, o in obs["outposts"].items()},
        positions={k: dict(controller=p.get("controller"), control=p.get("control"),
                           progress=float(p.get("progress") or 0.0), active=list(p.get("active") or []))
                   for k, p in positions.items()},
        economy=dict(coins=float(teams[team].get("resources", 0.0)),
                     logistics=float(obs["referee"].get("logistics", 1.0)),
                     control_own=float(teams[team].get("control", 0.0)),
                     control_enemy=float(teams.get(enemy, {}).get("control", 0.0)),
                     damage_own=float(teams[team].get("attack_damage", 0.0)),
                     damage_enemy=float(teams.get(enemy, {}).get("attack_damage", 0.0)),
                     purchased_17=float(teams[team].get("purchased_17", 0.0))))
