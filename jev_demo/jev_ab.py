# -*- coding: utf-8 -*-
"""A/B：同一个真实对局秒，thin prompt（70 token 摘要）vs full briefing（规则手册 + 字段语义）。

规则依据 RoboMaster 2026 超级对抗赛规则手册 V2.1.0（20260716）。
briefing 放进 state 而不是 instructions：state 只 ingest 一次、所有问题并行评估，更省 token。
"""
import io, json, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(HERE, ".env")):
    if os.environ.get("TYPESAFE_API_KEY") or not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env()
S = json.load(open(os.path.join(HERE, "sample.json"), encoding="utf-8"))
CELL, NX, NY = 2.0, 14, 8

THIN = ("RoboMaster university championship, 7-minute 1v1 team match on a 28m x 15m arena. "
        "This robot is a 17mm infantry driven toward winning the round: the round is decided by "
        "outpost and base HP, so trading HP for map control near the enemy outpost is usually "
        "worth more than farming kills. Firing while 17mm heat is near its cap causes "
        "self-damage. All coordinates are absolute arena metres; bearings are math bearings "
        "atan2(dy,dx) in degrees.")

BRIEF = {
    "objective": (
        "Win the 7-minute round. Destroying the enemy base ends the round immediately. Otherwise "
        "the tiebreak order is fixed: higher remaining own base HP -> higher own outpost HP (if "
        "neither outpost was ever destroyed) -> the side whose outpost was never destroyed -> "
        "higher total team damage dealt -> higher total remaining HP of surviving robots -> draw."),
    "base": (
        "5000 HP plus a 150-point virtual shield. A base is INVINCIBLE while its own outpost is "
        "alive, so destroying the enemy outpost is the prerequisite for any base damage. Base "
        "armor must additionally be deployed before it can be hit; deployment is triggered by an "
        "enemy holding that base's 堡垒增益点 for 20 cumulative seconds, or by dart hits. 17mm "
        "deals only 20 damage per hit to base armor (5 on the upper front module); 42mm deals 200. "
        "A single 17mm infantry cannot meaningfully kill a 5000 HP base on its own."),
    "outpost": (
        "1500 HP with rotating middle armor. Destroying it removes the enemy base's invincibility "
        "and moves the dart guide light to the base. A destroyed outpost can be rebuilt to 750 HP "
        "(hero/infantry/sentry need 10s of continuous detection, engineer 5s); no rebuilds after "
        "the 5-minute mark. So enemy_outpost_hp == 0 means both that the enemy base is now "
        "attackable in principle and that the enemy may try to rebuild the outpost."),
    "vulnerability": (
        "Damage resolves as raw_damage * (1 - defense_buff). 易伤 (vulnerability) is negative "
        "defense: radar marking progress 100-120 gives +15% incoming damage, 120-150 gives +20%. "
        "Same-category buffs do not stack, the largest applies. A separate 双倍易伤 trigger (max 2 "
        "per round, 30s each) doubles radar-sourced 易伤 only. So radar_marked_vulnerable=true "
        "means roughly +15% to +40% incoming damage, NOT double damage."),
    "heat_17mm": (
        "Each 17mm round adds 10 heat; cooling is continuous at the robot's per-second cooling "
        "rate. Exceeding the cap does NOT cause self-damage: between the cap and cap+100 the "
        "launcher LOCKS and the operator's first-person video feed is cut until heat returns to 0; "
        "at cap+100 or above the launcher is locked for the rest of the round. Heat resets to 0 on "
        "death. Infantry launcher tiers range from 40 cap / 12 per second cooling up to 260 / 20."),
    "defense_buffs": (
        "Occupiable zones grant damage reduction, die with the occupier, and lapse 2s after "
        "leaving: 基地增益点 50% plus ammo purchase; 梯形高地 50%, own side only; 中央高地 25%, two "
        "independent zones, holding one locks the enemy out; 前哨站增益点 25% plus ammo; 补给区 "
        "heals 10% of max HP per second (25%/s after the 4-minute mark if out of combat) and "
        "speeds the respawn bar to 4/s; 堡垒增益点 activates only after your OWN outpost has been "
        "destroyed and gives one robot 50% defense plus heat cooling and reserve ammo, while an "
        "ENEMY standing on it takes +100% 易伤 and opens your base armor after 20 cumulative "
        "seconds. Terrain crossings (公路/高地/飞坡/隧道) grant 25-50% defense for 5-30s."),
    "economy": (
        "Both sides start at 400 gold and gain 50 at each of 5:59/4:59/3:59/2:59/1:59 plus 150 at "
        "0:59, so 800 is the baseline. Everything beyond that comes from the ENGINEER assembling "
        "科技核心, which also grants permanent team-wide defense (25% at tier 3, 50% at tier 4) and "
        "+2000 base HP at tier 4. Gold buys ammo (17mm: 10 gold per 10 rounds locally, 150 per 100 "
        "remotely, team cap 1000), air support, remote healing (60% of max HP after a 6s delay), "
        "and instant revives. HP, heat and power upgrades come from exp levels, not from gold."),
    "respawn": (
        "Respawn bar length = 10 + (420 - seconds_remaining)/10 + 20 * prior instant revives, "
        "filling at 1/s, or 4/s in the supply zone or when own base is below 2000 HP. On revival "
        "the robot returns at 10% HP with 30s invincibility and in 虚弱: launcher locked, cannot "
        "occupy any buff point or rebuild the outpost until 虚弱 breaks."),
    "roles": {
        "hero": "The only 42mm platform, 200 damage per hit to base armor, 260-600 HP, the biggest "
                "single threat. If the hero goes 4 seconds without firing 42mm, all enemy armor "
                "ignores 42mm entirely, so an idle hero is temporarily harmless.",
        "infantry": "17mm, two per team (numbered 3 and 4), 150-400 HP. Main map-control unit.",
        "engineer": "No launcher, fixed 250 HP, invincible in the assembly and supply zones, and "
                    "holds 50% defense just by being alive in the first 3 minutes. It is the "
                    "entire economy, so killing it denies gold and tech tiers. Rebuilds the "
                    "outpost in 5s rather than 10s.",
        "aerial": "17mm, a fixed 750 rounds with no resupply, immune to collision and chassis "
                  "power limits, present only during air-support windows.",
        "sentry": "Automated 17mm turret defending its own base area: 400 HP, 260 heat cap, 30 per "
                  "second cooling in auto mode. Postures trade defense against cooling and "
                  "mobility (防御 = 50% defense, 进攻 = 3x cooling but 25% 易伤).",
    },
    "what_you_are_not_given": (
        "You do not get the obstacle layout, line of sight, or the coordinates of any buff zone. "
        "Reason only from the positions, distances, HP and heat you are given."),
}

SEMANTICS = {
    "ego.heat_17mm / ego.heat_17mm_max": "current 17mm heat and its cap; each round adds 10; "
                                         "exceeding the cap locks the launcher and cuts the feed",
    "radar_marked_vulnerable": "true = currently radar-marked, taking +15% to +20% extra damage "
                               "(up to roughly +40% inside a 双倍易伤 window)",
    "hp / max_hp": "current and maximum HP; max_hp reflects the exp-level upgrade tier reached",
    "dist_m": "straight-line distance from the ego robot in metres, with no line-of-sight guarantee",
    "bearing_deg": "math bearing from ego to that entity, atan2(dy,dx) in degrees",
    "ego.turret_bearing_deg": "where the ego robot's muzzle currently points, same convention",
    "buildings.*_outpost_hp": "0 means destroyed; while an outpost is alive its own base is invincible",
    "economy.my_gold_total / my_gold_left": "team cumulative and currently unspent gold",
    "visible_enemies": "only the enemies this robot can currently perceive; an enemy absent from "
                       "the list is unseen, not absent from the field",
    "t_sec": "seconds elapsed in a 420-second round, so seconds_remaining = 420 - t_sec",
}

TARGETS_THIN = {
    "hero": "The enemy hero (42mm, highest damage output, usually the biggest threat).",
    "engineer": "The enemy engineer. It never shoots; killing it only denies economy.",
    "infantry_3": "The enemy infantry numbered 3 (17mm).",
    "infantry_4": "The enemy infantry numbered 4 (17mm).",
    "aerial": "The enemy aerial robot.",
    "sentry": "The enemy sentry, an automated turret defending its own base area.",
    "none": "No enemy is worth engaging this second (none visible, none in range, "
            "or the robot should be repositioning or cooling instead).",
}
TARGETS_FULL = {
    "hero": "The enemy hero: 42mm, the only unit that can meaningfully damage a base (200 per "
            "hit), so the biggest threat and the highest-value kill.",
    "engineer": "The enemy engineer: unarmed, 250 HP, but it is the whole enemy economy and tech "
                "tier, and it rebuilds their outpost in 5s. Holds 50% defense in the first 3 "
                "minutes and is invincible inside the assembly and supply zones.",
    "infantry_3": "The enemy infantry numbered 3: 17mm, 150-400 HP, their main map-control unit.",
    "infantry_4": "The enemy infantry numbered 4: 17mm, 150-400 HP, their main map-control unit.",
    "aerial": "The enemy aerial robot: 17mm, 750 rounds and no resupply, so it runs dry.",
    "sentry": "The enemy sentry: automated, 400 HP, 30/s cooling, guarding their base area. "
              "Engaging it means fighting into their defended ground.",
    "none": "Do not engage any enemy this second: nothing worth shooting, or the robot should be "
            "repositioning, cooling, or contesting a zone instead.",
}
AGGR_THIN = ["Break contact and retreat to heal or cool down", "Hold position and defend",
             "Trade fire at current range", "Push forward onto the enemy outpost"]
AGGR_FULL = [
    "Disengage: break contact and fall back toward own ground, supply or cover, to heal or to let "
    "heat cool",
    "Hold: keep the current position and defend the approach without advancing",
    "Trade: exchange fire at the current range without giving or taking ground",
    "Press: advance into enemy-held ground to contest their base area or buff zones",
]


def grid_criteria(red_home_low_x):
    out = {}
    for cx in range(NX):
        for cy in range(NY):
            x0, y0 = cx * CELL, cy * CELL
            half = "own half" if (cx < NX / 2) == red_home_low_x else "enemy half"
            lane = "bottom lane" if cy < NY / 3 else ("top lane" if cy > 2 * NY / 3 else "middle lane")
            depth = "deep" if (cx < 3 or cx > NX - 4) else "mid-field"
            out[f"c{cx:02d}_{cy:02d}"] = (f"x {x0:.0f}-{x0 + CELL:.0f} m, y {y0:.0f}-{y0 + CELL:.0f} m; "
                                         f"{half}, {lane}, {depth}")
    out["stay"] = "Hold the current position, do not relocate this second."
    return out


GRID = grid_criteria(S["red_home_low_x"])


def build(full):
    """返回 (state, questions)。full=True 时规则进 state，instructions 只留问题本身。"""
    pre = "" if full else THIN + " "
    state = dict(S["state_ego"])
    if full:
        state = {"rmuc_rules": BRIEF, "field_semantics": SEMANTICS, "match_state": state}
    q = {
        "weapons_free": Noul(instructions=pre + "Should this robot be firing 17mm rounds during "
                                                "the next second?"),
        "target": Choice(instructions=pre + "Which enemy should this robot put its turret on for "
                                            "the next second?",
                         criteria=TARGETS_FULL if full else TARGETS_THIN),
        "nav_cell": Choice(instructions=pre + "Which arena grid cell should this robot move toward "
                                              "over the next ~3 seconds?", criteria=GRID),
        "aggression": Score(instructions=pre + "How aggressive should this robot be right now?",
                            criteria=AGGR_FULL if full else AGGR_THIN),
    }
    return state, q


def run(full):
    state, q = build(full)
    with TypeSafeClient() as c:
        t0 = time.perf_counter()
        r = c.system_one(state=state, questions=q)
        return r, (time.perf_counter() - t0) * 1000


def top(d, n):
    return ", ".join(f"{k}={v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n])


H = S["human"]
print(f"样本 game_id={S['game_id']} t={int(S['t'])}s | 蓝方前哨站已毁(0) 红方前哨站 1300 | "
      f"ego 易伤={S['state_ego']['ego']['radar_marked_vulnerable']} heat={S['state_ego']['ego']['heat_17mm']}"
      f"/{S['state_ego']['ego']['heat_17mm_max']}")
print(f"人类实际: fire={H['weapons_free']}({H['rounds_fired_this_sec']}发) "
      f"target={H['target_by_muzzle_bearing']}(误差{H['muzzle_err_deg']}°) "
      f"cell {H['cell_now']}->+3s {H['cell_in_3s']}\n")

for label, full in (("THIN  (70 tok 摘要, 含 2 条假机制)", False), ("FULL  (规则手册 + 字段语义)", True)):
    r, ms = run(full)
    a = r.answers
    print(f"--- {label} | {ms:.0f} ms | input={r.usage.input_tokens} tok ---")
    print(f"  weapons_free P(fire) = {a['weapons_free'].noul:.3f}")
    print(f"  target       {a['target'].choice:12s} conf {a['target'].confidence:.3f}  | "
          f"{top(a['target'].probabilities, 4)}")
    print(f"  nav_cell     {a['nav_cell'].choice:12s} conf {a['nav_cell'].confidence:.3f}  | "
          f"{top(a['nav_cell'].probabilities, 5)}")
    print(f"  aggression   score {a['aggression'].score:.2f}  conf {a['aggression'].confidence:.3f}  | "
          + ", ".join(f"{k}={v:.3f}" for k, v in sorted(a['aggression'].probabilities.items())))
    print(f"  外推 25.7 万秒成本: ${(r.usage.input_tokens or 0) * 257000 * 0.042 / 1e6:.2f}\n")
