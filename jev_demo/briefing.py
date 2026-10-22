# -*- coding: utf-8 -*-
"""RMUC 规则 briefing，依据《RoboMaster 2026 超级对抗赛比赛规则手册 V2.1.0（20260716）》。

放进 Jev 的 `state` 而不是 `instructions`：state 只 ingest 一次、所有问题并行评估。
"""

BRIEF = {
    "objective": (
        "Win the 7-minute (420 s) round. Destroying the enemy base ends the round immediately. "
        "Otherwise the tiebreak order is fixed: higher remaining own base HP -> higher own outpost "
        "HP (if neither outpost was ever destroyed) -> the side whose outpost was never destroyed "
        "-> higher total team damage dealt -> higher total remaining HP of surviving robots -> draw."),
    "base": (
        "5000 HP plus a 150-point virtual shield. A base is INVINCIBLE while its own outpost is "
        "alive, so destroying the enemy outpost is the prerequisite for any base damage. Base armor "
        "must additionally be deployed before it can be hit; deployment is triggered by an enemy "
        "holding that base's 堡垒增益点 for 20 cumulative seconds, or by dart hits. 17mm deals only "
        "20 damage per hit to base armor (5 on the upper front module); 42mm deals 200. A single "
        "17mm infantry cannot meaningfully kill a 5000 HP base on its own."),
    "outpost": (
        "1500 HP with rotating middle armor. Destroying it removes the enemy base's invincibility "
        "and moves the dart guide light to the base. A destroyed outpost can be rebuilt to 750 HP "
        "(hero/infantry/sentry need 10 s of continuous detection, engineer 5 s); no rebuilds after "
        "the 5-minute mark. So enemy_outpost_hp == 0 means both that the enemy base is now "
        "attackable in principle and that the enemy may try to rebuild the outpost."),
    "vulnerability": (
        "Damage resolves as raw_damage * (1 - defense_buff). 易伤 (vulnerability) is negative "
        "defense: radar marking progress 100-120 gives +15% incoming damage, 120-150 gives +20%. "
        "Same-category buffs do not stack, the largest applies. A separate 双倍易伤 trigger (max 2 "
        "per round, 30 s each) doubles radar-sourced 易伤 only. So radar_marked_vulnerable=true "
        "means roughly +15% to +40% incoming damage, NOT double damage."),
    "heat_17mm": (
        "Each 17mm round adds 10 heat; cooling is continuous at the robot's per-second cooling "
        "rate. Exceeding the cap does NOT cause self-damage: between the cap and cap+100 the "
        "launcher LOCKS and the operator's first-person video feed is cut until heat returns to 0; "
        "at cap+100 or above the launcher is locked for the rest of the round. Heat resets to 0 on "
        "death. Infantry launcher tiers range from 40 cap / 12 per second cooling up to 260 / 20."),
    "defense_buffs": (
        "Occupiable zones grant damage reduction, die with the occupier, and lapse 2 s after "
        "leaving: 基地增益点 50% plus ammo purchase; 梯形高地 50%, own side only; 中央高地 25%, two "
        "independent zones, holding one locks the enemy out; 前哨站增益点 25% plus ammo; 补给区 heals "
        "10% of max HP per second (25%/s after the 4-minute mark if out of combat) and speeds the "
        "respawn bar to 4/s; 堡垒增益点 activates only after your OWN outpost has been destroyed and "
        "gives one robot 50% defense plus heat cooling and reserve ammo, while an ENEMY standing on "
        "it takes +100% 易伤 and opens your base armor after 20 cumulative seconds. Terrain "
        "crossings (公路/高地/飞坡/隧道) grant 25-50% defense for 5-30 s."),
    "economy": (
        "Both sides start at 400 gold and gain 50 at each of 5:59/4:59/3:59/2:59/1:59 plus 150 at "
        "0:59, so 800 is the baseline. Everything beyond that comes from the ENGINEER assembling "
        "科技核心, which also grants permanent team-wide defense (25% at tier 3, 50% at tier 4) and "
        "+2000 base HP at tier 4. Gold buys ammo (17mm: 10 gold per 10 rounds locally, 150 per 100 "
        "remotely, team cap 1000), air support, remote healing (60% of max HP after a 6 s delay), "
        "and instant revives. HP, heat and power upgrades come from exp levels, not from gold."),
    "respawn": (
        "Respawn bar length = 10 + (420 - seconds_remaining)/10 + 20 * prior instant revives, "
        "filling at 1/s, or 4/s in the supply zone or when own base is below 2000 HP. On revival "
        "the robot returns at 10% HP with 30 s invincibility and in 虚弱: launcher locked, cannot "
        "occupy any buff point or rebuild the outpost until 虚弱 breaks."),
    "roles": {
        "hero": "The only 42mm platform, 200 damage per hit to base armor, 260-600 HP, the biggest "
                "single threat. If the hero goes 4 seconds without firing 42mm, all enemy armor "
                "ignores 42mm entirely, so an idle hero is temporarily harmless.",
        "infantry": "17mm, two per team (numbered 3 and 4), 150-400 HP. Main map-control unit.",
        "engineer": "No launcher, fixed 250 HP, invincible in the assembly and supply zones, and "
                    "holds 50% defense just by being alive in the first 3 minutes. It is the entire "
                    "economy, so killing it denies gold and tech tiers. Rebuilds the outpost in 5 s.",
        "aerial": "17mm, a fixed 750 rounds with no resupply, immune to collision and chassis power "
                  "limits, present only during air-support windows.",
        "sentry": "Automated 17mm turret defending its own base area: 400 HP, 260 heat cap, 30 per "
                  "second cooling in auto mode. Postures trade defense against cooling and mobility "
                  "(防御 = 50% defense, 进攻 = 3x cooling but 25% 易伤).",
    },
    "frame": (
        "All coordinates are in a canonical field frame, 28 m (x) by 15 m (y), in which YOUR team's "
        "home end is at low x and the enemy's home end is at high x. Blue-side rounds are mirrored "
        "into this frame, so 'my' always means the acting robot's own team. dx/dy are offsets from "
        "the acting robot; bearings are math bearings atan2(dy, dx) in degrees."),
    "what_you_are_not_given": (
        "You do not get the obstacle layout, line of sight, or the coordinates of any buff zone. "
        "Reason only from the positions, distances, HP, heat and economy you are given."),
}

SEMANTICS = {
    "ego.heat_17mm / ego.heat17_max": "current 17mm heat and its cap; each round adds 10; exceeding "
                                      "the cap locks the launcher and cuts the operator's feed",
    "radar_marked_vulnerable": "true = currently radar-marked, taking +15% to +20% extra damage "
                               "(up to roughly +40% inside a 双倍易伤 window)",
    "hp / max_hp": "current and maximum HP; max_hp reflects the exp-level upgrade tier reached, so "
                   "it also tells you how far that unit has levelled",
    "heat17_max / heat42_max": "launcher upgrade tier of that unit",
    "rounds_fired_so_far": "cumulative 17mm rounds this robot has fired this round",
    "dx / dy / dist_m": "offset and straight-line distance from the acting robot, with no "
                        "line-of-sight guarantee",
    "bearing_deg": "math bearing from the acting robot to that entity, atan2(dy,dx) in degrees",
    "position_known": "false = the referee system has lost track of that unit; its dx/dy are absent",
    "engagement_prior": "empirical [0,1] prior that this pair of positions historically produced an "
                        "engagement — a data-driven stand-in for line of sight, which is not logged",
    "team_slots": "your own team in a FIXED 6-slot order including the acting robot itself "
                  "(is_ego=true marks it)",
    "buildings.*_outpost_hp": "0 means destroyed; while an outpost is alive its own base is invincible",
    "opponent_strength_prior": "leak-safe historical strength features for the opposing school, "
                               "built from past matches only",
    "t_sec / seconds_remaining": "position in the 420-second round",
}

TARGETS_FULL = {
    "hero": "The enemy hero: 42mm, the only unit that can meaningfully damage a base (200 per hit), "
            "so the biggest threat and the highest-value kill.",
    "engineer": "The enemy engineer: unarmed, 250 HP, but it is the whole enemy economy and tech "
                "tier, and it rebuilds their outpost in 5 s. Holds 50% defense in the first 3 "
                "minutes and is invincible inside the assembly and supply zones.",
    "infantry_3": "The enemy infantry numbered 3: 17mm, 150-400 HP, their main map-control unit.",
    "infantry_4": "The enemy infantry numbered 4: 17mm, 150-400 HP, their main map-control unit.",
    "aerial": "The enemy aerial robot: 17mm, 750 rounds and no resupply, so it runs dry.",
    "sentry": "The enemy sentry: automated, 400 HP, 30/s cooling, guarding their base area. "
              "Engaging it means fighting into their defended ground.",
    "none": "Do not engage any enemy this second: nothing worth shooting, or the robot should be "
            "repositioning, cooling, or contesting a zone instead.",
}

AGGR_FULL = [
    "Disengage: break contact and fall back toward own ground, supply or cover, to heal or to let "
    "heat cool",
    "Hold: keep the current position and defend the approach without advancing",
    "Trade: exchange fire at the current range without giving or taking ground",
    "Press: advance into enemy-held ground to contest their base area or buff zones",
]
