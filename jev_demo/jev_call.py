# -*- coding: utf-8 -*-
"""把 jev_sample.py 生成的真实对局秒喂给 Jev，打印输出并与人类操作手动作对照。"""
import io, json, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(HERE, ".env")):
    """本地化存储 key：jev_demo/.env（已在 .gitignore 里），环境变量优先。"""
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

TARGETS = {
    "hero": "The enemy hero (42mm, highest damage output, usually the biggest threat).",
    "engineer": "The enemy engineer. It never shoots; killing it only denies economy.",
    "infantry_3": "The enemy infantry numbered 3 (17mm).",
    "infantry_4": "The enemy infantry numbered 4 (17mm).",
    "aerial": "The enemy aerial robot.",
    "sentry": "The enemy sentry, an automated turret defending its own base area.",
    "none": "No enemy is worth engaging this second (none visible, none in range, "
            "or the robot should be repositioning or cooling instead).",
}


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


RULES = ("RoboMaster university championship, 7-minute 1v1 team match on a 28m x 15m arena. "
         "This robot is a 17mm infantry driven toward winning the round: the round is decided by "
         "outpost and base HP, so trading HP for map control near the enemy outpost is usually "
         "worth more than farming kills. Firing while 17mm heat is near its cap causes "
         "self-damage. All coordinates are absolute arena metres; bearings are math bearings "
         "atan2(dy,dx) in degrees.")

questions = {
    "weapons_free": Noul(instructions=RULES + " Should this robot be firing 17mm rounds during "
                                              "the next second?"),
    "target": Choice(instructions=RULES + " Which enemy should this robot put its turret on for "
                                          "the next second?", criteria=TARGETS),
    "nav_cell": Choice(instructions=RULES + " Which arena grid cell should this robot move toward "
                                            "over the next ~3 seconds?",
                       criteria=grid_criteria(S["red_home_low_x"])),
    "aggression": Score(instructions=RULES + " How aggressive should this robot be right now?",
                        criteria=["Break contact and retreat to heal or cool down",
                                  "Hold position and defend",
                                  "Trade fire at current range",
                                  "Push forward onto the enemy outpost"]),
}

with TypeSafeClient() as client:
    t0 = time.perf_counter()
    rA = client.system_one(state=S["state_ego"], questions=questions)
    msA = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    rB = client.system_one(state=S["state_full"], questions={
        "red_wins": Noul(instructions=RULES + " Will the RED team win this round?"),
    })
    msB = (time.perf_counter() - t0) * 1000

H = S["human"]
print(f"=== Jev {rA.model} | game_id={S['game_id']} t={int(S['t'])}s ===")
print(f"\n[A] ego 视角 4 个问题 一次调用: {msA:.0f} ms, "
      f"input_tokens={rA.usage.input_tokens}, output_tokens={rA.usage.output_tokens}")

a = rA.answers
print(f"\nweapons_free  Jev P(fire)={a['weapons_free'].noul:.3f}   | 人类实际: "
      f"{H['weapons_free']} ({H['rounds_fired_this_sec']} 发)")

tg = a["target"]
top = sorted(tg.probabilities.items(), key=lambda kv: -kv[1])[:4]
print(f"target        Jev={tg.choice} (conf {tg.confidence:.3f}) | 人类实际(枪口指向)="
      f"{H['target_by_muzzle_bearing']} (误差 {H['muzzle_err_deg']}°, 锥内 {H['n_enemies_in_cone']} 个)")
print("              分布 top4: " + ", ".join(f"{k}={v:.3f}" for k, v in top))

nv = a["nav_cell"]
topn = sorted(nv.probabilities.items(), key=lambda kv: -kv[1])[:5]
print(f"nav_cell      Jev={nv.choice} (conf {nv.confidence:.3f})   | 人类实际: now={H['cell_now']} "
      f"-> +3s={H['cell_in_3s']}")
print("              分布 top5: " + ", ".join(f"{k}={v:.3f}" for k, v in topn))

ag = a["aggression"]
print(f"aggression    Jev score={ag.score:.2f} (conf {ag.confidence:.3f}) "
      f"legend={json.dumps(ag.legend, ensure_ascii=False)}")
print("              分布: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(ag.probabilities.items())))

print(f"\n[B] 完整裁判系统状态 胜率问题: {msB:.0f} ms, input_tokens={rB.usage.input_tokens}")
print(f"red_wins      Jev P(红胜)={rB.answers['red_wins'].noul:.3f}  | 实际胜方: {H['match_winner']}")

tot = (rA.usage.input_tokens or 0) + (rB.usage.input_tokens or 0)
print(f"\n本次两调用共 {tot} input tokens -> ${tot * 0.042 / 1e6:.6f}")
print(f"按 A 调用的 {rA.usage.input_tokens} tok/状态外推 25.7 万个对局秒: "
      f"${(rA.usage.input_tokens or 0) * 257000 * 0.042 / 1e6:.2f}, "
      f"1200 RPM 下耗时 {257000 / 1200 / 60:.1f} 小时")
