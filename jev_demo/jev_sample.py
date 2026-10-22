# -*- coding: utf-8 -*-
"""Jev 样例：取一个真实 RMUC 对局秒喂给 TypeSafe Jev，并与人类操作手的实际动作对照。

两次调用：
  A) ego 视角（做了占位可见性遮挡）-> weapons_free / target / nav_cell   [可与操作手对比]
  B) 完整裁判系统状态             -> 红方胜率 Noul                      [验证实验 1 的单点预览]
"""
import io, json, math, os, sqlite3, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.environ.get("RMUC_DB", os.path.join(REPO, "dataset", "rmuc_2026_region_dataset.sqlite"))
EGO_TYPE, EGO_SIDE = "步兵3", "红"
NO_TURRET, CONE, VIS_R = -140.0, 15.0, 8.0   # VIS_R 是占位，真实版接 rm_rl/data/vis_map.py
CELL, NX, NY = 2.0, 14, 8                     # 28m x 15m -> 112 格，低于 Choice 的 255 上限

ROLE_EN = {"英雄": "hero", "工程": "engineer", "步兵3": "infantry_3", "步兵4": "infantry_4",
           "空中": "aerial", "哨兵": "sentry", "基地": "base", "前哨站": "outpost"}
SHOOTERS = {"英雄", "步兵3", "步兵4", "空中", "哨兵"}


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def cell_of(x, y):
    cx = min(NX - 1, max(0, int(x // CELL)))
    cy = min(NY - 1, max(0, int(y // CELL)))
    return f"c{cx:02d}_{cy:02d}"


def build_grid_criteria(red_home_low_x):
    out = {}
    for cx in range(NX):
        for cy in range(NY):
            x0, x1 = cx * CELL, (cx + 1) * CELL
            y0, y1 = cy * CELL, (cy + 1) * CELL
            half = ("own half" if (cx < NX / 2) == red_home_low_x else "enemy half")
            lane = "bottom lane" if cy < NY / 3 else ("top lane" if cy > 2 * NY / 3 else "middle lane")
            depth = "deep" if (cx < 3 or cx > NX - 4) else "mid-field"
            out[f"c{cx:02d}_{cy:02d}"] = (f"x {x0:.0f}-{x1:.0f} m, y {y0:.0f}-{y1:.0f} m; "
                                         f"{half}, {lane}, {depth}")
    out["stay"] = "Hold current position, do not relocate this second."
    return out


def load(cur, gid, t):
    cur.execute("select * from timeseries where game_id=? and 时刻秒=?", (gid, t))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def pick_moment(cur):
    """选一个中局、ego 正在开火、且场上有多个活敌的秒。"""
    cur.execute("select game_id, 胜方, 红方学校, 蓝方学校 from matches where 胜方 in ('红','蓝') "
                "and 时长秒>=400 order by game_id limit 1")
    gid, winner, red_school, blue_school = cur.fetchone()
    cur.execute("select 时刻秒, 累计17mm发弹 from timeseries where game_id=? and 机器人类型=? "
                "and 阵营=? order by 时刻秒", (gid, EGO_TYPE, EGO_SIDE))
    ser = [(t, a) for t, a in cur.fetchall() if a is not None]
    for i in range(len(ser) - 4):
        t, a = ser[i]
        if not (120 <= t <= 300):
            continue
        if ser[i + 1][1] - a <= 0:
            continue
        snap = load(cur, gid, t)
        ego = next((r for r in snap if r["机器人类型"] == EGO_TYPE and r["阵营"] == EGO_SIDE), None)
        if ego is None or ego["枪口朝向"] == NO_TURRET or (ego["x"] == 0 and ego["y"] == 0):
            continue
        foes = [r for r in snap if r["阵营"] != EGO_SIDE and r["机器人类型"] in SHOOTERS
                and r["当前血量"] > 0 and not (r["x"] == 0 and r["y"] == 0)]
        if len(foes) >= 2:
            return gid, t, winner, red_school, blue_school
    raise SystemExit("no suitable moment found")


def entity(r, ego, masked):
    d = math.hypot(r["x"] - ego["x"], r["y"] - ego["y"])
    e = {"role": ROLE_EN[r["机器人类型"]], "hp": round(r["当前血量"]), "max_hp": round(r["最大血量"]),
         "x": round(r["x"], 1), "y": round(r["y"], 1), "dist_m": round(d, 1),
         "bearing_deg": round(math.degrees(math.atan2(r["y"] - ego["y"], r["x"] - ego["x"])), 1),
         "radar_marked_vulnerable": bool(r["是否易伤"])}
    if not masked:
        e["heat"] = None if r["小热量"] is None else round(r["小热量"])
        e["heat_max"] = None if r["小热量上限"] is None else round(r["小热量上限"])
    return e


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    gid, t, winner, red_school, blue_school = pick_moment(cur)
    snap, nxt = load(cur, gid, t), load(cur, gid, t + 1)
    fut = load(cur, gid, min(t + 3, t + 3))

    def get(rowset, typ, side):
        return next((r for r in rowset if r["机器人类型"] == typ and r["阵营"] == side), None)

    ego = get(snap, EGO_TYPE, EGO_SIDE)
    ego_n, ego_f = get(nxt, EGO_TYPE, EGO_SIDE), get(fut, EGO_TYPE, EGO_SIDE)

    # 判断红方大本营在 x 小的一侧（用 t=1 的出生位置）
    first = load(cur, gid, 1.0)
    rx = [r["x"] for r in first if r["阵营"] == "红" and r["机器人类型"] in SHOOTERS and r["x"] > 0]
    bx = [r["x"] for r in first if r["阵营"] == "蓝" and r["机器人类型"] in SHOOTERS and r["x"] > 0]
    red_home_low_x = (sum(rx) / max(1, len(rx))) < (sum(bx) / max(1, len(bx)))

    ROBOTS = {"英雄", "工程", "步兵3", "步兵4", "空中", "哨兵"}
    allies = [r for r in snap if r["阵营"] == EGO_SIDE and r is not ego and r["机器人类型"] in ROBOTS]
    foes_all = [r for r in snap if r["阵营"] != EGO_SIDE and r["机器人类型"] in SHOOTERS and r["当前血量"] > 0
                and not (r["x"] == 0 and r["y"] == 0)]
    foes_vis = [r for r in foes_all if math.hypot(r["x"] - ego["x"], r["y"] - ego["y"]) <= VIS_R
                or r["是否易伤"]]
    if len(foes_vis) < 1:
        foes_vis = foes_all[:1]

    def building(rowset, typ, side):
        r = get(rowset, typ, side)
        return None if r is None else round(r["当前血量"])

    common = {
        "t_sec": int(t), "match_len_sec": 420, "my_team": "red",
        "buildings": {"my_outpost_hp": building(snap, "前哨站", "红"),
                      "enemy_outpost_hp": building(snap, "前哨站", "蓝"),
                      "my_base_hp": building(snap, "基地", "红"),
                      "enemy_base_hp": building(snap, "基地", "蓝")},
        "economy": {"my_gold_total": round(ego["队伍总金币"]), "my_gold_left": round(ego["队伍剩余金币"])},
        "field": {"size_m": [28, 15], "my_home_side": "low x" if red_home_low_x else "high x"},
    }
    ego_block = {"role": ROLE_EN[EGO_TYPE], "x": round(ego["x"], 1), "y": round(ego["y"], 1),
                 "hp": round(ego["当前血量"]), "max_hp": round(ego["最大血量"]),
                 "heat_17mm": round(ego["小热量"]), "heat_17mm_max": round(ego["小热量上限"]),
                 "chassis_power_w": round(ego["底盘功率"], 1),
                 "turret_bearing_deg": round(ego["枪口朝向"], 1),
                 "radar_marked_vulnerable": bool(ego["是否易伤"])}

    state_ego = dict(common, ego=ego_block,
                     allies=[entity(r, ego, True) for r in allies],
                     visible_enemies=[entity(r, ego, True) for r in foes_vis],
                     note="Enemies not listed are currently unseen by this robot.")
    state_full = dict(common, ego=ego_block,
                      allies=[entity(r, ego, False) for r in allies],
                      enemies=[entity(r, ego, False) for r in foes_all])

    # ---------- 人类操作手在这一秒的实际动作（ground truth） ----------
    fired = (ego_n["累计17mm发弹"] or 0) - (ego["累计17mm发弹"] or 0)
    aimed = sorted(((angdiff(ego["枪口朝向"],
                             math.degrees(math.atan2(r["y"] - ego["y"], r["x"] - ego["x"]))), r)
                    for r in foes_all), key=lambda p: p[0])
    human = {
        "weapons_free": fired > 0, "rounds_fired_this_sec": int(fired),
        "target_by_muzzle_bearing": (ROLE_EN[aimed[0][1]["机器人类型"]] if aimed and aimed[0][0] <= CONE
                                     else "none"),
        "muzzle_err_deg": round(aimed[0][0], 1) if aimed else None,
        "n_enemies_in_cone": sum(1 for e, _ in aimed if e <= CONE),
        "cell_now": cell_of(ego["x"], ego["y"]),
        "cell_in_3s": cell_of(ego_f["x"], ego_f["y"]) if ego_f else None,
        "match_winner": "red" if winner == "红" else "blue",
    }
    con.close()
    return gid, t, red_school, blue_school, state_ego, state_full, human, red_home_low_x


if __name__ == "__main__":
    gid, t, rs, bs, s_ego, s_full, human, rlx = main()
    print(f"=== 样本 game_id={gid} t={int(t)}s  红:{rs} vs 蓝:{bs} ===")
    print("\n--- state A (ego 视角, 已遮挡) ---")
    print(json.dumps(s_ego, ensure_ascii=False, indent=2))
    print("\n--- 人类操作手实际动作 ---")
    print(json.dumps(human, ensure_ascii=False, indent=2))
    json.dump({"state_ego": s_ego, "state_full": s_full, "human": human,
               "red_home_low_x": rlx, "game_id": gid, "t": t},
              open(os.path.join(os.path.dirname(__file__), "sample.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote sample.json")
