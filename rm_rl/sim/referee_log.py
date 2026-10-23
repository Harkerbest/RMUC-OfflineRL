"""Arena-1 engine frames -> the official RMUC 2026 referee log, one row per second.

`RefereeLog` is fed every engine tick and keeps the same three tables the DJI
dataset ships (`matches`, `timeseries`, `events`, Chinese column names).  It
serves two consumers:

* `rm_rl.sim.export_sqlite` writes the rows to a SQLite file, so the unchanged
  offline pipeline (vis_map -> team_prior -> build_dataset -> train) can run on
  simulated matches;
* `rm_rl.sim.obs_adapter` turns the rows recorded so far into `GameArrays`
  through `build_dataset.game_arrays_from_df`, so a live observation goes through
  the exact code path of an offline one.

Field mapping (Arena-1 robot -> timeseries column):

    当前血量/最大血量   hp / max_hp
    x, y, z            pose; buildings stay at (0, 0) like the official log
    枪口朝向            degrees(aim), wrapped to [-140, 220); engineer/buildings -140
    底盘功率            power_w
    小热量(上限)        heat / max_heat on 17 mm shooters
    大热量(上限)        heat / max_heat on the hero (42 mm)
    累计17/42mm发弹     cumulative `shot` events of that robot
    队伍总金币          team resources + everything spent so far
    队伍剩余金币        team resources
    是否易伤            0 (Arena has no radar marking)

Dataset conventions.  Several columns are *constant* in the official log, and a
policy normalises them with the 1e-3 standard-deviation floor, so any deviation
is amplified ~1000x and saturates the network (measured: a policy fed Arena's
raw aerial power emitted one constant command all match).  Where Arena-1 models
a robot differently from the log, the row follows the log:

    空中  (aerial)  100 / 100 HP and 0 W chassis power — the log's aerial row;
                    Arena 2026's aerial has no hit points and reports flight power
    哨兵  (slot 7)  health and 17 mm heat as fractions of the log's fixed sentry
                    caps, 400 HP / 260 heat — Arena 2026 fields an infantry there
    英雄  (hero)    小热量上限 50, the value the log carries for the hero's
                    unused 17 mm barrel
"""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from ..data import schema as S
from .arena import SLOT_TYPE_2026, TEAM_CAMP

TS_COLUMNS = ["赛区", "场次号", "赛程", "局号", "game_id", "时刻秒", "robot_id", "机器人类型",
              "阵营", "学校名", "对手学校", "当前血量", "最大血量", "x", "y", "z", "枪口朝向",
              "底盘功率", "小热量", "小热量上限", "大热量", "大热量上限", "累计17mm发弹",
              "累计42mm发弹", "队伍总金币", "队伍剩余金币", "是否易伤"]
EV_COLUMNS = ["赛区", "场次号", "赛程", "局号", "game_id", "时刻秒", "事件类型", "robot_id",
              "机器人类型", "阵营", "学校名", "目标robot_id", "目标类型", "类别", "数值", "备注"]
MATCH_COLUMNS = ["赛区", "场次号", "赛程", "局号", "game_id", "web_game_id", "红方学校",
                 "蓝方学校", "胜方", "开始时间", "时长秒"]
REGION = "Arena-1"
AERIAL_HP = S.MAX_HP[S.TYPE_AERIAL]                # 100
SENTRY_HP = S.MAX_HP[S.TYPE_SENTRY]                # 400
SENTRY_HEAT17 = S.HEAT17_MAX_REF                   # 260
HERO_HEAT17_MAX = 50.0

# timeseries column -> ASCII alias, for game_arrays_from_df
_ALIAS = {k: v for k, v in S.TS_COLS.items()}


def _yaw_deg(aim_rad: float) -> float:
    d = math.degrees(aim_rad)
    return (d + 140.0) % 360.0 - 140.0            # the log's [-140, 220) range


class RefereeLog:
    def __init__(self, game_id: int, schools: dict, flip_y: bool = False):
        """schools: {"red": name, "blue": name} (the strategy names play the schools).
        flip_y mirrors y (15 - y) for a dataset whose y axis runs the other way."""
        self.game_id, self.schools, self.flip_y = int(game_id), schools, flip_y
        self.ts_rows: list[list] = []
        self.ev_rows: list[list] = []
        self.shots = defaultdict(lambda: [0, 0])          # robot id -> [17mm, 42mm]
        self.spent = {"red": 0.0, "blue": 0.0}
        self.last_res: dict = {}
        self.ids: dict = {}                                # arena id -> (robot_id, 类型, team)
        self.last_second = -1

    # ------------------------------------------------------------------ helpers
    def _rid(self, robot):
        key = robot["id"]
        if key not in self.ids:
            rtype = SLOT_TYPE_2026[int(robot["slot"])]
            self.ids[key] = (S.robot_id(rtype, TEAM_CAMP[robot["team"]]), rtype, robot["team"])
        return self.ids[key]

    def _y(self, y):
        return S.FIELD_Y - y if self.flip_y else y

    def _common(self, team):
        return [REGION, self.game_id, f"【{REGION}】第{self.game_id}场", 1, self.game_id]

    def _event(self, second, etype, team, rid, rtype, category, value, target=(None, None)):
        self.ev_rows.append(self._common(team) + [
            second, etype, rid, rtype, TEAM_CAMP[team], self.schools[team],
            target[0], target[1], category, value, None])

    # ------------------------------------------------------------------ feed
    def on_frame(self, frame: dict):
        """Call with every engine frame (20 Hz)."""
        second = int(math.ceil(frame["time"] - 1e-9))
        for team in ("red", "blue"):
            res = float(frame["teams"][team].get("resources", 0.0))
            prev = self.last_res.get(team)
            if prev is not None and res < prev:
                self.spent[team] += prev - res
            self.last_res[team] = res
        robots = {r["id"]: r for r in frame["robots"]}
        for e in frame["events"]:
            kind = e.get("type")
            if kind == "shot" and e.get("robot") in robots:
                r = robots[e["robot"]]
                rid, rtype, team = self._rid(r)
                big = r["role"] == "hero"
                self.shots[r["id"]][1 if big else 0] += 1
                self._event(second, S.EV_SHOOT, team, rid, rtype, "42mm" if big else "17mm", 1)
            elif kind == "hit" and e.get("robot") in robots:
                rid, rtype, team = self._rid(robots[e["robot"]])
                self._event(second, S.EV_HIT, team, rid, rtype, "装甲", -float(e.get("damage", 0.0)))
            elif kind in ("base_hit", "outpost_hit") and e.get("target_team") in TEAM_CAMP:
                team = e["target_team"]
                rtype = S.TYPE_OUTPOST if kind == "outpost_hit" else S.TYPE_BASE
                self._event(second, S.EV_HIT, team, S.robot_id(rtype, TEAM_CAMP[team]), rtype,
                            "装甲", -float(e.get("damage", 0.0)))
        if abs(frame["time"] - round(frame["time"])) < 1e-6 and round(frame["time"]) > self.last_second:
            self._record(int(round(frame["time"])), frame)

    def _record(self, second: int, frame: dict):
        self.last_second = second
        coins = {t: (self.last_res[t] + self.spent[t], self.last_res[t]) for t in ("red", "blue")}
        enemy = {"red": "blue", "blue": "red"}
        for r in frame["robots"]:
            rid, rtype, team = self._rid(r)
            hero, eng = r["role"] == "hero", r["role"] == "engineer"
            s17, s42 = self.shots[r["id"]]
            hp, max_hp, power = float(r["hp"]), float(r["max_hp"]), float(r.get("power_w", 0.0))
            h17 = h17max = h42 = h42max = None
            if hero:
                h17, h17max, h42, h42max = 0.0, HERO_HEAT17_MAX, float(r["heat"]), float(r["max_heat"])
            elif not eng:
                h17, h17max = float(r["heat"]), float(r["max_heat"])
            if rtype == S.TYPE_AERIAL:
                hp = max_hp = AERIAL_HP
                power = 0.0
            elif rtype == S.TYPE_SENTRY:
                hp, max_hp = (hp / max_hp * SENTRY_HP if max_hp > 0 else 0.0), SENTRY_HP
                h17, h17max = (h17 / h17max * SENTRY_HEAT17 if h17max else 0.0), SENTRY_HEAT17
            self.ts_rows.append(self._common(team) + [
                second, rid, rtype, TEAM_CAMP[team], self.schools[team], self.schools[enemy[team]],
                hp, max_hp, float(r["x"]), float(self._y(r["y"])), float(r["z"]),
                S.YAW_SENTINEL if eng else _yaw_deg(r["aim"]) if not self.flip_y else _yaw_deg(-r["aim"]),
                power, h17, h17max, h42, h42max,
                None if (hero or eng) else s17, s42 if hero else None,
                coins[team][0], coins[team][1], 0])
        for team in ("red", "blue"):
            for rtype, src in ((S.TYPE_BASE, frame["bases"][team]), (S.TYPE_OUTPOST, frame["outposts"][team])):
                self.ts_rows.append(self._common(team) + [
                    second, S.robot_id(rtype, TEAM_CAMP[team]), rtype, TEAM_CAMP[team],
                    self.schools[team], self.schools[enemy[team]],
                    float(src["hp"]), float(src["max_hp"]), 0.0, 0.0, 0.0, S.YAW_SENTINEL, 0.0,
                    None, None, None, None, None, None, coins[team][0], coins[team][1], 0])

    # ------------------------------------------------------------------ outputs
    def timeseries_df(self, ascii_names: bool = True) -> pd.DataFrame:
        """All rows so far; with ASCII aliases ready for game_arrays_from_df."""
        df = pd.DataFrame(self.ts_rows, columns=TS_COLUMNS)
        if ascii_names:
            df = df[list(_ALIAS)].rename(columns=_ALIAS)
            for c in df.columns:
                if c != "rtype" and c != "camp" and c != "school":
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def game_arrays(self):
        from ..data.build_dataset import game_arrays_from_df
        return game_arrays_from_df(self.timeseries_df())

    def match_row(self, winner_team: str | None, duration: float, start_index: int):
        winner = TEAM_CAMP.get(winner_team) if winner_team else None
        start = f"2026-01-01 00:00:{start_index:06d}"      # orders matches for team_prior
        return [REGION, self.game_id, f"【{REGION}】第{self.game_id}场", 1, self.game_id,
                self.game_id, self.schools["red"], self.schools["blue"], winner, start,
                int(round(duration))]
