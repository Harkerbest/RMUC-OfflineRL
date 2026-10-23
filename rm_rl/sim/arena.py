"""Run Arena-1 matches in-process: engine transport, scripted sides, team messages.

The Arena runner executes every strategy in its own sandboxed process under a
200 ms wall-clock deadline.  For data generation and RL that is the wrong
contract — we want a synchronous, deterministic loop with no deadlines — so this
module drives the engine directly and imports the (trusted, bundled) example
strategies as Python modules.

    arena = load_arena()                       # ARENA1_ROOT or ../Arena-1
    red = ScriptedSide(arena, "examples/claude_opus55_xhigh_team", "red")
    blue = ScriptedSide(arena, "examples/claude_opus55_xhigh_hold", "blue")
    m = ArenaMatch(arena, red, blue, seed=0)
    while not m.finished:
        m.decide()                             # scripted actions for both sides
        m.advance()                            # one decision interval (0.1 s)
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ARENA_ENV = "ARENA1_ROOT"

# Arena-1 RMUC 2026 roster slot -> the dataset's 机器人类型.  Arena's 2026 rules
# replace the sentry with a third infantry on slot 7; it keeps the sentry's
# referee id, so it fills the dataset's 哨兵 column.
SLOT_TYPE_2026 = {1: "英雄", 2: "工程", 3: "步兵3", 4: "步兵4", 6: "空中", 7: "哨兵"}
TYPE_SLOT_2026 = {v: k for k, v in SLOT_TYPE_2026.items()}
TEAM_CAMP = {"red": "红", "blue": "蓝"}
CAMP_TEAM = {v: k for k, v in TEAM_CAMP.items()}

_module_ids = itertools.count()


def arena_root(path: str | os.PathLike | None = None) -> Path:
    for cand in (path, os.environ.get(ARENA_ENV),
                 Path(__file__).resolve().parents[3] / "Arena-1"):
        if cand and (Path(cand) / "rmuc_arena" / "engine.py").is_file():
            return Path(cand).resolve()
    raise SystemExit(
        f"Arena-1 checkout not found. Clone https://github.com/RM-Arena/Arena-1, build "
        f"its engine (cargo build --release --manifest-path simulation/Cargo.toml) and "
        f"set {ARENA_ENV}=/path/to/Arena-1 or pass --arena.")


def load_arena(path=None) -> SimpleNamespace:
    """Import Arena-1's engine transport, rules and public strategy SDK."""
    root = arena_root(path)
    for p in (root, root / "rmuc_arena"):    # strategies do `from sdk import ...`
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from rmuc_arena import engine, seasons, submissions
    from rmuc_arena.communication import TeamCommunication
    import sdk
    engine.binary_path()                      # fail early if the engine is not built
    return SimpleNamespace(root=root, engine=engine, seasons=seasons,
                           submissions=submissions, TeamCommunication=TeamCommunication,
                           sdk=sdk)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"arena_strategy_{next(_module_ids)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strip_messages(action: dict) -> tuple[dict, list]:
    action = dict(action)
    return action, action.pop("messages", [])


class ScriptedSide:
    """One Arena-1 submission (team, squad or distributed mode) playing one side."""

    def __init__(self, arena, path, team: str):
        path = Path(path)
        if not path.is_absolute():
            path = arena.root / path
        manifest, entries = arena.submissions.load_submission(path)
        self.name, self.team, self.mode = path.name, team, manifest["mode"]
        if self.mode not in ("team", "squad", "distributed"):
            raise ValueError(f"{path.name}: mode {self.mode!r} is not a full-team submission")
        # hardware the strategy was registered with (None = the season default)
        self.lineup = manifest.get("team_lineup")
        # (controlled slots or None for the whole team, module, private state)
        self.units = [[None if self.mode == "team" else [int(key)], _load_module(file), None]
                      for key, file in sorted(entries.items())]

    def viewers(self):
        return [(self.team, slots) for slots, _, _ in self.units]

    def act(self, observations):
        """observations: one dict per unit (same order as viewers()).
        Returns ({slot: action}, [(team, sender_slot, messages)])."""
        actions, outgoing = {}, []
        for unit, obs in zip(self.units, observations):
            out, unit[2] = unit[1].act(obs, unit[2])
            for slot, action in out.items():
                action, messages = strip_messages(action)
                if messages:
                    outgoing.append((self.team, str(slot), messages))
                actions[str(slot)] = action
        return actions, outgoing


class ArenaMatch:
    """Engine + scripted sides + message routing, stepped one decision at a time.

    `decide()` gathers observations, runs the scripted sides and returns the
    team views so a caller can override any slot's action (see `override`).
    `advance()` holds the commands for one decision interval (POLICY_INTERVAL
    engine ticks = 0.1 s) and returns the per-tick frames.
    """

    def __init__(self, arena, red: ScriptedSide | None, blue: ScriptedSide | None,
                 seed=0, duration=420.0, map_name="rmuc2026", lineups=None):
        self.arena, self.sides = arena, {"red": red, "blue": blue}
        random.seed(seed)        # strategies draw from the shared `random` module
        season = arena.seasons.season_of_map(map_name)
        if lineups is None and any(s is not None and s.lineup for s in self.sides.values()):
            # each side fields the hardware its strategy was registered with
            default = arena.seasons.default_lineup(season)
            lineups = {team: [{"slot": int(k), "model": int(v)}
                              for k, v in ((s and s.lineup) or default).items()]
                       for team, s in self.sides.items()}
        self.match = arena.engine.Match(seed=seed, duration=duration, map_name=map_name,
                                        lineups=lineups)
        self.slots = [str(s) for s in arena.seasons.slots(season)]
        self.comm = arena.TeamCommunication(None, self.slots)
        self.commands = {"red": {}, "blue": {}}
        self.frame = self.match.snapshot()
        self.team_obs = {}

    @property
    def finished(self) -> bool:
        return self.match.status == "finished"

    @property
    def time(self) -> float:
        return self.frame["time"]

    def _with_messages(self, team, raw_line):
        obs = json.loads(raw_line)
        inboxes = {slot: self.comm.inboxes[team].get(slot, []) for slot in obs["controlled_ids"]}
        obs["message_inboxes"] = inboxes
        obs["messages"] = [m for box in inboxes.values() for m in box]
        return obs

    def decide(self):
        """Run every scripted side; returns {team: full-team observation}."""
        viewers = [("red", None), ("blue", None)]
        unit_views = []
        for team, side in self.sides.items():
            if side is not None and side.mode != "team":
                unit_views += side.viewers()
        _, lines = self.match.observations(viewers + unit_views)
        team_obs = {t: self._with_messages(t, l) for (t, _), l in zip(viewers, lines)}
        extra = {(t, tuple(s)): self._with_messages(t, l) for (t, s), l in zip(unit_views, lines[2:])}
        commands, outgoing = {"red": {}, "blue": {}}, []
        for team, side in self.sides.items():
            if side is None:
                continue
            views = [team_obs[team] if slots is None else extra[(team, tuple(slots))]
                     for _, slots in side.viewers()]
            actions, sent = side.act(views)
            commands[team].update(actions)
            outgoing += sent
        self.commands, self._outgoing, self.team_obs = commands, outgoing, team_obs
        return team_obs

    def override(self, team, slot, action):
        """Replace one slot's command for the coming interval (messages allowed)."""
        action, messages = strip_messages(action)
        self.commands[team][str(slot)] = action
        self._outgoing = [o for o in self._outgoing if not (o[0] == team and o[1] == str(slot))]
        if messages:
            self._outgoing.append((team, str(slot), messages))

    def advance(self):
        tick = self.match.tick
        interval = self.arena.engine.POLICY_INTERVAL
        self.comm.route(self._outgoing, tick, tick + interval)
        frames = []
        for _ in range(interval):
            if self.finished:
                break
            self.frame = self.match.step(self.commands)
            frames.append(self.frame)
        return frames

    def close(self):
        self.match.close()
