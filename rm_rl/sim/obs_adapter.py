"""Live Arena-1 match -> the offline 161-D observation, through features.build_obs.

The adapter does not re-implement any feature: it rebuilds `GameArrays` from the
per-second referee rows recorded so far (`RefereeLog`) and calls the very same
`build_obs` the dataset builder uses, then keeps the newest row.  Parity with
the offline pipeline therefore holds by construction, and `check_parity`
verifies it against a log written to SQLite and read back.

Enemy information
-----------------
The official log is a full-information referee feed — every enemy position is
known — and the policies were trained on it (unless built with --vis-radius).

* ``enemy_info="full"``   reproduces that: the training distribution.
* ``enemy_info="vision"`` keeps an enemy's position only while one of the ego
  team's cameras sees it this second (Arena-1's field-of-view and occlusion
  model).  Unseen enemies are written as (0, 0) — exactly how the log encodes a
  robot the referee lost — so `pos_known`, the relative position and the
  engagement prior all switch off the way `build_obs` already handles.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..data import schema as S
from ..data.features import GameArrays, build_obs
from ..data.team_prior import DEFAULTS as TEAM_DEFAULTS
from .arena import SLOT_TYPE_2026, TEAM_CAMP


class ObsAdapter:
    def __init__(self, log, vis_map=None, use_team_prior=True, enemy_info="full"):
        """use_team_prior: the run was trained with --team-prior.  Arena matches
        have no season history, so the cold-start prior (as for a team's first
        official match) is used for both sides; without it the block is zeros."""
        if enemy_info not in ("full", "vision"):
            raise ValueError(enemy_info)
        self.log, self.vis_map, self.enemy_info = log, vis_map, enemy_info
        self.team_feat = (np.asarray(list(TEAM_DEFAULTS) * 2, np.float32)
                          if use_team_prior else None)
        self.game: GameArrays | None = None

    def refresh(self) -> GameArrays:
        """Rebuild the arrays from the rows logged so far; call once per decision."""
        self.game = self.log.game_arrays()
        return self.game

    def _gated(self, camp: str, visible_slots) -> GameArrays:
        game = self.game
        ecamp = S.enemy_camp(camp)
        ent = dict(game.ent)
        for slot, rtype in SLOT_TYPE_2026.items():
            if slot in visible_slots:
                continue
            rid = S.robot_id(rtype, ecamp)
            e = ent.get(rid)
            if e is None:
                continue
            x, y = e.x.copy(), e.y.copy()
            x[-1] = y[-1] = 0.0                      # the log's "position unknown"
            ent[rid] = dataclasses.replace(e, x=x, y=y)
        return GameArrays(game.T, ent)

    def observe(self, team: str, agent_type: str, visible_slots=None) -> np.ndarray:
        """161-D observation for `agent_type` of `team` at the latest second.
        visible_slots: enemy slots currently seen by `team` (vision mode only)."""
        camp = TEAM_CAMP[team]
        game = self.game
        if self.enemy_info == "vision":
            game = self._gated(camp, set(visible_slots or ()))
        obs = build_obs(game, camp, agent_type, vis_map=self.vis_map, team_feat=self.team_feat)
        return obs[-1]

    def full_obs(self, team: str, agent_type: str) -> np.ndarray:
        """Full-information observation regardless of `enemy_info` (critic input)."""
        obs = build_obs(self.game, TEAM_CAMP[team], agent_type, vis_map=self.vis_map,
                        team_feat=self.team_feat)
        return obs[-1]
