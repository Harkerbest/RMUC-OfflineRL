"""Per-second rewards for live Arena-1 matches, from `rm_rl.data.reward`.

No new reward is defined here.  The simulator's per-second referee rows are
turned into the same `GameArrays` the offline builder uses, and
`reward.compute_reward` runs on them unchanged, so every term keeps the meaning
and the weights that were validated against match outcomes (AUC 0.976):

    outpost/base atk & def   前哨站 / 基地 当前血量          <- Arena outposts/bases hp
    dmg_deal / dmg_take      sum of mobile 当前血量 drops    <- Arena robot hp
    kill / death             当前血量 > 0 -> 0 transitions  <- Arena robot hp
    ego_surv / ego_hp        the ego's own 当前血量         <- Arena robot hp
    heat                     小热量 over 小热量上限          <- Arena heat / max_heat
    econ                     队伍总金币 advantage            <- resources + spend
    terminal (win, base)     胜方 + final base HP            <- Arena winner, bases

`compute_reward` always adds the terminal terms to the last transition it sees,
which is only right once the match is over; `step_rewards` strips them while a
match is still running.
"""
from __future__ import annotations

import numpy as np

from ..data import schema as S
from ..data.reward import BASE_NORM, RewardConfig, compute_reward


def step_rewards(game, camp: str, agent_type: str, cfg: RewardConfig,
                 winner_camp: str | None = None, terminal: bool = False) -> np.ndarray:
    """Rewards for every transition 1..T of the match so far ([T-1]).

    terminal=False: an unfinished match — no win term, no final-base term.
    terminal=True:  the match is over; `winner_camp` ('红'/'蓝'/None) applies.
    """
    if game.T < 2:
        return np.zeros(0, np.float32)
    r, _ = compute_reward(game, camp, agent_type, winner_camp if terminal else "", cfg)
    if not terminal:
        own = game.get(S.TYPE_BASE, camp).hp
        en = game.get(S.TYPE_BASE, S.enemy_camp(camp)).hp
        r[-1] -= cfg.w_final_base * float((own[-1] - en[-1]) / BASE_NORM)
    return r
