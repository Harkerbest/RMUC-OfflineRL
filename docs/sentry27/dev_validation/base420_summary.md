# 2027 sentry evaluation (paired against `bt:balanced`)

| arm | n | score | 95% CI | W/D/L | control lead | damage lead | Δscore vs ref [95% CI] | Δcontrol vs ref [95% CI] | Δdamage vs ref [95% CI] | deaths | no-ammo | on own central |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| script | 8 | 0.375 | [0.14, 0.69] | 3/0/5 | -391.3 | +750 | -0.125 [-0.375, +0.000] | -179.3 [-495.4, +129.5] | -621 [-1856, +643] | — | — | — |
| bt:balanced | 8 | 0.500 | [0.22, 0.78] | 4/0/4 | -211.9 | +1372 | — | — | — | 2.12 | 0.36 | 0.50 |

score: win 1, draw 0.5, loss 0.  Leads are own minus opponent at the end of the match.
Δ columns: mean paired difference to the reference arm on the same (own, opp, seed, side), bootstrap 95% CI.
