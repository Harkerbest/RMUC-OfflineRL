# 2027 sentry evaluation (paired against `bt:balanced`)

| arm | n | score | 95% CI | W/D/L | control lead | damage lead | Δscore vs ref [95% CI] | Δcontrol vs ref [95% CI] | Δdamage vs ref [95% CI] | deaths | no-ammo | on own central | on enemy base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bt:balanced | 16 | 0.375 | [0.18, 0.61] | 6/0/10 | -271.9 | +773 | — | — | — | 1.75 | 0.40 | 0.54 | 0.00 |
| bt:raid | 16 | 0.750 | [0.51, 0.90] | 12/0/4 | +405.8 | +495 | +0.375 [+0.125, +0.625] | +677.7 [+472.6, +859.9] | -278 [-1199, +723] | 0.56 | 0.46 | 0.00 | 0.70 |

score: win 1, draw 0.5, loss 0.  Leads are own minus opponent at the end of the match.
Δ columns: mean paired difference to the reference arm on the same (own, opp, seed, side), bootstrap 95% CI.
