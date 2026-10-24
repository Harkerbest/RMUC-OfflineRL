# 2027 sentry evaluation (paired against `bt:balanced`)

| arm | n | score | 95% CI | W/D/L | control lead | damage lead | Δscore vs ref [95% CI] | Δcontrol vs ref [95% CI] | Δdamage vs ref [95% CI] | deaths | no-ammo | on own central |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rl_0020 | 18 | 0.389 | [0.20, 0.61] | 7/0/11 | +0.1 | +272 | -0.222 [-0.444, -0.056] | -7.2 [-15.4, -0.1] | -5 [-168, +194] | 0.39 | 0.24 | 0.53 |
| imit | 18 | 0.667 | [0.44, 0.84] | 12/0/6 | +8.2 | +346 | +0.056 [-0.111, +0.222] | +0.9 [-12.8, +14.9] | +68 [-135, +267] | 0.44 | 0.25 | 0.64 |
| bt:balanced | 18 | 0.611 | [0.39, 0.80] | 11/0/7 | +7.3 | +277 | — | — | — | 0.33 | 0.18 | 0.63 |
| script | 18 | 0.556 | [0.34, 0.75] | 10/0/8 | -4.3 | +161 | -0.056 [-0.389, +0.278] | -11.6 [-38.4, +15.3] | -116 [-406, +186] | — | — | — |
| rl | 18 | 0.556 | [0.34, 0.75] | 10/0/8 | +6.5 | +201 | -0.056 [-0.278, +0.167] | -0.8 [-12.3, +10.8] | -76 [-328, +202] | 0.44 | 0.27 | 0.51 |

score: win 1, draw 0.5, loss 0.  Leads are own minus opponent at the end of the match.
Δ columns: mean paired difference to the reference arm on the same (own, opp, seed, side), bootstrap 95% CI.
