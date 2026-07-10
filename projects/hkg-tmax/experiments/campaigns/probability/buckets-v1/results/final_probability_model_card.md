# HKG Tmax Probability Bucket V1 Model Card

Champion: `B4_hierarchical_residual_pmf`

Scope: weather probability distribution only. No market prices, EV, order books, Kelly sizing, PnL, or trade recommendations are used or emitted.

Primary target: HKO Daily Extract one-decimal HKG daily maximum temperature bucket.

Primary normalized RPS: 0.041524
NLL: 1.037181
Brier: 0.045921
ECE: 0.019859

Leakage audit status: `pass` with total violations `0`.
Label first-publication audit: `ok`, bucket changes `0`.

Selection rule: leaderboard sorted by normalized RPS ascending; methods must also pass no-worse NLL/Brier gates versus B4.