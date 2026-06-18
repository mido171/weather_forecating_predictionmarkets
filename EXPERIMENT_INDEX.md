# Experiment Index

Generated from experiment `STATUS.yaml` files.

| ID | Title | Status | Primary conclusion | OOS delta | Leakage | Reproducible |
|---|---|---|---|---:|---|---|
| [EXP-0001](experiments/EXP-0001-g0-repository-and-archive-smoke-test/README.md) | G0 repository and archive smoke test | ACCEPTED | G0 repository and immutable raw archive smoke test passed; predictive modelling remains gated. | None | PASS | PASS |
| [EXP-0002](experiments/EXP-0002-g1-daily-extract-and-clmmaxt-target-parity/README.md) | G1 Daily Extract and CLMMAXT target parity | BLOCKED | Latest HKO Daily Extract and latest CLMMAXT matched 31/31 May 2026 rows, but G1 is blocked pending first-publication Daily Extract evidence. | None | PASS_TARGET_ONLY_NO_MODEL | PASS_FOR_CHECKPOINT |
| [EXP-0003](experiments/EXP-0003-g1-daily-extract-first-publication-polling/README.md) | G1 Daily Extract first-publication polling | ACCEPTED | Daily Extract polling and first-observation ledger mechanics passed for 17 June 2026 rows; G1 remains blocked pending provider first-publication evidence. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |
| [EXP-0004](experiments/EXP-0004-g1-daily-extract-bounded-polling-candidate-gating/README.md) | G1 Daily Extract bounded polling candidate gating | ACCEPTED | Bounded Daily Extract polling and watched-date candidate gating passed; G1 remains blocked pending actual provider first-publication evidence. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |
| [EXP-0005](experiments/EXP-0005-g1-daily-extract-active-first-publication-watch-2026-06-18/README.md) | G1 Daily Extract active first-publication watch 2026-06-18 | ACCEPTED | Active watch completed four iterations; 2026-06-18 remained absent, and stricter absent-before-present candidate gating passed focused tests. G1 remains blocked. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |
| [EXP-0006](experiments/EXP-0006-g1-daily-extract-active-first-publication-watch-2026-06-18-conti/README.md) | G1 Daily Extract active first-publication watch 2026-06-18 continuation | ACCEPTED | Continuation poll completed after adding tested bounded fetch retries; 2026-06-18 remained absent through 18:01:26Z. G1 remains blocked. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |
| [EXP-0007](experiments/EXP-0007-g1-daily-extract-active-first-publication-watch-2026-06-18-secon/README.md) | G1 Daily Extract active first-publication watch 2026-06-18 second continuation | ACCEPTED | Second continuation poll completed six iterations; 2026-06-18 remained absent through 18:09:27Z. G1 remains blocked. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |
| [EXP-0008](experiments/EXP-0008-g1-daily-extract-active-first-publication-watch-2026-06-18-third/README.md) | G1 Daily Extract active first-publication watch 2026-06-18 third continuation | ACCEPTED | Third continuation poll completed six iterations; 2026-06-18 remained absent through 18:20:46Z. G1 remains blocked. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |
| [EXP-0009](experiments/EXP-0009-g1-daily-extract-active-first-publication-watch-2026-06-18-fourt/README.md) | G1 Daily Extract active first-publication watch 2026-06-18 fourth continuation | ACCEPTED | Fourth continuation poll completed six iterations; 2026-06-18 remained absent through 18:30:12Z. G1 remains blocked. | None | PASS_TARGET_ONLY_NO_MODEL | PASS |

Regenerate with:

```bash
python -m hkg_tmax experiments index
```
