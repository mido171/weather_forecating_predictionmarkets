# Shared HKG T+24 Research Constitution

## 1. Forecast definition

- Target: daily maximum temperature at Hong Kong Observatory headquarters for local settlement date `T`.
- Decision cutoff: `T-1 15:00 Asia/Hong_Kong`, unless a repository-wide owner-approved contract supersedes it.
- The system must be executable at that instant using only information already available to the operator.
- “Meteorologically valid before T,” “present in a retrospective archive,” and “proved available before the decision cutoff” are different. Only the third supports deployable scoring.

## 2. Availability gate

Every predictor row must carry or inherit:

- source identifier;
- event/valid time;
- issue time where applicable;
- available-at or retrieval time;
- timezone;
- target date;
- cutoff time;
- eligibility tier;
- proof or conservative-lag assumption.

A predictor is rejected from the strict track when `available_at > cutoff`, when no availability proof exists and no owner-approved conservative lag applies, or when alignment is ambiguous. Rejection still creates a complete experiment folder.

## 3. Absolute prohibitions

Never use current-T target values, current-T residual/error fields, target-day observations, post-cutoff forecast revisions, retrospective best-track values as live predictors, full-history preprocessing, future-defined bins, future station normals, future-selected analogues, or 2024+ outcomes. The denylist in `evidence/HARD_FEATURE_DENYLIST.csv` is binding.

## 4. Mature labels and residual states

Online target/residual memory may update only after the label is proven available. Default maturity is T-7 because the existing strict experiment uses that conservative rule. A faster maturity is allowed only in a separately named branch after first-publication proof is stored and audited. For row T, state is computed first; prediction is emitted; only later, when the label becomes eligible, may state update.

## 5. Walk-forward fitting

All imputers, scalers, climatologies, PCA/graph factors, text vectorizers, feature selectors, model parameters, calibrators, routers, blend weights, quantile bins, analog distance scales and correction caps selected from data must be fitted using earlier eligible rows only. Outer-fold outcomes are for scoring, never tuning. Hyperparameter grids are preregistered.

## 6. Identical-row scoring

Candidate, official baseline, strict-core benchmark and parent expert must be compared on the identical ordered row set. Write a SHA-256 row-universe hash. Missing optional features must not silently change the scored frame; use prior-fit imputation, missingness/age features or parent fallback.

## 7. Separate evidence tracks

Every scoreboard row must declare one of:

- `STRICT_DEPLOYABLE`;
- `RESEARCH_PROXY_QC_DEPENDENT`;
- `DIAGNOSTIC_TEACHER_ONLY`;
- `PROSPECTIVE_SHORT_HISTORY`;
- `BLOCKED_MISSING_DATA`;
- `INVALID_LEAKAGE`.

Scores from different tracks or frames cannot be ranked together.

## 8. Mandatory metrics

Report MAE, RMSE, bias, median absolute error, P90/P95/P99 absolute error, maximum absolute error, >2 C and >3 C rates, hot-underforecast and cold-overforecast rates, year/month/season/source/era/fold metrics, late-window results, correction distribution and activation rate. Distributional lanes also report CRPS, pinball loss, PIT/rank diagnostics, interval coverage and threshold Brier scores.

## 9. Promotion gates

A candidate may be promoted only when it:

1. passes feature availability and leakage validation;
2. beats the correct parent and official baseline on identical rows;
3. shows positive lift in multiple nonempty chronological folds;
4. does not materially worsen MAM, the late RSS window or >3 C tails;
5. beats simpler baselines and survives feature-family ablation;
6. has self-contained reproducible code and immutable manifests;
7. leaves 2024+ sealed.

A lower global MAE alone is insufficient.

## 10. Confirmation rule

No task in this bundle opens target dates from 2024 onward. If development MAE reaches or beats 0.45 C, freeze the exact candidate, code hash, feature list, model state and decision protocol. Mark `DEVELOPMENT_GATE_REACHED`; do not claim confirmed performance and do not automatically open confirmation.
