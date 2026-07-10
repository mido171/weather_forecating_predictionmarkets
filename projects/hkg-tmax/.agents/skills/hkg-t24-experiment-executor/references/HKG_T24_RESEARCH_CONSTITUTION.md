# HKG T+24 Immutable Research Constitution

## 1. Mission

The scientific objective is to create the most accurate, robust, reproducible, and operationally legitimate point forecast of the Hong Kong Observatory target station's daily maximum temperature for target date **T**, using only information genuinely available by the repository's canonical **T−24 operational decision cutoff**. The aspirational external target is **0.45 °C MAE**. That number is a research objective, not permission to manipulate evaluation, search indefinitely on a fixed holdout, or claim a result that has not survived a frozen protocol.

The system must be aggressive in lawful information discovery and conservative in scientific claims. Competitive intensity never overrides timestamp truth, target integrity, reproducibility, or honest reporting.

## 2. Exact forecast object

- Target station: Hong Kong Observatory, as identified by the repository's canonical target mapping.
- Target variable: daily maximum temperature in degrees Celsius for target local date T.
- Forecast horizon: T−24, using the exact cutoff timestamp defined in the repository's canonical data contract.
- Timezone: Asia/Hong_Kong unless the canonical repository contract explicitly states otherwise.
- Unit: degrees Celsius.
- Every experiment must record the target definition, target timezone, daily boundary, cutoff timestamp function, and source of that definition.

The phrase “day T−1” is not enough by itself. The exact operational timestamp must be resolved from repository evidence. If the repository does not contain a precise canonical cutoff, a deployable experiment must be rejected with reason `CUTOFF_NOT_PRECISELY_DEFINED`. No skill may invent a clock time merely to continue.

## 3. The three-time test

For every candidate value, distinguish:

1. **Meteorological valid time** — when the weather value describes the atmosphere.
2. **Archive presence time** — when the value is found in a retrospective archive or downloaded file.
3. **Operational available-at time** — when the live forecasting system could demonstrably have possessed the value before the cutoff.

Only the third establishes deployability. A value with an old valid time is not safe merely because it later appeared in an archive. Retrieval time is not issue time. Final publication time is not necessarily first publication time. A daily summary may contain post-cutoff observations even if its date label is T−1. A model field may be valid before T while its cycle was issued after the decision cutoff. Every deployable feature must pass all relevant timestamp checks.

## 4. Point-in-time admissibility classes

Every source and feature must receive exactly one current status:

- `DEPLOYABLE_PROVEN`: availability before cutoff is demonstrated row by row or by an audited deterministic latency contract covering all rows.
- `DEPLOYABLE_LAGGED_ONLY`: the raw source is not safe contemporaneously, but a specified lag makes availability provable.
- `DIAGNOSTIC_ONLY`: physically or statistically useful for mechanism discovery, but not allowed in production-style scoring.
- `PROSPECTIVE_ONLY`: usable for future live collection after retrieval instrumentation, but insufficient retrospective history or vintage proof.
- `BLOCKED`: necessary evidence, parsing, metadata, legality, or timestamps are missing.
- `REJECTED`: known to violate the contract for the proposed use.

Unknown is never silently converted to deployable. Ambiguity resolves toward blocking, not convenience.

## 5. Prohibited forward-looking information

Unless explicit, documented, row-level proof says the value is available before cutoff, prohibit:

- target-date observations from T;
- target-day daily climate values;
- target-day station summaries that include observations after cutoff;
- rolling windows accidentally containing T;
- same-source forecast revisions issued after cutoff;
- final HKO daily extracts used as predictors without first-publication proof;
- finalized upper-air, marine, climate, best-track, radar, satellite, lightning, or reanalysis values whose vintage is retrospective;
- climatologies, scalers, encoders, PCA transforms, graph modes, imputation values, feature selection, thresholds, or model parameters fitted using future rows;
- residual memory updated before the current target outcome is known;
- confirmation-period outcomes used for feature design, model selection, or hyperparameter choice;
- archive retrieval order treated as historical publication order;
- any manual correction derived after seeing the target.

## 6. Confirmation seal

Rows dated 2024-01-01 or later are sealed confirmation evidence unless the project owner explicitly opens them. The Research Director and Experiment Executor must not read target outcomes, scores, residuals, or target-derived summaries from this period during development. If files mix development and confirmation rows, code must filter before any target-dependent statistic, inspection, plotting, feature selection, or model fitting.

A one-time confirmation run may occur only after:

1. the candidate and all rules are frozen;
2. the feature whitelist is frozen;
3. the model code and hyperparameters are frozen;
4. the cutoff contract passes audit;
5. the project owner explicitly authorizes opening confirmation;
6. the confirmation run is logged as one immutable event.

Repeated confirmation access converts confirmation into development and invalidates the claim.

## 7. Causal feature construction

For row T, every target-derived feature uses target outcomes from dates strictly earlier than T and known by cutoff. Every source-derived feature uses rows with `available_at <= cutoff(T)`. Rolling statistics must use explicit shifts before rolling. Exponentially weighted states must update only after an outcome becomes available. Quantile bins, normals, anomalies, climatologies, encoders, and station baselines must be fitted from prior history within each fold.

Static geospatial features may be globally deterministic if they contain no outcome information and their provenance is fixed. All other transformations must state whether they are static, fold-fitted, expanding, or rolling.

## 8. Sequential experiment integrity

Experiments are executed one at a time. The Research Director defines one experiment specification before the Experiment Executor sees results. The Executor does not redesign the hypothesis after observing outcomes. Any post-result modification is a new experiment with a new folder and ID.

An experiment may include a predeclared finite parameter grid. The grid, selection rule, and inner validation must be declared before execution. Searching additional parameters after looking at outer-fold results is a new experiment. Silent retries with changed thresholds are forbidden.

## 9. Canonical frames and identical-row comparison

Every score must name its evaluation frame. Candidate and baseline must use:

- identical target dates;
- identical target values;
- identical source eligibility rules;
- identical missing-row policy;
- identical forecast horizon;
- identical scoring functions.

A candidate may not claim improvement by scoring an easier subset. If feature availability reduces coverage, report both the identical-row candidate-versus-baseline metrics and the coverage loss relative to the parent frame. Scores from different frames must never be presented as direct wins without a harmonized replay.

## 10. Required baselines

Every promotion-oriented experiment names the relevant baseline. Depending on the question, required comparisons can include causal climatology, target-memory baseline, raw official forecast, global bias correction, source-specific bias correction, online residual-memory baseline, current canonical champion, and the simplest model using the same new feature family.

The current champion is not a remembered number. It is an artifact in the champion ledger tied to a frame, code version, row set, and leakage status.

## 11. Validation

Use expanding or rolling walk-forward evaluation. Training for each prediction may include only earlier target dates. Nested tuning occurs inside each outer training history. Preprocessing is fold-local. At least four years of out-of-fold evaluation are preferred where source coverage permits. Long-history experiments use multiple temporal eras, not a single recent split.

Each experiment reports MAE, RMSE, bias, median absolute error, P90 and P95 absolute error, severe-error counts, signed hot underforecast and cold overforecast diagnostics, year/season/month/source/source-era/late-window slices, coverage, missingness, and temporal stability evidence.

## 12. Minimum support and shrinkage

No sparse slice is promoted merely because its mean looks attractive. The specification must declare minimum global, fold, season, source, and trigger counts. Small states require shrinkage toward a broader parent. Hard corrections on tiny cells are prohibited. Thresholds must be predeclared.

## 13. High-error tail and no-harm gates

Average MAE is necessary but insufficient. A candidate aimed at ordinary days may not create catastrophic misses. A specialist must improve its declared target slice and pass global no-harm gates. At minimum examine P90/P95 absolute error, >2 °C and >3 °C counts, worst signed tails, and seasonal damage.

## 14. Diagnostic versus realized lift

A relationship can be physically real yet add no incremental value beyond the official forecast. Every conclusion must distinguish:

- target-level explanatory signal;
- official-residual signal;
- uncertainty signal;
- routing/trust signal;
- high-error detection signal;
- diagnostic physical interpretation;
- realized point-MAE improvement.

Do not promote a feature for point correction solely because it predicts raw Tmax.

## 15. Negative results are permanent evidence

Failed, null, unstable, blocked, and duplicated lanes must be preserved. The Research Director reads and uses them. A result is not discarded because it failed to improve MAE. It may reveal redundancy, wrong functional form, insufficient coverage, a blocked mechanism, or a useful uncertainty signal. A failed lane may not be repeatedly renamed and rerun without a precise new mechanism or methodological improvement.

## 16. Experiment folder requirement

Every assigned experiment receives a folder under the canonical `experiments` directory, including rejected experiments. At minimum it contains:

- `README.md` — hypothesis, rationale, design, inputs, and high-level explanation;
- `RESULTS.md` — structured results and baseline comparison;
- `CONCLUSION.md` — post-analysis, interpretation, promotion decision, and next implications;
- `experiment_spec.json`;
- `leakage_audit.md`;
- `data_manifest.csv`;
- `feature_definitions.csv`;
- `scoreboard.csv` when scored;
- `slice_metrics.csv` when scored;
- `yearly_metrics.csv` when scored;
- `summary.json`;
- `run_manifest.json`;
- reproducible source code under `src/`;
- predictions or row identifiers sufficient for audit when scored.

The folder validator must pass before an experiment is complete.

## 17. Rejection is a valid scientific outcome

Reject before scoring if the cutoff is ambiguous, a source lacks operational availability proof, the target is contaminated, the frame cannot be reproduced, or the specification is internally inconsistent. Create the full rejection folder. State what evidence would unlock the experiment and whether a diagnostic-only variant is permitted.

## 18. The 0.45 °C loop

After each complete experiment, the Research Director validates the folder, updates the evidence and negative-results registries, recomputes the champion ledger only from comparable leakage-passed results, and checks the canonical development MAE against 0.45 °C. If greater than 0.45, it performs a new full-corpus synthesis, defines one next experiment, explicitly invokes the Executor, and repeats sequentially.

The loop never changes the frame or loosens leakage rules to cross the target. It never claims that reaching 0.45 is guaranteed. It persists state so work can resume across sessions. It stops only on verified development-gate success, explicit owner stop, a hard runtime/tool failure, or scientifically documented exhaustion of admissible lanes. In the last case it produces an exhaustion report; it does not fabricate a win.

## 19. Development success versus confirmed success

`DEVELOPMENT_GATE_REACHED` requires a frozen candidate with MAE <= 0.45 on the predeclared canonical development protocol plus all stability, coverage, leakage, and no-harm gates. It does not imply live profitability or confirmation success.

`CONFIRMED_0P45_OR_BETTER` requires explicit owner authorization and a one-time untouched confirmation result meeting the predeclared criterion. No skill can grant itself permission to open confirmation.

## 20. Scientific honesty

The skills must be ambitious, creative, and persistent. They must also state uncertainty. They may infer mechanisms but label inference. They may use external scientific knowledge to form hypotheses but must distinguish external rationale from evidence in the repository. They may never hide a failed fold, a damaged season, a coverage reduction, an unavailable timestamp, or a contradictory result.
