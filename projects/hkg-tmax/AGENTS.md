# HKG Tmax Elite Research — Operating Constitution

## Mission

Build, audit, and continuously improve the most accurate **point-in-time forecast distribution** for the official Hong Kong Observatory daily maximum temperature used by the applicable Polymarket Hong Kong highest-temperature contract.

The system is not permitted to optimize for impressive in-sample fit, retrospective storytelling, or raw temperature MAE alone. It must optimize for **reliable, leakage-free, reproducible out-of-sample forecasting and executable decision quality**.

The target level of effort is exceptionally high. No potentially useful lawful data source, mechanism, transformation, regime, station relationship, forecast vintage, or falsifiable hypothesis is dismissed without explicit evaluation. At the same time, complexity earns its place only through robust out-of-sample evidence.

## Non-negotiable principles

1. **Target truth before prediction.**
   - The canonical settlement target is the first-published value in the contract-named Hong Kong Observatory Daily Extract field, subject to the exact rules of each market.
   - Never assume a machine-readable climate series is identical to settlement truth until parity has been measured and documented.
   - Parse and archive the market rules for every event. Fail closed if the source, field, precision, date, range definitions, or revision language changes.

2. **Point-in-time truth only.**
   - Every feature must have `valid_at`, `issued_at`, `published_at`, `available_at`, and `retrieved_at` semantics where applicable.
   - A row may enter an as-of forecast only if the information was genuinely available by that forecast cutoff.
   - Revised observations, finalized climate files, reanalysis, best tracks, and retrospectively corrected model data must never masquerade as real-time inputs.

3. **Evidence over enthusiasm.**
   - Every claimed improvement requires a predeclared hypothesis, baseline, split, metric, uncertainty interval, and failure criteria.
   - Report negative and null results with the same care as positive results.
   - Never promote a feature or model because it “looks good” on a chart.

4. **Classical understanding before machine learning.**
   - Start with climatology, persistence, official forecasts, deterministic and ensemble numerical guidance, bias correction, regime analysis, physical diagnostics, station analogues, and transparent combinations.
   - Machine learning is allowed only after data provenance, baselines, leakage controls, and classical relationships are mature.
   - ML must beat strong classical baselines on locked, point-in-time out-of-sample periods and under sensitivity tests.

5. **Forecast a distribution.**
   - Produce a calibrated distribution for the continuous final Tmax and exact probabilities for every contract bucket.
   - Never rely solely on a point estimate.
   - Evaluate both meteorological accuracy and bucket/log-loss quality.

6. **Reproducibility is a deliverable.**
   - Every experiment lives in its own immutable experiment directory.
   - It must be rerunnable from a clean environment using only its manifest, code commit, configuration, and archived/raw source references.
   - Never overwrite an experiment result. Create a new experiment ID.

7. **No silent data changes.**
   - Store raw payloads before parsing.
   - Hash raw payloads, schemas, configs, rules, and code state.
   - Any parser/schema change requires a migration note and new derived-data version.

8. **No guaranteed-profit claims.**
   - Forecast edge is not trading profit. Include fees, spread, slippage, fill probability, latency, inventory, and model uncertainty.
   - Stop trading if live calibration, source parity, or execution assumptions deteriorate.

9. **Lawful and ethical operation only.**
   - Use public or properly licensed data and APIs.
   - Never interfere with, approach, manipulate, or attempt to influence a weather sensor, observation process, publication process, or market.
   - Respect rate limits, terms, attribution, and redistribution restrictions.

## Local credentials

- GribStream local credentials live in `secrets/local/gribstream.env` as `GRIBSTREAM_API_KEY`.
- `secrets/local/*` is ignored by Git except `secrets/local/README.md`.
- Never print, log, copy, commit, or include the token value in task artifacts, experiment outputs, documentation, screenshots, or shell transcripts.
- T03 may complete public catalog/licence work without the key, but authenticated GribStream probes must load this variable or be marked blocked. T06 and later GribStream acquisition tasks must not run live API requests without `GRIBSTREAM_API_KEY`.
- Credential details for future agents are documented in `documentation/GRIBSTREAM_CREDENTIAL_LOCATION.md`.

## GribStream runtime discipline

- Use one thread for GribStream work unless the user explicitly authorizes a higher provider-approved throughput.
- Default authenticated request spacing is 12 seconds with bounded retries.
- Honor `Retry-After`; stop the active probe loop after a real `429` instead of continuing to hammer the API.
- Never retry `401` or deterministic `400` request errors without fixing the credential/request first.
- For T03-T05 status, run `.\.venv\Scripts\python.exe scripts\check_t03_t05_status.py`.
- If T03 probes already completed, use `--reuse-existing-t03-artifacts` for recovery/finalization so authenticated probes are not repeated.
- T03-T05 GribStream logging, retry, background-run, and status-check details live in `documentation/T03_T05_GRIBSTREAM_AND_NWP_FOUNDATION_RUNBOOK.md`.
- For T06 GribStream `/runs` client status, run `.\.venv\Scripts\python.exe scripts\check_t06_gribstream_status.py`; do not rerun the live API unless the status/ledger says the raw object is absent or incomplete.
- T06 code, raw landing, resume ledger, and operator details live in `documentation/T06_GRIBSTREAM_RESUMABLE_RUNS_CLIENT_RUNBOOK.md`.
- The old split T07-T12 GribStream fetch tasks are superseded by `tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/not-completed/T07_T12_tactical_h24n_gribstream_backfill/`.
- The old broad `scripts/run_t07_t13_gribstream_backfill.py` runner is retired and blocks by default. Do not restart it for acquisition work.
- Tactical GribStream acquisition details, clean-slate reset commands, smoke commands, schema locations, and provider-scope notes live in `documentation/T07_T12_CONSOLIDATED_TACTICAL_GRIBSTREAM_BACKFILL_RUNBOOK.md`.
- For tactical GribStream storage cleanup, use `.\.venv\Scripts\python.exe scripts\reset_tactical_gribstream_store.py --apply-schema --purge-raw --execute`.
- For a tiny exact-cycle smoke, use `.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_h24n_smoke.py --max-data-calls 3`.
- For the measured first-week tactical pull, use `.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_first_week.py --apply-schema --api-min-interval-seconds 12`; this writes raw objects, `nwp_tactical.acquisition_chunk`, `nwp_tactical.raw_response_object`, `nwp_tactical.forecast_wide`, and `experiments/0214_tactical_h24n_gribstream_backfill/first_week_pull/first_week_summary.json`.
- For the 10-week batch sanity check, use `.\.venv\Scripts\python.exe scripts\run_tactical_gribstream_batch_smoke.py --apply-schema --days 70 --output-name batch_smoke_10w --api-min-interval-seconds 12 --api-max-attempts 2 --pause-on-429-seconds 300`; this uses model-specific batch sizes and writes `experiments/0214_tactical_h24n_gribstream_backfill/batch_smoke_10w/batch_summary.json`.
- Latest full tactical backfill result is documented in `documentation/T07_T12_FULL_TACTICAL_BACKFILL_20260625_RESULT.md`; do not declare T07-T12 complete from row counts alone because the run ended API-clean but with data-quality flags: recent `ifsenfo` chunks missing member `0`, empty `fourcastnetgfs` tail, and empty `nbmoc` probe.
- Latest full tactical deep sanity audit is documented in `documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md`; rerun it with `.\.venv\Scripts\python.exe scripts\audit_tactical_gribstream_deep_sanity.py --skip-file-hash` before modeling off `nwp_tactical.forecast_wide`.
- The human-facing inventory of all fetched GribStream data, models, date ranges, variable coverage, quality flags, and modeling gates lives in `documentation/strategy_implementation_documentation/GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md`; update it whenever GribStream data coverage, schemas, or usable-source decisions change.
- The implementation ledger for leakage-safe DB retrieval from `nwp_tactical.forecast_wide` lives in `documentation/strategy_implementation_documentation/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md`; read and update it before changing any feature query, model dataset builder, source allow/block list, H24N cutoff rule, or publication-buffer policy.
- For GribStream acquisition, prefer the project credential file `secrets/local/gribstream.env` over any stale legacy `GRIBSTREAM_API_TOKEN` process environment variable.
- Before using `nwp_tactical.forecast_wide` for features, read `documentation/GRIBSTREAM_TMAX_LEAKAGE_SAFETY.md`; raw rows are not automatically leakage-safe, and feature extraction must enforce `run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT`.
- Do not treat `aigfspres`, `aigefssfc`, or `nbmoc` as daily Tmax-producing sources unless a later provider/selector probe proves usable non-null 2m or Tmax coverage.

## Canonical research question

At a precisely defined cutoff \(c\), what is:

\[
P(T^{official}_{max,T} \in B_j \mid \mathcal{I}_{\le c})
\]

for every Polymarket bucket \(B_j\), where:

- \(T\) is the Hong Kong local calendar date;
- \(T^{official}_{max,T}\) is the first-published contract-authoritative daily maximum;
- \(\mathcal{I}_{\le c}\) contains only data demonstrably available by cutoff \(c\);
- the mapping to buckets is taken from the exact event rules, not inferred from labels.

## Primary forecast horizon

The user’s strategic preference is to act before the crowd, roughly 24 hours before the likely daytime peak. The repository therefore defines candidate horizons rather than hard-coding a misleading “T-24”:

- `H39`: 00:00 HKT on T-1 — 39 hours before a nominal 15:00 HKT peak.
- `H27`: 12:00 HKT on T-1 — 27 hours before a nominal 15:00 HKT peak.
- `H24N`: 15:00 HKT on T-1 — 24 hours before a nominal 15:00 HKT peak.
- `H15`: 00:00 HKT on T — 15 hours before a nominal 15:00 HKT peak.
- `LIVE_*`: intraday horizons used only after the pre-event system is mature.

Goal G2 must determine the primary horizon using actual event-open times, liquidity formation, data-vintage availability, and forecast skill. Until then, all four pre-event horizons must be evaluated.

## Mandatory workflow for every task

1. Read:
   - `CODEX_START_HERE.md`
   - `FIRST_GOALS.md`
   - `MILESTONES.md`
   - `EXPERIMENT_INDEX.md`
   - relevant documentation and prior experiment conclusions.

2. Determine whether the task changes target semantics, data availability, evaluation, or production behavior. If yes, require an explicit audit.

3. Create or reserve an experiment ID before analysis:
   ```bash
   python -m hkg_tmax experiments create --title "..."
   ```

4. Fill the experiment’s hypothesis and protocol **before** inspecting holdout outcomes.

5. Run data/as-of validation:
   ```bash
   make validate
   ```

6. Run the experiment deterministically. Save:
   - exact query URLs or API parameters;
   - source vintages;
   - raw hashes;
   - feature definitions;
   - code commit and dirty-state patch;
   - environment lock;
   - seeds;
   - metrics, uncertainty, plots, diagnostics, logs.

7. Have the leakage auditor and reproducibility reviewer independently review accepted candidates.

8. Update:
   - the experiment conclusion;
   - `EXPERIMENT_INDEX.md`;
   - `MILESTONES.md` only if acceptance gates pass;
   - `docs/PROJECT_STRUCTURE_AND_CODE_MAP.md` when code, tests, scripts, migrations, tasks, or experiment-output mappings are added, moved, renamed, or retired;
   - `CHANGELOG.md` for material system changes.

9. Never delete failed work. Mark it `REJECTED`, `INCONCLUSIVE`, or `BLOCKED` and record why.

## Mandatory code-map maintenance

`docs/PROJECT_STRUCTURE_AND_CODE_MAP.md` is the canonical map for where implementation files live, where tests live, and which experiment folders contain the evidence for completed implementation tasks.

Every agent that changes source code, tests, scripts, database migrations, A-to-Z tasks, or experiment-output locations must update that file in the same work unit. Do not leave path changes, new task completions, or new experiment/code relationships undocumented.

## Experiment acceptance gates

An experiment may be called a milestone only if all applicable gates pass:

- target parity: PASS;
- source provenance: PASS;
- as-of/leakage audit: PASS;
- reproducibility from clean checkout: PASS;
- locked out-of-sample improvement: PASS;
- uncertainty interval and practical effect: reported;
- multiple-regime robustness: PASS;
- ablation confirms the claimed contribution;
- result survives reasonable alternate cutoffs, seasons, years, and missing-data assumptions;
- no material degradation in calibration or tail behavior;
- complexity and operational cost justified;
- market claim, if any, includes executable prices and costs.

## Required experiment directory

Every `experiments/EXP-####-slug/` must contain:

- `README.md` — plain-language overview and status;
- `HYPOTHESIS.md` — mechanism, prediction, falsification;
- `PROTOCOL.md` — predeclared sample, split, metrics, comparisons;
- `ASOF_CONTRACT.md` — exact timestamp/availability rules;
- `DATA_MANIFEST.yaml` — source IDs, versions, hashes, date ranges;
- `RUN_CONFIG.yaml` — all tunable values and seeds;
- `RESULTS.md` — complete results, not selected highlights;
- `CONCLUSION.md` — accept/reject/inconclusive and next implications;
- `REPRODUCE.md` — one-command rerun instructions;
- `STATUS.yaml` — state, reviewers, gate results;
- `results/metrics.json`;
- `results/predictions.parquet` or a documented pointer;
- `artifacts/`;
- `logs/`.

## Research conduct

- Search aggressively for mechanisms, not p-values.
- Analyze residuals by month, monsoon phase, wind sector, cloud/rain regime, tropical-cyclone proximity, temperature level, model spread, forecast horizon, and station-network pattern.
- Use causal and physical reasoning to propose transformations, but treat causality claims conservatively.
- Perform negative controls and placebo tests.
- Inspect whether a “new edge” is merely a proxy for season, trend, issue time, missingness, or data revision.
- Prefer simple robust gains over fragile complex gains.
- Keep a model graveyard; rejected ideas should prevent repeated wasted work.
- Never open the locked test period repeatedly to tune decisions.

## Definition of done

The project is not “done” when a model has a low MAE. It reaches production eligibility only after:

- exact settlement parity is established;
- point-in-time data archive is operational;
- primary horizon is fixed;
- strong baselines are beaten on a locked test;
- probabilistic forecasts are calibrated;
- live shadow performance confirms backtests;
- market replay is executable and cost-aware;
- source/data/model monitoring and kill switches are active;
- all results are reproducible and independently audited.
