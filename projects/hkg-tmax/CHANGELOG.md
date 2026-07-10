# Changelog

## Script runtime-path consolidation - 2026-07-10

- Routed the 36-script migration census through `ProjectPaths`, replacing repository-local
  `data`, `reports`, `artifacts`, `models`, `predictions`, and `tmp` defaults with the
  configured external data/run roots.
- Closed equivalent slash-form and multiline paths found by the structural audit, including
  tactical GribStream raw storage and two additional research readers missed by the original
  text scan.
- Classified the census as 26 retained reproduction scripts, 9 active operator utilities,
  and 1 active research entry point in the generated script registry. Retained scripts stay
  flat only because tests and historical manifests import them by filename; they are not
  current research authority.
- Added an AST-based regression test that rejects any top-level script anchored at a mutable
  `REPO_ROOT` child and verifies the classified census continues to use the path layer.

## Public weather optimized DB pipeline validation - 2026-07-09

- Added opt-in optimized execution to `scripts/backfill_public_weather_to_postgres.py` with safe `gap=0` GRIB range coalescing, bounded model fetch workers, model normalization process workers, Himawari workers, serialized DB writes, per-phase metrics, CPU/staging telemetry hooks, and short transient staging roots.
- Added optimized flag passthrough to `scripts/run_public_weather_backfill_day_shards.py` for multi-day robustness runs.
- Added experiment `experiments/hkg_tmax/0012_public_weather_backfill_optimized_pipeline_validation_20260709` and focused tests for range coalescing, optimized CLI defaults, and bounded raw deletion.
- Added an experiment-local `documentation/` handoff covering implementation, failure recovery, leakage contracts, validation evidence, live Postgres relation measurements, and the `121.4 GB` retained-capacity estimate for the 2017-2026 three-source backfill.

## Public weather speed optimization experiment - 2026-07-09

- Added `scripts/benchmark_public_weather_speed_optimization.py` and experiment `experiments/hkg_tmax/0011_public_weather_speed_optimization_20260709` to benchmark bounded S3 byte-range GFS/GEFS fetching, Himawari fetch/decode worker counts, local `wgrib2` availability, CPU telemetry, staging bytes, and raw cleanup.

## Public GFS/GEFS/Himawari/radar lean DB backfill draft — 2026-07-08

- Added `scripts/backfill_public_weather_to_postgres.py` for day-streamed public weather acquisition into `weather_backfill` Postgres tables with source issues, station features, area features, ingest runs/events, leakage-safe availability clocks, and immediate raw deletion.
- Added NOAA S3 `.idx` byte-range fallback for aged-out GFS/GEFS dates so selected GRIB messages can be extracted without downloading full raw model files.
- Added focused tests for source parsing, expected inventory counts, inclusive date windows, and GRIB index range selection.

## HKG Polymarket demo cutoff-profile trading gates — 2026-07-07

- Added explicit validated as-of profiles for 12:00, 13:00, 14:00, 15:00, 16:00, 17:00, and 17:59 Stockholm entries, including the matching HKT cutoffs and observed MAE/RPS/NLL quality metadata.
- Updated demo probability snapshots so validated profiles use the same cutoff profile for forecast-anchor selection and B4 training-row filtering; unsupported or live profiles fail closed as display-only/non-tradeable.
- Hardened live current-day forecast fetching so validated cutoff profiles only use the HKO local forecast (`flw`) family, parse "tonight and tomorrow" temperature ranges, reject future cutoff profiles, and fail closed instead of falling back to 9-day (`fnd`) rows.
- Hardened demo snapshot caching so stale unavailable snapshots and old wrong-source live-cutoff snapshots are refreshed or rejected instead of being reused as successful market edges.
- Fixed the React backtester profile/date state flow so profile switches clear stale market data and async market responses cannot overwrite the panel after the selected date/profile has changed.
- Added `/api/profiles`, profile-filtered market snapshot lookup, persisted trade entry metadata for profile/forecast/model context, and server-side trade gates requiring executable CLOB ask price, edge `>= 15.0 pp`, and model win probability `>= 70.0%`.
- Updated the React backtester with a profile selector, explicit apples-to-apples status, active profile and cutoff metadata in the engine/ticket/ledger, 5% stake shortcut, executable-price indicators, and disabled trade states with concrete blocker reasons.
- Extended focused demo-trading tests for B4 profile filtering, API profile contract, trade metadata, CLOB entry gates, fallback snapshot rendering, and settlement/account views.

## HKG Tmax official residual-memory point forecast 0003 — 2026-07-06

- Added `code/src/hkg_tmax/data/official_residual_memory_features.py` to build same-cutoff official forecast residual-memory features with a strict lag-2 floor, min-count rolling windows, source-date lineage, and publication-safety audits.
- Added `code/src/hkg_tmax/features/residual_memory_policy.py` and updated `code/src/hkg_tmax/features/pruned_feature_policy.py` so residual-memory predictors are guarded from target/evaluation leakage and reported as `official_residual_memory`.
- Added `code/src/hkg_tmax/evaluation/official_residual_memory_runner.py` with the governed D0-D5 benchmark: A7 reproduction, D1 shrinkage, D2/D3 LGBM residual challengers, D4 constrained stack, and D5 conservative A7-plus-memory blend.
- Added `configs/hkg_tmax/residual_ml_official_memory.yaml` and `scripts/run_hkg_tmax_residual_ml_official_memory.py` for the 0003 experiment.
- Added `code/tests/test_hkg_tmax_official_residual_memory.py`; verified the residual-memory contract with `7 passed`.
- Generated `experiments/hkg_tmax/0003_official_residual_memory_20260706/results/` and compatibility copy `experiments/hkg_tmax_residual_ml_official_memory/results/`.
- Result: `D5_conservative_A7_plus_memory_blend` did not promote. Primary `T-1 23:59 HKT` MAE was `0.898594 C` versus A7 `0.898665 C` and raw official `0.930858 C`, but D5 failed the predeclared development and presealed gain gates. Leakage, row identity, residual-memory publication safety, and slice no-harm gates passed.

## HKG Tmax probability distribution methods V2 — 2026-07-06

- Added `code/src/hkg_tmax_probability/distribution_methods_v2.py` with true location-scale EMOS challengers (`E1_normal_emos`, `E2_student_t_emos`, `E3_two_piece_normal_emos`), `G1_gamlss_tree_location_scale`, `Q1_quantile_cdf_gb`, `Q2_threshold_cdf_gb`, `T1_time_decay_b4`, and `H1_b4_challenger_linear_pool`.
- Added `code/src/hkg_tmax_probability/leaderboard_v2.py` to enforce the predeclared B4 promotion gates: 1.5% fold1-4 RPS gain, 1.0% presealed RPS gain, NLL no worse than B4 by 0.005, Brier no worse than B4 by 0.002, and passing leakage/row-identity gates.
- Added `configs/hkg_tmax/probability_distribution_methods_v2.yaml` and `scripts/run_hkg_tmax_probability_distribution_methods_v2.py` to run the governed V2 benchmark while preserving the V1 B4 probability implementation unchanged.
- Generated `experiments/hkg_tmax_probability_distribution_methods_v2/results/` with the required scoreboards, predictions, continuous distribution parameters, bootstrap deltas, diagnostics, method-selection log, leakage audit, row-identity gate, model card, supreme-method summary, and reproducibility manifest.
- Added `code/tests/test_hkg_tmax_probability_distribution_methods_v2.py`; verified the V2 contract with `9 passed`.
- V2 result: `B4_hierarchical_residual_pmf` remains supreme with normalized RPS `0.0415235723`; raw lowest RPS was `B5_kernel_analog_pmf` at `0.0412874816`, but it failed fold/presealed gain and NLL gates. Leakage audit, row-identity gate, and live no-trading audit all passed.

## HKG Polymarket demo backtester UI — 2026-07-06

- Added the local-only `hkg_tmax_demo_trading` backend with FastAPI endpoints for HKG daily market views, B4 edge snapshots, fictitious trade entry, idempotent settlement, account reset, open-trade marking, and account/PnL summaries.
- Added `migrations/postgres/20260706_0009_demo_trading_backtester.sql` with the `demo_trading` schema, account sessions, frozen market snapshots, trade metadata, settlement, and PnL columns.
- Added the React/Vite `apps/hkg-polymarket-backtester` frontend for the July 1-10, 2026 HKG daily ladder, YES/NO ticketing, manual historical price overrides, account KPIs, open trades, and performance panels.
- Added focused tests for bucket boundaries, EV/PnL math, manual price validation, API surface behavior, and migration contract checks.
- Added the `hkg-tmax-demo-trading` console script, defaulting the local Uvicorn app to port 6000 when launched through the server entrypoint.

## HKG Tmax probability bucket calibration V1 — 2026-07-05

- Added the weather-probability-only `hkg_tmax_probability` package for Decimal-safe HKG Tmax bucket rules, official forecast cutoff selection, modeling-table construction, label-publication audit, residual PMFs, MOS distributions, direct classifiers, CDF calibration, conservative stacking, scoring, diagnostics, leakage audit, live probability-only inference, and reporting.
- Added `configs/hkg_tmax/probability_bucket_v1.yaml` and `scripts/run_hkg_tmax_probability_bucket_v1.py` to benchmark bucket distributions from strict Info.gov local forecasts against canonical and sealed HKO Daily Extract Tmax labels.
- Generated `experiments/hkg_tmax_probability_buckets_v1/results/` with scoreboards, predictions, probability PMFs, diagnostics, stack weights, model-selection log, leakage/label audits, model card, live inference example, and reproducibility manifest.
- Added `code/tests/test_hkg_tmax_probability_bucket_v1.py`; verified the V1 contract with `10 passed` and reran the full benchmark against PostgreSQL `hkg_tmax_research`.
- Champion under the configured simplicity gate is `B4_hierarchical_residual_pmf`; `S1_conservative_simplex_stack` had slightly lower RPS but did not clear the configured improvement threshold, and `B5_kernel_analog_pmf` failed the NLL no-worse gate.

## HKG-T24-003 router, specialists, final formula, and distribution — 2026-06-27

- Added Jira003 router training for R0/R1 with OOF chronology checks, strict proxy/shadow refusal, static SLSQP weights, dynamic expected-error weights, promotion/demotion gates, DB persistence, and CLI smoke support.
- Added all six Jira003 specialists with fold-local prior scoring, neutral missing components, support/no-harm gates, bounded corrections, activation reports, and specialist persistence.
- Added final strict system replay with R1/R0/E0/E2 fallback, specialist total cap, official +/-1.20C clipping, component provenance, distribution calibration/fallback, exact 41 threshold probabilities, confidence states, and no-trade flags.
- Added Jira003 database tables, report outputs, focused tests, context documentation, contract coverage, and smoke verification for the five required CLI commands.
- Real DB smoke was not run in this implementation pass because no `HKG_TMAX_DATABASE_URL` or `HKG_TMAX_DB_DSN` was configured.

## Tactical H24N GribStream reset — 2026-06-25

- Consolidated active GribStream fetch work from T07-T12 into `T07_T12_tactical_h24n_gribstream_backfill`; moved the old split folders under `tasks/superseded/T07_T12_legacy_split_gribstream_fetch_tasks/`.
- Added `migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql` with `nwp_tactical` model plans, variable plans, 12-point HKO stencil, exact `timesList` chunk ledger, raw object manifest, wide forecast table, and validation issue table.
- Added `scripts/reset_tactical_gribstream_store.py` for clean-slate GribStream DB/raw cleanup and `scripts/run_tactical_gribstream_h24n_smoke.py` for the bounded exact-cycle smoke.
- Retired the broad `scripts/run_t07_t13_gribstream_backfill.py` runner by making it block by default.
- Added `documentation/T07_T12_CONSOLIDATED_TACTICAL_GRIBSTREAM_BACKFILL_RUNBOOK.md` and `code/tests/test_tactical_gribstream_h24n.py`.

## A-to-Z T07-T13 GribStream backfill wave — 2026-06-24

- Added `scripts/run_t07_t13_gribstream_backfill.py` for one-thread, resumable, fair-wave GribStream acquisition across T07-T12 with live shared-parameter selectors, member chunking, credit budgeting, raw NDJSON gzip landing, and PostgreSQL lineage ingest.
- Added `scripts/check_t07_t13_gribstream_status.py`, `documentation/T07_T13_GRIBSTREAM_BACKFILL_RUNBOOK.md`, and focused tests in `code/tests/test_t07_t13_gribstream_backfill.py`.
- Fixed credential precedence so the project credential file `secrets/local/gribstream.env` is preferred over stale legacy `GRIBSTREAM_API_TOKEN` values, and hardened the runner to stop on `401/403`.
- Started the full background backfill wave with `--mode full --credit-budget 85000 --api-min-interval-seconds 12`.
- Verified the latest CWA WRF smoke chunk: `cwawrf15` `temperature_2m` for `2026-06-23`, 7,920 raw rows, 7,920 DB point rows, zero rejected rows, HTTP 200.
- Recorded T13 as a blocker because HKO ARWF exact-vintage collection is outside GribStream and requires a separate HKO collector.

## A-to-Z foundation T06 completion — 2026-06-24

- Added the reusable `hkg_tmax.gribstream` package for secret-safe one-thread `/runs` acquisition, live selector resolution, request planning, NDJSON gzip raw landing, normalization, and PostgreSQL lineage inserts.
- Added `scripts/run_t06_gribstream_resumable_runs_client.py` and `scripts/check_t06_gribstream_status.py`.
- Added `config/acquisition_policy.yaml` as the canonical T06+ policy entrypoint tied to the existing gridded acquisition policy.
- Completed the T06 GFS smoke acquisition using `TMP` at `2 m above ground`: one `2026-06-23T00:00:00Z` run, valid range `2026-06-23T00:00:00Z` through `2026-06-25T00:00:00Z`, 132 locations, 6,468 raw NDJSON rows, 6,468 normalized DB point rows, and zero rejected rows.
- Verified no `429`, `400`, `401`, or `5xx` responses in the T06 API event log; two HTTP 200 events were recorded because the first successful response exposed a local Windows long-path write bug and the second saved the raw object before resume.
- Added T06 focused tests and documented the T06 runbook/evidence map.

## A-to-Z foundation T03-T05 completion — 2026-06-24

- Completed T03, T04, and T05 and moved their task folders into `tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/`.
- Recorded the GribStream probe outcome: 23 coverage rows, 52 API events, 1 HTTP 200 success for `gfs` `/runs`, 17 `ConnectTimeout` blockers, 5 exact-selector blockers, and no recorded `429`, `400`, `401`, or `5xx`.
- Added long-path-safe file copy/write handling and offline T03 artifact recovery via `--reuse-existing-t03-artifacts`.
- Verified PostgreSQL migrations/loads, isolated temp DB migration test, focused tests, and secret scan.

## A-to-Z foundation T02 census reconciliation — 2026-06-24

- Added T02 full current data census reconciliation artifacts under
  `experiments/0209_full_current_data_census_reconciliation`.
- Added registry compatibility views `catalog.source_registry` and
  `governance.attribute_contract` for downstream A-to-Z tasks.
- Added a T02 generator script and focused regression tests for the required
  registry aliases and generated census schemas.

## G1 target-station checkpoint — 2026-06-18

- Added HKO Daily Extract backing-payload parsing for `Absolute Daily Max (deg. C)`.
- Fixed CLMMAXT parsing for real HKO bilingual CSV headers and footer rows.
- Added fail-closed target adapter tests for missing source/field/value, ambiguous
  date, unsupported precision, station mismatch, and source failure.
- Generated a May 2026 latest-payload parity sample: 31/31 Daily Extract rows
  matched CLMMAXT HKO.
- Marked EXP-0002 `BLOCKED` pending first-publication Daily Extract evidence;
  no predictive modelling or Polymarket backtesting was run.
- Added source contracts for Daily Extract, CLMMAXT HKO, and HKO station metadata.
- Added EXP-0003 Daily Extract polling infrastructure and a first-observation
  ledger for June 2026; it is accepted as infrastructure, while G1 remains
  blocked pending provider first-publication evidence.
- Added EXP-0004 bounded Daily Extract polling with explicit metrics output and
  watched-date provider-first candidate gating; G1 remains blocked until actual
  provider first-publication evidence is observed.
- Added EXP-0005 active Daily Extract watch for `2026-06-18` and tightened
  provider-first candidate gating to require active absent-before-present raw
  snapshots before candidate status.
- Added EXP-0006 continuation polling for `2026-06-18`, per-iteration poll
  snapshot metrics, and explicit bounded fetch retries for transient provider
  disconnects.
- Added EXP-0007 second continuation polling for `2026-06-18`; the Daily
  Extract monthly payload remained unchanged and the watched row was still
  absent.
- Added EXP-0008 third continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:20:46Z`.
- Added EXP-0009 fourth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:30:12Z`.
- Added EXP-0010 fifth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:38:03Z`.
- Added EXP-0011 sixth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:45:51Z`.
- Added EXP-0012 seventh continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:55:41Z`.
- Added EXP-0013 eighth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:03:26Z`.
- Added EXP-0014 ninth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:10:57Z`.
- Added EXP-0015 tenth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:18:33Z`.
- Added EXP-0016 eleventh continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:27:34Z`.
- Added EXP-0017 twelfth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:35:51Z`.
- Added EXP-0018 thirteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:42:55Z`.
- Added EXP-0019 fourteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:50:50Z`.
- Added EXP-0020 fifteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:58:46Z`.
- Added EXP-0021 sixteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:06:57Z`.
- Added EXP-0022 seventeenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:15:05Z`.
- Added EXP-0023 eighteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:24:33Z`.
- Added EXP-0024 nineteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:33:50Z`.
- Added EXP-0025 twentieth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:42:02Z`.
- Added EXP-0026 twenty-first continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:50:08Z`.
- Added EXP-0027 twenty-second continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:57:57Z`.
- Added EXP-0028 twenty-third continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T21:10:15Z`.
- Added EXP-0029 twenty-fourth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T21:21:19Z`.
- Added EXP-0030 twenty-fifth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T21:29:42Z`.
- Added EXP-0031 twenty-sixth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T21:39:42Z`.

## G0 smoke test — 2026-06-18

- Archived every `bootstrap_now` source twice and documented EXP-0001.
- Generated `reports/source_inventory.md` and refreshed `MANIFEST.json`.
- Added fetch failure-path tests for HTTP errors and empty payloads.
- Added malformed HKO climate CSV parser coverage.
- Made experiment creation tests independent of the live registry counter.
- Fixed `scripts/bootstrap.ps1` so native command failures stop the bootstrap.
- Added compatibility docs for requested `HKG_TMAX_FIRST_GOALS.md`,
  `docs/00_PROJECT_OVERVIEW.md`, and `docs/06_LEAKAGE_CONTROL.md` paths.

## 0.1.0 — 2026-06-18

- Created complete Codex research bootstrap.
- Added target-parity and rules-change gates.
- Added point-in-time timestamp model and leakage controls.
- Added immutable raw archive and source catalog scaffolding.
- Added experiment ledger, milestone renderer, specialist agents, and skills.
- Added initial goal program from target verification through production eligibility.
- Added tests for bucket mapping, as-of availability, immutability, and config integrity.
# 2026-06-24

- Added the T03-T05 foundation runner with GribStream-safe one-thread probing, bounded retries, `Retry-After` handling, structured status files, and secret-safe API event logs.
- Added T03-T05 PostgreSQL migrations for GribStream catalog registries, NWP storage/lineage tables, and canonical location/station/geospatial registries.
- Added `scripts/check_t03_t05_status.py` and `documentation/T03_T05_GRIBSTREAM_AND_NWP_FOUNDATION_RUNBOOK.md` so background progress can be checked without rerunning API calls.
