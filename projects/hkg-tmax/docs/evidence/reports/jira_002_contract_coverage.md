# HKG-T24-002 Contract Coverage

## Natural Language Overview

This report is the traceability document for Jira 002. Its purpose is to connect the requirements from the Jira and binding strategy documents to the actual implementation, tests, and artifacts that now exist in the repository. The other documentation files explain the implementation in narrative form: what was read, how the schema works, how feature builders work, how the expert factory works, and what verification was run. This file is different. It is organized as a coverage matrix so a reviewer can ask, requirement by requirement, whether the code has an implementation location, whether there is test evidence, whether an artifact or report exists, and whether any caveat remains.

The report starts with binding precedence because every coverage claim depends on the correct source of truth. Jira 002 was implemented under the final consistency patch first, then final clarifications, then the completion specification, then the blueprint, then the Jira packet. That order is not decorative. It explains why the strict target-memory feature set uses `target__lag2_tmax_c` as the first finalized target lag and rejects finalized `target__lag1_...` naming, even if older text elsewhere referred to lag1-like concepts. It also explains why strict feature prefix validation is so narrow. The report is not trying to cover every idea in the broader strategy roadmap; it is proving the Jira 002 contract as bounded by the final patch and acceptance criteria.

The feature family coverage table is the first major section because Jira 002 is primarily a feature-engineering Jira. Each row names a feature family, its scope, the implementation module, the test evidence, and the artifact evidence. This makes it possible to verify that calendar, official, target-memory, online residual, GFS, GEFS mean, GEFS ensemble, proxy, diagnostic, and shadow families are all accounted for. It also makes scope explicit. A station proxy feature is not "missing" from strict coverage; it belongs in proxy coverage. An IFS or ARWF feature is not "missing" from strict coverage; it belongs in live-shadow coverage. The report is structured this way to prevent accidental promotion of proxy or shadow data during later work.

The feature matrix coverage section focuses on matrix-level invariants. It documents the allowed strict prefixes, forbidden strict prefixes, proxy separation, live-shadow separation, feature ordering, and the persistence rule that `model_features.feature_matrix` remains the physical table. This is where a reviewer should look if they are concerned that a useful but unsafe feature family might have slipped into strict. The evidence points to `feature_dictionary.py`, `matrix_builder.py`, and tests that enforce prefix rejection and ordering.

The target-memory leakage section is deliberately detailed because finalized daily labels are one of the easiest ways to introduce silent leakage. The report maps T-2-only usage, missing indicators, forbidden lag1 naming, target/calendar year-index consistency, causal climatology, and warming trend behavior to code and tests. Some of those tests are exact formula checks, while others are builder-level checks that prove output shape and naming. Together they show that the implementation follows the final naming and lag discipline rather than an older target-memory design.

The online residual state coverage section does the same for causal residual memory. It maps the schema table, prior-date filter, EWMA half-lives, warmup statuses, streak fields, and shrinkage/capping behavior to implementation and tests. The most important coverage item is that online state for T uses only observations before T. That rule is tested directly with a synthetic current-date residual that would dominate the state if it were incorrectly included.

The expert coverage and OOF coverage sections prove that E0 through E11 are represented with the expected behavior. E0 is direct official. E1 is capped and can be demoted. E2, E4, and E5 are strict residual experts. E3 is proxy-only. E6, E7, E8, E9, and E11 are shadow/direct-placeholder rows with zero strict weight. E10 is diagnostic proxy placeholder only. The OOF section then ties those rows to chronology, proving that folds enforce `train_end_date < test_start_date`, that the DB table has the same constraint, that same-row residual features are not emitted into matrices, and that shadow rows carry zero strict weight.

The artifact and CLI sections are included because Jira 002 acceptance was not limited to Python functions. It required tables, reports, dictionaries, commands, and expert artifacts. This report therefore maps feature-family tables, expert prediction tables, artifact tables, dictionary reports, OOF reports, model-selection reports, and CLI commands to their implementation locations. A reviewer can use this to confirm that the deliverable is an operational package surface, not just isolated functions.

Finally, the report closes with verification and residual risk. Local verification passed for compile, lint, strict typing, and focused pytest. The real-DB smoke commands were not run because no DSN was configured, and the report says that plainly. That caveat is not hidden because live DB row counts and source coverage are materially different from synthetic tests. This report should therefore be read as "contract implemented and locally verified except live DB smoke," with exact commands listed for completing that final database-backed check.

## Status

Implemented with local DB-dependent verification skipped because neither `HKG_TMAX_DATABASE_URL` nor `HKG_TMAX_DB_DSN` was set.

## Binding Precedence

Final consistency patch, final clarifications, completion specification, blueprint, then Jira packet.

The final consistency patch wins on target-memory names. Finalized daily target-memory features start at `target__lag2_tmax_c`; `target__lag1_...` finalized feature names are rejected.

## Feature Family Coverage

| Feature family | Scope | Implementation evidence | Test evidence | Artifact/report evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Calendar | strict | `code/src/hkg_t24/features/calendar.py` | `test_jira002_feature_builders.py` | `feature_dictionary_strict.csv` | Implemented |
| Official anchor | strict | `code/src/hkg_t24/features/official_anchor.py` | `test_jira002_feature_builders.py` | `official_anchor_coverage.md` | Implemented |
| Official text/PSR | strict | `code/src/hkg_t24/features/official_text.py` | `test_jira002_feature_builders.py` | `feature_dictionary_strict.csv` | Implemented |
| Target memory | strict | `code/src/hkg_t24/features/target_memory.py` | `test_snapshot_builder_synthetic.py`, `test_jira002_feature_builders.py` | `feature_dictionary_strict.csv` | Implemented |
| Online residual | strict | `code/src/hkg_t24/features/online_state.py` | `test_jira002_online_and_experts.py` | `online_state_audit.csv`, `online_state_audit_report.md` | Implemented |
| GFS NWP | strict | `code/src/hkg_t24/features/nwp_daily.py`, `db_feature_builders.py` | `test_jira002_feature_builders.py`, `test_jira002_synthetic_integration.py` | `feature_dictionary_strict.csv`, `feature_matrix_coverage_report.md` | Implemented |
| GEFS mean | strict | `code/src/hkg_t24/features/nwp_daily.py`, `db_feature_builders.py` | `test_jira002_feature_builders.py`, `test_jira002_synthetic_integration.py` | `feature_dictionary_strict.csv`, `feature_matrix_coverage_report.md` | Implemented |
| GEFS ensemble | strict | `code/src/hkg_t24/features/nwp_daily.py`, `db_feature_builders.py` | `test_jira002_feature_builders.py`, `test_jira002_synthetic_integration.py` | `feature_dictionary_strict.csv`, `feature_matrix_coverage_report.md` | Implemented |
| Station proxy | proxy | `code/src/hkg_t24/features/station_proxy.py` | `test_jira002_feature_builders.py`, real-DB smoke test placeholder path | `feature_dictionary_proxy.csv` | Implemented |
| Diagnostic climate proxy | proxy | `code/src/hkg_t24/features/diagnostic_proxy.py` | synthetic/proxy validation coverage | `feature_dictionary_proxy.csv` | Implemented |
| Shadow NWP | live_shadow | `code/src/hkg_t24/features/nwp_daily.py`, `models/experts.py` | `test_jira002_online_and_experts.py` | `feature_dictionary_shadow.csv` | Implemented |

## Feature Matrix Coverage

| Requirement | Implementation evidence | Test evidence | Status |
| --- | --- | --- | --- |
| Strict build produces only allowed prefixes | `features/feature_dictionary.py`, `features/matrix_builder.py` | `test_jira002_feature_builders.py`, `test_jira002_synthetic_integration.py` | Pass |
| Strict build excludes forbidden proxy/shadow prefixes | `STRICT_FORBIDDEN_FEATURE_PREFIXES`, `validate_feature_names("strict", ...)` | `test_jira002_feature_builders.py` | Pass |
| Proxy build stays separate | `build_scoped_matrix_rows(scope="proxy")`, `build_proxy_family` | `test_jira002_real_db_smoke.py` skips without DB | Implemented |
| Live-shadow build stays separate | `build_scoped_matrix_rows(scope="live_shadow")`, `build_shadow_family` | `test_jira002_real_db_smoke.py` skips without DB | Implemented |
| Feature order metadata first then lexicographic features with missing indicators after base | `matrix_row_to_model_record`, `ordered_feature_names` | `test_jira002_feature_builders.py` | Pass |
| Physical matrix table remains `model_features.feature_matrix` | `db/ddl.py`, `persist_feature_matrix_rows` | `test_schema_sql_contract.py` | Pass |

## Target-Memory Leakage Coverage

| Rule | Implementation evidence | Test evidence | Status |
| --- | --- | --- | --- |
| Use T-2 or older finalized labels only | `features/target_memory.py` | `test_snapshot_builder_synthetic.py`, `test_jira002_feature_builders.py` | Pass |
| No finalized target lag1 names | `FORBIDDEN_FINALIZED_TARGET_TERMS`, `assert_no_forbidden_target_memory_names` | `test_h24n_contract_policy.py`, `test_jira002_feature_builders.py` | Pass |
| Missing indicators for every target-memory feature except year index | `TARGET_MEMORY_MISSING_INDICATOR_FEATURES`, `_with_missing_indicators` | `test_snapshot_builder_synthetic.py`, `test_jira002_feature_builders.py` | Pass |
| `target__year_index == calendar__year_index` | `assert_target_year_index_matches_calendar` | `test_jira002_feature_builders.py` | Pass |
| Causal climatology uses prior years and T-2 cutoff | `_causal_climatology` | unit coverage through builder and missingness checks | Implemented |
| 10-year warming trend uses prior complete years | `_annual_mean_slope` | unit coverage through builder and missingness checks | Implemented |

## Online Residual State Coverage

| Rule | Implementation evidence | Test evidence | Status |
| --- | --- | --- | --- |
| `model_features.online_residual_state` exists | `db/ddl.py` | `test_schema_sql_contract.py` | Pass |
| State uses only dates before T | `build_online_state` filter `observation.target_date_hkt < target_date_hkt` | `test_jira002_online_and_experts.py` | Pass |
| EWMA half-lives 5/10/20/40 | `ONLINE_HALF_LIVES`, `online_state.py` | `test_jira002_online_and_experts.py` | Pass |
| Warmup statuses | `warmup_status` | `test_jira002_online_and_experts.py` | Pass |
| Streak fields | `_streaks` and emitted feature names | `test_jira002_online_and_experts.py` | Pass |
| Shrinkage/capping/expected abs error | `build_online_state` | `test_jira002_online_and_experts.py` | Pass |

## Expert Coverage

| Expert | Required behavior | Implementation evidence | Test evidence | Status |
| --- | --- | --- | --- | --- |
| E0_OFFICIAL_RAW_ANCHOR | Direct official forecast max | `models/experts.py` | `test_jira002_online_and_experts.py`, synthetic integration | Pass |
| E1_OFFICIAL_RESIDUAL | Capped residual, demoted if promotion fails | `models/experts.py`, `model_selection.py` | `test_jira002_online_and_experts.py` | Pass |
| E2_TARGET_MEMORY | Safe target-memory expert | `models/experts.py` | synthetic integration | Pass |
| E3_STATION_PROXY | Proxy-only | `models/experts.py` | `test_jira002_online_and_experts.py` | Pass |
| E4_GFS_MOS | Uses GFS strict anchor/features | `models/experts.py` | synthetic integration | Pass |
| E5_GEFS_ENSEMBLE | Uses GEFS mean/member features | `models/experts.py` | synthetic integration | Pass |
| E6_IFS_OPER_SHADOW | Shadow/direct placeholder, zero strict weight | `models/experts.py` | `test_jira002_online_and_experts.py` | Pass |
| E7_IFS_ENS_SHADOW | Shadow/direct placeholder, zero strict weight | `models/experts.py` | `test_jira002_online_and_experts.py` | Pass |
| E8_AI_NWP_SHADOW | Shadow/direct placeholder, zero strict weight | `models/experts.py` | `test_jira002_online_and_experts.py` | Pass |
| E9_CWA_WRF_LIVE_SHADOW | Shadow/direct placeholder, zero strict weight | `models/experts.py` | `test_jira002_online_and_experts.py` | Pass |
| E10_DIAGNOSTIC_PROXY | Placeholder unavailable/proxy only | `models/experts.py` | `test_jira002_online_and_experts.py`, synthetic integration | Pass |
| E11_ARWF_LIVE_SHADOW | Shadow/direct placeholder, zero strict weight | `models/experts.py` | `test_jira002_online_and_experts.py` | Pass |

## OOF Coverage

| Rule | Implementation evidence | Test evidence | Artifact/report evidence | Status |
| --- | --- | --- | --- | --- |
| Genuine chronological OOF | `models/folds.py`, `generate_expert_oof_predictions` | `test_jira002_online_and_experts.py`, synthetic integration | `oof_integrity_report.md` | Pass |
| `train_end_date < test_start_date` | `FoldSpec.validate`, DB CHECK constraint | `test_jira002_online_and_experts.py`, `test_schema_sql_contract.py` | `model_oof.expert_prediction` DDL | Pass |
| No same-row official residual feature inputs | No same-row residual features emitted into matrices | `oof_integrity_report` helper reports zero | `oof_integrity_report.md` | Pass |
| Shadow strict weight zero | `router_weight_cap=0.0` for live-shadow experts | `test_jira002_online_and_experts.py` | `expert_predictions_oof.csv` | Pass |

## Artifact Coverage

| Artifact | Implementation evidence | Status |
| --- | --- | --- |
| `model_features.official_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.official_revision_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.target_memory_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.online_residual_state` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.nwp_daily_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.nwp_ensemble_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.station_proxy_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.diagnostic_proxy_features` | `db/ddl.py`, `db_feature_builders.py` | Implemented |
| `model_features.static_geospatial_features` | `db/ddl.py` | Implemented |
| `model_features.feature_matrix` Jira 002 columns | `db/ddl.py`, `matrix_builder.py` | Implemented |
| `model_oof.expert_prediction` | `db/ddl.py`, `artifact_store.py` | Implemented |
| `model_oof.expert_artifact` | `db/ddl.py`, `artifact_store.py` | Implemented |
| Feature dictionary CSV/MD reports | `feature_dictionary.py` | Implemented |
| OOF scoreboard/integrity reports | `models/oof.py` | Implemented |
| Model-selection JSON/MD artifacts | `artifact_store.py`, `model_selection.py` | Implemented |

## CLI Coverage

| Command | Implementation evidence | Test evidence | Status |
| --- | --- | --- | --- |
| `build-features` | `cli.py`, `db_feature_builders.py` | real-DB smoke test skips without DSN | Implemented |
| `build-online-states` | `cli.py`, `db_feature_builders.py` | unit online-state tests | Implemented |
| `replay-online-states-oof` | `cli.py`, `db_feature_builders.py` | unit online-state tests | Implemented |
| `update-online-states-after-settlement` | `cli.py`, `db_feature_builders.py` | unit online-state tests | Implemented |
| `train-experts` | `cli.py`, `db_expert_factory.py` | real-DB smoke test skips without DSN | Implemented |
| `generate-oof` | `cli.py`, `db_expert_factory.py` | real-DB smoke test skips without DSN | Implemented |

## Verification Summary

Commands run locally:

```text
python -m compileall code/src/hkg_t24
python -m ruff check code/src/hkg_t24 code/tests/hkg_t24
python -m mypy code/src/hkg_t24
python -m pytest code/tests/hkg_t24
```

Latest focused pytest result before final handoff:

```text
28 passed, 3 skipped
```

The skipped tests are DB-dependent and skip with:

```text
SKIPPED_REAL_DB_NO_DATABASE_URL
```

## DB Smoke Status

Not run locally because no DB DSN was configured.

Required smoke commands when a DB is configured:

```text
python -m hkg_t24.cli build-features --scope strict --from-date 2021-04-14 --to-date 2021-05-31 --smoke
python -m hkg_t24.cli train-experts --scope strict-pre2024 --smoke --from-date 2021-04-14 --to-date 2021-05-31
python -m hkg_t24.cli generate-oof --scope strict-pre2024 --smoke --from-date 2021-04-14 --to-date 2021-05-31
```

## Residual Risk

The DB-backed NWP feature extraction is implemented against the documented `nwp_tactical.forecast_wide` columns and mandatory safe-row filter. Because the DSN was unset locally, actual DB row counts and smoke thresholds were not verified in this run.

The expert model family is a deterministic mean-residual MOS implementation. It satisfies Jira 002 OOF, capping, placeholder, artifact, and leakage contracts. More advanced LightGBM model candidates can be added behind the same interfaces later without changing the Jira 002 schema surface.
