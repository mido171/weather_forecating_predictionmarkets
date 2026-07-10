# EXP-0050 / HKG-T24-R18 Long-Form Experiment Report

## Title

HKO Official Forecast Baseline, Bias Correction, and MOS

## Research Question

Score the last eligible HKO official Tmax forecast and transparent MOS corrections.

## Intended Design

This experiment is part of the ordered HKG T-24 Tmax research plan. The intended input families are: RSS/JSON official forecast vintage parser, issue/publication/retrieval time extraction, valid-date numerical Tmax extraction, last-eligible-vintage selection before cutoff. The scientific goal is not to make a superficial model row; it is to test a specific physical or statistical mechanism under the T-1 15:00 Asia/Hong_Kong as-of contract. The required forecast target remains the official Hong Kong Observatory Headquarters daily maximum air temperature for local day T. All predictor information must be available before the cutoff, and all model-selection decisions before R30 must use development OOF evidence only.

## Current Evidence Audited

The current repository and data root contain the following relevant evidence: HKO RSS forecast archives are downloaded from 2020/2021 through 2026-06-18, current JSON forecast snapshots are collected prospectively, bronze current forecast snapshots exist but do not cover historical vintages. This evidence is useful but insufficient for a lawful scored experiment. The data acquisition inventory says many source families were downloaded as immutable raw payloads, but a raw payload alone is not model-ready. For R13-R30, the blocker is usually not that nothing exists; it is that the source-native parser, point-in-time issue/vintage contract, station metadata reconciliation, or complete upstream expert table is not yet available.

## Precondition Gate Result

The status for HKG-T24-R18 is `BLOCKED_PRECONDITIONS_NOT_MET`. The explicit blockers are: official_forecast_vintage_parser_missing, historical_json_versions_not_found, last_eligible_forecast_table_missing, mos_oof_predictions_missing. Because these blockers affect the actual predictor set or validation authorization, the experiment does not train a model, does not score OOF predictions, and does not promote any feature. This is a deliberate fail-closed result. It avoids fabricating features from target-day daily climate values, retrospective best tracks, current-only snapshots, unparsed raw archives, or post-hoc validation knowledge.

## Readiness Table

| category | item | status | disposition |
| --- | --- | --- | --- |
| required_input | RSS/JSON official forecast vintage parser | required | not scored until available |
| required_input | issue/publication/retrieval time extraction | required | not scored until available |
| required_input | valid-date numerical Tmax extraction | required | not scored until available |
| required_input | last-eligible-vintage selection before cutoff | required | not scored until available |
| available_evidence | HKO RSS forecast archives are downloaded from 2020/2021 through 2026-06-18 | available_or_partial | audited for blocker gate |
| available_evidence | current JSON forecast snapshots are collected prospectively | available_or_partial | audited for blocker gate |
| available_evidence | bronze current forecast snapshots exist but do not cover historical vintages | available_or_partial | audited for blocker gate |
| blocker | official_forecast_vintage_parser_missing | blocked | Parse the RSS archives into immutable issue/vintage rows and extract numerical Tmax before scoring raw official forecast and MOS. |
| blocker | historical_json_versions_not_found | blocked | Parse the RSS archives into immutable issue/vintage rows and extract numerical Tmax before scoring raw official forecast and MOS. |
| blocker | last_eligible_forecast_table_missing | blocked | Parse the RSS archives into immutable issue/vintage rows and extract numerical Tmax before scoring raw official forecast and MOS. |
| blocker | mos_oof_predictions_missing | blocked | Parse the RSS archives into immutable issue/vintage rows and extract numerical Tmax before scoring raw official forecast and MOS. |

## Leakage and As-Of Controls

No validation-2024 outcomes are read by this gate. No 2025-2026 locked-test target rows are read. No Polymarket, market data, backtesting, trading, or profitability logic is touched. If a required historical source is only available as a finalized retrospective dataset, this gate treats it as mechanism evidence until a publication or issue-time contract is proven. If a current/live feed exists only prospectively from June 2026, it is not backfilled into historical OOF rows. If a source is downloaded raw but not parsed, it is not treated as a usable feature table.

## Why A Blocked Folder Is Still An Experiment Artifact

The uploaded goal explicitly requires each research-plan experiment to have its own immutable folder and conclusion. A blocked precondition gate is therefore represented as an experiment artifact rather than buried in terminal output. The folder contains the same handoff shape as scored experiments: README, hypothesis, information-gain note, as-of contract, data manifest, feature spec, run config, protocol, ablation plan, negative controls, status, empty OOF prediction table, metrics JSON, subgroup metrics placeholder, readiness artifacts, results, conclusion, and reproduction command. This makes the null/blocker durable and prevents future work from accidentally claiming that the mechanism was tested.

## Date Ranges

No scored OOF prediction period exists for this experiment because the gate stops before feature construction or model training. The available upstream modern high-frequency diagnostics continue to cover the pre-validation target-date period through 2023-12-31, with the strict four-year OOF limitation documented in R04-R12. Long-history target-only evidence exists from 1884 onward in R02, but this specific experiment does not get to borrow that span unless its required predictors are parsed and eligible over that span. Validation 2024 is not accessed. Locked test dates from 2025-01-01 onward are not accessed.

## What Would Be Wrong To Do

It would be wrong to fill missing predictor families with target-day daily climate values, use retrospective best-track or full-day products as if they were known at T-1 15:00, infer station coordinates from names, train on current-only June 2026 snapshots and pretend they support 2021-2023 OOF, or run validation 2024 before a final challenger is predeclared. It would also be wrong to weaken the user's four-year OOF rule by silently promoting a short modern diagnostic. This gate records those constraints in machine-readable form.

## Required Next Action

Parse the RSS archives into immutable issue/vintage rows and extract numerical Tmax before scoring raw official forecast and MOS.

## Decision Record

HKG-T24-R18 is complete as a precondition gate, not as a scored model experiment. The decision is conservative and reproducible: blocked inputs are listed, available evidence is retained, no model output is fabricated, and the exact next engineering task is written down. This satisfies documentation discipline without pretending the system has information it does not yet have.

## Handoff Detail

A future Codex or GPT-Pro conversation should start from this folder before attempting the experiment again. It should first verify whether the listed blockers have been removed by new parsers, source requests, or approved credentials. Only then should it replace the empty OOF table with scored predictions. If the blocker remains, the correct action is to update the readiness evidence and keep the experiment blocked. For R30 specifically, validation remains unauthorized until every prerequisite is complete, a single final challenger is selected from development OOF evidence, and a predeclaration file is written and hashed before validation access.

This record also protects the research ledger from silent optimism. A blocked gate is not a failure to document work; it is the documented boundary between acquired raw data and scientifically usable, leakage-safe predictor evidence.
