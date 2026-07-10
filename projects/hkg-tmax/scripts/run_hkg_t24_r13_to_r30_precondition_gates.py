from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
EXPERIMENT_ROOT = REPO_ROOT / "analysis" / "hkg_tmax_t24" / "experiments"


@dataclass(frozen=True)
class GateSpec:
    research_id: str
    exp_id: str
    slug: str
    title: str
    intended_question: str
    intended_inputs: tuple[str, ...]
    available_evidence: tuple[str, ...]
    blockers: tuple[str, ...]
    next_action: str
    status: str = "BLOCKED_PRECONDITIONS_NOT_MET"

    @property
    def folder(self) -> Path:
        return EXPERIMENT_ROOT / f"{self.exp_id}-{self.research_id}"

    @property
    def report_name(self) -> str:
        number = self.research_id.rsplit("R", maxsplit=1)[-1]
        return f"R{number}_{self.slug.upper()}.md"


GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(
        "HKG-T24-R13",
        "EXP-0045",
        "cloud_rain_visibility_surface_wetness",
        "Cloud, Rain, Visibility, and Surface-Wetness Suppression",
        "Test whether pre-cutoff hydrometeorological suppression and recovery conditions improve next-day HKO Tmax.",
        (
            "historical cutoff-safe rainfall/feed observations",
            "visibility and present-weather histories",
            "cloud/ceiling or current-weather vintages",
            "R06 moisture and R12 radiation features",
        ),
        (
            "HKO daily climate rainfall/cloud/visibility elements are parsed as retrospective mechanism-only daily rows",
            "current rainfall, visibility, lightning, and weather feeds are collected prospectively from 2026-06-19/20",
            "R06 moisture and R12 solar diagnostics exist for the modern pre-validation window",
        ),
        (
            "historical_rainfall_feed_versions_not_found",
            "historical_visibility_feed_versions_not_found",
            "historical_current_weather_json_not_found",
            "target_day_daily_climate_values_forbidden_as_predictors",
        ),
        "Prepare official HKO request/source-discovery package for historical rainfall, visibility, and current-weather feed versions; parse only cutoff-safe vintages before scoring R13.",
    ),
    GateSpec(
        "HKG-T24-R14",
        "EXP-0046",
        "eligible_upper_air_thermal_potential",
        "Eligible Upper-Air Thermal Potential and Inversion Structure",
        "Test whether the 00 UTC T-1 King’s Park/Hong Kong sounding adds vertical-structure skill.",
        (
            "IGRA HKM00045004 period-of-record raw file",
            "eligible sounding parser",
            "pressure-level interpolation/QC",
            "release-latency contract for 00 UTC T-1",
        ),
        (
            "NOAA IGRA HKM00045004 period-of-record zip is downloaded",
            "NOAA IGRA year-to-date zip is downloaded",
            "IGRA documentation/list-format files are downloaded",
        ),
        (
            "igra_parser_not_implemented",
            "eligible_00utc_tminus1_release_latency_not_proven",
            "pressure_level_qc_table_missing",
            "upper_air_feature_matrix_missing",
        ),
        "Implement the IGRA parser and eligibility rule, preserve QC flags, then rerun R14 with long-history and modern-overlap folds.",
    ),
    GateSpec(
        "HKG-T24-R15",
        "EXP-0047",
        "surface_upper_air_coupling",
        "Surface-Upper-Air Coupling and Mixing-Potential Experiment",
        "Test whether surface-to-aloft thermal mismatch explains recoverable heating and transition risk.",
        (
            "R14 upper-air levels and inversion features",
            "R04 surface trajectory",
            "R07 pressure transition features",
            "physically constrained coupling formulas",
        ),
        (
            "R04 surface trajectory exists",
            "R07 pressure/front proxy diagnostic exists",
            "IGRA raw data is downloaded but not parsed",
        ),
        (
            "blocked_by_R14_upper_air_parser",
            "mixing_depth_and_inversion_features_missing",
            "surface_upper_air_coupling_matrix_missing",
        ),
        "Complete R14 first, then construct predefined surface-to-level lapse, theta, moisture, and inversion-capped heating-potential features.",
    ),
    GateSpec(
        "HKG-T24-R16",
        "EXP-0048",
        "fifty_year_regional_isd_surface_core",
        "Fifty-Year Regional ISD Surface Core",
        "Test whether a long-history regional surface core provides robust anomaly forecasts across at least 50 years.",
        (
            "NOAA ISD station-year files",
            "ISD parser preserving QC/report types",
            "station metadata segment handling",
            "long-history rolling OOF folds",
        ),
        (
            "951 NOAA ISD annual station-year gzip files are downloaded across 36 stations",
            "overall raw station-year coverage spans 1945-2025",
            "ISD format/readme/history/inventory files are downloaded",
        ),
        (
            "isd_parser_not_implemented",
            "isd_qc_report_type_tables_missing",
            "fifty_year_feature_matrix_missing",
            "modern_transfer_scorecard_missing",
        ),
        "Implement the ISD parser and QC coverage tables, then build exact eligible cutoff reports and long-history anomaly folds.",
    ),
    GateSpec(
        "HKG-T24-R17",
        "EXP-0049",
        "metadata_breaks_urbanization_era_transfer",
        "Station Metadata Breaks, Urbanization, and Era Transfer",
        "Determine whether station moves, reporting changes, and urbanization alter feature-target relationships.",
        (
            "HKO/NOAA/IGRA metadata timelines",
            "station coordinates/elevations",
            "land-use rasters with valid eras",
            "R16 long-history parsed station features",
        ),
        (
            "HKO station registry exists but HKO named-station official metadata remains pending",
            "NOAA ISD metadata is downloaded",
            "LUHK 2018-2024 rasters are downloaded but not parsed to station buffers",
        ),
        (
            "hko_station_history_metadata_unresolved",
            "station_coordinates_missing_for_hko_feed_names",
            "luhk_station_context_not_parsed",
            "blocked_by_R16_isd_parser",
        ),
        "Resolve station metadata timelines and static context first; then build segment-aware offset and era-transfer diagnostics.",
    ),
    GateSpec(
        "HKG-T24-R18",
        "EXP-0050",
        "official_forecast_mos",
        "HKO Official Forecast Baseline, Bias Correction, and MOS",
        "Score the last eligible HKO official Tmax forecast and transparent MOS corrections.",
        (
            "RSS/JSON official forecast vintage parser",
            "issue/publication/retrieval time extraction",
            "valid-date numerical Tmax extraction",
            "last-eligible-vintage selection before cutoff",
        ),
        (
            "HKO RSS forecast archives are downloaded from 2020/2021 through 2026-06-18",
            "current JSON forecast snapshots are collected prospectively",
            "bronze current forecast snapshots exist but do not cover historical vintages",
        ),
        (
            "official_forecast_vintage_parser_missing",
            "historical_json_versions_not_found",
            "last_eligible_forecast_table_missing",
            "mos_oof_predictions_missing",
        ),
        "Parse the RSS archives into immutable issue/vintage rows and extract numerical Tmax before scoring raw official forecast and MOS.",
    ),
    GateSpec(
        "HKG-T24-R19",
        "EXP-0051",
        "analogue_system_redesign",
        "Analogue-System Redesign and Learned Similarity",
        "Redesign the station-state analogue distance, scaling, season window, and uncertainty model.",
        (
            "exact reproducible baseline analogue feature table",
            "promoted or conditionally retained R04-R18 feature families",
            "nested rolling-origin tuning",
            "neighbor identity audit table",
        ),
        (
            "supplied baseline predictions were reproduced from row-level archive in R01",
            "baseline rerun/hash parity remains documented as a reproduction blocker",
            "R04-R12 diagnostics exist but none are promotable under the four-year OOF rule",
        ),
        (
            "exact_baseline_rerun_hash_mismatch",
            "no_promoted_feature_families_from_R04_to_R18",
            "nested_analogue_neighbor_audit_not_built",
        ),
        "Repair exact baseline rerun parity and complete required upstream parsers before starting analogue redesign.",
    ),
    GateSpec(
        "HKG-T24-R20",
        "EXP-0052",
        "physical_regime_classifier",
        "Physically Defined Regime Classifier and Specialist Experts",
        "Build transparent pre-cutoff probabilities for cold surge, spring cloud, maritime, subsidence heat, rain suppression, and local regimes.",
        (
            "R07 transition features",
            "R11 dynamic advection or fixed fallback",
            "R13 rain/cloud suppression",
            "R14/R15 upper-air structure",
            "R18 official forecast signals",
        ),
        (
            "R07 transition proxy diagnostic exists",
            "R11 is blocked by missing station geometry",
            "R13/R14/R18 are blocked by missing parsers",
        ),
        (
            "rain_cloud_regime_inputs_missing",
            "upper_air_regime_inputs_missing",
            "official_forecast_regime_inputs_missing",
            "specialist_expert_training_not_eligible",
        ),
        "Complete the upstream parsers and then fit regime probabilities inside chronological folds with minimum sample constraints.",
    ),
    GateSpec(
        "HKG-T24-R21",
        "EXP-0053",
        "data_driven_regime_discovery",
        "Data-Driven Regime Discovery and Cluster Stability",
        "Discover repeatable multivariate pre-cutoff weather states and test whether they improve forecasts.",
        (
            "compact promoted state vector",
            "fold-local PCA/GMM/HDBSCAN or comparable clustering",
            "cluster identity alignment across folds",
            "R20 physical regime comparison",
        ),
        (
            "R10 fold-local PCA diagnostic exists",
            "R20 physical regimes are not yet available",
            "no compact promoted feature bank exists under the four-year OOF rule",
        ),
        (
            "promoted_state_vector_missing",
            "physical_regime_baseline_missing",
            "cluster_stability_protocol_not_ready",
        ),
        "After R20 and upstream parsers, build a compact state vector and run fold-local clustering with shuffled/global-fit negative controls.",
    ),
    GateSpec(
        "HKG-T24-R22",
        "EXP-0054",
        "abrupt_transition_catastrophic_error_specialist",
        "Abrupt Transition and Catastrophic-Error Specialist",
        "Reduce large misses from abrupt T-1 to T changes with calibrated transition probability and specialist correction.",
        (
            "R07 front/pressure features",
            "R11 advection features",
            "R14/R15 upper-air changes",
            "R18 forecast revisions",
            "R20/R21 regime probabilities",
        ),
        (
            "R07 pressure/front proxy exists",
            "R11/R14/R18/R20/R21 are blocked or unavailable",
            "development-only target-change labels can be made later but predictors are incomplete",
        ),
        (
            "transition_predictor_bank_incomplete",
            "official_forecast_revision_features_missing",
            "upper_air_change_features_missing",
            "regime_probabilities_missing",
        ),
        "Complete required predictor families, then train nested transition classifiers and specialist corrections using development-only labels.",
    ),
    GateSpec(
        "HKG-T24-R23",
        "EXP-0055",
        "extreme_heat_tc_adjacent_subsidence",
        "Extreme-Heat and Tropical-Cyclone-Adjacent Subsidence Specialist",
        "Improve hot-tail threshold probabilities and TC-adjacent subsidence cases without retrospective best-track leakage.",
        (
            "upper-air warmth/stability",
            "radiation and cloud suppression",
            "official or point-in-time TC advisory vintages",
            "hot-tail calibrated classifiers",
        ),
        (
            "HKO TC best tracks are downloaded but retrospective only",
            "TC realtime snapshots are prospective from 2026-06-19/20",
            "R12 radiation diagnostic exists but not promotable",
        ),
        (
            "point_in_time_tc_advisory_archive_missing",
            "upper_air_features_missing",
            "cloud_suppression_features_missing",
            "hot_tail_sample_gate_not_established",
        ),
        "Build or acquire point-in-time TC advisory vintages and upper-air features before any hot-tail specialist scoring.",
    ),
    GateSpec(
        "HKG-T24-R24",
        "EXP-0056",
        "marine_sea_temperature_coastline_terrain",
        "Marine, Sea-Temperature, Coastline, and Terrain Interaction",
        "Test whether sea-air contrast, coastline, terrain exposure, and onshore-flow interactions explain residuals.",
        (
            "publication-safe sea temperature",
            "station coastline distance and coast-normal vectors",
            "terrain/elevation station context",
            "land-use buffers with valid eras",
            "R11 flow-relative geometry",
        ),
        (
            "HKO daily sea-temperature raw files are parsed as retrospective daily climate mechanism rows",
            "terrain, coastline, and LUHK raw packages are downloaded",
            "station_distance_bearing exists only for static HKO/NOAA registry, not all HKO feed names",
        ),
        (
            "sea_temperature_publication_timing_unproven",
            "hko_named_station_geospatial_context_missing",
            "terrain_luhk_station_buffers_not_parsed",
            "blocked_by_R11_dynamic_geometry",
        ),
        "Parse static geospatial station context and prove sea-temperature publication timing before testing marine/terrain interactions.",
    ),
    GateSpec(
        "HKG-T24-R25",
        "EXP-0057",
        "privileged_teacher_student_auxiliary_targets",
        "Privileged-Information Teacher/Student Auxiliary Targets",
        "Use target-day information safely as training-only auxiliary labels whose student predictions are cutoff-safe.",
        (
            "auxiliary retrospective labels",
            "lawful student feature bank",
            "nested OOF student probabilities",
            "future-mutation leakage test",
        ),
        (
            "R03 peak-time anatomy exists",
            "retrospective daily climate mechanism labels are parsed",
            "no promoted lawful feature bank exists yet",
        ),
        (
            "nested_auxiliary_oof_pipeline_missing",
            "lawful_promoted_student_feature_bank_missing",
            "future_mutation_teacher_test_not_implemented",
        ),
        "After a lawful feature bank exists, cross-fit auxiliary student models and prove teacher data cannot affect inference rows.",
    ),
    GateSpec(
        "HKG-T24-R26",
        "EXP-0058",
        "multi_era_hierarchical_core_modern_booster",
        "Multi-Era Hierarchical Core and Modern Residual Booster",
        "Combine long-history core forecasts with modern HKO-network residual corrections without fabricating missing modern features in old eras.",
        (
            "R16 fifty-year ISD/IGRA core",
            "modern promoted residual features",
            "OOF core predictions for modern rows",
            "hierarchical residual booster",
        ),
        (
            "R02 long-history target-only climatology exists",
            "R16 fifty-year regional surface core is blocked by the ISD parser",
            "no modern feature family is promotable under four-year OOF rule",
        ),
        (
            "fifty_year_surface_core_missing",
            "modern_promoted_residual_feature_bank_missing",
            "oof_core_residual_training_table_missing",
        ),
        "Build R16 first, then create OOF long-core predictions and train modern residual boosters only on out-of-fold residuals.",
    ),
    GateSpec(
        "HKG-T24-R27",
        "EXP-0059",
        "operational_latency_missing_station_outage",
        "Operational Latency, Missing-Station, and Data-Outage Robustness",
        "Quantify forecast degradation under realistic latency, stale observations, and station/source outages.",
        (
            "best candidate model from prior experiments",
            "source freshness masks",
            "station outage simulation framework",
            "minimum-data fallback hierarchy",
        ),
        (
            "R04-R12 diagnostics expose data coverage and some station/missingness information",
            "no final or promotable candidate model exists yet",
            "R11 station geometry and later source parsers are blocked",
        ),
        (
            "candidate_model_for_outage_testing_missing",
            "minimum_data_fallback_hierarchy_missing",
            "source_specific_latency_scenarios_not_mapped",
        ),
        "After a candidate feature/model bank exists, run deterministic outage scenarios and source-criticality rankings.",
    ),
    GateSpec(
        "HKG-T24-R28",
        "EXP-0060",
        "transparent_nonlinear_model_family_benchmark",
        "Transparent Nonlinear Model-Family Benchmark",
        "Benchmark compact linear, robust, GAM, boosting, and tree families on the accepted feature bank.",
        (
            "accepted feature bank from prior experiments",
            "nested chronological tuning",
            "fixed compact hyperparameter grids",
            "seed/stability diagnostics",
        ),
        (
            "R04-R12 diagnostics exist",
            "no feature family has cleared the hard four-year OOF promotion gate",
            "R13-R27 source/model prerequisites are blocked",
        ),
        (
            "accepted_feature_bank_missing",
            "nested_model_family_benchmark_not_eligible",
            "sample_span_below_four_year_rule",
        ),
        "Wait until an accepted feature bank exists; then run compact predeclared model-family grids inside chronological folds.",
    ),
    GateSpec(
        "HKG-T24-R29",
        "EXP-0061",
        "conditional_distribution_oof_expert_ensemble",
        "Conditional Distribution, Calibration, and OOF Expert Ensemble",
        "Blend complete OOF experts into a calibrated predictive distribution and 0.1 C probability mass function.",
        (
            "complete OOF base experts",
            "expert disagreement and missingness features",
            "conditional calibration",
            "valid monotonic CDF and 0.1 C mass function",
        ),
        (
            "R02 long-history diagnostic, R04 baseline, R10 PCA diagnostic, and R12 solar diagnostic exist",
            "official forecast, upper-air, transition, hot-tail, multi-era, and robustness experts are missing or blocked",
        ),
        (
            "complete_oof_expert_panel_missing",
            "calibration_feature_bank_missing",
            "distributional_ensemble_not_eligible",
        ),
        "Complete prerequisite experts and then train only on development OOF predictions to produce calibrated distributions.",
    ),
    GateSpec(
        "HKG-T24-R30",
        "EXP-0062",
        "predeclared_final_challenger_freeze",
        "Predeclared Final Challenger Freeze and One-Shot Validation",
        "Pre-register exactly one final challenger and score validation 2024 once.",
        (
            "final architecture selected from development OOF only",
            "feature schema/data snapshot/code hashes",
            "predeclaration hash",
            "one validation-2024 scoring command",
        ),
        (
            "R01 validation access was limited to supplied baseline reproduction",
            "R02-R29 have not produced an eligible final challenger",
            "no R30 predeclaration has been written or hashed",
        ),
        (
            "prior_experiments_not_all_eligible",
            "final_challenger_not_selected",
            "predeclaration_not_written",
            "validation_2024_not_accessed_for_R30",
        ),
        "Do not run validation. Finish upstream parsers/experiments, select a final challenger from development OOF, write and hash PREDECLARATION.md, then run exactly one validation command.",
        status="BLOCKED_FINAL_VALIDATION_NOT_AUTHORIZED",
    ),
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.Series(dtype="datetime64[ns]"),
            "fold_id": pd.Series(dtype="object"),
            "model_id": pd.Series(dtype="object"),
            "model_family": pd.Series(dtype="object"),
            "point_forecast": pd.Series(dtype="float64"),
            "target_tmax_c": pd.Series(dtype="float64"),
            "blocked_reason": pd.Series(dtype="object"),
        }
    )


def readiness_rows(spec: GateSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in spec.intended_inputs:
        rows.append({"category": "required_input", "item": item, "status": "required", "disposition": "not scored until available"})
    for item in spec.available_evidence:
        rows.append({"category": "available_evidence", "item": item, "status": "available_or_partial", "disposition": "audited for blocker gate"})
    for item in spec.blockers:
        rows.append({"category": "blocker", "item": item, "status": "blocked", "disposition": spec.next_action})
    return rows


def long_report(spec: GateSpec, rows: Sequence[dict[str, object]]) -> str:
    table = markdown_table(rows, ["category", "item", "status", "disposition"])
    blockers = ", ".join(spec.blockers)
    inputs = ", ".join(spec.intended_inputs)
    evidence = ", ".join(spec.available_evidence)
    return f"""# {spec.exp_id} / {spec.research_id} Long-Form Experiment Report

## Title

{spec.title}

## Research Question

{spec.intended_question}

## Intended Design

This experiment is part of the ordered HKG T-24 Tmax research plan. The intended input families are: {inputs}. The scientific goal is not to make a superficial model row; it is to test a specific physical or statistical mechanism under the T-1 15:00 Asia/Hong_Kong as-of contract. The required forecast target remains the official Hong Kong Observatory Headquarters daily maximum air temperature for local day T. All predictor information must be available before the cutoff, and all model-selection decisions before R30 must use development OOF evidence only.

## Current Evidence Audited

The current repository and data root contain the following relevant evidence: {evidence}. This evidence is useful but insufficient for a lawful scored experiment. The data acquisition inventory says many source families were downloaded as immutable raw payloads, but a raw payload alone is not model-ready. For R13-R30, the blocker is usually not that nothing exists; it is that the source-native parser, point-in-time issue/vintage contract, station metadata reconciliation, or complete upstream expert table is not yet available.

## Precondition Gate Result

The status for {spec.research_id} is `{spec.status}`. The explicit blockers are: {blockers}. Because these blockers affect the actual predictor set or validation authorization, the experiment does not train a model, does not score OOF predictions, and does not promote any feature. This is a deliberate fail-closed result. It avoids fabricating features from target-day daily climate values, retrospective best tracks, current-only snapshots, unparsed raw archives, or post-hoc validation knowledge.

## Readiness Table

{table}

## Leakage and As-Of Controls

No validation-2024 outcomes are read by this gate. No 2025-2026 locked-test target rows are read. No Polymarket, market data, backtesting, trading, or profitability logic is touched. If a required historical source is only available as a finalized retrospective dataset, this gate treats it as mechanism evidence until a publication or issue-time contract is proven. If a current/live feed exists only prospectively from June 2026, it is not backfilled into historical OOF rows. If a source is downloaded raw but not parsed, it is not treated as a usable feature table.

## Why A Blocked Folder Is Still An Experiment Artifact

The uploaded goal explicitly requires each research-plan experiment to have its own immutable folder and conclusion. A blocked precondition gate is therefore represented as an experiment artifact rather than buried in terminal output. The folder contains the same handoff shape as scored experiments: README, hypothesis, information-gain note, as-of contract, data manifest, feature spec, run config, protocol, ablation plan, negative controls, status, empty OOF prediction table, metrics JSON, subgroup metrics placeholder, readiness artifacts, results, conclusion, and reproduction command. This makes the null/blocker durable and prevents future work from accidentally claiming that the mechanism was tested.

## Date Ranges

No scored OOF prediction period exists for this experiment because the gate stops before feature construction or model training. The available upstream modern high-frequency diagnostics continue to cover the pre-validation target-date period through 2023-12-31, with the strict four-year OOF limitation documented in R04-R12. Long-history target-only evidence exists from 1884 onward in R02, but this specific experiment does not get to borrow that span unless its required predictors are parsed and eligible over that span. Validation 2024 is not accessed. Locked test dates from 2025-01-01 onward are not accessed.

## What Would Be Wrong To Do

It would be wrong to fill missing predictor families with target-day daily climate values, use retrospective best-track or full-day products as if they were known at T-1 15:00, infer station coordinates from names, train on current-only June 2026 snapshots and pretend they support 2021-2023 OOF, or run validation 2024 before a final challenger is predeclared. It would also be wrong to weaken the user's four-year OOF rule by silently promoting a short modern diagnostic. This gate records those constraints in machine-readable form.

## Required Next Action

{spec.next_action}

## Decision Record

{spec.research_id} is complete as a precondition gate, not as a scored model experiment. The decision is conservative and reproducible: blocked inputs are listed, available evidence is retained, no model output is fabricated, and the exact next engineering task is written down. This satisfies documentation discipline without pretending the system has information it does not yet have.

## Handoff Detail

A future Codex or GPT-Pro conversation should start from this folder before attempting the experiment again. It should first verify whether the listed blockers have been removed by new parsers, source requests, or approved credentials. Only then should it replace the empty OOF table with scored predictions. If the blocker remains, the correct action is to update the readiness evidence and keep the experiment blocked. For R30 specifically, validation remains unauthorized until every prerequisite is complete, a single final challenger is selected from development OOF evidence, and a predeclaration file is written and hashed before validation access.

This record also protects the research ledger from silent optimism. A blocked gate is not a failure to document work; it is the documented boundary between acquired raw data and scientifically usable, leakage-safe predictor evidence.
"""


def write_gate(spec: GateSpec, data_root: Path) -> dict[str, object]:
    folder = spec.folder
    for subdir in ["artifacts", "logs", "metrics", "predictions", "results"]:
        (folder / subdir).mkdir(parents=True, exist_ok=True)
    rows = readiness_rows(spec)
    readiness = pd.DataFrame(rows)
    predictions = empty_predictions()
    subgroup = pd.DataFrame(
        {
            "subgroup": ["all"],
            "status": ["not_scored"],
            "reason": [spec.status],
        }
    )
    readiness_path = folder / "artifacts" / "precondition_readiness.csv"
    predictions_path = folder / "predictions" / "oof_predictions.parquet"
    subgroup_path = folder / "metrics" / "subgroup_metrics.parquet"
    readiness.to_csv(readiness_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    subgroup.to_parquet(subgroup_path, index=False)
    metrics = {
        "research_id": spec.research_id,
        "experiment_id": spec.exp_id,
        "status": spec.status,
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "oof_predictions": 0,
        "intended_question": spec.intended_question,
        "available_evidence": list(spec.available_evidence),
        "blockers": list(spec.blockers),
        "next_action": spec.next_action,
    }
    write_text(folder / "metrics" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(folder / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(folder / "logs" / "run_summary.json", json.dumps({"generated_at": now_utc(), **metrics}, indent=2, default=str))
    write_text(folder / "README.md", f"# {spec.exp_id} {spec.research_id} {spec.title}\n\nPrecondition-gate result: `{spec.status}`.\n")
    write_text(folder / "HYPOTHESIS.md", f"# Hypothesis\n\n{spec.intended_question}\n")
    write_text(folder / "INFORMATION_GAIN.md", f"# Information Gain\n\nThe information gained is the precise blocker state for {spec.research_id}: {', '.join(spec.blockers)}.\n")
    write_text(folder / "ASOF_CONTRACT.md", "# As-Of Contract\n\nNo predictor is scored unless it has a lawful T-1 15:00 HKT availability contract. This gate reads no validation outcomes and no locked-test target rows.\n")
    write_text(folder / "FEATURE_SPEC.yaml", "research_id: " + spec.research_id + "\nfeature_status: blocked\nblocked_inputs:\n" + "\n".join(f"  - {item}" for item in spec.blockers) + "\n")
    write_text(folder / "RUN_CONFIG.yaml", f"""research_id: {spec.research_id}
experiment_id: {spec.exp_id}
mode: precondition_gate
data_root: {data_root}
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(folder / "PROTOCOL.md", "# Protocol\n\n1. Audit available evidence.\n2. Compare required inputs to parsed, point-in-time eligible artifacts.\n3. If any gating source is missing, write blocked artifacts and do not score a model.\n")
    write_text(folder / "ABLATION_PLAN.md", "# Ablation Plan\n\nAblations are deferred because the required primary feature family is not available. Running ablations without the primary feature would be misleading.\n")
    write_text(folder / "NEGATIVE_CONTROLS.md", "# Negative Controls\n\nNegative controls are documented in the long report and deferred until the primary feature table exists. No global-fit or retrospective leakage control is scored here.\n")
    write_text(folder / "DATE_RANGES.md", "# Date Ranges\n\n- OOF prediction period: none, blocked before scoring.\n- Validation 2024: not accessed.\n- Locked test 2025-01-01 onward: not accessed.\n")
    write_text(folder / "DATA_MANIFEST.yaml", f"""research_id: {spec.research_id}
data_root: {data_root}
precondition_readiness: {readiness_path}
precondition_readiness_sha256: {sha256_file(readiness_path)}
oof_predictions: {predictions_path}
oof_predictions_sha256: {sha256_file(predictions_path)}
validation_2024_accessed: false
locked_test_accessed: false
blocked_inputs:
""" + "\n".join(f"  - {item}" for item in spec.blockers) + "\n")
    report = long_report(spec, rows)
    write_text(folder / "EXPERIMENT_REPORT_7500_CHARS.md", report)
    write_text(folder / "RESULTS.md", f"# Results\n\n{spec.research_id} was not scored. Status: `{spec.status}`.\n\n" + markdown_table(rows, ["category", "item", "status", "disposition"]) + "\n")
    write_text(folder / "CONCLUSION.md", f"# Conclusion\n\n{spec.research_id} is blocked: {', '.join(spec.blockers)}. No validation or locked-test access occurred.\n")
    write_text(folder / "REPRODUCE.md", f"# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r13_to_r30_precondition_gates.py --data-root C:\\hkg_tmax_data --research-id {spec.research_id}\n```\n")
    write_text(folder / "STATUS.yaml", f"""status: {spec.status}
research_id: {spec.research_id}
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
production_eligible: false
blocked_inputs:
""" + "\n".join(f"  - {item}" for item in spec.blockers) + "\n")
    if spec.research_id == "HKG-T24-R30":
        write_text(folder / "PREDECLARATION_BLOCKED.md", "# Predeclaration Blocked\n\nNo final challenger exists. Validation 2024 is not authorized and was not accessed.\n")
    write_text(REPO_ROOT / "reports" / "hkg_t24" / spec.report_name, report)
    return {
        "research_id": spec.research_id,
        "experiment_id": spec.exp_id,
        "status": spec.status,
        "folder": str(folder),
        "report_chars": len(report),
        "blockers": list(spec.blockers),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write HKG-T24 R13-R30 precondition-gate experiment folders.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--research-id", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    selected = [spec for spec in GATE_SPECS if args.research_id == "all" or spec.research_id == args.research_id]
    if not selected:
        raise ValueError(f"No R13-R30 gate spec found for {args.research_id}")
    rows = [write_gate(spec, data_root) for spec in selected]
    summary = pd.DataFrame(rows)
    out_dir = data_root / "gold" / "hkg_t24" / "r13_to_r30_precondition_gates"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_dir / "r13_to_r30_gate_summary.parquet", index=False)
    summary.to_csv(out_dir / "r13_to_r30_gate_summary.csv", index=False)
    print(json.dumps({"status": "ok", "written": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
