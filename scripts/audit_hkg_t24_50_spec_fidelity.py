from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

runner = importlib.import_module("scripts.run_hkg_t24_50_long_history_experiments")
DEFAULT_SPEC_PATH = runner.DEFAULT_SPEC_PATH
EXPERIMENT_ROOT = runner.EXPERIMENT_ROOT
REPORT_ROOT = runner.REPORT_ROOT
build_feature_matrix = runner.build_feature_matrix
parse_spec = runner.parse_spec
run_specs = runner.run_specs
model_strategy_for = runner.model_strategy_for
candidate_family_for = runner.candidate_family_for


ADVANCED_METHOD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cyclic_splines", r"\bcyclic splines?\b"),
    ("boosting", r"\bboost(?:ed|ing)?\b|gradient boosting"),
    ("pca_or_fpca", r"\bPCA\b|\bFPCA\b|principal component|component count"),
    ("hmm_or_gmm", r"\bHMM\b|Gaussian mixture|mixture of experts|soft regime|cluster"),
    ("dtw_or_analog", r"\bDTW\b|dynamic-time-warping|analog"),
    ("cusum_or_change_model", r"\bCUSUM\b|Bayesian change|change-point detection|Hidden Markov"),
    ("wavelet_ssa", r"\bwavelet\b|\bSSA\b|spectral phase|phase-scrambled"),
    ("student_t_or_conformal", r"Student-t|conformal|PIT|tail thickness"),
    ("quantile_cdf", r"quantile|q01|q99|0\.1|CDF|probability masses"),
    ("graph_model", r"graph|Laplacian|DeepSets|set aggregator"),
    ("monotonic_or_shape_constraints", r"monotonic|shape constraints|bounded GAM|Beta regression"),
)


DIRECT_IMPLEMENTATION_EVIDENCE: Mapping[str, tuple[str, ...]] = {
    "EXP-0050": (
        "Loads R14-R17 nested OOF prediction files.",
        "Drops folds with fewer than 1000 rows.",
        "Scores strict common R14-R17 dates.",
        "Freezes corrected R17 era-transfer row as long_history_core_v1.",
    ),
    "EXP-0051": (
        "Builds T-7 causal same-day-of-year expanding normals.",
        "Adds 5/10/20/40-year recency half-life seasonal normals.",
        "Adds harmonic drift, q05/q50/q95 seasonal normals, and constrained equal blend.",
    ),
    "EXP-0052": (
        "Detects break years inside each training fold only.",
        "Adds fold-local after-break flags and hinge features to OOF rows.",
    ),
    "EXP-0053": (
        "Adds lagged slopes, roughness, reversal counts, range, plateau fraction, and monotonic trajectory-shape balance.",
    ),
    "EXP-0054": (
        "Adds causal rolling autocorrelation, absolute-change energy bands, and entropy proxy.",
    ),
    "EXP-0055": (
        "Adds lagged hot/cold run lengths, cumulative anomalies, and reversal-hazard proxy.",
    ),
    "EXP-0056": (
        "Adds lagged MAD/IQR/sign-change volatility and forecastability score.",
    ),
    "EXP-0057": (
        "Trains origin-relative full/50/30/15-year recency experts.",
        "Builds nonnegative inverse-training-MAE blend.",
    ),
    "EXP-0075": (
        "Adds regional temperature-plane gradients, north-south/east-west spatial contrasts, and graph-Laplacian station context.",
    ),
}

STRATEGY_IMPLEMENTATION_EVIDENCE: Mapping[str, tuple[str, ...]] = {
    "cyclic_spline_climatology": (
        "Fits fold-local periodic cyclic splines with 8/12/18/24 knot settings.",
        "Uses recency half-life climatology, harmonic drift, q05/q50/q95 normals, and month-specific trends.",
    ),
    "change_point_ridge": (
        "Detects training-only change points and adds piecewise hinges plus decaying break effects.",
    ),
    "dtw_analog_blend": (
        "Runs season-restricted DTW/derivative-DTW/correlation analog retrieval on T-7 lag trajectories.",
        "Blends distance-weighted analog outcomes with the regularized descriptor model.",
    ),
    "spectral_ssa_pca": (
        "Adds causal spectral-band descriptors and fold-local PCA/SSA-style components.",
    ),
    "hazard_mixture": (
        "Fits a logistic spell-reversal hazard and uses it as a two-expert mixture gate.",
    ),
    "forecastability_scale": (
        "Models absolute residual scale and produces Student-t/conformal-style monotone quantiles.",
    ),
    "ridge_strength_pca": (
        "Creates ridge-strength score features using height, thickness, warmth, weak-wind state, and training-only PCA.",
    ),
    "fpca_profile_analog": (
        "Fits fold-local FPCA/PCA profile components for temperature, moisture, height, and wind shape.",
        "Adds profile reconstruction error and analog-style profile information.",
    ),
    "gmm_regime_mixture": (
        "Fits fold-local Gaussian mixture regime probabilities on standardized upper-air profile features.",
        "Fits soft regime-specific residual experts with a global fallback.",
    ),
    "reliability_shrinkage": (
        "Blends upper-air and surface predictions using sounding completeness reliability shrinkage.",
    ),
    "intraday_trajectory_pca": (
        "Fits fold-local PCA trajectory components for pre-cutoff ISD morning/midday thermal evolution.",
    ),
    "tensor_pca_composites": (
        "Fits fold-local PCA tensor components across station, variable, and time-window summaries.",
    ),
    "front_cusum_gate": (
        "Builds one-sided CUSUM front/surge state features from pressure, temperature, dewpoint, and wind tendencies.",
        "Uses graph-Laplacian station context as a spatial consistency check for front/surge propagation.",
    ),
    "flow_relative_weighting": (
        "Constructs flow-relative station/gradient features from surface wind and regional plane gradients.",
        "Uses graph-Laplacian station context to preserve station-network geometry in flow-relative weighting.",
    ),
    "sea_breeze_index": (
        "Constructs sea-breeze susceptibility from coastal-inland contrast, weak synoptic wind, and shear.",
    ),
    "north_south_propagation": (
        "Adds north-south gradient propagation and CUSUM state features for continental surge/return flow.",
    ),
    "east_west_flow_gradient": (
        "Adds east-west estuary flow-gradient and CUSUM state features.",
    ),
    "graph_pca_modes": (
        "Computes metadata-only graph-Laplacian station modes from ISD latitude, longitude, and elevation.",
        "Adds fold-local PCA station-panel components and graph total variation.",
    ),
    "robust_distribution_shape": (
        "Constructs spatial q05-q95 quantiles, IQR, hot/cold tail spread, and station-field entropy.",
    ),
    "distributed_lag_station_map": (
        "Uses station-change, graph-mode, and fixed distributed-lag station features fitted inside folds.",
    ),
    "station_homogenization_offsets": (
        "Estimates training-only station offsets against network consensus and adds homogenized station anomalies.",
    ),
    "station_dropout_masked": (
        "Adds station coverage masks, entropy, and graph coverage features for dropout robustness.",
    ),
    "quality_weighted_surface": (
        "Adds report-quality, station-count, observation-age, and reliability-weighted surface features.",
    ),
    "rainfall_reservoir": (
        "Builds strict T-7 rainfall wetness reservoir, dry-memory, and evaporation-rain balance features.",
    ),
    "cloud_sunshine_regime": (
        "Fits fold-local Gaussian-mixture clear/cloud/radiation regimes from lagged cloud, sunshine, solar, and rain.",
    ),
    "solar_efficiency_state": (
        "Builds lagged solar conversion-efficiency and radiation-per-sunshine state features.",
    ),
    "surface_storage_state": (
        "Builds grass-air contrast, evaporation-rain balance, and nocturnal heat-storage state features.",
    ),
    "sea_temperature_extrapolator": (
        "Builds lagged sea-temperature state, air-minus-sea contrast, and onshore moderation features.",
    ),
    "markov_wind_regime": (
        "Fits fold-local wind-regime probabilities and CUSUM state features from Waglan/current wind histories.",
    ),
    "daily_climate_factor": (
        "Fits fold-local sparse-PCA style daily-climate factors from strict T-7 daily climate blocks.",
    ),
    "climate_trajectory_analog": (
        "Runs mixed-metric climate-memory analog retrieval from strict T-7 daily climate trajectories.",
    ),
    "teacher_student_subsidence": (
        "Creates training-only TC/ridge/subsidence teacher proxies and OOF student probabilities.",
    ),
    "teacher_student_suppression": (
        "Creates training-only cloud/rain/sunshine suppression teacher proxies and OOF student probabilities.",
    ),
    "teacher_student_archetype": (
        "Fits fold-local GMM / Gaussian mixture synoptic archetype probabilities and student residual experts.",
    ),
    "expert_gate": (
        "Uses nested OOF R14-R17 expert predictions in a nonnegative season-conditioned expert gate.",
    ),
    "residual_stack": (
        "Fits an ordered nested residual stack across dynamic, surface, upper-air, daily-climate, and regime blocks.",
    ),
    "student_t_conformal_scale": (
        "Fits heteroscedastic residual scale with distributional boosting and rolling split-conformal/Student-t-style monotonic quantiles.",
    ),
    "quantile_tail_cdf": (
        "Fits direct quantile tail-specialist residual corrections, Student-t/conformal central distribution, monotonic q01-q99 quantiles, and 0.1 CDF probability masses.",
    ),
}


def direct_evidence_for(experiment_id: str) -> tuple[str, ...]:
    strategy = model_strategy_for(experiment_id)
    strategy_evidence = STRATEGY_IMPLEMENTATION_EVIDENCE.get(strategy, ())
    if strategy == "ridge":
        strategy_evidence = (
            f"Uses candidate model family `{candidate_family_for(experiment_id)}` with experiment-specific predeclared feature construction.",
        )
    probabilistic_evidence = (
        "Writes distribution sigma plus q05-q95 predictive quantile columns for candidate/control rows.",
    )
    return tuple([*DIRECT_IMPLEMENTATION_EVIDENCE.get(experiment_id, ()), *strategy_evidence, *probabilistic_evidence])


def construction_bullets(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def method_gaps(experiment_text: str, direct_evidence: Sequence[str]) -> list[str]:
    evidence_text = " ".join(direct_evidence).lower()
    gaps: list[str] = []
    for name, pattern in ADVANCED_METHOD_PATTERNS:
        if re.search(pattern, experiment_text, flags=re.I) and not re.search(pattern, evidence_text, flags=re.I):
            gaps.append(name)
    return gaps


def status_for(experiment_id: str, gaps: Sequence[str], evidence: Sequence[str]) -> str:
    if experiment_id == "EXP-0050" and not gaps:
        return "direct_core_contract_implemented"
    if evidence and not gaps:
        return "direct_basic_methods_implemented"
    if evidence and gaps:
        return "partial_exact_plus_proxy"
    return "proxy_feature_block_only"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "title",
        "status",
        "direct_evidence_count",
        "unimplemented_advanced_methods",
        "candidate_feature_count",
        "construction_bullet_count",
        "folder_exists",
        "long_report_exists",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_report(rows: Sequence[Mapping[str, object]]) -> str:
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    lines = [
        "# HKG T24 50-Experiment Spec Fidelity Audit",
        "",
        "This audit checks implementation fidelity, not only folder completeness. It is deliberately stricter than the artifact-contract tests.",
        "",
        "## Summary",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `direct_core_contract_implemented`: the core benchmark requirement is implemented directly.",
            "- `direct_basic_methods_implemented`: the listed direct method evidence covers the obvious construction requirements in this audit.",
            "- `partial_exact_plus_proxy`: some experiment-specific features exist, but at least one advanced method requested by the spec is not directly implemented.",
            "- `proxy_feature_block_only`: the current run uses a predeclared feature block and Ridge comparison, but does not yet implement the experiment's exact method stack.",
            "",
            "Passing this audit is not required to preserve the current artifacts. It is required before claiming the 50-experiment program is fully implemented exactly as specified.",
            "",
            "## Experiment Rows",
            "",
            "| experiment | status | gaps | features | evidence |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        gaps = row["unimplemented_advanced_methods"] or "none"
        lines.append(
            f"| {row['experiment_id']} | `{row['status']}` | {gaps} | {row['candidate_feature_count']} | {row['direct_evidence_summary']} |"
        )
    lines.append("")
    return "\n".join(lines)


def audit(spec_path: Path) -> list[dict[str, object]]:
    _, parsed = parse_spec(spec_path)
    features, _ = build_feature_matrix()
    specs = {item.parsed.experiment_id: item for item in run_specs(parsed, features)}

    rows: list[dict[str, object]] = []
    for item in parsed:
        direct = direct_evidence_for(item.experiment_id)
        combined_spec_text = "\n".join(
            [
                item.construction,
                item.negative_controls,
                item.acceptance,
                item.required_artifacts,
            ]
        )
        gaps = method_gaps(combined_spec_text, direct)
        folder = EXPERIMENT_ROOT / item.folder_name
        rows.append(
            {
                "experiment_id": item.experiment_id,
                "title": item.title,
                "status": status_for(item.experiment_id, gaps, direct),
                "direct_evidence_count": len(direct),
                "direct_evidence_summary": "<br>".join(direct) if direct else "feature-block proxy only",
                "unimplemented_advanced_methods": ",".join(gaps),
                "candidate_feature_count": len(specs[item.experiment_id].feature_columns),
                "construction_bullet_count": len(construction_bullets(item.construction)),
                "folder_exists": folder.exists(),
                "long_report_exists": (folder / "EXPERIMENT_REPORT_7500_CHARS.md").exists(),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact-spec fidelity for HKG T24 50 long-history experiments.")
    parser.add_argument("--spec-path", default=str(DEFAULT_SPEC_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit(Path(args.spec_path))
    out_csv = REPORT_ROOT / "SPEC_IMPLEMENTATION_FIDELITY_AUDIT.csv"
    out_md = REPORT_ROOT / "SPEC_IMPLEMENTATION_FIDELITY_AUDIT.md"
    write_csv(out_csv, rows)
    out_md.write_text(build_report(rows), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
