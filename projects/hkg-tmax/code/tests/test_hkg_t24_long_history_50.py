from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_hkg_t24_50_long_history_experiments import (
    BUNDLE_ZIP_PATH,
    DEFAULT_SPEC_PATH,
    EXPERIMENT_ROOT,
    REPORT_ROOT,
    candidate_family_for,
    columns_matching,
    model_strategy_for,
    parse_spec,
    strict_confirmation_guard,
)


def test_long_history_50_spec_parses_exact_experiment_sequence() -> None:
    if not DEFAULT_SPEC_PATH.exists():
        pytest.skip(f"Spec file not present: {DEFAULT_SPEC_PATH}")

    _, parsed = parse_spec(DEFAULT_SPEC_PATH)

    assert len(parsed) == 50
    assert [item.experiment_id for item in parsed] == [f"EXP-{number:04d}" for number in range(50, 100)]
    assert parsed[0].title == "Corrected Common-Row Benchmark and Fold-Geometry Repair"
    assert parsed[-1].title.startswith("Tail-Specialist Quantile Mixture")


def test_strict_confirmation_guard_blocks_2024_and_allows_2023() -> None:
    strict_confirmation_guard([pd.Timestamp("2023-12-31")], context="unit-test allowed")

    with pytest.raises(RuntimeError, match="confirmation dates"):
        strict_confirmation_guard([pd.Timestamp("2024-01-01")], context="unit-test blocked")


def test_columns_matching_never_selects_target_label_as_feature() -> None:
    frame = pd.DataFrame(
        {
            "target_tmax_c": [28.0],
            "target_date": [pd.Timestamp("2020-01-01")],
            "target_lag7_tmax_c": [27.0],
            "target_roll30_mean_lag7_c": [26.5],
        }
    )

    selected = columns_matching(frame, "target_")

    assert "target_tmax_c" not in selected
    assert "target_date" not in selected
    assert "target_lag7_tmax_c" in selected


def test_generated_exp_0050_to_0099_have_required_artifacts_when_present() -> None:
    if not (EXPERIMENT_ROOT / "EXP-0050").exists():
        pytest.skip("EXP-0050..0099 have not been generated in this checkout")

    required = {
        "README.md",
        "HYPOTHESIS.md",
        "INFORMATION_GAIN.md",
        "ASOF_CONTRACT.md",
        "DATA_MANIFEST.yaml",
        "DATE_RANGES.md",
        "EXPERIMENT_REPORT_7500_CHARS.md",
        "FEATURE_SPEC.yaml",
        "PROTOCOL.md",
        "RUN_CONFIG.yaml",
        "NEGATIVE_CONTROLS.md",
        "ABLATION_PLAN.md",
        "STATUS.yaml",
        "RESULTS.md",
        "CONCLUSION.md",
        "REPRODUCE.md",
        "predictions/oof_predictions.parquet",
        "artifacts/feature_diagnostics.csv",
        "artifacts/fold_score_deltas.csv",
        "artifacts/subgroup_metrics.csv",
        "metrics/subgroup_metrics.parquet",
        "results/metrics.json",
        "results/predictions.parquet",
        "results/headline_oof_2020_2023_predictions.parquet",
        "results/decision.json",
        "results/feature_diagnostics.csv",
        "results/fold_metrics.csv",
        "results/subgroup_metrics.csv",
        "results/top_50_error_cases.csv",
    }

    for number in range(50, 100):
        folder = EXPERIMENT_ROOT / f"EXP-{number:04d}"
        assert folder.exists()
        for relative in required:
            assert (folder / Path(relative)).exists(), f"missing {relative} in {folder.name}"
        report = (folder / "EXPERIMENT_REPORT_7500_CHARS.md").read_text(encoding="utf-8")
        assert len(report) >= 7500, f"long-form report too short in {folder.name}: {len(report)} chars"
        assert f"EXP-{number:04d}" in report
        assert "## Data Used And Date Ranges" in report
        assert "## Leakage Review" in report
        assert "2024, 2025, or 2026 confirmation rows" in report


def test_generated_experiments_persist_strategy_metadata_and_keep_confirmation_sealed() -> None:
    if not (EXPERIMENT_ROOT / "EXP-0050").exists():
        pytest.skip("EXP-0050..0099 have not been generated in this checkout")

    for number in range(50, 100):
        experiment_id = f"EXP-{number:04d}"
        folder = EXPERIMENT_ROOT / experiment_id
        feature_spec = (folder / "FEATURE_SPEC.yaml").read_text(encoding="utf-8")
        run_config = (folder / "RUN_CONFIG.yaml").read_text(encoding="utf-8")
        assert f"candidate_strategy: {model_strategy_for(experiment_id)}" in feature_spec
        assert f"candidate_model_family: {candidate_family_for(experiment_id)}" in feature_spec
        assert f"candidate_strategy: {model_strategy_for(experiment_id)}" in run_config
        predictions = pd.read_parquet(folder / "results" / "predictions.parquet")
        assert pd.to_datetime(predictions["target_date"]).max() < pd.Timestamp("2024-01-01")


def test_exp0099_has_direct_tail_quantiles_and_0p1_cdf_probability_grid() -> None:
    folder = EXPERIMENT_ROOT / "EXP-0099"
    if not folder.exists():
        pytest.skip("EXP-0099 has not been generated in this checkout")

    predictions = pd.read_parquet(folder / "results" / "predictions.parquet")
    candidate = predictions[predictions["model_id"].eq("exp-0099_candidate")]
    assert not candidate.empty
    for column in ("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"):
        assert column in candidate.columns
        assert candidate[column].notna().all()
    ordered = candidate[["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]]
    assert (ordered.diff(axis=1).iloc[:, 1:] >= -1e-9).all().all()

    cdf_path = folder / "results" / "cdf_0p1_probability_grid.parquet"
    assert cdf_path.exists()
    cdf = pd.read_parquet(cdf_path)
    assert cdf["target_date"].nunique() == candidate[candidate["headline_oof"]]["target_date"].nunique()
    assert (cdf["temperature_bin_c"].round(1) == cdf["temperature_bin_c"]).all()
    mass_sums = cdf.groupby("target_date", observed=True)["probability_mass"].sum()
    assert ((mass_sums - 1.0).abs() < 1e-6).all()


def test_spec_fidelity_audit_has_no_partial_or_proxy_rows_when_generated() -> None:
    audit_csv = REPORT_ROOT / "SPEC_IMPLEMENTATION_FIDELITY_AUDIT.csv"
    if not audit_csv.exists():
        pytest.skip("spec fidelity audit has not been generated in this checkout")

    audit = pd.read_csv(audit_csv)
    assert set(audit["status"]) <= {"direct_core_contract_implemented", "direct_basic_methods_implemented"}
    assert audit["unimplemented_advanced_methods"].fillna("").eq("").all()


def test_final_long_history_report_deliverables_exist_when_generated() -> None:
    if not REPORT_ROOT.exists():
        pytest.skip("long-history reports have not been generated in this checkout")

    required = {
        "VERIFIED_BUNDLE_STATE.md",
        "BUNDLE_FILE_INVENTORY.csv",
        "DATASET_ELIGIBILITY_MATRIX.csv",
        "EXPERIMENT_LEDGER.csv",
        "COMMON_OOF_SCOREBOARD.csv",
        "FAMILY_FDR_AND_STABILITY.csv",
        "RESIDUAL_COMPLEMENTARITY.parquet",
        "MECHANISM_FINDINGS.md",
        "NULL_AND_REJECTED_FINDINGS.md",
        "FAILURE_TAXONOMY.md",
        "SYNTHESIS_AND_FINAL_CHALLENGER.md",
        "CONFIRMATION_LOCK_STATUS.json",
        "SPEC_IMPLEMENTATION_FIDELITY_AUDIT.md",
        "SPEC_IMPLEMENTATION_FIDELITY_AUDIT.csv",
    }
    missing = [relative for relative in sorted(required) if not (REPORT_ROOT / relative).exists()]
    assert not missing

    inventory = pd.read_csv(REPORT_ROOT / "BUNDLE_FILE_INVENTORY.csv")
    if BUNDLE_ZIP_PATH.exists():
        assert len(inventory) == 173
        assert int(inventory["uncompressed_size"].sum()) == 169_382_005
        assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
