from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_hkg_t24_0105_0183_beastmode_roadmap import parse_roadmap, required_artifact_names

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments"
ROADMAP_GLOBAL_ARTIFACT_DIR = (
    EXPERIMENT_ROOT / "0000_research_state_and_data_contract" / "r0105_0183"
)
EXTERNAL_HF_ROOT = Path(r"C:\hkg_tmax_data")


REQUIRED_ROOT_FILES = {
    "README.md",
    "HYPOTHESIS.md",
    "PROTOCOL.md",
    "ASOF_CONTRACT.md",
    "DATA_MANIFEST.yaml",
    "RUN_CONFIG.yaml",
    "RESULTS.md",
    "CONCLUSION.md",
    "REPRODUCE.md",
    "STATUS.yaml",
    "FEATURE_SPEC.yaml",
    "INFORMATION_GAIN.md",
    "NEGATIVE_CONTROLS.md",
    "ABLATION_PLAN.md",
    "DATE_RANGES.md",
    "leakage_audit.md",
    "run.py",
}

REQUIRED_OUTPUT_FILES = {
    "artifacts/feature_diagnostics.csv",
    "artifacts/input_hashes.csv",
    "metrics/subgroup_metrics.csv",
    "metrics/subgroup_metrics.parquet",
    "results/metrics.json",
    "results/predictions.parquet",
    "results/decision.json",
    "results/fold_metrics.csv",
    "results/feature_diagnostics.csv",
    "results/subgroup_metrics.csv",
    "results/top_50_error_cases.csv",
    "predictions/oof_predictions.parquet",
    "logs/.gitkeep",
}

REQUIRED_COMMON_CONTRACT_FILES = {
    "summary.json",
    "data_range.csv",
    "input_hashes.json",
    "feature_definitions.csv",
    "feature_eligibility.csv",
    "scoreboard.csv",
    "fold_metrics.csv",
    "year_stability.csv",
    "season_stability.csv",
    "source_stability.csv",
    "high_error_tail.csv",
    "negative_results.md",
    "next_recommendation.md",
}


def test_roadmap_parser_extracts_exact_0105_to_0183_sequence() -> None:
    _, specs = parse_roadmap()

    assert len(specs) == 79
    assert [spec.experiment_id for spec in specs] == [f"{number:04d}" for number in range(105, 184)]
    assert specs[0].folder_name == "0105_station_dossier_geography_coverage"
    assert specs[-1].folder_name == "0183_hf_teacher_long_history_distillation"


def test_generated_0105_to_0183_folders_have_clean_required_layout() -> None:
    _, specs = parse_roadmap()

    for spec in specs:
        folder = EXPERIMENT_ROOT / spec.folder_name
        assert folder.is_dir(), f"missing folder for {spec.experiment_id}: {folder}"
        for relative in REQUIRED_ROOT_FILES | REQUIRED_OUTPUT_FILES:
            assert (folder / relative).exists(), f"missing {relative} in {spec.folder_name}"

        metrics = json.loads((folder / "results" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["experiment_id"] == spec.experiment_id
        assert metrics["uses_2024_plus_target_rows"] is False

        predictions = pd.read_parquet(folder / "results" / "predictions.parquet", columns=["target_date"])
        assert not predictions.empty
        assert pd.to_datetime(predictions["target_date"]).max() < pd.Timestamp("2024-01-01")


def test_generated_0105_to_0183_folders_materialize_roadmap_named_artifacts() -> None:
    _, specs = parse_roadmap()

    for spec in specs:
        folder = EXPERIMENT_ROOT / spec.folder_name
        for relative in REQUIRED_COMMON_CONTRACT_FILES:
            assert (folder / relative).exists(), f"missing common contract artifact {relative} in {spec.folder_name}"

        required_names = required_artifact_names(spec)
        fidelity_path = folder / "artifacts" / "roadmap_required_artifact_fidelity.csv"
        assert fidelity_path.exists(), f"missing roadmap fidelity ledger in {spec.folder_name}"
        fidelity = pd.read_csv(fidelity_path)
        assert len(fidelity) == len(required_names)
        assert fidelity["required_artifact"].tolist() == required_names

        for artifact_name in required_names:
            artifact_path = folder / "artifacts" / artifact_name
            if Path(artifact_name).suffix:
                assert artifact_path.is_file(), f"missing roadmap artifact {artifact_name} in {spec.folder_name}"
            else:
                assert artifact_path.is_dir(), f"missing roadmap directory artifact {artifact_name} in {spec.folder_name}"
                assert (artifact_path / "README.md").exists()


def test_roadmap_execution_ledger_has_all_rows_and_no_production_promotions() -> None:
    ledger_path = ROADMAP_GLOBAL_ARTIFACT_DIR / "roadmap_0105_0183_execution_ledger.csv"
    assert ledger_path.exists()

    ledger = pd.read_csv(ledger_path, dtype={"experiment_id": str})
    assert len(ledger) == 79
    assert ledger["experiment_id"].tolist() == [f"{number:04d}" for number in range(105, 184)]
    assert not ledger["status"].astype(str).str.startswith("ACCEPTED").any()
    assert set(ledger["status"]) <= {
        "COMPLETE_AUDIT",
        "COMPLETE_RESEARCH_LIFT_NOT_PRODUCTION_PROMOTED",
        "BLOCKED_DATA_UNAVAILABLE",
        "COMPLETE_HF_DIAGNOSTIC_NOT_PRODUCTION_PROMOTED",
    }

    blocked = ledger[ledger["status"].eq("BLOCKED_DATA_UNAVAILABLE")]
    if EXTERNAL_HF_ROOT.exists():
        assert blocked.empty
    else:
        assert blocked["experiment_id"].tolist() == ["0179", "0180", "0181", "0182", "0183"]
    for experiment_id in blocked["experiment_id"]:
        folder = next(spec.folder_name for spec in parse_roadmap()[1] if spec.experiment_id == experiment_id)
        assert (EXPERIMENT_ROOT / folder / "artifacts" / "blocker_evidence.csv").exists()


def test_high_frequency_0179_to_0183_use_external_source_when_available() -> None:
    if not EXTERNAL_HF_ROOT.exists():
        return
    _, specs = parse_roadmap()
    spec_by_id = {spec.experiment_id: spec for spec in specs}
    ledger = pd.read_csv(
        ROADMAP_GLOBAL_ARTIFACT_DIR / "roadmap_0105_0183_execution_ledger.csv",
        dtype={"experiment_id": str},
    )
    hf_rows = ledger[ledger["experiment_id"].isin(["0179", "0180", "0181", "0182", "0183"])]
    assert hf_rows["status"].eq("COMPLETE_HF_DIAGNOSTIC_NOT_PRODUCTION_PROMOTED").all()
    assert hf_rows["feature_count"].astype(int).gt(0).all()
    assert hf_rows["rows"].astype(int).ge(300).all()

    for experiment_id in ["0179", "0180", "0181", "0182", "0183"]:
        folder = EXPERIMENT_ROOT / spec_by_id[experiment_id].folder_name
        predictions = pd.read_parquet(folder / "results" / "predictions.parquet")
        assert set(predictions["model_family"]) == {"high_frequency_leave_year_diagnostic"}
        assert predictions["feature_count"].max() > 0
        assert pd.to_datetime(predictions["target_date"]).max() < pd.Timestamp("2024-01-01")
        assert not (folder / "artifacts" / "blocker_evidence.csv").exists()

        input_hashes = json.loads((folder / "input_hashes.json").read_text(encoding="utf-8"))
        source_ids = {row["source_id"] for row in input_hashes}
        assert any(source_id.startswith("external_hf_") for source_id in source_ids)
