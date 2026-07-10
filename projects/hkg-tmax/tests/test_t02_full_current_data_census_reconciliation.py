from __future__ import annotations

import csv
from pathlib import Path

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
MIGRATION = (
    PROJECT_PATHS.db_root
    / "migrations/postgres/20260624_0003_t02_census_registry_compatibility.sql"
)
SCRIPT = REPO_ROOT / "scripts/run_t02_full_current_data_census_reconciliation.py"


def test_t02_migration_declares_required_registry_aliases() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CREATE OR REPLACE VIEW catalog.source_registry AS" in sql
    assert "FROM catalog.source_file_registry" in sql
    assert "CREATE OR REPLACE VIEW governance.attribute_contract AS" in sql
    assert "FROM catalog.attribute_contract" in sql
    assert "20260624_0003_t02_census_registry_compatibility" in sql


def test_t02_generator_declares_mandatory_output_artifacts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for artifact in [
        "source_eligibility_matrix.csv",
        "table_reconciliation.csv",
        "attribute_reconciliation.csv",
        "station_reconciliation.csv",
        "experiment_evidence_linkage.csv",
        "updated_quality_blockers.csv",
        "expected_count_reconciliation.csv",
        "unmapped_object_zero_check.csv",
        "duplicate_physical_representation_check.csv",
        "handoff_manifest.json",
    ]:
        assert artifact in source


def test_t02_generated_artifacts_preserve_required_schema_if_present() -> None:
    experiment_dir = REPO_ROOT / "experiments/0209_full_current_data_census_reconciliation"
    if not experiment_dir.exists():
        return

    expected_headers = {
        "source_eligibility_matrix.csv": {
            "source_key",
            "source_kind",
            "disposition",
            "strict_feature_eligible",
            "availability_proof",
            "blocker",
        },
        "table_reconciliation.csv": {
            "actual_source_key",
            "object_kind",
            "disposition",
            "status",
        },
        "attribute_reconciliation.csv": {
            "attribute_key",
            "contract_object",
            "reconciliation_status",
            "rationale",
        },
        "experiment_evidence_linkage.csv": {
            "experiment_key",
            "canonical_live_feature_source",
            "disposition",
        },
    }
    for name, required_columns in expected_headers.items():
        with (experiment_dir / name).open(newline="", encoding="utf-8") as handle:
            header = set(next(csv.reader(handle)))
        assert required_columns <= header
