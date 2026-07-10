from __future__ import annotations

import csv
from pathlib import Path

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
SCRIPT = REPO_ROOT / "scripts/run_t03_t05_foundation_tasks.py"
STATUS_SCRIPT = REPO_ROOT / "scripts/check_t03_t05_status.py"
MIGRATION_T03 = (
    PROJECT_PATHS.db_root / "migrations/postgres/20260624_0004_t03_gribstream_catalog_registry.sql"
)
MIGRATION_T04 = PROJECT_PATHS.db_root / "migrations/postgres/20260624_0005_t04_nwp_storage_lineage.sql"
MIGRATION_T05 = (
    PROJECT_PATHS.db_root / "migrations/postgres/20260624_0006_t05_location_station_geospatial_registry.sql"
)


def test_t03_migration_declares_catalog_license_and_selector_objects() -> None:
    sql = MIGRATION_T03.read_text(encoding="utf-8")

    for object_name in [
        "catalog.catalog_snapshot",
        "catalog.model_registry",
        "catalog.selector_snapshot",
        "catalog.source_license",
        "governance.gribstream_usage_constraint",
    ]:
        assert object_name in sql
    assert "20260624_0004_t03_gribstream_catalog_registry" in sql


def test_t04_migration_declares_nwp_storage_lineage_boundaries() -> None:
    sql = MIGRATION_T04.read_text(encoding="utf-8")

    for object_name in [
        "raw_audit.acquisition_request",
        "raw_audit.response_object",
        "nwp_core.model_run",
        "nwp_core.point_value",
        "feature_store.target_snapshot_manifest",
        "research.expert_oof_prediction",
        "live.issued_forecast",
        "quarantine.rejected_payload",
    ]:
        assert object_name in sql
    assert "PARTITION BY RANGE (valid_time_utc)" in sql
    assert "point_value_default" in sql
    assert "request_sha256 char(64) NOT NULL UNIQUE" in sql


def test_t05_migration_declares_station_location_registry() -> None:
    sql = MIGRATION_T05.read_text(encoding="utf-8")

    for object_name in [
        "catalog.location",
        "catalog.station",
        "catalog.station_metadata_history",
        "catalog.location_group",
        "catalog.location_group_member",
    ]:
        assert object_name in sql
    assert "idx_location_lat_lon" in sql
    assert "20260624_0006_t05_location_station_geospatial_registry" in sql


def test_runner_declares_mandatory_task_artifacts_and_secret_redaction() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for artifact in [
        "catalog_snapshot.json",
        "selector_map.csv",
        "coverage_probe_results.csv",
        "quota_storage_estimate.csv",
        "licence_register.md",
        "final_model_disposition.csv",
        "schema_diagram.md",
        "index_partition_plan.md",
        "rollback_plan.md",
        "migration_test_log.md",
        "location_registry.csv",
        "station_dossier_complete.csv",
        "location_groups.csv",
        "local_domain.geojson",
        "synoptic_domain.geojson",
        "unresolved_station_blockers.csv",
    ]:
        assert artifact in source
    assert "[REDACTED_GRIBSTREAM_API_KEY]" in source
    assert "GRIBSTREAM_API_KEY=" in source
    assert "Retry-After" in source
    assert "api-min-interval-seconds" in source
    assert "gribstream_api_events.jsonl" in source
    assert "blocked_rate_limit_safety_stop" in source
    assert "reuse-existing-coverage-probes" in source
    assert "reuse-existing-t03-artifacts" in source
    assert "def load_existing_t03_artifacts(" in source
    assert "def copy_file(" in source


def test_status_checker_reports_background_progress_files() -> None:
    source = STATUS_SCRIPT.read_text(encoding="utf-8")

    assert "t03_t05_background_status.json" in source
    assert "coverage_probe_results.csv" in source
    assert "gribstream_api_events.jsonl" in source
    assert "coverage_probe_status_counts" in source


def test_generated_artifact_headers_if_present() -> None:
    expected_headers = {
        "experiments/0210_gribstream_catalog_coverage_licence_quota_audit/final_model_disposition.csv": {
            "model_code",
            "source_matrix_disposition",
            "coverage_probe_status",
            "final_status",
        },
        "experiments/0210_gribstream_catalog_coverage_licence_quota_audit/selector_map.csv": {
            "model_code",
            "semantic_variable",
            "selector_status",
            "native_name",
            "blocker",
        },
        "experiments/0210_gribstream_catalog_coverage_licence_quota_audit/logs/manual_gribstream_3_request_unblock_probe.csv": {
            "http_status",
            "row_count",
            "retry_after",
        },
        "experiments/0211_nwp_database_object_storage_migrations/t04_db_object_verification.csv": {
            "object_name",
            "exists",
            "status",
        },
        "experiments/0212_canonical_location_station_geospatial_registry/location_registry.csv": {
            "location_code",
            "latitude",
            "longitude",
            "metadata_sha256",
        },
        "experiments/0212_canonical_location_station_geospatial_registry/station_dossier_complete.csv": {
            "station_code",
            "station_name",
            "network",
            "metadata_status",
        },
    }

    for relative_path, required in expected_headers.items():
        path = REPO_ROOT / relative_path
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            header = set(next(csv.reader(handle)))
        assert required <= header
