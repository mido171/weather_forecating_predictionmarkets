from __future__ import annotations

from datetime import UTC, date, timedelta
from pathlib import Path

from hkg_tmax.paths import ProjectPaths
from hkg_tmax_db.contracts import validate_audit_bundle
from hkg_tmax_db.cutoff import hkg_t24_cutoff_utc
from hkg_tmax_db.reconciliation import reconcile_sources, table_name_for_source

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
AUDIT_ROOT = (
    REPO_ROOT
    / "docs/archive/legacy-data-scaffold/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT"
)
DATASETS_ROOT = PROJECT_PATHS.data_root / "datasets"
PROFILE_JSON = DATASETS_ROOT / "DATASET_ATTRIBUTE_VALUE_PROFILE_FOR_GPT_PRO.json"
MIGRATION = PROJECT_PATHS.db_root / "migrations/postgres/20260623_0001_audit_driven_ingestion.sql"


def test_hkg_t24_cutoff_is_1500_hkt_t_minus_1_as_0700_utc() -> None:
    cutoff = hkg_t24_cutoff_utc(date(2026, 6, 23))

    assert cutoff.tzinfo == UTC
    assert cutoff.isoformat() == "2026-06-22T07:00:00+00:00"


def test_hong_kong_cutoff_has_no_dst_shift_between_winter_and_summer() -> None:
    winter = hkg_t24_cutoff_utc(date(2026, 1, 15))
    summer = hkg_t24_cutoff_utc(date(2026, 7, 15))

    assert winter.hour == 7
    assert summer.hour == 7
    assert summer - winter == timedelta(days=181)


def test_audit_bundle_hashes_and_counts_match_contract() -> None:
    bundle = validate_audit_bundle(AUDIT_ROOT)

    assert bundle.summary["dataset_count"] == 13
    assert bundle.summary["table_decision_count"] == 52
    assert bundle.summary["attribute_decision_count"] == 1869
    assert bundle.summary["quality_issue_count"] == 22
    assert bundle.summary["station_dossier_count"] == 36
    assert bundle.profile_summary["row_table_rows_total"] == 7_219_745


def test_source_reconciliation_accounts_for_all_52_files() -> None:
    bundle = validate_audit_bundle(AUDIT_ROOT)
    rows = reconcile_sources(bundle, datasets_root=DATASETS_ROOT, profile_json=PROFILE_JSON)

    assert len(rows) == 52
    assert {row.status for row in rows} == {"PASS"}
    assert sum(1 for row in rows if row.db_action == "SKIP_DUPLICATE_FORMAT") == 2
    assert sum(1 for row in rows if row.disposition == "REGISTER_OBJECT_OR_ARTIFACT") == 10


def test_table_name_for_source_is_stable_and_mysql_legacy_independent() -> None:
    name = table_name_for_source("05_hko_historical_rss_forecasts/hko_press_archive_forecast_days.parquet")

    assert name == "ds_05_hko_historical_rss_forecasts__hko_press_archive_forecast_"
    assert len(name) <= 63


def test_migration_contains_sealing_firewall_and_safe_cutoff_view() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CREATE SCHEMA IF NOT EXISTS sealed_confirmation" in sql
    assert "CREATE OR REPLACE FUNCTION governance.hkg_t24_cutoff_utc" in sql
    assert "available_at_utc <= cutoff_utc" in sql
    assert "REVOKE ALL ON SCHEMA sealed_confirmation FROM PUBLIC" in sql
    assert "TO hkg_tmax_live_inference" in sql
    assert "feature_safe.hko_t24_official_anchor" in sql
    assert "CHECK (local_date >= date '2024-01-01')" in sql
