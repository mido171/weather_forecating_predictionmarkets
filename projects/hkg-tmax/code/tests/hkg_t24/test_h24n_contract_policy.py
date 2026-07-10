from __future__ import annotations

from datetime import date

import pytest
from hkg_t24.audit.source_registry import (
    source_registry_rows,
    validate_source_registry_contract,
)
from hkg_t24.constants import (
    BLOCKED_DAILY_TMAX_DATASETS,
    DATASET_FEATURE_PREFIX,
    STRICT_SCHEMA_VERSION,
    TARGET_MEMORY_FEATURE_WHITELIST,
    assert_no_forbidden_target_memory_names,
)
from hkg_t24.timeutils import (
    calendar_row,
    formal_cutoff_utc,
    operational_freeze_utc,
)


def test_h24n_cutoff_and_freeze_are_final_patch_times() -> None:
    assert formal_cutoff_utc(date(2024, 2, 29)).isoformat() == "2024-02-28T07:00:00+00:00"
    assert operational_freeze_utc(date(2024, 2, 29)).isoformat() == "2024-02-28T06:45:00+00:00"


def test_cutoff_calendar_partitions_and_snapshot_id() -> None:
    assert calendar_row(date(2023, 12, 31)).partition_name == "pre2024_development"
    assert calendar_row(date(2024, 1, 1)).partition_name == "sealed_2024"
    assert calendar_row(date(2025, 6, 1)).partition_name == "sealed_2025"
    row = calendar_row(date(2026, 6, 21))
    assert row.partition_name == "prospective_2026"
    assert row.snapshot_id == "H24N:2026-06-21"


def test_source_registry_contract_rows_are_final_patch_compatible() -> None:
    validate_source_registry_contract()
    rows = {row.source_code: row for row in source_registry_rows()}

    assert rows["gfs"].strict_allowed
    assert rows["gefsatmosmean"].feature_prefix == "gefsmean"
    assert rows["gefsatmos"].feature_prefix == "gefsens"
    assert rows["aigfspres"].support_only
    assert rows["aigefssfc"].blocked
    assert rows["nbmoc"].blocked
    assert rows["arwf_live"].live_only
    assert all(rows[source].blocked or source == "aigfspres" for source in BLOCKED_DAILY_TMAX_DATASETS)


def test_dataset_prefix_mapping_and_schema_version_are_final_patch_values() -> None:
    assert DATASET_FEATURE_PREFIX["fourcastnetgfs"] == "fourcastnet"
    assert DATASET_FEATURE_PREFIX["ifsenfo"] == "ifsens"
    assert STRICT_SCHEMA_VERSION == "hkg_t24_h24n_strict_v1_20260626_patch1"


def test_finalized_target_memory_lag1_names_are_forbidden() -> None:
    assert_no_forbidden_target_memory_names(TARGET_MEMORY_FEATURE_WHITELIST)
    with pytest.raises(ValueError, match="Forbidden finalized target-memory"):
        assert_no_forbidden_target_memory_names(("target__lag1_tmax_c",))
