from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from hkg_tmax.validation import (
    LeakageError,
    assert_records_asof,
    assert_split_disjoint,
    validate_repository,
)


def test_repository_configs_validate(repo_root) -> None:
    report = validate_repository(repo_root)
    assert any("source catalog" in check for check in report.checks)
    assert any("G1" in warning for warning in report.warnings)


def test_future_record_is_rejected() -> None:
    cutoff = datetime(2026, 6, 18, 15, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    records = [
        {"id": "ok", "available_at": datetime(2026, 6, 18, 6, tzinfo=UTC)},
        {"id": "leak", "available_at": datetime(2026, 6, 18, 8, tzinfo=UTC)},
    ]
    with pytest.raises(LeakageError, match="leaks future"):
        assert_records_asof(records, cutoff_at=cutoff, id_field="id")


def test_missing_available_at_is_rejected() -> None:
    cutoff = datetime(2026, 6, 18, 7, tzinfo=UTC)
    with pytest.raises(LeakageError, match="missing"):
        assert_records_asof([{"id": "x"}], cutoff_at=cutoff, id_field="id")


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(LeakageError, match="overlap"):
        assert_split_disjoint([1, 2], [3], [2, 4])
