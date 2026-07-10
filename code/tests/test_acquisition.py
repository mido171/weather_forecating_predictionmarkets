import csv
from datetime import UTC, datetime, timedelta

import pytest

from hkg_tmax.acquisition import (
    ensure_data_root,
    resolve_data_root,
    store_content_addressed_retrieval,
)
from hkg_tmax.collector import _daily_hkt_due


def test_data_root_env_override_is_resolved(repo_root, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HKG_TMAX_DATA_ROOT", str(tmp_path / "hkg_data"))

    data_root = resolve_data_root(repo_root)
    ensure_data_root(repo_root)

    assert data_root == tmp_path / "hkg_data"
    for name in ("raw", "bronze", "silver", "gold", "metadata", "manifests", "state"):
        assert (data_root / name).is_dir()


def test_content_addressed_store_deduplicates_but_appends_ledger(tmp_path) -> None:
    data_root = tmp_path
    content = b"station,value\nHKO,31.2\n"
    first = store_content_addressed_retrieval(
        data_root,
        source_id="hko_test",
        provider="HKO",
        content=content,
        retrieved_at=datetime(2026, 6, 18, 1, tzinfo=UTC),
        extension="csv",
        metadata={"requested_url": "https://example.test/a.csv", "http_status": 200},
    )
    second = store_content_addressed_retrieval(
        data_root,
        source_id="hko_test",
        provider="HKO",
        content=content,
        retrieved_at=datetime(2026, 6, 18, 2, tzinfo=UTC),
        extension="csv",
        metadata={"requested_url": "https://example.test/a.csv", "http_status": 200},
    )

    assert first.content_path == second.content_path
    assert not first.deduplicated
    assert second.deduplicated

    with (data_root / "manifests" / "retrieval_ledger.csv").open(newline="") as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == 2
    assert {row["content_sha256"] for row in ledger} == {first.content_sha256}
    assert ledger[1]["deduplicated"] == "true"

    with (data_root / "manifests" / "file_manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 1


def test_daily_extract_schedule_allows_one_success_per_hkt_day() -> None:
    now = datetime(2026, 6, 19, 1, 30, tzinfo=UTC)  # 09:30 HKT
    due, reason = _daily_hkt_due(
        schedule={"at_local_time": "09:00", "retry_after_failed_hours": 6},
        state={},
        now_utc=now,
    )
    assert due, reason

    due, reason = _daily_hkt_due(
        schedule={"at_local_time": "09:00", "retry_after_failed_hours": 6},
        state={"last_success_utc": "2026-06-19T01:05:00Z"},
        now_utc=now,
    )
    assert not due
    assert "already succeeded today" in reason


def test_daily_extract_retry_waits_six_hours_after_failure() -> None:
    now = datetime(2026, 6, 19, 7, 30, tzinfo=UTC)
    failed_recently = {
        "last_status": "failed",
        "last_attempt_utc": (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
    }
    due, reason = _daily_hkt_due(
        schedule={"at_local_time": "09:00", "retry_after_failed_hours": 6},
        state=failed_recently,
        now_utc=now,
    )
    assert not due
    assert "retry window" in reason

    failed_earlier = {
        "last_status": "failed",
        "last_attempt_utc": (now - timedelta(hours=7)).isoformat().replace("+00:00", "Z"),
    }
    due, reason = _daily_hkt_due(
        schedule={"at_local_time": "09:00", "retry_after_failed_hours": 6},
        state=failed_earlier,
        now_utc=now,
    )
    assert due, reason


def test_daily_extract_schedule_rejects_naive_now() -> None:
    with pytest.raises(ValueError):
        _daily_hkt_due(
            schedule={"at_local_time": "09:00"},
            state={},
            now_utc=datetime(2026, 6, 19, 1, 30),
        )
