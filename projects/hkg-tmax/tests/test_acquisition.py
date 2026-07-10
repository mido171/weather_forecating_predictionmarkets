import csv
from datetime import UTC, datetime, timedelta

import pytest

from hkg_tmax.acquisition import (
    ensure_data_root,
    resolve_data_root,
    store_content_addressed_retrieval,
)
from hkg_tmax.collector import _daily_hkt_due
from hkg_tmax.hashing import sha256_bytes


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
    assert {row["storage_schema_version"] for row in ledger} == {"2"}
    assert {row["storage_root_id"] for row in ledger} == {"hkg-tmax-data"}
    assert ledger[0]["content_relpath"] == (
        f"raw/objects/{first.content_sha256[:2]}/{first.content_sha256}.csv"
    )
    assert ledger[0]["legacy_content_path"] == str(first.content_path.resolve())

    with (data_root / "manifests" / "file_manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 1
    assert manifest[0]["content_relpath"] == ledger[0]["content_relpath"]


def test_file_manifest_enriches_legacy_row_without_losing_provenance(tmp_path) -> None:
    content = b"legacy then relocated"
    digest = sha256_bytes(content)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    with (manifests / "retrieval_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "retrieval_id",
                "source_id",
                "provider",
                "retrieved_at",
                "status",
                "http_status",
                "request_url",
                "final_url",
                "etag",
                "last_modified",
                "content_sha256",
                "content_length",
                "content_path",
                "sidecar_path",
                "deduplicated",
                "error",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "retrieval_id": "legacy",
                "source_id": "legacy_source",
                "provider": "legacy",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "status": "success",
                "content_sha256": digest,
                "content_length": len(content),
                "content_path": r"C:\old-data\raw\objects\legacy.csv",
                "sidecar_path": r"C:\old-data\raw\objects\legacy.metadata.json",
            }
        )

    record = store_content_addressed_retrieval(
        tmp_path,
        source_id="new_source",
        provider="fixture",
        content=content,
        retrieved_at=datetime(2026, 1, 2, tzinfo=UTC),
        extension="csv",
        metadata={},
    )

    with (manifests / "file_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 1
    assert manifest[0]["legacy_content_path"] == r"C:\old-data\raw\objects\legacy.csv"
    assert manifest[0]["content_relpath"] == (
        f"raw/objects/{digest[:2]}/{digest}.csv"
    )
    assert record.content_path.is_file()


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
