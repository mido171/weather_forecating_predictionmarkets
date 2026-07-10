from datetime import UTC, datetime

import pytest

from hkg_tmax.publication import (
    ARCHIVE_FIRST_OBSERVED,
    PROVIDER_FIRST_CANDIDATE,
    REVISION_OBSERVED,
    PublicationError,
    build_daily_extract_publication_ledger,
    daily_extract_month_source_id,
    daily_extract_month_url,
)
from hkg_tmax.storage import store_raw_bytes


def _payload(day_values: list[tuple[str, str]]) -> bytes:
    rows = ",".join(
        f'["{day}","1008.0","{value}","28.0","26.0","24.0","80","70","0.0"]'
        for day, value in day_values
    )
    return f'{{"stn":{{"data":[{{"month":6,"dayData":[{rows}]}}]}}}}'.encode()


def test_daily_extract_month_helpers() -> None:
    assert daily_extract_month_source_id(2026, 6) == "hko_daily_extract_202606"
    assert (
        daily_extract_month_url(2026, 6)
        == "https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_202606.xml"
    )


def test_publication_ledger_tracks_first_latest_and_revisions(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    source_id = "hko_daily_extract_202606"
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0"), ("02", "31.0")]),
        retrieved_at=datetime(2026, 6, 18, 10, tzinfo=UTC),
        extension="xml",
        metadata={"final_url": "https://example.test/first.xml"},
    )
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.1"), ("02", "31.0"), ("03", "30.5")]),
        retrieved_at=datetime(2026, 6, 18, 11, tzinfo=UTC),
        extension="xml",
        metadata={"final_url": "https://example.test/second.xml"},
    )

    rows = build_daily_extract_publication_ledger(
        raw_root=raw_root,
        year=2026,
        month=6,
        source_id=source_id,
    )

    by_date = {row.local_date: row for row in rows}
    assert by_date["2026-06-01"].first_value == "32.0"
    assert by_date["2026-06-01"].latest_value == "32.1"
    assert by_date["2026-06-01"].evidence_class == REVISION_OBSERVED
    assert by_date["2026-06-01"].revision_observed == "true"
    assert by_date["2026-06-02"].evidence_class == ARCHIVE_FIRST_OBSERVED
    assert by_date["2026-06-03"].evidence_class == ARCHIVE_FIRST_OBSERVED
    assert "not proof of provider first publication" in by_date["2026-06-03"].notes


def test_publication_ledger_candidates_require_active_absence_and_watched_date(
    tmp_path,
) -> None:
    raw_root = tmp_path / "raw"
    source_id = "hko_daily_extract_202606"
    active_start = datetime(2026, 6, 18, 10, 30, tzinfo=UTC)
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0")]),
        retrieved_at=datetime(2026, 6, 18, 10, tzinfo=UTC),
        extension="xml",
    )
    absent_snapshot = store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0")]),
        retrieved_at=datetime(2026, 6, 18, 10, 45, tzinfo=UTC),
        extension="xml",
    )
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0"), ("02", "31.0"), ("03", "30.5")]),
        retrieved_at=datetime(2026, 6, 18, 11, tzinfo=UTC),
        extension="xml",
    )

    rows = build_daily_extract_publication_ledger(
        raw_root=raw_root,
        year=2026,
        month=6,
        source_id=source_id,
        provider_first_candidate_after=active_start,
        watched_candidate_dates=["2026-06-02"],
    )

    by_date = {row.local_date: row for row in rows}
    assert by_date["2026-06-01"].evidence_class == ARCHIVE_FIRST_OBSERVED
    assert by_date["2026-06-02"].evidence_class == PROVIDER_FIRST_CANDIDATE
    assert by_date["2026-06-02"].last_absent_archive_sha256 == absent_snapshot.sha256
    assert by_date["2026-06-02"].last_absent_archive_retrieved_at == "2026-06-18T10:45:00Z"
    assert by_date["2026-06-03"].evidence_class == ARCHIVE_FIRST_OBSERVED


def test_publication_ledger_rejects_candidate_without_active_absence(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    source_id = "hko_daily_extract_202606"
    active_start = datetime(2026, 6, 18, 10, 30, tzinfo=UTC)
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0")]),
        retrieved_at=datetime(2026, 6, 18, 10, tzinfo=UTC),
        extension="xml",
    )
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0"), ("02", "31.0")]),
        retrieved_at=datetime(2026, 6, 18, 11, tzinfo=UTC),
        extension="xml",
    )

    rows = build_daily_extract_publication_ledger(
        raw_root=raw_root,
        year=2026,
        month=6,
        source_id=source_id,
        provider_first_candidate_after=active_start,
        watched_candidate_dates=["2026-06-02"],
    )

    by_date = {row.local_date: row for row in rows}
    assert by_date["2026-06-02"].evidence_class == ARCHIVE_FIRST_OBSERVED
    assert by_date["2026-06-02"].last_absent_archive_retrieved_at == ""
    assert "no active absent snapshot" in by_date["2026-06-02"].notes


def test_publication_ledger_revision_overrides_candidate(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    source_id = "hko_daily_extract_202606"
    active_start = datetime(2026, 6, 18, 10, 30, tzinfo=UTC)
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0")]),
        retrieved_at=datetime(2026, 6, 18, 10, 45, tzinfo=UTC),
        extension="xml",
    )
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0"), ("02", "31.0")]),
        retrieved_at=datetime(2026, 6, 18, 11, tzinfo=UTC),
        extension="xml",
    )
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0"), ("02", "31.1")]),
        retrieved_at=datetime(2026, 6, 18, 12, tzinfo=UTC),
        extension="xml",
    )

    rows = build_daily_extract_publication_ledger(
        raw_root=raw_root,
        year=2026,
        month=6,
        source_id=source_id,
        provider_first_candidate_after=active_start,
        watched_candidate_dates=["2026-06-02"],
    )

    by_date = {row.local_date: row for row in rows}
    assert by_date["2026-06-02"].evidence_class == REVISION_OBSERVED


def test_publication_ledger_rejects_invalid_watched_date(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    source_id = "hko_daily_extract_202606"
    store_raw_bytes(
        raw_root,
        source_id=source_id,
        content=_payload([("01", "32.0")]),
        retrieved_at=datetime(2026, 6, 18, 10, tzinfo=UTC),
        extension="xml",
    )

    with pytest.raises(PublicationError, match="Invalid watched candidate date"):
        build_daily_extract_publication_ledger(
            raw_root=raw_root,
            year=2026,
            month=6,
            source_id=source_id,
            watched_candidate_dates=["20260601"],
        )


def test_publication_ledger_requires_timezone_aware_candidate_marker(tmp_path) -> None:
    with pytest.raises(PublicationError, match="timezone-aware"):
        build_daily_extract_publication_ledger(
            raw_root=tmp_path,
            year=2026,
            month=6,
            source_id="hko_daily_extract_202606",
            provider_first_candidate_after=datetime(2026, 6, 18, 10),
        )


def test_publication_ledger_requires_snapshots(tmp_path) -> None:
    with pytest.raises(PublicationError, match="No raw snapshots"):
        build_daily_extract_publication_ledger(
            raw_root=tmp_path,
            year=2026,
            month=6,
            source_id="hko_daily_extract_202606",
        )
