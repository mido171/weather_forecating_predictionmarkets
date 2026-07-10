from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .fetch import FetchPolicy, fetch_and_archive
from .hko import DailyExtractRow, parse_daily_extract_json
from .paths import (
    ArchivePathResolutionError,
    ProjectPaths,
    infer_storage_root,
    resolve_archive_content_path,
)
from .storage import RawSnapshot


class PublicationError(RuntimeError):
    """Raised when target-publication evidence cannot be built safely."""


ARCHIVE_FIRST_OBSERVED = "ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST"
PROVIDER_FIRST_CANDIDATE = "PROVIDER_FIRST_PUBLICATION_CANDIDATE"
REVISION_OBSERVED = "REVISION_OBSERVED"


@dataclass(frozen=True)
class SnapshotMetadata:
    source_id: str
    content_path: Path
    sidecar_path: Path
    sha256: str
    retrieved_at: datetime
    final_url: str | None


@dataclass(frozen=True)
class DailyExtractPublicationRow:
    local_date: str
    source_id: str
    first_value: str
    first_value_precision: str
    first_completeness: str
    first_parse_issue: str
    evidence_class: str
    first_archive_retrieved_at: str
    first_archive_sha256: str
    first_archive_path: str
    last_absent_archive_retrieved_at: str
    last_absent_archive_sha256: str
    last_absent_archive_path: str
    latest_value: str
    latest_value_precision: str
    latest_completeness: str
    latest_parse_issue: str
    latest_archive_retrieved_at: str
    latest_archive_sha256: str
    latest_archive_path: str
    revision_observed: str
    distinct_values: str
    observation_count: str
    notes: str


def daily_extract_month_source_id(year: int, month: int) -> str:
    _validate_year_month(year, month)
    return f"hko_daily_extract_{year:04d}{month:02d}"


def daily_extract_month_url(year: int, month: int) -> str:
    _validate_year_month(year, month)
    return f"https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{year:04d}{month:02d}.xml"


def _validate_year_month(year: int, month: int) -> None:
    if year < 1884 or year > 2100:
        raise PublicationError(f"Unsupported Daily Extract year: {year}")
    if month < 1 or month > 12:
        raise PublicationError(f"Unsupported Daily Extract month: {month}")


def fetch_daily_extract_month(
    *,
    root: Path,
    year: int,
    month: int,
    policy: FetchPolicy | None = None,
) -> tuple[RawSnapshot, RawSnapshot]:
    raw_root = ProjectPaths.from_project_root(root).data_root / "raw"
    catalog = fetch_and_archive(
        url="https://www.hko.gov.hk/cis/hko.xml",
        source_id="hko_daily_extract_catalog",
        raw_root=raw_root,
        policy=policy,
    )
    monthly = fetch_and_archive(
        url=daily_extract_month_url(year, month),
        source_id=daily_extract_month_source_id(year, month),
        raw_root=raw_root,
        policy=policy,
    )
    return catalog, monthly


def load_snapshot_metadata(raw_root: Path, source_id: str) -> list[SnapshotMetadata]:
    sidecars = sorted((raw_root / source_id).glob("**/*.metadata.json"))
    snapshots: list[SnapshotMetadata] = []
    for sidecar in sidecars:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("source_id") != source_id:
            raise PublicationError(f"Sidecar source mismatch in {sidecar}")
        content_path = _resolve_content_path(sidecar, data, data_root=infer_storage_root(raw_root))
        if not content_path.is_file():
            raise PublicationError(f"Raw content missing for {source_id}: {content_path}")
        retrieved_at = _parse_retrieved_at(str(data["retrieved_at"]), sidecar)
        metadata = data.get("metadata", {})
        final_url = metadata.get("final_url") if isinstance(metadata, dict) else None
        snapshots.append(
            SnapshotMetadata(
                source_id=source_id,
                content_path=content_path,
                sidecar_path=sidecar,
                sha256=str(data["content_sha256"]),
                retrieved_at=retrieved_at,
                final_url=str(final_url) if final_url is not None else None,
            )
        )
    return sorted(snapshots, key=lambda item: (item.retrieved_at, str(item.sidecar_path)))


def _resolve_content_path(sidecar: Path, data: dict[str, Any], *, data_root: Path) -> Path:
    try:
        return resolve_archive_content_path(data, data_root=data_root, sidecar_path=sidecar)
    except ArchivePathResolutionError as exc:
        raise PublicationError(str(exc)) from exc


def _parse_retrieved_at(value: str, sidecar: Path) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationError(f"Invalid retrieved_at in {sidecar}: {value}") from exc
    if parsed.tzinfo is None:
        raise PublicationError(f"Naive retrieved_at in {sidecar}: {value}")
    return parsed.astimezone(UTC)


def build_daily_extract_publication_ledger(
    *,
    raw_root: Path,
    year: int,
    month: int,
    source_id: str | None = None,
    provider_first_candidate_after: datetime | None = None,
    watched_candidate_dates: Iterable[str] = (),
) -> list[DailyExtractPublicationRow]:
    source_id = source_id or daily_extract_month_source_id(year, month)
    if provider_first_candidate_after is not None and provider_first_candidate_after.tzinfo is None:
        raise PublicationError("provider_first_candidate_after must be timezone-aware")
    watched = _validated_watched_dates(watched_candidate_dates)
    snapshots = load_snapshot_metadata(raw_root, source_id)
    if not snapshots:
        raise PublicationError(f"No raw snapshots available for {source_id}")

    snapshot_rows: list[tuple[SnapshotMetadata, dict[str, DailyExtractRow]]] = []
    observations: dict[str, list[tuple[SnapshotMetadata, DailyExtractRow]]] = {}
    for snapshot in snapshots:
        rows = parse_daily_extract_json(snapshot.content_path.read_bytes(), year=year, month=month)
        row_map = {row.local_date.isoformat(): row for row in rows}
        snapshot_rows.append((snapshot, row_map))
        for local_date, row in row_map.items():
            observations.setdefault(local_date, []).append((snapshot, row))

    ledger: list[DailyExtractPublicationRow] = []
    for local_date in sorted(observations):
        entries = sorted(observations[local_date], key=lambda item: item[0].retrieved_at)
        first_snapshot, first_row = entries[0]
        latest_snapshot, latest_row = entries[-1]
        distinct_values = sorted({_value_key(row) for _, row in entries})
        revision_observed = len(distinct_values) > 1
        last_absent_snapshot = _last_active_absence_snapshot(
            snapshot_rows,
            local_date=local_date,
            first_present_at=first_snapshot.retrieved_at,
            active_start=provider_first_candidate_after,
        )
        evidence_class = _evidence_class(
            local_date,
            first_snapshot.retrieved_at,
            provider_first_candidate_after=provider_first_candidate_after,
            revision_observed=revision_observed,
            watched_candidate_dates=watched,
            last_absent_snapshot=last_absent_snapshot,
        )
        watched_without_active_absence = (
            provider_first_candidate_after is not None
            and local_date in watched
            and first_snapshot.retrieved_at >= provider_first_candidate_after.astimezone(UTC)
            and last_absent_snapshot is None
        )
        notes = _notes_for(
            evidence_class,
            revision_observed,
            watched_without_active_absence=watched_without_active_absence,
        )
        ledger.append(
            DailyExtractPublicationRow(
                local_date=local_date,
                source_id=source_id,
                first_value=_decimal_text(first_row.absolute_daily_max_c),
                first_value_precision=_decimal_text(first_row.value_precision),
                first_completeness=first_row.completeness,
                first_parse_issue=first_row.parse_issue or "",
                evidence_class=evidence_class,
                first_archive_retrieved_at=_iso_utc(first_snapshot.retrieved_at),
                first_archive_sha256=first_snapshot.sha256,
                first_archive_path=str(first_snapshot.content_path),
                last_absent_archive_retrieved_at=(
                    "" if last_absent_snapshot is None else _iso_utc(last_absent_snapshot.retrieved_at)
                ),
                last_absent_archive_sha256=(
                    "" if last_absent_snapshot is None else last_absent_snapshot.sha256
                ),
                last_absent_archive_path=(
                    "" if last_absent_snapshot is None else str(last_absent_snapshot.content_path)
                ),
                latest_value=_decimal_text(latest_row.absolute_daily_max_c),
                latest_value_precision=_decimal_text(latest_row.value_precision),
                latest_completeness=latest_row.completeness,
                latest_parse_issue=latest_row.parse_issue or "",
                latest_archive_retrieved_at=_iso_utc(latest_snapshot.retrieved_at),
                latest_archive_sha256=latest_snapshot.sha256,
                latest_archive_path=str(latest_snapshot.content_path),
                revision_observed=str(revision_observed).lower(),
                distinct_values="|".join(distinct_values),
                observation_count=str(len(entries)),
                notes=notes,
            )
        )
    return ledger


def _validated_watched_dates(values: Iterable[str]) -> set[str]:
    watched: set[str] = set()
    for value in values:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise PublicationError(f"Invalid watched candidate date: {value}") from exc
        watched.add(value)
    return watched


def _last_active_absence_snapshot(
    snapshot_rows: list[tuple[SnapshotMetadata, dict[str, DailyExtractRow]]],
    *,
    local_date: str,
    first_present_at: datetime,
    active_start: datetime | None,
) -> SnapshotMetadata | None:
    if active_start is None:
        return None
    start = active_start.astimezone(UTC)
    active_absences = [
        snapshot
        for snapshot, row_map in snapshot_rows
        if start <= snapshot.retrieved_at < first_present_at and local_date not in row_map
    ]
    return active_absences[-1] if active_absences else None


def _evidence_class(
    local_date: str,
    first_retrieved_at: datetime,
    *,
    provider_first_candidate_after: datetime | None,
    revision_observed: bool,
    watched_candidate_dates: set[str],
    last_absent_snapshot: SnapshotMetadata | None,
) -> str:
    if revision_observed:
        return REVISION_OBSERVED
    if (
        provider_first_candidate_after is not None
        and local_date in watched_candidate_dates
        and first_retrieved_at >= provider_first_candidate_after.astimezone(UTC)
        and last_absent_snapshot is not None
    ):
        return PROVIDER_FIRST_CANDIDATE
    return ARCHIVE_FIRST_OBSERVED


def _notes_for(
    evidence_class: str,
    revision_observed: bool,
    *,
    watched_without_active_absence: bool,
) -> str:
    if revision_observed:
        return "Later archived payload differs from the first archived value for this date."
    if evidence_class == PROVIDER_FIRST_CANDIDATE:
        return "Candidate provider first publication; active archive observed absence before first presence."
    if watched_without_active_absence:
        return "Watched date first observed after active start, but no active absent snapshot preceded it."
    return "First time observed by this archive; not proof of provider first publication."


def _value_key(row: DailyExtractRow) -> str:
    value = _decimal_text(row.absolute_daily_max_c)
    precision = _decimal_text(row.value_precision)
    issue = row.parse_issue or ""
    return f"{value},{precision},{row.completeness},{issue}"


def _decimal_text(value: Decimal | None) -> str:
    return "" if value is None else str(value)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def write_publication_ledger(path: Path, rows: Iterable[DailyExtractPublicationRow]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DailyExtractPublicationRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: getattr(row, field) for field in fieldnames})
    return len(materialized)


def summarize_publication_rows(rows: Iterable[DailyExtractPublicationRow]) -> dict[str, Any]:
    materialized = list(rows)
    evidence_counts: dict[str, int] = {}
    revision_count = 0
    for row in materialized:
        evidence_counts[row.evidence_class] = evidence_counts.get(row.evidence_class, 0) + 1
        if row.revision_observed == "true":
            revision_count += 1
    return {
        "row_count": len(materialized),
        "evidence_counts": evidence_counts,
        "provider_first_candidate_count": evidence_counts.get(PROVIDER_FIRST_CANDIDATE, 0),
        "revision_count": revision_count,
        "provider_first_publication_proven": False,
    }
