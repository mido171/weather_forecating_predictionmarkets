from __future__ import annotations

import csv
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .acquisition import (
    AcquisitionRecord,
    ensure_data_root,
    fetch_http_to_acquisition,
    inspect_data_root,
)
from .config import Source, SourceCatalog, load_yaml
from .fetch import FetchPolicy
from .paths import ProjectPaths
from .sources import is_http_url, unresolved_template


class CollectorError(RuntimeError):
    """Raised when collector configuration or collection fails."""


HKT = ZoneInfo("Asia/Hong_Kong")
MARKET_STATUS = "MARKET_ONLY"
BATCH_LEDGER_PREFIXES = {
    "arwf-current": ("hko_arwf",),
    "daily-extract": ("hko_daily_extract",),
    "ncep-operational-current": ("ncep_gfs", "ncep_gefs"),
    "radar-lightning": ("hko_radar", "hko_lightning", "hko_gridded_rainfall"),
    "satellite-current": ("hko_satellite",),
    "upper-air": ("noaa_igra",),
}


@dataclass(frozen=True)
class CollectOutcome:
    source_id: str
    status: str
    message: str
    record: AcquisitionRecord | None = None


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _read_state(data_root: Path, source_id: str) -> dict[str, Any]:
    path = data_root / "state" / f"{source_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CollectorError(f"Invalid collector state: {path}")
    return data


def _write_state(data_root: Path, source_id: str, state: Mapping[str, object]) -> None:
    path = data_root / "state" / f"{source_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(state), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _merge_state(data_root: Path, source_id: str, updates: Mapping[str, object]) -> None:
    state = _read_state(data_root, source_id)
    state.update(dict(updates))
    _write_state(data_root, source_id, state)


def _is_collectable_source(source: Source) -> tuple[bool, str]:
    if source.point_in_time_status == MARKET_STATUS:
        return False, "market source excluded by data-acquisition reset"
    url = source.url
    if unresolved_template(url):
        return False, "URL template requires parameters"
    if not is_http_url(url):
        return False, "non-HTTP source"
    method = str(source.access.get("method", ""))
    if "discover" in method or "interactive" in method or "template" in method:
        return False, f"method requires discovery or parameters: {method}"
    return True, "collectable"


def collect_source_ids(
    root: Path,
    *,
    source_ids: Sequence[str],
    continue_on_error: bool = False,
) -> list[CollectOutcome]:
    data_root = ensure_data_root(root)
    catalog = SourceCatalog.from_path(root / "config" / "sources" / "data_sources.yaml")
    timeout = float(os.getenv("HKG_TMAX_HTTP_TIMEOUT_SECONDS", "60"))
    user_agent = os.getenv(
        "HKG_TMAX_USER_AGENT",
        "HKG-Tmax-Research/0.1 (+research-contact-required)",
    )
    policy = FetchPolicy(
        timeout_seconds=timeout,
        user_agent=user_agent,
        max_attempts=2,
        retry_sleep_seconds=2,
    )
    outcomes: list[CollectOutcome] = []
    for source_id in source_ids:
        source = catalog.get(source_id)
        allowed, reason = _is_collectable_source(source)
        if not allowed:
            outcome = CollectOutcome(source_id, "skipped", reason)
            outcomes.append(outcome)
            if not continue_on_error:
                raise CollectorError(f"{source_id}: {reason}")
            continue
        try:
            record = fetch_http_to_acquisition(
                data_root,
                source_id=source.id,
                provider=source.provider,
                url=source.url,
                policy=policy,
            )
        except Exception as exc:
            _write_state(
                data_root,
                source.id,
                {
                    "source_id": source.id,
                    "last_attempt_utc": datetime.now(UTC),
                    "last_status": "failed",
                    "last_error": str(exc),
                    "consecutive_failures": int(_read_state(data_root, source.id).get("consecutive_failures", 0))
                    + 1,
                },
            )
            outcome = CollectOutcome(source_id, "failed", str(exc))
            outcomes.append(outcome)
            if not continue_on_error:
                raise CollectorError(f"{source_id}: {exc}") from exc
            continue
        _write_state(
            data_root,
            source.id,
            {
                "source_id": source.id,
                "last_attempt_utc": datetime.now(UTC),
                "last_success_utc": record.retrieved_at,
                "last_status": "success",
                "last_error": "",
                "last_content_sha256": record.content_sha256,
                "last_content_length": record.content_length,
                "last_content_path": str(record.content_path),
                "last_deduplicated": record.deduplicated,
                "consecutive_failures": 0,
            },
        )
        outcomes.append(
            CollectOutcome(
                source_id,
                "success",
                "deduplicated" if record.deduplicated else "new_content",
                record,
            )
        )
    return outcomes


def load_collector_schedules(root: Path) -> dict[str, Any]:
    return load_yaml(root / "config" / "acquisition" / "collector_schedules.yaml")


def _collector_execution_enabled(schedules: Mapping[str, Any]) -> tuple[bool, str]:
    policy = schedules.get("policy", {})
    if not isinstance(policy, Mapping) or not policy.get("enabled", False):
        return False, "collector policy is disabled"
    acknowledgement = str(
        policy.get("execution_acknowledgement", "HKG_TMAX_ENABLE_COLLECTORS")
    ).strip()
    if not acknowledgement:
        return False, "collector acknowledgement variable is not configured"
    if os.getenv(acknowledgement, "").strip() != "1":
        return False, f"set {acknowledgement}=1 to acknowledge network execution"
    return True, "collector execution explicitly enabled"


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _daily_hkt_due(
    *,
    schedule: Mapping[str, Any],
    state: Mapping[str, Any],
    now_utc: datetime,
) -> tuple[bool, str]:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone aware")
    local_now = now_utc.astimezone(HKT)
    at_text = str(schedule.get("at_local_time", "09:00"))
    hour, minute = (int(part) for part in at_text.split(":", 1))
    scheduled_time = time(hour, minute, tzinfo=HKT)
    last_success = _parse_iso(state.get("last_success_utc"))
    if last_success and last_success.astimezone(HKT).date() == local_now.date():
        return False, "already succeeded today"
    if local_now.timetz() < scheduled_time:
        return False, f"before scheduled local time {at_text}"
    last_status = str(state.get("last_status", ""))
    last_attempt = _parse_iso(state.get("last_attempt_utc"))
    retry_hours = float(schedule.get("retry_after_failed_hours", 6))
    if last_status == "failed" and last_attempt:
        if now_utc - last_attempt < timedelta(hours=retry_hours):
            return False, "waiting for failed-request retry window"
        return True, "daily retry after failure"
    if last_attempt and last_attempt.astimezone(HKT).date() == local_now.date():
        return False, "already attempted today; unchanged successful fetches are not repeated"
    return True, "daily scheduled collection due"


def _interval_due(
    *,
    schedule: Mapping[str, Any],
    state: Mapping[str, Any],
    now_utc: datetime,
) -> tuple[bool, str]:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone aware")
    interval_minutes = float(schedule.get("interval_minutes", 60))
    last_attempt = _parse_iso(state.get("last_attempt_utc"))
    if last_attempt is None:
        return True, "no previous attempt"
    elapsed = now_utc - last_attempt
    if elapsed >= timedelta(minutes=interval_minutes):
        return True, "interval elapsed"
    return False, f"not due for {interval_minutes:g} minute interval"


def due_schedules(root: Path, *, now_utc: datetime | None = None) -> list[str]:
    now = now_utc or datetime.now(UTC)
    if now.tzinfo is None:
        raise CollectorError("now_utc must be timezone aware")
    schedules = load_collector_schedules(root)
    enabled, _ = _collector_execution_enabled(schedules)
    if not enabled:
        return []
    data_root = ensure_data_root(root)
    entries = schedules.get("sources", [])
    if not isinstance(entries, list):
        raise CollectorError("config/collector_schedules.yaml sources must be a list")
    due: list[str] = []
    for item in entries:
        if not isinstance(item, dict) or not item.get("enabled", False):
            continue
        source_id = str(item.get("source_id", ""))
        if not source_id:
            continue
        state = _read_state(data_root, source_id)
        schedule_type = str(item.get("schedule_type", "interval"))
        if schedule_type == "daily_hkt":
            is_due, _ = _daily_hkt_due(schedule=item, state=state, now_utc=now)
        else:
            is_due, _ = _interval_due(schedule=item, state=state, now_utc=now)
        if is_due:
            due.append(source_id)
    return due


def run_due_schedules(root: Path) -> list[CollectOutcome]:
    schedules = load_collector_schedules(root)
    enabled, reason = _collector_execution_enabled(schedules)
    if not enabled:
        raise CollectorError(f"Scheduled collectors are fail-closed: {reason}")
    data_root = ensure_data_root(root)
    entries = schedules.get("sources", [])
    if not isinstance(entries, list):
        raise CollectorError("config/collector_schedules.yaml sources must be a list")
    now = datetime.now(UTC)
    outcomes: list[CollectOutcome] = []
    simple_due: list[str] = []
    max_sources = max(1, int(os.getenv("HKG_TMAX_COLLECTOR_MAX_SOURCES", "1")))
    due_sources = 0
    for item in entries:
        if not isinstance(item, dict) or not item.get("enabled", False):
            continue
        source_id = str(item.get("source_id", ""))
        if not source_id:
            continue
        state = _read_state(data_root, source_id)
        schedule_type = str(item.get("schedule_type", "interval"))
        if schedule_type == "daily_hkt":
            is_due, reason = _daily_hkt_due(schedule=item, state=state, now_utc=now)
        else:
            is_due, reason = _interval_due(schedule=item, state=state, now_utc=now)
        if not is_due:
            continue
        due_sources += 1
        if due_sources > max_sources:
            raise CollectorError(
                "Due collector count exceeds HKG_TMAX_COLLECTOR_MAX_SOURCES="
                f"{max_sources}; narrow the schedule before execution"
            )
        adapter = str(item.get("adapter", "direct_source"))
        if adapter == "direct_source":
            simple_due.append(source_id)
            continue
        if adapter == "hko_backfill_batch":
            batch = str(item.get("batch", ""))
            try:
                from .hko_backfill import run_hko_backfill_batch

                result = run_hko_backfill_batch(
                    root,
                    batch=batch,
                    continue_on_error=True,
                    delay_seconds=float(item.get("delay_seconds", 0.2)),
                    skip_existing_successes=bool(item.get("skip_existing_successes", True)),
                    max_requests=max(
                        1, int(os.getenv("HKG_TMAX_COLLECTOR_MAX_REQUESTS", "1"))
                    ),
                )
            except Exception as exc:
                _merge_state(
                    data_root,
                    source_id,
                    {
                        "source_id": source_id,
                        "adapter": adapter,
                        "batch": batch,
                        "last_attempt_utc": now,
                        "last_status": "failed",
                        "last_error": str(exc),
                        "last_due_reason": reason,
                        "consecutive_failures": int(state.get("consecutive_failures", 0)) + 1,
                    },
                )
                outcomes.append(CollectOutcome(source_id, "failed", str(exc)))
                continue
            status = "success" if result.failed == 0 else "partial"
            _merge_state(
                data_root,
                source_id,
                {
                    "source_id": source_id,
                    "adapter": adapter,
                    "batch": batch,
                    "last_attempt_utc": now,
                    "last_success_utc": now if result.succeeded else state.get("last_success_utc", ""),
                    "last_status": status,
                    "last_error": "; ".join(result.failures[:5]),
                    "last_due_reason": reason,
                    "last_requested": result.requested,
                    "last_succeeded": result.succeeded,
                    "last_failed": result.failed,
                    "last_skipped": result.skipped,
                    "consecutive_failures": 0 if status == "success" else int(state.get("consecutive_failures", 0)) + 1,
                },
            )
            outcomes.append(
                CollectOutcome(
                    source_id,
                    status,
                    (
                        f"batch={batch} requested={result.requested} "
                        f"succeeded={result.succeeded} skipped={result.skipped} failed={result.failed}"
                    ),
                )
            )
            continue
        _merge_state(
            data_root,
            source_id,
            {
                "source_id": source_id,
                "adapter": adapter,
                "last_attempt_utc": now,
                "last_status": "skipped",
                "last_error": f"unknown schedule adapter: {adapter}",
                "last_due_reason": reason,
            },
        )
        outcomes.append(CollectOutcome(source_id, "skipped", f"unknown schedule adapter: {adapter}"))
    if simple_due:
        outcomes.extend(collect_source_ids(root, source_ids=simple_due, continue_on_error=True))
    return outcomes


def _read_ledger(data_root: Path) -> list[dict[str, str]]:
    path = data_root / "manifests" / "retrieval_ledger.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pa: Any = importlib.import_module("pyarrow")
        pq: Any = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError:
        return
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd")


def write_machine_source_catalog(root: Path) -> Path:
    data = load_yaml(root / "config" / "sources" / "data_source_catalog.yaml")
    rows = data.get("sources", [])
    if not isinstance(rows, list):
        raise CollectorError("config/data_source_catalog.yaml sources must be a list")
    flat_rows = [{key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows if isinstance(row, dict)]
    metadata_dir = ensure_data_root(root) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / "source_catalog.csv"
    if flat_rows:
        fieldnames = sorted({key for row in flat_rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
    _write_parquet(metadata_dir / "source_catalog.parquet", flat_rows)
    return metadata_dir / "source_catalog.parquet"


def write_health_report(root: Path) -> Path:
    data_root = ensure_data_root(root)
    states: dict[str, dict[str, Any]] = {}
    for path in sorted((data_root / "state").glob("*.json")):
        states[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    schedules = load_collector_schedules(root)
    entries = schedules.get("sources", [])
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source_id = str(entry.get("source_id", ""))
            if not source_id or not entry.get("enabled", False):
                continue
            states.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "adapter": str(entry.get("adapter", "direct_source")),
                    "batch": str(entry.get("batch", "")),
                    "last_status": "not_started",
                    "consecutive_failures": 0,
                },
            )
    ledger = _read_ledger(data_root)
    successful_ledger = [row for row in ledger if row.get("status") == "success"]
    latest_by_prefix: dict[str, str] = {}
    for row in successful_ledger:
        source_id = row.get("source_id", "")
        retrieved_at = row.get("retrieved_at", "")
        if not source_id or not retrieved_at:
            continue
        for batch, prefixes in BATCH_LEDGER_PREFIXES.items():
            if any(source_id.startswith(prefix) for prefix in prefixes):
                latest_by_prefix[batch] = max(latest_by_prefix.get(batch, ""), retrieved_at)
    lines = [
        "# Live Collector Health",
        "",
        f"- data root: `{data_root}`",
        f"- generated_at_utc: `{datetime.now(UTC).isoformat().replace('+00:00', 'Z')}`",
        "",
        "| Source | Adapter | Last status | Last attempt UTC | Last success UTC | Changed payload | Batch counts | Last content hash | Consecutive failures |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    if not states:
        lines.append("| none |  | not started |  |  |  |  |  | 0 |")
    for source_id in sorted(states):
        state = states[source_id]
        batch = str(state.get("batch", ""))
        if batch and batch in latest_by_prefix:
            ledger_success = latest_by_prefix[batch]
            state_last_success = str(state.get("last_success_utc", ""))
            state_last_attempt = str(state.get("last_attempt_utc", ""))
            if not state_last_success or ledger_success > max(state_last_success, state_last_attempt):
                state["last_success_utc"] = ledger_success
                state["last_status"] = "ledger_success"
                state["ledger_supersedes_state"] = True
                state["consecutive_failures"] = 0
        deduplicated = state.get("last_deduplicated", "")
        changed_payload = "" if deduplicated == "" else str(not bool(deduplicated))
        batch_counts = ""
        if state.get("ledger_supersedes_state"):
            batch_counts = "latest success from ledger; manual batch counts not stored"
        elif "last_requested" in state:
            batch_counts = (
                f"requested={state.get('last_requested', 0)}, "
                f"succeeded={state.get('last_succeeded', 0)}, "
                f"skipped={state.get('last_skipped', 0)}, "
                f"failed={state.get('last_failed', 0)}"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(state.get("source_id", source_id)),
                    str(state.get("adapter", "direct_source")),
                    str(state.get("last_status", "")),
                    str(state.get("last_attempt_utc", "")),
                    str(state.get("last_success_utc", "")),
                    changed_payload,
                    batch_counts,
                    str(state.get("last_content_sha256", ""))[:16],
                    str(state.get("consecutive_failures", 0)),
                ]
            )
            + " |"
        )
    report = ProjectPaths.from_project_root(root).run_root / "reports" / "live_collector_health.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_inventory_reports(root: Path) -> list[Path]:
    data_root = ensure_data_root(root)
    catalog = load_yaml(root / "config" / "sources" / "data_source_catalog.yaml")
    rows = catalog.get("sources", [])
    if not isinstance(rows, list):
        raise CollectorError("config/data_source_catalog.yaml sources must be a list")
    ledger = _read_ledger(data_root)
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in ledger:
        by_source.setdefault(row.get("source_id", ""), []).append(row)

    reports_dir = ProjectPaths.from_project_root(root).run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    inventory_lines = [
        "# Data Inventory",
        "",
        "Polymarket is explicitly excluded from the current acquisition goal.",
        "",
        "| Source | Family | Priority | Status | Point-in-time class | Last success | Unique hashes | Blocker |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    blocker_lines = ["# Source Blockers", "", "| Source | Status | Blocker | Evidence | Next action |", "|---|---|---|---|---|"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id", ""))
        source_rows = [entry for entry in by_source.get(source_id, []) if entry.get("status") == "success"]
        unique_hashes = len({entry.get("content_sha256", "") for entry in source_rows if entry.get("content_sha256")})
        last_success = max((entry.get("retrieved_at", "") for entry in source_rows), default="")
        inventory_lines.append(
            "| "
            + " | ".join(
                [
                    source_id,
                    str(row.get("family", "")),
                    str(row.get("priority", "")),
                    str(row.get("backfill_status", "")),
                    str(row.get("point_in_time_class", "")),
                    last_success,
                    str(unique_hashes),
                    str(row.get("blocker", "")),
                ]
            )
            + " |"
        )
        blocker = str(row.get("blocker", ""))
        status = str(row.get("backfill_status", ""))
        if blocker or status in {
            "UNAVAILABLE",
            "PAID",
            "CREDENTIAL_BLOCKED",
            "TERMS_BLOCKED",
            "TECHNICALLY_BLOCKED",
            "DISCOVERY_REQUIRED",
        }:
            blocker_lines.append(
                "| "
                + " | ".join(
                    [
                        source_id,
                        status,
                        blocker,
                        str(row.get("evidence", "")),
                        str(row.get("next_action", "")),
                    ]
                )
                + " |"
            )

    inventory_path = reports_dir / "data_inventory.md"
    inventory_path.write_text("\n".join(inventory_lines) + "\n", encoding="utf-8")
    written.append(inventory_path)

    coverage_lines = [
        "# Data Coverage",
        "",
        f"- data root: `{data_root}`",
        "",
        "| Source | Retrieval attempts | Successes | Unique hashes | Bytes | Earliest retrieved | Latest retrieved |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for source_id in sorted(by_source):
        source_rows = by_source[source_id]
        successes = [row for row in source_rows if row.get("status") == "success"]
        bytes_total = sum(int(row.get("content_length") or 0) for row in successes)
        retrieved = [row.get("retrieved_at", "") for row in successes if row.get("retrieved_at")]
        coverage_lines.append(
            "| "
            + " | ".join(
                [
                    source_id,
                    str(len(source_rows)),
                    str(len(successes)),
                    str(len({row.get("content_sha256", "") for row in successes if row.get("content_sha256")})),
                    str(bytes_total),
                    min(retrieved) if retrieved else "",
                    max(retrieved) if retrieved else "",
                ]
            )
            + " |"
        )
    coverage_path = reports_dir / "data_coverage.md"
    coverage_path.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")
    written.append(coverage_path)

    blocker_path = reports_dir / "source_blockers.md"
    blocker_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")
    written.append(blocker_path)

    storage = inspect_data_root(root)
    storage_lines = [
        "# Storage and Volume",
        "",
        f"- data root: `{storage.path}`",
        f"- path length: `{storage.path_length}`",
        f"- exists: `{storage.exists}`",
        f"- long path risk: `{storage.long_path_risk}`",
        f"- free GB: `{storage.free_bytes / (1024 ** 3):.2f}`",
        f"- total GB: `{storage.total_bytes / (1024 ** 3):.2f}`",
        "",
        "The configured data root uses content-addressed raw objects and append-only retrieval ledgers.",
    ]
    storage_path = reports_dir / "storage_and_volume.md"
    storage_path.write_text("\n".join(storage_lines) + "\n", encoding="utf-8")
    written.append(storage_path)

    station_lines = [
        "# Station Registry",
        "",
        "Initial station-registry work is metadata-only. Full station timelines remain a P0 acquisition task.",
        "",
        "| Source | Status | Notes |",
        "|---|---|---|",
        "| hko_station_metadata | discovery/backfill pending | Must preserve station code, name, lat/lon, elevation, operating dates, measured elements, and distance/bearing to HKO. |",
    ]
    station_path = reports_dir / "station_registry.md"
    station_path.write_text("\n".join(station_lines) + "\n", encoding="utf-8")
    written.append(station_path)

    bronze_metadata = sorted((data_root / "bronze").glob("*/*.metadata.json"))
    bronze_sources = sorted({path.parent.name for path in bronze_metadata})
    schema_lines = [
        "# Schema and Quality",
        "",
        "No model features are built. Current QC is limited to retrieval integrity, source-native bronze parsing, content length, hashes, and schema/version metadata.",
        "",
        f"- bronze dataset metadata files: `{len(bronze_metadata)}`",
        f"- bronze sources: `{', '.join(bronze_sources) if bronze_sources else 'none'}`",
        "",
        "| Check | Status |",
        "|---|---|",
        "| content-addressed raw dedupe | implemented |",
        "| retrieval ledger per attempt | implemented |",
        "| first HKO bronze rebuilds | implemented for acquired CLMMAXT, live temperature, since-midnight max/min, local forecast, and nine-day forecast |",
        "| station-level schema validation | pending deeper source-specific adapters |",
        "| silver/gold rebuilds | pending after source-specific bronze QA |",
    ]
    schema_path = reports_dir / "schema_and_quality.md"
    schema_path.write_text("\n".join(schema_lines) + "\n", encoding="utf-8")
    written.append(schema_path)

    written.append(write_health_report(root))
    return written
