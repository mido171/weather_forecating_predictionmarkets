#!/usr/bin/env python3
"""Resumable operator for the HKO official press forecast detail backfill.

This script does not reimplement the downloader/parser. It wraps
``hko_forecast_archive_downloader_rss/hko_archive.py official-details`` with
state inspection, network preflight, batching, locking, and clear status files.
Every run recalculates what candidate URLs are still missing a successful raw
HTML retrieval, then fetches only those URLs via ``--missing-success-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATA_ROOT = Path(r"C:\hko_press_2000_2026")
DEFAULT_TYPES = ("local", "5day", "7day", "9day")
DEFAULT_START = "2013-11-15"
DEFAULT_END = "2026-06-20"
DEFAULT_DATASET_DIR = PROJECT_PATHS.data_root / "datasets"
DEFAULT_MONITOR_DIR = (
    PROJECT_PATHS.run_root
    / "experiments"
    / "legacy"
    / "0000_research_state_and_data_contract"
    / "hko_official_backfill_monitor"
    / "artifacts"
)
DEFAULT_BUNDLE_STEM = "hko_official_press_weather_forecasts_20000101_20260620"
DOWNLOADER = REPO_ROOT / "scripts" / "hko_forecast_archive_downloader_rss" / "hko_archive.py"
MONITOR = REPO_ROOT / "scripts" / "monitor_hko_official_backfill.py"
FINALIZER = REPO_ROOT / "scripts" / "finalize_hko_official_backfill.py"


class BackfillError(RuntimeError):
    exit_code = 1


class ValidationError(BackfillError):
    exit_code = 2


class NetworkPreflightError(BackfillError):
    exit_code = 3


class DownloaderBatchError(BackfillError):
    exit_code = 4


class StalledBackfillError(BackfillError):
    exit_code = 5


class FinalizeError(BackfillError):
    exit_code = 6


@dataclass(frozen=True)
class Config:
    python: str
    data_root: Path
    archive_db: Path
    start: date
    end: date
    product_types: tuple[str, ...]
    batch_size: int
    max_batches: int
    max_stalled_batches: int
    delay_seconds: float
    timeout_seconds: float
    max_retries: int
    progress_interval_seconds: float
    probe_url: str | None
    probe_timeout_seconds: float
    skip_network_check: bool
    check_only: bool
    finalize: bool
    monitor_each_batch: bool
    output_dir: Path
    monitor_output_dir: Path
    bundle_stem: str
    force_lock: bool


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def emit_log(
    event: str,
    message: str,
    *,
    level: str = "INFO",
    stream: Any = sys.stdout,
    **fields: Any,
) -> None:
    payload = {
        "ts": utc_now_iso(),
        "level": level,
        "event": event,
        "message": message,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True, default=str), file=stream, flush=True)


def emit_summary(event: str, summary: dict[str, Any]) -> None:
    emit_log(
        event,
        "Archive state summary",
        candidate_urls=summary["candidate_urls"],
        successful_raw_urls=summary["successful_raw_urls"],
        remaining_urls=summary["remaining_urls"],
        remaining_by_year=summary["remaining_by_year"],
        remaining_by_type=summary["remaining_by_type"],
        retrieval_status_counts=summary["retrieval_status_counts"],
        first_remaining=summary["first_remaining"][:3],
        recent_failed_retrievals=summary["recent_failed_retrievals"][:3],
    )


def parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{field} must be YYYY-MM-DD, got {value!r}") from exc


def parse_types(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValidationError("--types must include at least one product type")
    return parsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(zip(row.keys(), row, strict=True))


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ValidationError(f"Archive DB does not exist: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def require_tables(conn: sqlite3.Connection) -> None:
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing = {"candidates", "retrievals"} - tables
    if missing:
        raise ValidationError(f"Archive DB is missing required tables: {sorted(missing)}")


def validate_config(args: argparse.Namespace) -> Config:
    data_root = args.data_root.resolve()
    archive_db = (args.archive_db or data_root / "metadata" / "archive.sqlite3").resolve()
    start = parse_date(args.start, "--start")
    end = parse_date(args.end, "--end")
    if end < start:
        raise ValidationError("--end must be on or after --start")
    if args.batch_size <= 0:
        raise ValidationError("--batch-size must be positive")
    if args.max_batches < 0:
        raise ValidationError("--max-batches must be zero or positive")
    if args.max_stalled_batches <= 0:
        raise ValidationError("--max-stalled-batches must be positive")
    if args.delay_seconds < 0:
        raise ValidationError("--delay-seconds must be zero or positive")
    if args.timeout_seconds <= 0:
        raise ValidationError("--timeout-seconds must be positive")
    if args.max_retries < 0:
        raise ValidationError("--max-retries must be zero or positive")
    if args.progress_interval_seconds <= 0:
        raise ValidationError("--progress-interval-seconds must be positive")
    if args.probe_timeout_seconds <= 0:
        raise ValidationError("--probe-timeout-seconds must be positive")
    if not DOWNLOADER.exists():
        raise ValidationError(f"Downloader script does not exist: {DOWNLOADER}")
    if args.finalize and not MONITOR.exists():
        raise ValidationError(f"Monitor script does not exist: {MONITOR}")
    if args.finalize and not FINALIZER.exists():
        raise ValidationError(f"Finalizer script does not exist: {FINALIZER}")

    return Config(
        python=args.python,
        data_root=data_root,
        archive_db=archive_db,
        start=start,
        end=end,
        product_types=parse_types(args.types),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_stalled_batches=args.max_stalled_batches,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        progress_interval_seconds=args.progress_interval_seconds,
        probe_url=args.probe_url,
        probe_timeout_seconds=args.probe_timeout_seconds,
        skip_network_check=args.skip_network_check,
        check_only=args.check_only,
        finalize=args.finalize,
        monitor_each_batch=args.monitor_each_batch,
        output_dir=args.output_dir.resolve(),
        monitor_output_dir=args.monitor_output_dir.resolve(),
        bundle_stem=args.bundle_stem,
        force_lock=args.force_lock,
    )


def candidate_rows(
    conn: sqlite3.Connection,
    *,
    start: date,
    end: date,
    product_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in product_types)
    query = f"""
        SELECT index_date, product_type, title, url
        FROM candidates
        WHERE source = 'info_gov'
          AND product_type IN ({placeholders})
          AND index_date >= ?
          AND index_date <= ?
        ORDER BY index_date, product_type, url
    """
    params: list[Any] = [*product_types, start.isoformat(), end.isoformat()]
    return [row_to_dict(row) for row in conn.execute(query, params)]


def successful_raw_urls(
    conn: sqlite3.Connection,
    *,
    progress_interval_seconds: float | None = None,
) -> set[str]:
    urls: set[str] = set()
    query = """
        SELECT url, raw_path
        FROM retrievals
        WHERE source = 'info_gov_bulletin'
          AND status_code >= 200
          AND status_code < 300
          AND raw_path IS NOT NULL
          AND raw_path != ''
    """
    count_query = """
        SELECT COUNT(*) AS row_count
        FROM retrievals
        WHERE source = 'info_gov_bulletin'
          AND status_code >= 200
          AND status_code < 300
          AND raw_path IS NOT NULL
          AND raw_path != ''
    """
    total_rows = int(conn.execute(count_query).fetchone()["row_count"])
    checked_rows = 0
    missing_raw_paths = 0
    next_progress_at = (
        time.monotonic() + progress_interval_seconds
        if progress_interval_seconds is not None
        else None
    )
    if progress_interval_seconds is not None:
        emit_log(
            "raw_success_scan_start",
            "Checking successful retrieval rows against raw files on disk",
            retrieval_rows=total_rows,
        )
    for row in conn.execute(query):
        checked_rows += 1
        raw_path = Path(row["raw_path"])
        if raw_path.exists() and raw_path.is_file():
            urls.add(row["url"])
        else:
            missing_raw_paths += 1
        if next_progress_at is not None and time.monotonic() >= next_progress_at:
            emit_log(
                "raw_success_scan_progress",
                "Still checking raw HTML file existence",
                checked_rows=checked_rows,
                retrieval_rows=total_rows,
                verified_success_urls=len(urls),
                missing_raw_paths=missing_raw_paths,
            )
            next_progress_at = time.monotonic() + progress_interval_seconds
    if progress_interval_seconds is not None:
        emit_log(
            "raw_success_scan_done",
            "Finished checking successful retrieval rows against raw files on disk",
            checked_rows=checked_rows,
            retrieval_rows=total_rows,
            verified_success_urls=len(urls),
            missing_raw_paths=missing_raw_paths,
        )
    return urls


def retrieval_status_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT
            status_code,
            COUNT(*) AS retrieval_rows,
            COUNT(DISTINCT url) AS unique_urls,
            MAX(attempted_at_utc) AS latest_attempted_at_utc
        FROM retrievals
        WHERE source = 'info_gov_bulletin'
        GROUP BY status_code
        ORDER BY status_code
    """
    return [row_to_dict(row) for row in conn.execute(query)]


def recent_failed_retrievals(conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    query = """
        SELECT attempted_at_utc, status_code, error, url
        FROM retrievals
        WHERE source = 'info_gov_bulletin'
          AND (status_code IS NULL OR status_code < 200 OR status_code >= 300)
        ORDER BY attempted_at_utc DESC
        LIMIT ?
    """
    return [row_to_dict(row) for row in conn.execute(query, (limit,))]


def summarize_remaining(
    config: Config,
    sample_limit: int = 10,
    *,
    log_progress: bool = False,
) -> dict[str, Any]:
    with connect_readonly(config.archive_db) as conn:
        require_tables(conn)
        if log_progress:
            emit_log(
                "candidate_scan_start",
                "Loading candidate URLs from archive DB",
                start=config.start.isoformat(),
                end=config.end.isoformat(),
                types=list(config.product_types),
            )
        candidates = candidate_rows(
            conn,
            start=config.start,
            end=config.end,
            product_types=config.product_types,
        )
        if log_progress:
            emit_log("candidate_scan_done", "Loaded candidate URLs", candidate_urls=len(candidates))
        successes = successful_raw_urls(
            conn,
            progress_interval_seconds=config.progress_interval_seconds if log_progress else None,
        )
        if log_progress:
            emit_log("remaining_calc_start", "Calculating candidate URLs still missing successful raw HTML")
        remaining = [row for row in candidates if row["url"] not in successes]

        remaining_by_year: dict[str, int] = {}
        remaining_by_type: dict[str, int] = {}
        for row in remaining:
            index_date = str(row["index_date"])
            remaining_by_year[index_date[:4]] = remaining_by_year.get(index_date[:4], 0) + 1
            product_type = str(row["product_type"])
            remaining_by_type[product_type] = remaining_by_type.get(product_type, 0) + 1
        if log_progress:
            emit_log("remaining_calc_done", "Finished calculating remaining work", remaining_urls=len(remaining))

        return {
            "checked_at_utc": utc_now_iso(),
            "data_root": str(config.data_root),
            "archive_db": str(config.archive_db),
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "types": list(config.product_types),
            "candidate_urls": len(candidates),
            "successful_raw_urls": len(successes.intersection({row["url"] for row in candidates})),
            "remaining_urls": len(remaining),
            "remaining_by_year": [
                {"index_year": year, "remaining_urls": count}
                for year, count in sorted(remaining_by_year.items())
            ],
            "remaining_by_type": [
                {"product_type": product_type, "remaining_urls": count}
                for product_type, count in sorted(remaining_by_type.items())
            ],
            "first_remaining": remaining[:sample_limit],
            "retrieval_status_counts": retrieval_status_counts(conn),
            "recent_failed_retrievals": recent_failed_retrievals(conn, limit=sample_limit),
        }


def choose_probe_url(config: Config, summary: dict[str, Any]) -> str | None:
    if config.probe_url:
        return config.probe_url
    first_remaining = summary.get("first_remaining") or []
    if first_remaining:
        return str(first_remaining[0]["url"])
    return None


def network_probe(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "hko-official-backfill-runner/1.0",
            "Range": "bytes=0-0",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response.read(1)
            return {
                "ok": True,
                "url": url,
                "status": getattr(response, "status", None),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": True,
            "http_ok": False,
            "url": url,
            "status": exc.code,
            "reason": exc.reason,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - this is an operator preflight.
        return {
            "ok": False,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def detail_command(config: Config) -> list[str]:
    return [
        config.python,
        "-u",
        str(DOWNLOADER),
        "official-details",
        "--types",
        ",".join(config.product_types),
        "--start",
        config.start.isoformat(),
        "--end",
        config.end.isoformat(),
        "--data-root",
        str(config.data_root),
        "--delay-seconds",
        str(config.delay_seconds),
        "--timeout-seconds",
        str(config.timeout_seconds),
        "--max-retries",
        str(config.max_retries),
        "--missing-success-only",
        "--limit",
        str(config.batch_size),
    ]


def pump_stream(
    stream: Any,
    log_handle: Any,
    *,
    event: str,
    batch_number: int,
) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            log_handle.write(line)
            log_handle.flush()
            emit_log(
                event,
                line.rstrip("\r\n"),
                batch=batch_number,
            )
    finally:
        stream.close()


def run_logged_command(
    command: list[str],
    *,
    stdout_log: Path,
    stderr_log: Path,
    batch_number: int,
    progress_interval_seconds: float,
) -> dict[str, Any]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    started = time.monotonic()
    with stdout_log.open("w", encoding="utf-8") as out_handle, stderr_log.open(
        "w",
        encoding="utf-8",
    ) as err_handle:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=pump_stream,
            args=(process.stdout, out_handle),
            kwargs={"event": "downloader_stdout", "batch_number": batch_number},
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump_stream,
            args=(process.stderr, err_handle),
            kwargs={"event": "downloader_stderr", "batch_number": batch_number},
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        next_progress_at = started + progress_interval_seconds
        while process.poll() is None:
            now = time.monotonic()
            if now >= next_progress_at:
                emit_log(
                    "batch_running",
                    "Downloader batch still running",
                    batch=batch_number,
                    elapsed_seconds=round(now - started, 1),
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                )
                next_progress_at = now + progress_interval_seconds
            time.sleep(0.5)

        returncode = process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    return {
        "started_at_utc": started_at,
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "returncode": returncode,
        "command": command,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def run_monitor(config: Config) -> dict[str, Any]:
    emit_log(
        "monitor_start",
        "Running coverage monitor",
        archive_db=str(config.archive_db),
        output_dir=str(config.monitor_output_dir),
    )
    command = [
        config.python,
        str(MONITOR),
        "--archive-db",
        str(config.archive_db),
        "--output-dir",
        str(config.monitor_output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        raise FinalizeError(f"Monitor failed: {payload}")
    try:
        payload["summary"] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FinalizeError(f"Monitor did not return JSON: {completed.stdout}") from exc
    emit_log(
        "monitor_done",
        "Coverage monitor completed",
        returncode=completed.returncode,
        completion_status=payload["summary"].get("completion", {}).get("completion_status"),
    )
    return payload


def run_finalizer(config: Config) -> dict[str, Any]:
    emit_log(
        "finalizer_start",
        "Running export/package finalizer",
        data_root=str(config.data_root),
        output_dir=str(config.output_dir),
        bundle_stem=config.bundle_stem,
    )
    command = [
        config.python,
        str(FINALIZER),
        "--archive-db",
        str(config.archive_db),
        "--data-root",
        str(config.data_root),
        "--output-dir",
        str(config.output_dir),
        "--bundle-stem",
        config.bundle_stem,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        raise FinalizeError(f"Finalizer failed: {payload}")
    with suppress(json.JSONDecodeError):
        payload["summary"] = json.loads(completed.stdout)
    emit_log(
        "finalizer_done",
        "Export/package finalizer completed",
        returncode=completed.returncode,
        stdout=completed.stdout.strip()[:2000],
    )
    return payload


class RunLock:
    def __init__(self, lock_path: Path, force: bool) -> None:
        self.lock_path = lock_path
        self.force = force
        self.fd: int | None = None

    def __enter__(self) -> RunLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.force and self.lock_path.exists():
            emit_log("lock_force_remove", "Removing stale lock because --force-lock was passed", lock_path=str(self.lock_path))
            self.lock_path.unlink()
        try:
            self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValidationError(
                f"Another backfill run appears to be active: {self.lock_path}. "
                "Remove it only if you are sure no run is active, or pass --force-lock.",
            ) from exc
        payload = {
            "pid": os.getpid(),
            "started_at_utc": utc_now_iso(),
            "script": str(Path(__file__).resolve()),
        }
        os.write(self.fd, json.dumps(payload, indent=2).encode("utf-8"))
        os.close(self.fd)
        self.fd = None
        emit_log("lock_acquired", "Acquired single-run lock", lock_path=str(self.lock_path), pid=os.getpid())
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        with suppress(FileNotFoundError):
            self.lock_path.unlink()
        emit_log("lock_released", "Released single-run lock", lock_path=str(self.lock_path))


def maybe_finalize(config: Config, status: dict[str, Any], status_path: Path) -> bool:
    monitor_payload = run_monitor(config)
    status["monitor"] = monitor_payload
    write_json(status_path, status)
    completion_status = (
        monitor_payload.get("summary", {})
        .get("completion", {})
        .get("completion_status")
    )
    if completion_status != "complete_no_gap":
        emit_log(
            "finalize_skip",
            "Monitor did not report complete_no_gap; not packaging as complete",
            completion_status=completion_status,
            status_path=str(status_path),
        )
        return False
    if config.finalize:
        status["finalizer"] = run_finalizer(config)
        write_json(status_path, status)
    return True


def run_backfill(config: Config) -> dict[str, Any]:
    run_logs = config.data_root / "run_logs"
    status_path = run_logs / "official_remaining_backfill_status.json"
    lock_path = run_logs / "official_remaining_backfill.lock"
    status: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "started_at_utc": utc_now_iso(),
        "data_root": str(config.data_root),
        "archive_db": str(config.archive_db),
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "types": list(config.product_types),
        "batch_size": config.batch_size,
        "batches": [],
    }

    emit_log(
        "run_start",
        "Starting resumable HKO official detail backfill",
        data_root=str(config.data_root),
        archive_db=str(config.archive_db),
        start=config.start.isoformat(),
        end=config.end.isoformat(),
        types=list(config.product_types),
        batch_size=config.batch_size,
        delay_seconds=config.delay_seconds,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        progress_interval_seconds=config.progress_interval_seconds,
        finalize=config.finalize,
    )
    with RunLock(lock_path, config.force_lock):
        emit_log(
            "status_path",
            "Run status JSON will be updated after each phase",
            status_path=str(status_path),
        )
        emit_log("summary_start", "Inspecting archive DB and raw HTML files")
        initial_summary = summarize_remaining(config, log_progress=True)
        status["initial_summary"] = initial_summary
        write_json(status_path, status)
        emit_summary("summary_done", initial_summary)

        if config.check_only:
            status["completed_at_utc"] = utc_now_iso()
            status["result"] = "check_only"
            write_json(status_path, status)
            emit_log("run_complete", "Check-only run complete", result=status["result"])
            return status

        if initial_summary["remaining_urls"] == 0:
            status["result"] = "nothing_to_fetch"
            emit_log("nothing_to_fetch", "No candidate URLs are missing successful raw HTML")
            if config.finalize:
                status["finalize_ready"] = maybe_finalize(config, status, status_path)
            status["completed_at_utc"] = utc_now_iso()
            write_json(status_path, status)
            emit_log("run_complete", "Run complete", result=status["result"])
            return status

        probe_url = choose_probe_url(config, initial_summary)
        if not config.skip_network_check and probe_url:
            emit_log(
                "network_preflight_start",
                "Testing live HTTP access from this PowerShell process",
                url=probe_url,
                timeout_seconds=config.probe_timeout_seconds,
            )
            probe = network_probe(probe_url, config.probe_timeout_seconds)
            status["network_preflight"] = probe
            write_json(status_path, status)
            emit_log("network_preflight_done", "Network preflight finished", **probe)
            if not probe["ok"]:
                raise NetworkPreflightError(
                    "Network preflight failed from this shell. "
                    f"URL={probe_url} error={probe.get('error_type')}: {probe.get('error')}",
                )

        current_summary = initial_summary
        stalled_batches = 0
        batches_run = 0
        command = detail_command(config)
        emit_log(
            "downloader_command",
            "Prepared child downloader command",
            command=subprocess.list2cmdline(command),
        )

        while current_summary["remaining_urls"] > 0:
            if config.max_batches and batches_run >= config.max_batches:
                status["result"] = "max_batches_reached"
                status["completed_at_utc"] = utc_now_iso()
                write_json(status_path, status)
                emit_log(
                    "run_pause",
                    "Reached configured maximum number of batches",
                    max_batches=config.max_batches,
                    remaining_urls=current_summary["remaining_urls"],
                    status_path=str(status_path),
                )
                return status

            batch_number = batches_run + 1
            stdout_log = run_logs / f"official_remaining_backfill_batch_{batch_number:06d}.out.log"
            stderr_log = run_logs / f"official_remaining_backfill_batch_{batch_number:06d}.err.log"
            before = current_summary
            emit_log(
                "batch_start",
                "Starting downloader batch",
                batch=batch_number,
                remaining_urls=before["remaining_urls"],
                batch_size=config.batch_size,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
            )
            batch_result = run_logged_command(
                command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                batch_number=batch_number,
                progress_interval_seconds=config.progress_interval_seconds,
            )
            batch_result["batch"] = batch_number
            if batch_result["returncode"] != 0:
                status["batches"].append(batch_result)
                status["completed_at_utc"] = utc_now_iso()
                status["result"] = "downloader_failed"
                write_json(status_path, status)
                emit_log(
                    "batch_failed",
                    "Downloader batch failed",
                    level="ERROR",
                    batch=batch_number,
                    returncode=batch_result["returncode"],
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    status_path=str(status_path),
                )
                raise DownloaderBatchError(
                    f"Downloader batch {batch_number} failed with return code "
                    f"{batch_result['returncode']}. See {stderr_log}",
                )

            after = summarize_remaining(config, log_progress=True)
            batch_result["before_remaining_urls"] = before["remaining_urls"]
            batch_result["after_remaining_urls"] = after["remaining_urls"]
            batch_result["remaining_delta"] = before["remaining_urls"] - after["remaining_urls"]
            batch_result["successful_raw_urls"] = after["successful_raw_urls"]
            status["batches"].append(batch_result)
            status["latest_summary"] = after
            write_json(status_path, status)
            emit_log(
                "batch_done",
                "Downloader batch completed and archive state was recalculated",
                **batch_result,
            )
            emit_summary("summary_after_batch", after)

            if config.monitor_each_batch:
                emit_log("monitor_each_batch", "Running monitor because --monitor-each-batch is enabled", batch=batch_number)
                status["monitor"] = run_monitor(config)
                write_json(status_path, status)

            if batch_result["remaining_delta"] <= 0:
                stalled_batches += 1
                emit_log(
                    "batch_no_progress",
                    "Batch did not reduce remaining URL count",
                    level="WARNING",
                    batch=batch_number,
                    stalled_batches=stalled_batches,
                    remaining_urls=current_summary["remaining_urls"],
                )
            else:
                stalled_batches = 0
            batches_run += 1
            current_summary = after

            if stalled_batches >= config.max_stalled_batches:
                complete = False
                if config.finalize:
                    complete = maybe_finalize(config, status, status_path)
                if complete:
                    status["result"] = "complete_no_gap_with_unfetchable_candidate_urls"
                    status["completed_at_utc"] = utc_now_iso()
                    write_json(status_path, status)
                    emit_log(
                        "run_complete",
                        "Monitor is complete_no_gap even though some candidate URLs remain unfetchable",
                        result=status["result"],
                        status_path=str(status_path),
                    )
                    return status
                status["result"] = "stalled_with_remaining_urls"
                status["completed_at_utc"] = utc_now_iso()
                write_json(status_path, status)
                emit_log(
                    "run_stalled",
                    "Backfill stalled with remaining candidate URLs",
                    level="ERROR",
                    remaining_urls=current_summary["remaining_urls"],
                    stalled_batches=stalled_batches,
                    status_path=str(status_path),
                )
                raise StalledBackfillError(
                    f"Backfill stalled for {stalled_batches} batch(es); "
                    f"{current_summary['remaining_urls']} candidate URLs still lack successful raw HTML. "
                    f"See {status_path}",
                )

        status["latest_summary"] = current_summary
        status["result"] = "all_candidate_urls_have_successful_raw_html"
        if config.finalize:
            status["finalize_ready"] = maybe_finalize(config, status, status_path)
            if not status["finalize_ready"]:
                raise FinalizeError(
                    "Raw candidate fetch completed, but monitor did not report complete_no_gap. "
                    f"See {status_path}",
                )
        status["completed_at_utc"] = utc_now_iso()
        write_json(status_path, status)
        emit_log(
            "run_complete",
            "All candidate URLs have successful raw HTML",
            result=status["result"],
            status_path=str(status_path),
        )
        return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the HKO official press forecast detail backfill. "
            "Each run fetches only candidate URLs missing a successful raw HTML retrieval."
        ),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for child scripts.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive-db", type=Path, default=None)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--max-stalled-batches", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
        help="Print a live still-running line this often while a downloader batch is active.",
    )
    parser.add_argument("--probe-url", default=None)
    parser.add_argument("--probe-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--skip-network-check", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--monitor-each-batch", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--monitor-output-dir", type=Path, default=DEFAULT_MONITOR_DIR)
    parser.add_argument("--bundle-stem", default=DEFAULT_BUNDLE_STEM)
    parser.add_argument("--force-lock", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = validate_config(args)
        result = run_backfill(config)
        emit_log("process_exit", "Process exiting successfully", result=result.get("result"))
        return 0
    except BackfillError as exc:
        emit_log("process_error", str(exc), level="ERROR", stream=sys.stderr, exit_code=exc.exit_code)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
