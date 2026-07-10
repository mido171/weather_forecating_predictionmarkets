from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ID = "0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709"
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / DEFAULT_EXPERIMENT_ID


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def date_span(start: date, end: date) -> list[date]:
    if end < start:
        raise argparse.ArgumentTypeError("end-date must be >= start-date")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def wp(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def ensure_dir(path: Path) -> None:
    Path(wp(path)).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    Path(wp(path)).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def file_size(path: Path) -> int:
    if not os.path.exists(wp(path)):
        return 0
    if os.path.isfile(wp(path)):
        return os.path.getsize(wp(path))
    total = 0
    for dirpath, _dirnames, filenames in os.walk(wp(path)):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except FileNotFoundError:
                continue
    return total


def drive_free_bytes(path: Path) -> int:
    resolved = path.resolve()
    return int(shutil.disk_usage(str(resolved.anchor or resolved)).free)


def add_int(summary: dict[str, Any], key: str, amount: int) -> None:
    summary[key] = int(summary.get(key, 0)) + int(amount)


@dataclass
class DayJob:
    day: date
    process: subprocess.Popen[bytes]
    shard_dir: Path
    stdout_path: Path
    stderr_path: Path
    stdout_handle: TextIO
    stderr_handle: TextIO
    started_at: datetime


def build_worker_command(args: argparse.Namespace, day: date, shard_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "backfill_public_weather_to_postgres.py"),
        "--start-date",
        day.isoformat(),
        "--end-date",
        day.isoformat(),
        "--sources",
        args.sources,
        "--cycles",
        args.cycles,
        "--leads",
        args.leads,
        "--experiment-id",
        args.experiment_id,
        "--experiment-dir",
        str(shard_dir),
        "--progress-every",
        str(args.progress_every),
        "--max-attempts",
        str(args.max_attempts),
        "--max-staging-gb",
        str(args.max_staging_gb),
        "--stop-free-gb",
        str(args.stop_free_gb),
        "--execution-mode",
        args.execution_mode,
        "--model-fetch-workers",
        str(args.model_fetch_workers),
        "--model-range-workers",
        str(args.model_range_workers),
        "--model-normalize-workers",
        str(args.model_normalize_workers),
        "--himawari-workers",
        str(args.himawari_workers),
        "--model-range-coalesce-gap-bytes",
        str(args.model_range_coalesce_gap_bytes),
        "--staging-root",
        str(args.staging_root / args.experiment_id / day.isoformat()),
    ]
    if args.cpu_telemetry:
        command.append("--cpu-telemetry")
    command.append("--no-skip-existing-complete" if args.no_skip_existing_complete else "--skip-existing-complete")
    return command


def launch_day(args: argparse.Namespace, day: date) -> DayJob:
    shard_dir = args.experiment_dir / "shards" / day.isoformat()
    ensure_dir(shard_dir / "logs")
    stdout_path = shard_dir / "logs" / "worker_stdout.log"
    stderr_path = shard_dir / "logs" / "worker_stderr.log"
    stdout_handle = open(wp(stdout_path), "w", encoding="utf-8")  # noqa: SIM115
    stderr_handle = open(wp(stderr_path), "w", encoding="utf-8")  # noqa: SIM115
    env = os.environ.copy()
    if args.database_url:
        env["HKG_TMAX_DATABASE_URL"] = args.database_url
    command = build_worker_command(args, day, shard_dir)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=env,
    )
    return DayJob(
        day=day,
        process=process,
        shard_dir=shard_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        started_at=utc_now(),
    )


def read_day_metrics(shard_dir: Path) -> dict[str, Any] | None:
    metrics_path = shard_dir / "results" / "metrics.json"
    if not os.path.exists(wp(metrics_path)):
        return None
    return json.loads(Path(wp(metrics_path)).read_text(encoding="utf-8"))


def aggregate_metrics(
    args: argparse.Namespace,
    *,
    started_at: datetime,
    completed: dict[str, dict[str, Any]],
    exit_codes: dict[str, int],
    monitor_summary: dict[str, Any],
) -> dict[str, Any]:
    elapsed_seconds = (utc_now() - started_at).total_seconds()
    aggregate: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "sources": args.sources,
        "cycles": args.cycles,
        "leads": args.leads,
        "max_workers": args.max_workers,
        "started_at_utc": iso(started_at),
        "completed_at_utc": iso(utc_now()),
        "elapsed_seconds": elapsed_seconds,
        "exit_codes": exit_codes,
        "day_metrics": completed,
        "day_elapsed_seconds": {},
        "failed_days": [day for day, code in sorted(exit_codes.items()) if code != 0],
        "max_aggregate_staging_bytes": monitor_summary.get("max_aggregate_staging_bytes", 0),
        "min_free_disk_bytes": monitor_summary.get("min_free_disk_bytes"),
    }
    sum_keys = [
        "source_issues_touched",
        "fetch_ok",
        "fetch_failed",
        "normalize_ok",
        "normalize_failed",
        "task_errors",
        "station_features_upserted",
        "area_features_upserted",
        "raw_bytes_deleted",
        "raw_files_deleted",
    ]
    by_source: dict[str, dict[str, int]] = {}
    for day, metrics in sorted(completed.items()):
        aggregate["day_elapsed_seconds"][day] = metrics.get("elapsed_seconds")
        for key in sum_keys:
            add_int(aggregate, key, int(metrics.get(key, 0) or 0))
        aggregate["max_raw_object_bytes"] = max(
            int(aggregate.get("max_raw_object_bytes", 0)),
            int(metrics.get("max_raw_object_bytes", 0) or 0),
        )
        aggregate["max_worker_staging_bytes"] = max(
            int(aggregate.get("max_worker_staging_bytes", 0)),
            int(metrics.get("max_staging_bytes", 0) or 0),
        )
        for source, row in (metrics.get("by_source") or {}).items():
            source_out = by_source.setdefault(source, {})
            for key, value in row.items():
                if key.startswith("max_"):
                    source_out[key] = max(int(source_out.get(key, 0)), int(value or 0))
                elif isinstance(value, int):
                    source_out[key] = int(source_out.get(key, 0)) + value
    aggregate["by_source"] = by_source
    aggregate["status"] = "complete" if not aggregate["failed_days"] else "complete_with_failed_shards"
    return aggregate


def write_markdown_report(experiment_dir: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Parallel Day-Sharded Backfill Results",
        "",
        f"Status: `{aggregate.get('status')}`",
        f"Date range: `{aggregate.get('start_date')}` to `{aggregate.get('end_date')}`",
        f"Elapsed seconds: `{aggregate.get('elapsed_seconds')}`",
        f"Max workers: `{aggregate.get('max_workers')}`",
        "",
        "## Counts",
        "",
        f"- Source issues touched: `{aggregate.get('source_issues_touched', 0)}`",
        f"- Fetch ok: `{aggregate.get('fetch_ok', 0)}`",
        f"- Fetch failed: `{aggregate.get('fetch_failed', 0)}`",
        f"- Normalize ok: `{aggregate.get('normalize_ok', 0)}`",
        f"- Normalize failed: `{aggregate.get('normalize_failed', 0)}`",
        f"- Task errors: `{aggregate.get('task_errors', 0)}`",
        f"- Station features upserted: `{aggregate.get('station_features_upserted', 0)}`",
        f"- Area features upserted: `{aggregate.get('area_features_upserted', 0)}`",
        f"- Raw files deleted: `{aggregate.get('raw_files_deleted', 0)}`",
        f"- Raw bytes deleted: `{aggregate.get('raw_bytes_deleted', 0)}`",
        "",
        "## Disk",
        "",
        f"- Max aggregate staging bytes: `{aggregate.get('max_aggregate_staging_bytes')}`",
        f"- Max single-worker staging bytes: `{aggregate.get('max_worker_staging_bytes')}`",
        f"- Max raw object bytes: `{aggregate.get('max_raw_object_bytes')}`",
        f"- Minimum free disk bytes observed: `{aggregate.get('min_free_disk_bytes')}`",
        "",
        "## By Source",
        "",
    ]
    for source, row in sorted((aggregate.get("by_source") or {}).items()):
        lines.append(f"- `{source}`: {row}")
    if aggregate.get("failed_days"):
        lines.extend(["", "## Failed Days", ""])
        for day in aggregate["failed_days"]:
            lines.append(f"- `{day}` exit code `{aggregate['exit_codes'].get(day)}`")
    Path(wp(experiment_dir / "RESULTS_PARALLEL.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    ensure_dir(args.experiment_dir / "logs")
    ensure_dir(args.experiment_dir / "results")
    days = date_span(args.start_date, args.end_date)
    pending = list(days)
    running: dict[str, DayJob] = {}
    completed: dict[str, dict[str, Any]] = {}
    exit_codes: dict[str, int] = {}
    monitor_path = args.experiment_dir / "logs" / "parallel_monitor.jsonl"
    started_at = utc_now()
    monitor_summary: dict[str, Any] = {"max_aggregate_staging_bytes": 0, "min_free_disk_bytes": None}
    print(f"[parallel-start] days={len(days)} max_workers={args.max_workers}", flush=True)

    while pending or running:
        while pending and len(running) < args.max_workers:
            day = pending.pop(0)
            job = launch_day(args, day)
            running[day.isoformat()] = job
            print(f"[launch] {day} pid={job.process.pid}", flush=True)

        for day_key, job in list(running.items()):
            code = job.process.poll()
            if code is None:
                continue
            job.stdout_handle.close()
            job.stderr_handle.close()
            exit_codes[day_key] = int(code)
            metrics = read_day_metrics(job.shard_dir)
            if metrics is not None:
                completed[day_key] = metrics
            running.pop(day_key)
            print(f"[complete] {day_key} exit={code}", flush=True)

        aggregate_staging = sum(
            file_size(args.staging_root / args.experiment_id / day.isoformat())
            for day in days
        )
        free_bytes = drive_free_bytes(args.experiment_dir)
        monitor_summary["max_aggregate_staging_bytes"] = max(
            int(monitor_summary.get("max_aggregate_staging_bytes", 0)),
            aggregate_staging,
        )
        current_min_free = monitor_summary.get("min_free_disk_bytes")
        monitor_summary["min_free_disk_bytes"] = (
            free_bytes if current_min_free is None else min(int(current_min_free), free_bytes)
        )
        event = {
            "observed_at_utc": iso(utc_now()),
            "running_days": sorted(running),
            "pending_days": len(pending),
            "completed_days": len(completed),
            "aggregate_staging_bytes": aggregate_staging,
            "free_disk_bytes": free_bytes,
        }
        with open(wp(monitor_path), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        print(
            "[monitor] "
            f"running={len(running)} pending={len(pending)} completed={len(completed)} "
            f"staging={aggregate_staging} free_gb={free_bytes / 1024**3:.2f}",
            flush=True,
        )
        if pending or running:
            time.sleep(args.monitor_interval_seconds)

    aggregate = aggregate_metrics(
        args,
        started_at=started_at,
        completed=completed,
        exit_codes=exit_codes,
        monitor_summary=monitor_summary,
    )
    write_json(args.experiment_dir / "results" / "parallel_aggregate_metrics.json", aggregate)
    write_markdown_report(args.experiment_dir, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return 0 if aggregate["status"] == "complete" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--sources", default="gfs,gefs_control,himawari_b13_s0510,radar")
    parser.add_argument("--cycles", default="0,6,12,18")
    parser.add_argument("--leads", default="0:48:3")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--execution-mode", choices=["serial", "optimized"], default="serial")
    parser.add_argument("--model-fetch-workers", type=int, default=8)
    parser.add_argument("--model-range-workers", type=int, default=4)
    parser.add_argument("--model-normalize-workers", type=int, default=2)
    parser.add_argument("--himawari-workers", type=int, default=8)
    parser.add_argument("--model-range-coalesce-gap-bytes", type=int, default=0)
    parser.add_argument("--cpu-telemetry", action="store_true", default=False)
    parser.add_argument("--staging-root", type=Path, default=REPO_ROOT / "_weather_backfill_staging" / "day_shards")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-staging-gb", type=float, default=4.0)
    parser.add_argument("--stop-free-gb", type=float, default=50.0)
    parser.add_argument("--monitor-interval-seconds", type=int, default=30)
    parser.add_argument("--no-skip-existing-complete", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
