from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATA_ROOT = PROJECT_PATHS.run_root / "work" / "hko_press_2000_2026_resume"
DEFAULT_SOURCE_DB = Path(r"C:\hko_press_2000_2026\metadata\archive.sqlite3")
DEFAULT_ARCHIVE_DB = DEFAULT_DATA_ROOT / "metadata" / "archive.sqlite3"
DEFAULT_START = "2013-11-15"
DEFAULT_END = "2026-06-20"
DEFAULT_PROBE_URL = "https://www.info.gov.hk/gia/wr/201311/15/P201311150757.htm"
DEFAULT_BUNDLE_STEM = "hko_official_press_weather_forecasts_20000101_20260620"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_resume_db(source_db: Path, archive_db: Path) -> dict[str, Any]:
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    if archive_db.exists():
        return {
            "created": False,
            "archive_db": str(archive_db),
            "bytes": archive_db.stat().st_size,
        }
    if not source_db.exists():
        raise FileNotFoundError(f"Missing source DB for resume backup: {source_db}")

    source = sqlite3.connect(str(source_db))
    destination = sqlite3.connect(str(archive_db))
    try:
        source.backup(destination, pages=5000)
    finally:
        destination.close()
        source.close()
    return {
        "created": True,
        "source_db": str(source_db),
        "archive_db": str(archive_db),
        "bytes": archive_db.stat().st_size,
    }


def network_probe(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return {
                "ok": True,
                "url": url,
                "status": int(response.status),
                "reason": response.reason,
            }
    except urllib.error.HTTPError as exc:
        # HTTP errors still prove sockets and TLS worked. The downloader can handle
        # non-2xx detail URLs; this preflight is only about environment access.
        return {
            "ok": True,
            "url": url,
            "status": int(exc.code),
            "reason": str(exc.reason),
        }
    except Exception as exc:  # noqa: BLE001 - surfaced as a preflight failure.
        return {
            "ok": False,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_json_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(command)}\n{completed.stdout}") from exc


def compact_monitor_summary(summary: dict[str, Any]) -> dict[str, Any]:
    coverage = summary.get("combined_selected_target_coverage") or {}
    return {
        "completion_status": summary.get("completion_status"),
        "target_window_start": summary.get("target_window_start"),
        "target_window_end": summary.get("target_window_end"),
        "observed_target_days": coverage.get("observed_target_days"),
        "expected_target_days": coverage.get("expected_target_days"),
        "missing_target_days": coverage.get("missing_target_days"),
        "first_missing_scored_gap": summary.get("first_missing_scored_gap"),
        "largest_missing_scored_gap": summary.get("largest_missing_scored_gap"),
        "last_press_selected_target_date": summary.get("last_press_selected_target_date"),
        "candidate_raw_missing_urls": summary.get("candidate_raw_missing_urls"),
    }


def run_monitor(python: Path, archive_db: Path, output_dir: Path) -> dict[str, Any]:
    command = [
        str(python),
        str(REPO_ROOT / "scripts" / "monitor_hko_official_backfill.py"),
        "--archive-db",
        str(archive_db),
        "--output-dir",
        str(output_dir),
    ]
    return run_json_command(command, REPO_ROOT)


def finalize_if_complete(
    python: Path,
    data_root: Path,
    archive_db: Path,
    details_log: Path,
    output_dir: Path,
    monitor_output_dir: Path,
    bundle_stem: str,
) -> dict[str, Any]:
    command = [
        str(python),
        str(REPO_ROOT / "scripts" / "finalize_hko_official_backfill.py"),
        "--data-root",
        str(data_root),
        "--archive-db",
        str(archive_db),
        "--details-log",
        str(details_log),
        "--output-dir",
        str(output_dir),
        "--monitor-output-dir",
        str(monitor_output_dir),
        "--bundle-stem",
        bundle_stem,
    ]
    return run_json_command(command, REPO_ROOT)


def detail_command(
    python: Path,
    data_root: Path,
    start: str,
    end: str,
    delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    limit: int | None,
) -> list[str]:
    command = [
        str(python),
        str(REPO_ROOT / "scripts" / "hko_forecast_archive_downloader_rss" / "hko_archive.py"),
        "official-details",
        "--types",
        "local,5day,7day,9day",
        "--start",
        start,
        "--end",
        end,
        "--data-root",
        str(data_root),
        "--delay-seconds",
        str(delay_seconds),
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-retries",
        str(max_retries),
        "--missing-success-only",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def launch_details(
    command: list[str],
    *,
    stdout_log: Path,
    stderr_log: Path,
    foreground: bool,
) -> dict[str, Any]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_log.open("w", encoding="utf-8")
    stderr_handle = stderr_log.open("w", encoding="utf-8")
    try:
        if foreground:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
            return {
                "mode": "foreground",
                "returncode": completed.returncode,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            creationflags=creationflags,
        )
        return {
            "mode": "background",
            "pid": process.pid,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }
    finally:
        stdout_handle.close()
        stderr_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume the HKO official press forecast detail crawl and enforce no-gap completion gates."
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-url", default=DEFAULT_PROBE_URL)
    parser.add_argument("--probe-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--skip-network-check", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--bundle-stem", default=DEFAULT_BUNDLE_STEM)
    parser.add_argument("--watch", action="store_true", help="Keep retrying until the preflight succeeds and the crawl can run.")
    parser.add_argument("--watch-interval-seconds", type=float, default=300.0)
    parser.add_argument("--watch-max-probes", type=int, default=3, help="Maximum failed probes before exiting (default: 3).")
    return parser.parse_args()


def attempt_resume(args: argparse.Namespace, data_root: Path, archive_db: Path, probe_attempt: int) -> str:
    monitor_output_dir = (
        PROJECT_PATHS.run_root
        / "experiments"
        / "legacy"
        / "0000_research_state_and_data_contract"
        / "hko_official_backfill_monitor"
        / "artifacts"
    )
    output_dir = PROJECT_PATHS.data_root / "datasets"
    stdout_log = data_root / "run_logs" / f"official_details_resume_{args.start.replace('-', '')}_{args.end.replace('-', '')}.out.log"
    stderr_log = data_root / "run_logs" / f"official_details_resume_{args.start.replace('-', '')}_{args.end.replace('-', '')}.err.log"

    status: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "data_root": str(data_root),
        "archive_db": str(archive_db),
        "start": args.start,
        "end": args.end,
        "probe_attempt": probe_attempt,
        "network_preflight": None,
    }
    status["resume_db"] = ensure_resume_db(args.source_db, archive_db)
    monitor_summary = run_monitor(args.python, archive_db, monitor_output_dir)
    status["monitor_before"] = compact_monitor_summary(monitor_summary)
    if monitor_summary.get("completion_status") == "complete_no_gap":
        status["action"] = "already_complete_no_gap"
        if args.finalize:
            status["finalize"] = finalize_if_complete(
                args.python,
                data_root,
                archive_db,
                stdout_log,
                output_dir,
                monitor_output_dir,
                args.bundle_stem,
            )
        status_path = data_root / "run_logs" / "resume_status.json"
        write_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return "complete"

    if not args.skip_network_check:
        probe = network_probe(args.probe_url, args.probe_timeout_seconds)
        status["network_preflight"] = probe
        if not probe["ok"]:
            status["action"] = "watch_waiting_for_network" if args.watch else "not_started_network_unavailable"
            status_path = data_root / "run_logs" / "resume_status.json"
            write_json(status_path, status)
            print(json.dumps(status, indent=2, sort_keys=True))
            return "network_unavailable"

    if args.check_only:
        status["action"] = "check_only"
        status_path = data_root / "run_logs" / "resume_status.json"
        write_json(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return "check_only"

    command = detail_command(
        args.python,
        data_root,
        args.start,
        args.end,
        args.delay_seconds,
        args.timeout_seconds,
        args.max_retries,
        args.limit,
    )
    status["detail_command"] = command
    status["details"] = launch_details(
        command,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        foreground=args.foreground,
    )

    if args.foreground:
        monitor_after = run_monitor(args.python, archive_db, monitor_output_dir)
        status["monitor_after"] = compact_monitor_summary(monitor_after)
        if args.finalize and monitor_after.get("completion_status") == "complete_no_gap":
            status["finalize"] = finalize_if_complete(
                args.python,
                data_root,
                archive_db,
                stdout_log,
                output_dir,
                monitor_output_dir,
                args.bundle_stem,
            )
    status["action"] = "details_started"
    status_path = data_root / "run_logs" / "resume_status.json"
    write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return "details_started"


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    archive_db = args.archive_db
    if archive_db == DEFAULT_ARCHIVE_DB and data_root != DEFAULT_DATA_ROOT:
        archive_db = data_root / "metadata" / "archive.sqlite3"
    data_root.mkdir(parents=True, exist_ok=True)
    for child in ("raw", "bronze", "reports", "run_logs"):
        (data_root / child).mkdir(parents=True, exist_ok=True)

    probe_attempt = 1
    if args.watch_max_probes < 1:
        raise ValueError("--watch-max-probes must be at least 1")
    while True:  # repo-doctor: allow-unsafe-default - watch_max_probes is mandatory and finite
        outcome = attempt_resume(args, data_root, archive_db, probe_attempt)
        if outcome == "network_unavailable" and args.watch:
            if probe_attempt >= args.watch_max_probes:
                raise SystemExit(100)
            probe_attempt += 1
            time.sleep(args.watch_interval_seconds)
            continue
        if outcome == "network_unavailable":
            raise SystemExit(100)
        return


if __name__ == "__main__":
    main()
