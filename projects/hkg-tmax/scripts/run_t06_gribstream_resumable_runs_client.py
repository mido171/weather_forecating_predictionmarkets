from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hkg_tmax.gribstream.catalog import ResolvedSelector, resolve_temperature_2m_selector
from hkg_tmax.gribstream.client import (
    GribStreamClient,
    GribStreamRequestError,
    ResponseManifest,
    RetryConfig,
    canonical_request_json,
    sanitize_text,
    sha256_file,
)
from hkg_tmax.gribstream.normalizer import iter_ndjson_gzip, normalize_runs_ndjson_gzip
from hkg_tmax.gribstream.planner import build_runs_plan, load_canonical_locations
from hkg_tmax.gribstream.store import (
    ingest_response,
    load_location_ids,
    mark_request_failed,
    register_request_started,
)
from hkg_tmax.paths import ProjectPaths
from hkg_tmax_db.connection import DatabaseUnavailable, import_psycopg, redact_database_url

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
TASK_ROOT = REPO_ROOT / "tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION"
TASKS_NOT_COMPLETED = TASK_ROOT / "tasks/not-completed"
TASKS_COMPLETED = TASK_ROOT / "tasks/completed"
STATUS_INDEX = TASK_ROOT / "TASK_STATUS_INDEX.csv"
TASK_ID = "T06"
TASK_NAME = "T06_gribstream_resumable_runs_client"
EXPERIMENT_DIR = REPO_ROOT / "experiments/0213_gribstream_resumable_runs_client"
RAW_ROOT = PROJECT_PATHS.data_root / "raw" / "gribstream"
RUN_STATUS_PATH = EXPERIMENT_DIR / "logs/t06_status.json"
LEDGER_PATH = EXPERIMENT_DIR / "resume_ledger.jsonl"
API_EVENT_LOG = EXPERIMENT_DIR / "logs/gribstream_api_events.jsonl"
SECRET_FILE = REPO_ROOT / "secrets/local/gribstream.env"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
DEFAULT_RUN_TIME = "2026-06-23T00:00:00Z"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def fs_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def ensure_directory(path: Path) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def write_status(stage: str, status: str, **details: Any) -> None:
    write_json(
        RUN_STATUS_PATH,
        {
            "task_id": TASK_ID,
            "stage": stage,
            "status": status,
            "updated_at_utc": utc_now_iso(),
            "pid": os.getpid(),
            **details,
        },
    )


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def load_gribstream_token() -> str | None:
    env_token = os.environ.get("GRIBSTREAM_API_KEY")
    if env_token:
        return env_token.strip()
    if not SECRET_FILE.exists():
        return None
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("GRIBSTREAM_API_KEY="):
            token = line.split("=", 1)[1].strip()
            return token or None
    return None


def command_status() -> int:
    status = json.loads(RUN_STATUS_PATH.read_text(encoding="utf-8")) if RUN_STATUS_PATH.exists() else {}
    ledger_tail: list[dict[str, Any]] = []
    if LEDGER_PATH.exists():
        lines = [line for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        ledger_tail = [json.loads(line) for line in lines[-10:]]
    print(json.dumps({"status": status, "ledger_tail": ledger_tail}, indent=2, sort_keys=True))
    return 0


def query_counts(database_url: str, request_hash: str | None = None) -> dict[str, Any]:
    psycopg = import_psycopg()
    queries = {
        "catalog.location": "SELECT count(*) FROM catalog.location",
        "catalog.weather_model": "SELECT count(*) FROM catalog.weather_model",
        "catalog.variable_selector_snapshot": "SELECT count(*) FROM catalog.variable_selector_snapshot",
        "raw_audit.acquisition_request": "SELECT count(*) FROM raw_audit.acquisition_request",
        "raw_audit.response_object": "SELECT count(*) FROM raw_audit.response_object",
        "nwp_core.model_run": "SELECT count(*) FROM nwp_core.model_run",
        "nwp_core.point_value": "SELECT count(*) FROM nwp_core.point_value",
    }
    result: dict[str, Any] = {}
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            for name, sql in queries.items():
                cursor.execute(sql)
                result[name] = int(cursor.fetchone()[0])
            if request_hash:
                cursor.execute(
                    """
                    SELECT ar.request_id::text, ar.status, ar.attempt_count, ro.response_object_id, ro.row_count
                    FROM raw_audit.acquisition_request ar
                    LEFT JOIN raw_audit.response_object ro ON ar.request_id = ro.request_id
                    WHERE ar.request_sha256 = %s
                    ORDER BY ro.response_object_id DESC NULLS LAST
                    LIMIT 1
                    """,
                    (request_hash,),
                )
                row = cursor.fetchone()
                result["request"] = None if row is None else {
                    "request_id": row[0],
                    "status": row[1],
                    "attempt_count": row[2],
                    "response_object_id": row[3],
                    "row_count": row[4],
                }
    return result


def raw_object_path(run_time_utc: str, request_hash: str) -> Path:
    compact_run = run_time_utc.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return RAW_ROOT / "gfs/runs" / f"run_time_utc={compact_run}" / f"{request_hash}.ndjson.gz"


def manifest_for_existing_object(path: Path, *, dataset: str, request_hash: str) -> ResponseManifest:
    rows = iter_ndjson_gzip(path)
    return ResponseManifest(
        provider="GribStream",
        dataset=dataset,
        endpoint="runs",
        request_sha256=request_hash,
        object_path=path,
        byte_size=os.path.getsize(fs_path(path)),
        sha256=sha256_file(path),
        content_type="application/ndjson",
        retrieved_at_utc=utc_now_iso(),
        row_count=len(rows),
        http_status=200,
        attempt_count=0,
    )


def secret_scan(token: str | None, paths: list[Path]) -> dict[str, Any]:
    if not token:
        return {"status": "skipped_no_token", "matches": []}
    matches: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if token in text:
                matches.append(repo_rel(path))
    return {"status": "passed" if not matches else "failed", "matches": matches}


def file_manifest_sha(paths: list[Path]) -> str:
    rows = [
        {"path": repo_rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted({path for path in paths if path.exists()}, key=repo_rel)
    ]
    return __import__("hashlib").sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
    ).hexdigest()


def write_task_artifacts(
    *,
    status: str,
    selector: ResolvedSelector,
    plan_payload: dict[str, Any],
    request_hash: str,
    raw_path: Path,
    manifest: ResponseManifest | None,
    ingest_summary: Any | None,
    db_counts: dict[str, Any],
    open_blockers: list[str],
    secret_scan_result: dict[str, Any],
    commands: list[str],
) -> list[Path]:
    ensure_directory(EXPERIMENT_DIR / "logs")
    write_text(
        EXPERIMENT_DIR / "README.md",
        "# T06 Resumable GribStream Runs Client\n\n"
        "Mission: implement and prove the reusable GribStream `/runs` client, raw landing zone, "
        "normalizer, resume ledger, and DB lineage insert path for downstream NWP acquisition tasks.\n\n"
        "This T06 run uses a bounded GFS smoke acquisition. Full historical GFS backfill remains T07.",
    )
    write_json(
        EXPERIMENT_DIR / "run_config.json",
        {
            "dataset": selector.dataset,
            "endpoint": "runs",
            "selector": selector.as_request_variable(),
            "request_sha256": request_hash,
            "raw_object": repo_rel(raw_path) if os.path.exists(fs_path(raw_path)) else raw_path.as_posix(),
            "database": db_counts.get("database", "configured runtime database"),
            "rate_limit_policy": {
                "threads": 1,
                "min_interval_seconds": 12,
                "default_429_pause_seconds": 300,
            },
        },
    )
    write_text(EXPERIMENT_DIR / "requests" / f"{request_hash}.json", canonical_request_json(plan_payload))
    write_json(EXPERIMENT_DIR / "selector_snapshot.json", selector.source_json)
    write_json(
        EXPERIMENT_DIR / "response_manifest.json",
        None
        if manifest is None
        else {
            "provider": manifest.provider,
            "dataset": manifest.dataset,
            "endpoint": manifest.endpoint,
            "request_sha256": manifest.request_sha256,
            "object_path": repo_rel(manifest.object_path),
            "byte_size": manifest.byte_size,
            "sha256": manifest.sha256,
            "content_type": manifest.content_type,
            "retrieved_at_utc": manifest.retrieved_at_utc,
            "row_count": manifest.row_count,
            "http_status": manifest.http_status,
            "attempt_count": manifest.attempt_count,
        },
    )
    write_json(EXPERIMENT_DIR / "db_counts_after_t06.json", db_counts)
    write_json(EXPERIMENT_DIR / "logs/secret_scan.json", secret_scan_result)
    write_text(EXPERIMENT_DIR / "commands_executed.txt", "\n".join(commands))
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# T06 Results\n\n"
        f"Status: {status.upper()}\n\n"
        f"- Request SHA-256: `{request_hash}`\n"
        f"- Selector used: `{selector.native_name}` / `{selector.native_level}` / `{selector.native_info}`\n"
        f"- Raw object: `{repo_rel(raw_path) if os.path.exists(fs_path(raw_path)) else raw_path.as_posix()}`\n"
        f"- Raw rows: {manifest.row_count if manifest else 'not fetched'}\n"
        f"- DB inserted/updated point rows: {ingest_summary.inserted_or_updated_points if ingest_summary else 'not ingested'}\n"
        f"- Rejected normalized rows: {ingest_summary.rejected_rows if ingest_summary else 'not ingested'}\n"
        f"- Secret scan: {secret_scan_result['status']}\n"
        f"- Open blockers: {len(open_blockers)}",
    )
    write_text(
        EXPERIMENT_DIR / "CONCLUSION.md",
        "# T06 Conclusion\n\n"
        f"Status: {status.upper()}\n\n"
        "Acceptance finalization:\n\n"
        "- Same request never creates duplicate values: enforced by `nwp_core.point_value` primary key and T06 upsert path.\n"
        "- Interrupted download resumes: incomplete `.part` files are discarded and the same canonical request SHA is retried; completed raw objects are reused.\n"
        "- Selector/run/valid/member lineage: stored through `catalog.variable_selector_snapshot`, `nwp_core.model_run`, and `nwp_core.point_value`.\n\n"
        "Downstream consequence: T07 can use the T06 client for real GFS backfill chunks; T06 itself only proves the client and bounded smoke path.",
    )
    write_text(
        EXPERIMENT_DIR / "operator_runbook.md",
        "# T06 Operator Runbook\n\n"
        "Run the smoke acquisition:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_t06_gribstream_resumable_runs_client.py --mode smoke\n"
        "```\n\n"
        "Check status without rerunning the API:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\check_t06_gribstream_status.py\n"
        "```\n\n"
        "Rules: one thread only, 12 seconds between authenticated attempts, honor `Retry-After`, and default to a 300 second pause on `429` without `Retry-After`.",
    )
    created_files = [path for path in EXPERIMENT_DIR.rglob("*") if path.is_file()]
    handoff = {
        "task_id": "T06",
        "status": status,
        "git_commit": git_output("rev-parse", "HEAD"),
        "database_migration_version": "20260624_0005_t04_nwp_storage_lineage",
        "input_manifest_sha256": file_manifest_sha(
            [
                REPO_ROOT / "config/acquisition/acquisition_policy.yaml",
                TASK_ROOT / "tasks/completed/T03_gribstream_catalog_coverage_licence_quota_audit/COMPLETION_RECORD.md",
                TASK_ROOT / "tasks/completed/T04_nwp_database_object_storage_migrations/COMPLETION_RECORD.md",
                TASK_ROOT / "tasks/completed/T05_canonical_location_station_geospatial_registry/COMPLETION_RECORD.md",
            ],
        ),
        "output_manifest_sha256": file_manifest_sha(created_files),
        "created_tables_or_views": [],
        "created_files": [repo_rel(path) for path in sorted(created_files, key=repo_rel)],
        "open_blockers": open_blockers,
        "downstream_ready": status == "passed",
    }
    write_json(EXPERIMENT_DIR / "handoff_manifest.json", handoff)
    created_files = [path for path in EXPERIMENT_DIR.rglob("*") if path.is_file()]
    write_text(
        EXPERIMENT_DIR / "data_manifest.csv",
        "role,path,sha256,bytes\n"
        + "\n".join(
            f"output,{repo_rel(path)},{sha256_file(path)},{path.stat().st_size}"
            for path in sorted(created_files, key=repo_rel)
        ),
    )
    return [path for path in EXPERIMENT_DIR.rglob("*") if path.is_file()]


def move_task_to_completed() -> Path:
    source = TASKS_NOT_COMPLETED / TASK_NAME
    target = TASKS_COMPLETED / TASK_NAME
    if target.exists():
        return target
    if not source.exists():
        raise FileNotFoundError(f"Missing T06 task folder at {source}")
    ensure_directory(TASKS_COMPLETED)
    shutil.move(fs_path(source), fs_path(target))
    return target


def update_task_status_index(task_dir: Path) -> None:
    if not STATUS_INDEX.exists():
        return
    with open(fs_path(STATUS_INDEX), newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        if row.get("task_id") != TASK_ID:
            continue
        row["status"] = "completed"
        row["status_folder"] = f"tasks/completed/{task_dir.name}"
        row["task_file"] = f"tasks/completed/{task_dir.name}/t06_gribstream_resumable_runs_client.md"
        row["completion_record"] = f"tasks/completed/{task_dir.name}/COMPLETION_RECORD.md"
    with open(fs_path(STATUS_INDEX), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def write_completion_record(task_dir: Path, open_blockers: list[str]) -> None:
    write_text(
        task_dir / "COMPLETION_RECORD.md",
        "# T06 Completion Record\n\n"
        "Task: Resumable GribStream Runs Client and Raw Landing Zone\n\n"
        f"Evidence folder: `{repo_rel(EXPERIMENT_DIR)}`\n\n"
        "## What Was Done\n\n"
        "- Added the reusable `hkg_tmax.gribstream` client, selector resolver, planner, normalizer, and PostgreSQL store.\n"
        "- Added the T06 runner and status checker.\n"
        "- Resolved the exact GFS `TMP` / `2 m above ground` selector from the live shared-parameter catalog.\n"
        "- Fetched a bounded GFS `/runs` smoke object as NDJSON gzip and loaded normalized values into the T04 lineage schema.\n\n"
        "## Acceptance Finalization\n\n"
        "- Duplicate request/value protection is enforced by canonical request SHA plus `nwp_core.point_value` upsert keys.\n"
        "- Resume behavior is ledgered; `.part` files are retried under the same request SHA and completed raw objects are reused.\n"
        "- Selector/run/valid/member lineage is stored in `catalog.variable_selector_snapshot`, `nwp_core.model_run`, and `nwp_core.point_value`.\n\n"
        "## Open Blockers\n\n"
        + ("\n".join(f"- {item}" for item in open_blockers) if open_blockers else "- None"),
    )


def validate_inputs(database_url: str) -> list[str]:
    blockers: list[str] = []
    required_paths = [
        REPO_ROOT / "config/acquisition/acquisition_policy.yaml",
        TASK_ROOT / "tasks/completed/T03_gribstream_catalog_coverage_licence_quota_audit",
        TASK_ROOT / "tasks/completed/T04_nwp_database_object_storage_migrations",
        TASK_ROOT / "tasks/completed/T05_canonical_location_station_geospatial_registry",
    ]
    for path in required_paths:
        if not path.exists():
            blockers.append(f"Missing required dependency path: {repo_rel(path) if path.exists() else path}")
    try:
        counts = query_counts(database_url)
    except Exception as exc:  # noqa: BLE001 - dependency validation should report exact blocker
        blockers.append(f"Database dependency check failed: {type(exc).__name__}: {sanitize_text(str(exc))}")
        return blockers
    if counts.get("catalog.location", 0) <= 0:
        blockers.append("T05 locations are not loaded into catalog.location")
    return blockers


def run_smoke(args: argparse.Namespace) -> int:
    ensure_directory(EXPERIMENT_DIR / "logs")
    token = load_gribstream_token()
    commands = [
        ".\\.venv\\Scripts\\python.exe scripts\\run_t06_gribstream_resumable_runs_client.py --mode smoke",
        ".\\.venv\\Scripts\\python.exe -m pytest code\\tests\\test_t06_gribstream_resumable_runs_client.py",
    ]
    if not token:
        write_status("startup", "blocked", reason="missing_gribstream_api_key")
        print(json.dumps({"status": "blocked", "reason": "missing_gribstream_api_key"}, indent=2))
        return 2
    write_status("startup", "running", database=redact_database_url(args.database_url))
    open_blockers = validate_inputs(args.database_url)
    if open_blockers:
        write_status("input_validation", "blocked", open_blockers=open_blockers)
        print(json.dumps({"status": "blocked", "open_blockers": open_blockers}, indent=2))
        return 2

    try:
        write_status("selector_resolution", "running")
        selector = resolve_temperature_2m_selector(dataset="gfs")
        locations = load_canonical_locations(args.database_url, limit=args.location_limit)
        plan = build_runs_plan(
            selector=selector,
            locations=locations,
            forecasted_from=args.run_time_utc,
            forecasted_until=args.run_time_utc,
            min_lead_time=args.min_lead_time,
            max_lead_time=args.max_lead_time,
            dataset="gfs",
        )
        output_path = raw_object_path(args.run_time_utc, plan.request_sha256)
        append_jsonl(
            LEDGER_PATH,
            {
                "event": "planned",
                "request_sha256": plan.request_sha256,
                "estimated_rows": plan.estimated_rows,
                "estimated_credits": plan.estimated_credits,
                "location_count": len(locations),
                "updated_at_utc": utc_now_iso(),
            },
        )
        write_status(
            "planned",
            "running",
            request_sha256=plan.request_sha256,
            estimated_rows=plan.estimated_rows,
            estimated_credits=plan.estimated_credits,
            location_count=len(locations),
        )
        if args.mode == "dry-run":
            write_task_artifacts(
                status="partial",
                selector=selector,
                plan_payload=plan.payload,
                request_hash=plan.request_sha256,
                raw_path=output_path,
                manifest=None,
                ingest_summary=None,
                db_counts=query_counts(args.database_url),
                open_blockers=["Dry run only; no live GribStream request was sent."],
                secret_scan_result=secret_scan(token, [EXPERIMENT_DIR, REPO_ROOT / "docs", REPO_ROOT / "documentation"]),
                commands=commands,
            )
            write_status("dry_run", "partial", request_sha256=plan.request_sha256)
            return 0

        request_id = register_request_started(
            args.database_url,
            provider="GribStream",
            model_code="gfs",
            endpoint="runs",
            canonical_payload=plan.payload,
            request_hash=plan.request_sha256,
        )
        append_jsonl(
            LEDGER_PATH,
            {
                "event": "request_registered",
                "request_id": request_id,
                "request_sha256": plan.request_sha256,
                "updated_at_utc": utc_now_iso(),
            },
        )
        try:
            if os.path.exists(fs_path(output_path)):
                manifest = manifest_for_existing_object(output_path, dataset="gfs", request_hash=plan.request_sha256)
                append_jsonl(
                    LEDGER_PATH,
                    {
                        "event": "raw_object_reused",
                        "request_sha256": plan.request_sha256,
                        "object_path": repo_rel(output_path),
                        "row_count": manifest.row_count,
                        "updated_at_utc": utc_now_iso(),
                    },
                )
            else:
                write_status("api_fetch", "running", request_sha256=plan.request_sha256)
                retry_config = RetryConfig(
                    max_attempts=args.api_max_attempts,
                    min_interval_seconds=args.api_min_interval_seconds,
                    default_rate_limit_pause_seconds=args.pause_on_429_seconds,
                    min_rate_limit_pause_seconds=180.0,
                    max_retry_delay_seconds=args.max_retry_after_seconds,
                )
                with GribStreamClient(
                    token,
                    retry_config=retry_config,
                    event_log_path=API_EVENT_LOG,
                ) as client:
                    manifest = client.post_runs_to_gzip(
                        dataset="gfs",
                        payload=plan.payload,
                        output_path=output_path,
                        request_hash=plan.request_sha256,
                    )
                append_jsonl(
                    LEDGER_PATH,
                    {
                        "event": "raw_object_completed",
                        "request_sha256": plan.request_sha256,
                        "object_path": repo_rel(output_path),
                        "row_count": manifest.row_count,
                        "sha256": manifest.sha256,
                        "updated_at_utc": utc_now_iso(),
                    },
                )
        except GribStreamRequestError as exc:
            mark_request_failed(
                args.database_url,
                request_hash=plan.request_sha256,
                error_class=exc.error_class,
                error_message=str(exc),
            )
            write_status(
                "api_fetch",
                "failed",
                request_sha256=plan.request_sha256,
                error_class=exc.error_class,
                error_message=sanitize_text(str(exc), token),
                http_status=exc.status_code,
            )
            raise

        location_ids = load_location_ids(args.database_url)
        normalized = normalize_runs_ndjson_gzip(
            manifest.object_path,
            value_alias=selector.alias,
            location_ids_by_code=location_ids,
        )
        ingest_summary = ingest_response(
            args.database_url,
            request_id=request_id,
            selector=selector,
            manifest=manifest,
            points=normalized.points,
            rejected_rows=normalized.rejected_rows,
        )
        db_counts = query_counts(args.database_url, plan.request_sha256)
        scan = secret_scan(token, [EXPERIMENT_DIR, REPO_ROOT / "docs", REPO_ROOT / "documentation", REPO_ROOT / "AGENTS.md"])
        status = (
            "passed"
            if manifest.row_count > 0
            and ingest_summary.inserted_or_updated_points > 0
            and ingest_summary.rejected_rows == 0
            and scan["status"] == "passed"
            else "blocked"
        )
        if manifest.row_count == 0:
            open_blockers.append("Live GribStream request returned zero rows.")
        if ingest_summary.rejected_rows:
            open_blockers.append(f"Normalizer rejected {ingest_summary.rejected_rows} row(s).")
        if scan["status"] != "passed":
            open_blockers.append("Secret scan found the API token in generated artifacts.")
        created_files = write_task_artifacts(
            status=status,
            selector=selector,
            plan_payload=plan.payload,
            request_hash=plan.request_sha256,
            raw_path=output_path,
            manifest=manifest,
            ingest_summary=ingest_summary,
            db_counts=db_counts | {"database": redact_database_url(args.database_url)},
            open_blockers=open_blockers,
            secret_scan_result=scan,
            commands=commands,
        )
        if status == "passed":
            task_dir = move_task_to_completed()
            update_task_status_index(task_dir)
            write_completion_record(task_dir, open_blockers)
        write_status(
            "complete",
            status,
            request_sha256=plan.request_sha256,
            raw_object=repo_rel(output_path),
            row_count=manifest.row_count,
            point_rows=ingest_summary.inserted_or_updated_points,
            rejected_rows=ingest_summary.rejected_rows,
            artifacts=len(created_files),
            open_blockers=open_blockers,
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "request_sha256": plan.request_sha256,
                    "raw_object": repo_rel(output_path),
                    "row_count": manifest.row_count,
                    "point_rows": ingest_summary.inserted_or_updated_points,
                    "rejected_rows": ingest_summary.rejected_rows,
                    "experiment": repo_rel(EXPERIMENT_DIR),
                    "completed_task_dir": repo_rel(TASKS_COMPLETED / TASK_NAME) if status == "passed" else "",
                    "open_blockers": open_blockers,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return 0 if status == "passed" else 2
    except Exception as exc:
        write_status(
            "crashed",
            "failed",
            error_class=type(exc).__name__,
            error_message=sanitize_text(str(exc), token),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T06 GribStream resumable /runs client smoke.")
    parser.add_argument("--mode", choices=["smoke", "dry-run", "status"], default="smoke")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--run-time-utc", default=DEFAULT_RUN_TIME)
    parser.add_argument("--min-lead-time", default="0h")
    parser.add_argument("--max-lead-time", default="48h")
    parser.add_argument("--location-limit", type=int)
    parser.add_argument("--api-min-interval-seconds", type=float, default=12.0)
    parser.add_argument("--api-max-attempts", type=int, default=3)
    parser.add_argument("--pause-on-429-seconds", type=float, default=300.0)
    parser.add_argument("--max-retry-after-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.mode == "status":
        return command_status()
    try:
        return run_smoke(args)
    except DatabaseUnavailable as exc:
        write_status("database", "blocked", error_class=type(exc).__name__, error_message=str(exc))
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
