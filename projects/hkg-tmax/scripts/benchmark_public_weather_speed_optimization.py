from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_public_weather_to_postgres as lean  # noqa: E402
import run_public_gfs_gefs_himawari_7day_backfill_rehearsal as public  # noqa: E402


DEFAULT_EXPERIMENT_ID = "0011_public_weather_speed_optimization_20260709"
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / DEFAULT_EXPERIMENT_ID
DEFAULT_STAGING_ROOT = REPO_ROOT / "_s0011"
DEFAULT_TRIALS = (
    "wgrib2_probe,"
    "model_fetch_s3_w8_r4,"
    "model_fetch_s3_w8_c0,"
    "model_fetch_s3_w16_r4,"
    "model_fetch_s3_w16_r8,"
    "himawari_fetch_normalize_w4,"
    "himawari_fetch_normalize_w8,"
    "model_fetch_normalize_sample_w2"
)
TRIAL_DIR_NAMES = {
    "wgrib2_probe": "wg",
    "model_fetch_s3_w8_r4": "m8r4",
    "model_fetch_s3_w8_c0": "m8c0",
    "model_fetch_s3_w8_c1m": "m8c1m",
    "model_fetch_s3_w16_r4": "m16r4",
    "model_fetch_s3_w16_r8": "m16r8",
    "himawari_fetch_normalize_w4": "h4",
    "himawari_fetch_normalize_w8": "h8",
    "model_fetch_normalize_sample_w2": "mn2",
}


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_int_list(value: str) -> list[int]:
    if ":" in value:
        start_s, end_s, step_s = value.split(":")
        start = int(start_s)
        end = int(end_s)
        step = int(step_s)
        return list(range(start, end + 1, step))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_rmtree(path: Path, root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not str(resolved).lower().startswith(str(root_resolved).lower()):
        raise RuntimeError(f"Refusing to remove outside experiment root: {resolved}")
    if resolved == root_resolved:
        raise RuntimeError(f"Refusing to remove experiment root: {resolved}")
    try:
        shutil.rmtree(lean.wp(resolved))
    except FileNotFoundError:
        pass


def safe_delete_raw(path: Path, trial_dir: Path) -> int:
    path_s = lean.wp(path)
    if not os.path.exists(path_s) or not os.path.isfile(path_s):
        return 0
    resolved = path.resolve()
    root = trial_dir.resolve()
    if not str(resolved).lower().startswith(str(root).lower()):
        raise RuntimeError(f"Refusing to delete outside trial root: {resolved}")
    size = os.path.getsize(path_s)
    os.remove(path_s)
    return size


def trial_output_dir(experiment_dir: Path, trial_name: str) -> Path:
    return experiment_dir / "r" / TRIAL_DIR_NAMES.get(trial_name, trial_name[:16])


def dir_size(path: Path) -> int:
    if not os.path.exists(lean.wp(path)):
        return 0
    total = 0
    for root, _dirs, files in os.walk(lean.wp(path)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class ResourceSampler:
    def __init__(self, watch_dir: Path, interval_seconds: float = 1.0) -> None:
        self.watch_dir = watch_dir
        self.interval_seconds = interval_seconds
        self.cpu_samples: list[float] = []
        self.staging_samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:  # noqa: BLE001
            self._psutil = None

    def __enter__(self) -> "ResourceSampler":
        if self._psutil is not None:
            self._psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._psutil is not None:
                try:
                    self.cpu_samples.append(float(self._psutil.cpu_percent(interval=None)))
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.staging_samples.append(dir_size(self.watch_dir))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        cpu = self.cpu_samples
        staging = self.staging_samples
        actual_end = dir_size(self.watch_dir)
        return {
            "cpu_sampler_available": self._psutil is not None,
            "cpu_mean_percent": statistics.fmean(cpu) if cpu else None,
            "cpu_max_percent": max(cpu) if cpu else None,
            "staging_max_bytes": max(staging) if staging else None,
            "staging_end_bytes": actual_end,
        }


def task_to_dict(task: public.FetchTask) -> dict[str, Any]:
    return asdict(task)


def task_from_dict(payload: dict[str, Any]) -> public.FetchTask:
    return public.FetchTask(**payload)


def build_tasks(
    *,
    day: date,
    sources: set[str],
    cycles: list[int],
    leads: list[int],
    trial_dir: Path,
) -> list[public.FetchTask]:
    return lean.build_tasks_for_day(day, sources, cycles, leads, trial_dir)


def merge_selected_ranges(ranges: list[dict[str, Any]], max_gap_bytes: int) -> list[dict[str, Any]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: int(item["offset"]))
    current = {
        "offset": int(ordered[0]["offset"]),
        "end_offset": int(ordered[0]["end_offset"]),
        "message_count": 1,
    }
    merged: list[dict[str, Any]] = []
    for item in ordered[1:]:
        offset = int(item["offset"])
        end_offset = int(item["end_offset"])
        gap = offset - int(current["end_offset"]) - 1
        if gap <= max_gap_bytes:
            current["end_offset"] = max(int(current["end_offset"]), end_offset)
            current["message_count"] = int(current["message_count"]) + 1
        else:
            merged.append(current)
            current = {"offset": offset, "end_offset": end_offset, "message_count": 1}
    merged.append(current)
    return merged


def fetch_s3_idx_range_model_coalesced(
    task: public.FetchTask,
    full_path: Path,
    *,
    timeout: int,
    max_gap_bytes: int,
) -> tuple[public.FetchResult, dict[str, Any]]:
    if task.issue_day_utc is None or task.cycle_hour is None or task.lead_hour is None:
        raise ValueError("Model task is missing issue day, cycle, or lead")
    day = lean.parse_date(task.issue_day_utc)
    object_url = lean.model_object_url(task.source, day, task.cycle_hour, task.lead_hour)
    idx_url = lean.model_idx_url(task.source, day, task.cycle_hour, task.lead_hour)
    object_length = lean.http_head_content_length(object_url, timeout=timeout)
    idx_data, idx_headers, _status = lean.http_get_bytes(idx_url, timeout=timeout)
    ranges = lean.parse_grib_idx_ranges(idx_data.decode("utf-8", errors="replace"), object_length)
    if not ranges:
        return lean.fetch_s3_idx_range_model(task, full_path, timeout=timeout)
    merged_ranges = merge_selected_ranges(ranges, max_gap_bytes)

    def fetch_merged_range(index_and_item: tuple[int, dict[str, Any]]) -> tuple[int, bytes]:
        index, item = index_and_item
        byte_range = f"bytes={item['offset']}-{item['end_offset']}"
        chunk, _headers, _status = lean.http_get_bytes(object_url, timeout=timeout, headers={"Range": byte_range})
        if not chunk.startswith(b"GRIB"):
            raise RuntimeError(f"S3 coalesced range did not start with GRIB for {task.item_id}: {byte_range}")
        return index, chunk

    lean.ensure_dir(full_path.parent)
    chunks: list[bytes | None] = [None] * len(merged_ranges)
    range_workers = max(1, min(lean.S3_RANGE_WORKERS, len(merged_ranges)))
    with ThreadPoolExecutor(max_workers=range_workers) as executor:
        futures = [executor.submit(fetch_merged_range, item) for item in enumerate(merged_ranges)]
        for future in as_completed(futures):
            index, chunk = future.result()
            chunks[index] = chunk

    total_bytes = 0
    with open(lean.wp(full_path), "wb") as handle:
        for chunk in chunks:
            if chunk is None:
                raise RuntimeError(f"Missing selected S3 coalesced range for {task.item_id}")
            handle.write(chunk)
            total_bytes += len(chunk)
    data = Path(lean.wp(full_path)).read_bytes()
    invalid = public.payload_validation_error(task, data)
    selected_variables = [str(item["variable"]) for item in ranges]
    selected_level_pairs = [f"{item['variable']}:{item['level']}" for item in ranges]
    fetch = public.FetchResult(
        kind=task.kind,
        source=task.source,
        item_id=task.item_id,
        status="invalid_payload" if invalid else "ok",
        url=object_url,
        path=task.path,
        bytes=len(data),
        sha256=lean.sha256_bytes(data),
        elapsed_seconds=0.0,
        retrieved_at_utc=lean.iso(lean.utc_now()) or "",
        fetched_now=True,
        issue_day_utc=task.issue_day_utc,
        cycle_hour=task.cycle_hour,
        lead_hour=task.lead_hour,
        issued_at_utc=task.issued_at_utc,
        valid_at_utc=task.valid_at_utc,
        availability_proxy_utc=task.availability_proxy_utc,
        availability_proxy_method=task.availability_proxy_method,
        http_last_modified_utc=public.parse_http_datetime(idx_headers.get("last-modified")),
        content_length_header=object_length,
        content_type="application/octet-stream",
        error=invalid,
    )
    return (
        fetch,
        {
            "provider_mode": "s3_idx_range_coalesced",
            "idx_url": idx_url,
            "object_url": object_url,
            "object_length": object_length,
            "selected_message_count": len(ranges),
            "selected_variables": sorted(set(selected_variables)),
            "selected_variable_level_pairs": sorted(set(selected_level_pairs)),
            "selected_range_count": len(ranges),
            "merged_range_count": len(merged_ranges),
            "coalesce_gap_bytes": max_gap_bytes,
            "selected_bytes": sum(int(item["end_offset"]) - int(item["offset"]) + 1 for item in ranges),
            "downloaded_bytes": total_bytes,
            "selected_range_workers": range_workers,
            "nomads_filter_url": task.url,
        },
    )


def fetch_direct(
    task_payload: dict[str, Any],
    trial_dir_s: str,
    timeout: int,
    max_attempts: int,
    s3_range_workers: int,
    coalesce_gap_bytes: int | None,
) -> dict[str, Any]:
    trial_dir = Path(trial_dir_s)
    task = task_from_dict(task_payload)
    lean.S3_RANGE_WORKERS = s3_range_workers
    raw_path = trial_dir / task.path
    started = time.perf_counter()
    try:
        if task.kind == "model_grib":
            if coalesce_gap_bytes is None:
                fetch_fn = lambda: lean.fetch_s3_idx_range_model(task, raw_path, timeout=timeout)
            else:
                fetch_fn = lambda: fetch_s3_idx_range_model_coalesced(
                    task,
                    raw_path,
                    timeout=timeout,
                    max_gap_bytes=coalesce_gap_bytes,
                )
            (fetch, metadata), attempts = lean.request_with_retries(
                f"{task.item_id}:s3_idx_range",
                fetch_fn,
                max_attempts=max_attempts,
            )
        elif task.kind == "himawari_hsd":
            (fetch, metadata), attempts = lean.request_with_retries(
                f"{task.item_id}:himawari_hsd",
                lambda: lean.fetch_himawari(task, raw_path, timeout=timeout),
                max_attempts=max_attempts,
            )
        else:
            raise ValueError(f"Unsupported task kind: {task.kind}")
        fetch.elapsed_seconds = time.perf_counter() - started
        metadata["attempts"] = attempts
        return {
            "ok": fetch.status == "ok",
            "status": fetch.status,
            "task": task_payload,
            "fetch": asdict(fetch),
            "metadata": metadata,
            "raw_path": str(raw_path),
            "fetch_elapsed_seconds": fetch.elapsed_seconds,
            "normalization": None,
            "error": fetch.error,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "error",
            "task": task_payload,
            "fetch": None,
            "metadata": {"provider_mode": "direct_fetch", "attempts": None},
            "raw_path": str(raw_path),
            "fetch_elapsed_seconds": time.perf_counter() - started,
            "normalization": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def normalize_direct(result: dict[str, Any], trial_dir_s: str) -> dict[str, Any]:
    if not result.get("fetch"):
        return {"status": "skip", "elapsed_seconds": 0.0, "row_count": 0, "error": result.get("error")}
    fetch = public.FetchResult(**result["fetch"])
    started = time.perf_counter()
    if fetch.status != "ok":
        return {"status": "skip", "elapsed_seconds": 0.0, "row_count": 0, "error": fetch.error}
    try:
        trial_dir = Path(trial_dir_s)
        if fetch.kind == "model_grib":
            normalized = lean.normalize_model_result_full(trial_dir, asdict(fetch))
            row_count = (1 if normalized.get("station_row") else 0) + len(normalized.get("bbox_rows") or [])
        elif fetch.kind == "himawari_hsd":
            normalized = lean.normalize_himawari_result_full(trial_dir, fetch)
            row_count = 1 if normalized.get("row") else 0
        else:
            raise ValueError(f"Unsupported fetch kind: {fetch.kind}")
        return {
            "status": normalized.get("status"),
            "elapsed_seconds": time.perf_counter() - started,
            "row_count": row_count,
            "error": (
                (normalized.get("row") or {}).get("normalization_error")
                or (normalized.get("station_row") or {}).get("normalization_error")
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "elapsed_seconds": time.perf_counter() - started,
            "row_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def fetch_only_worker(
    task_payload: dict[str, Any],
    trial_dir_s: str,
    timeout: int,
    max_attempts: int,
    s3_range_workers: int,
    coalesce_gap_bytes: int | None,
) -> dict[str, Any]:
    result = fetch_direct(task_payload, trial_dir_s, timeout, max_attempts, s3_range_workers, coalesce_gap_bytes)
    raw_path = Path(result["raw_path"])
    try:
        result["raw_deleted_bytes"] = safe_delete_raw(raw_path, Path(trial_dir_s))
    except Exception as exc:  # noqa: BLE001
        result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return result


def fetch_normalize_delete_worker(
    task_payload: dict[str, Any],
    trial_dir_s: str,
    timeout: int,
    max_attempts: int,
    s3_range_workers: int,
    coalesce_gap_bytes: int | None,
) -> dict[str, Any]:
    result = fetch_direct(task_payload, trial_dir_s, timeout, max_attempts, s3_range_workers, coalesce_gap_bytes)
    try:
        result["normalization"] = normalize_direct(result, trial_dir_s)
    finally:
        raw_path = Path(result["raw_path"])
        try:
            result["raw_deleted_bytes"] = safe_delete_raw(raw_path, Path(trial_dir_s))
        except Exception as exc:  # noqa: BLE001
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def summarize_results(
    *,
    trial_name: str,
    trial_type: str,
    started_at_utc: str,
    completed_at_utc: str,
    wall_seconds: float,
    tasks: list[public.FetchTask],
    results: list[dict[str, Any]],
    resources: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    fetch_times = [float(r.get("fetch_elapsed_seconds") or 0.0) for r in results]
    norm_times = [
        float((r.get("normalization") or {}).get("elapsed_seconds") or 0.0)
        for r in results
        if r.get("normalization")
    ]
    ok_fetch = [r for r in results if r.get("fetch") and r["fetch"].get("status") == "ok"]
    ok_norm = [r for r in results if (r.get("normalization") or {}).get("status") == "ok"]
    bytes_ok = sum(int((r.get("fetch") or {}).get("bytes") or 0) for r in ok_fetch)
    rows = sum(int((r.get("normalization") or {}).get("row_count") or 0) for r in results)
    failures = [r for r in results if not r.get("ok") or (r.get("normalization") or {}).get("status") == "error"]
    by_source: dict[str, dict[str, Any]] = {}
    for result in results:
        task = result.get("task") or {}
        source = str(task.get("source") or "unknown")
        bucket = by_source.setdefault(source, {"tasks": 0, "fetch_ok": 0, "norm_ok": 0, "bytes": 0})
        bucket["tasks"] += 1
        if result.get("fetch") and result["fetch"].get("status") == "ok":
            bucket["fetch_ok"] += 1
            bucket["bytes"] += int(result["fetch"].get("bytes") or 0)
        if (result.get("normalization") or {}).get("status") == "ok":
            bucket["norm_ok"] += 1
    return {
        "trial_name": trial_name,
        "trial_type": trial_type,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "wall_seconds": wall_seconds,
        "tasks_total": len(tasks),
        "fetch_ok": len(ok_fetch),
        "normalize_ok": len(ok_norm),
        "failure_count": len(failures),
        "bytes_ok": bytes_ok,
        "normalized_row_count": rows,
        "fetch_seconds_p50": percentile(fetch_times, 0.5),
        "fetch_seconds_p90": percentile(fetch_times, 0.9),
        "normalization_seconds_p50": percentile(norm_times, 0.5),
        "normalization_seconds_p90": percentile(norm_times, 0.9),
        "tasks_per_minute_wall": len(tasks) / (wall_seconds / 60.0) if wall_seconds > 0 else None,
        "mb_per_second_wall": (bytes_ok / 1_000_000.0) / wall_seconds if wall_seconds > 0 else None,
        "by_source": by_source,
        "first_failures": [
            {
                "item_id": (item.get("task") or {}).get("item_id"),
                "source": (item.get("task") or {}).get("source"),
                "status": item.get("status"),
                "error": item.get("error") or (item.get("normalization") or {}).get("error"),
            }
            for item in failures[:10]
        ],
        "resources": resources,
        "config": config,
    }


def write_trial_outputs(trial_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    ensure_dir(trial_dir)
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (trial_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    csv_path = trial_dir / "results.csv"
    fieldnames = [
        "source",
        "item_id",
        "kind",
        "status",
        "bytes",
        "fetch_elapsed_seconds",
        "normalization_status",
        "normalization_elapsed_seconds",
        "row_count",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            task = result.get("task") or {}
            fetch = result.get("fetch") or {}
            norm = result.get("normalization") or {}
            writer.writerow(
                {
                    "source": task.get("source"),
                    "item_id": task.get("item_id"),
                    "kind": task.get("kind"),
                    "status": fetch.get("status") or result.get("status"),
                    "bytes": fetch.get("bytes"),
                    "fetch_elapsed_seconds": result.get("fetch_elapsed_seconds"),
                    "normalization_status": norm.get("status"),
                    "normalization_elapsed_seconds": norm.get("elapsed_seconds"),
                    "row_count": norm.get("row_count"),
                    "error": result.get("error") or norm.get("error"),
                }
            )


def run_wgrib2_probe(experiment_dir: Path) -> dict[str, Any]:
    started = utc_now_iso()
    started_perf = time.perf_counter()
    from shutil import which

    path = which("wgrib2")
    probe: dict[str, Any] = {
        "trial_name": "wgrib2_probe",
        "trial_type": "environment_probe",
        "started_at_utc": started,
        "wgrib2_path": path,
        "wgrib2_available": path is not None,
        "stdout": None,
        "stderr": None,
        "returncode": None,
    }
    if path:
        completed = subprocess.run([path, "-version"], text=True, capture_output=True, timeout=20, check=False)
        probe.update(
            {
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "returncode": completed.returncode,
            }
        )
    probe["completed_at_utc"] = utc_now_iso()
    probe["wall_seconds"] = time.perf_counter() - started_perf
    trial_dir = trial_output_dir(experiment_dir, "wgrib2_probe")
    write_trial_outputs(trial_dir, probe, [])
    return probe


def run_parallel_trial(
    *,
    experiment_dir: Path,
    staging_root: Path,
    trial_name: str,
    day: date,
    sources: set[str],
    cycles: list[int],
    leads: list[int],
    workers: int,
    range_workers: int,
    timeout: int,
    max_attempts: int,
    normalize: bool,
    use_processes: bool,
    coalesce_gap_bytes: int | None,
    max_tasks: int | None,
) -> dict[str, Any]:
    trial_dir = staging_root / trial_name
    output_dir = trial_output_dir(experiment_dir, trial_name)
    if trial_dir.exists():
        safe_rmtree(trial_dir, staging_root)
    ensure_dir(trial_dir)
    ensure_dir(output_dir)
    tasks = build_tasks(day=day, sources=sources, cycles=cycles, leads=leads, trial_dir=trial_dir)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    task_payloads = [task_to_dict(task) for task in tasks]
    worker_fn = fetch_normalize_delete_worker if normalize else fetch_only_worker
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    started = utc_now_iso()
    started_perf = time.perf_counter()
    print(
        f"[trial-start] {trial_name} tasks={len(tasks)} workers={workers} "
        f"range_workers={range_workers} normalize={normalize} processes={use_processes} "
        f"coalesce_gap={coalesce_gap_bytes}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ResourceSampler(trial_dir) as sampler:
        with executor_cls(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(
                    worker_fn,
                    payload,
                    str(trial_dir),
                    timeout,
                    max_attempts,
                    range_workers,
                    coalesce_gap_bytes,
                ): payload
                for payload in task_payloads
            }
            completed_count = 0
            for future in as_completed(future_to_task):
                completed_count += 1
                result = future.result()
                results.append(result)
                if completed_count == 1 or completed_count % 25 == 0 or completed_count == len(tasks):
                    print(
                        f"[trial-progress] {trial_name} {completed_count}/{len(tasks)} "
                        f"last_status={result.get('status')} item={(result.get('task') or {}).get('item_id')}",
                        flush=True,
                    )
        resources = sampler.summary()
    wall = time.perf_counter() - started_perf
    safe_rmtree(trial_dir, staging_root)
    config = {
        "day": day.isoformat(),
        "staging_root": str(staging_root),
        "sources": sorted(sources),
        "cycles": cycles,
        "leads": leads,
        "workers": workers,
        "range_workers": range_workers,
        "timeout": timeout,
        "max_attempts": max_attempts,
        "normalize": normalize,
        "use_processes": use_processes,
        "coalesce_gap_bytes": coalesce_gap_bytes,
        "max_tasks": max_tasks,
    }
    summary = summarize_results(
        trial_name=trial_name,
        trial_type="parallel_fetch_normalize" if normalize else "parallel_fetch_only",
        started_at_utc=started,
        completed_at_utc=utc_now_iso(),
        wall_seconds=wall,
        tasks=tasks,
        results=results,
        resources=resources,
        config=config,
    )
    write_trial_outputs(output_dir, summary, results)
    print(
        f"[trial-done] {trial_name} wall={wall:.1f}s ok={summary['fetch_ok']}/{summary['tasks_total']} "
        f"norm_ok={summary['normalize_ok']} staging_end={summary['resources'].get('staging_end_bytes')}",
        flush=True,
    )
    return summary


def run_named_trial(name: str, args: argparse.Namespace, experiment_dir: Path, staging_root: Path) -> dict[str, Any]:
    if name == "wgrib2_probe":
        return run_wgrib2_probe(experiment_dir)
    if name == "model_fetch_s3_w8_r4":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"gfs", "gefs_control"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=8,
            range_workers=4,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=False,
            use_processes=False,
            coalesce_gap_bytes=None,
            max_tasks=args.max_tasks,
        )
    if name == "model_fetch_s3_w8_c0":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"gfs", "gefs_control"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=8,
            range_workers=4,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=False,
            use_processes=False,
            coalesce_gap_bytes=0,
            max_tasks=args.max_tasks,
        )
    if name == "model_fetch_s3_w8_c1m":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"gfs", "gefs_control"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=8,
            range_workers=4,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=False,
            use_processes=False,
            coalesce_gap_bytes=1_000_000,
            max_tasks=args.max_tasks,
        )
    if name == "model_fetch_s3_w16_r4":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"gfs", "gefs_control"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=16,
            range_workers=4,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=False,
            use_processes=False,
            coalesce_gap_bytes=None,
            max_tasks=args.max_tasks,
        )
    if name == "model_fetch_s3_w16_r8":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"gfs", "gefs_control"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=16,
            range_workers=8,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=False,
            use_processes=False,
            coalesce_gap_bytes=None,
            max_tasks=args.max_tasks,
        )
    if name == "himawari_fetch_normalize_w4":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"himawari_b13_s0510"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=4,
            range_workers=1,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=True,
            use_processes=False,
            coalesce_gap_bytes=None,
            max_tasks=args.max_tasks,
        )
    if name == "himawari_fetch_normalize_w8":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"himawari_b13_s0510"},
            cycles=parse_int_list(args.cycles),
            leads=parse_int_list(args.leads),
            workers=8,
            range_workers=1,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=True,
            use_processes=False,
            coalesce_gap_bytes=None,
            max_tasks=args.max_tasks,
        )
    if name == "model_fetch_normalize_sample_w2":
        return run_parallel_trial(
            experiment_dir=experiment_dir,
            staging_root=staging_root,
            trial_name=name,
            day=parse_date(args.date),
            sources={"gfs", "gefs_control"},
            cycles=[0, 12],
            leads=[0, 24, 48],
            workers=2,
            range_workers=4,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            normalize=True,
            use_processes=True,
            coalesce_gap_bytes=None,
            max_tasks=args.max_tasks,
        )
    raise ValueError(f"Unknown trial: {name}")


def write_aggregate_outputs(experiment_dir: Path, summaries: list[dict[str, Any]]) -> None:
    results_dir = experiment_dir / "results"
    ensure_dir(results_dir)
    (results_dir / "metrics.json").write_text(json.dumps({"trials": summaries}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = [
        "trial_name",
        "trial_type",
        "wall_seconds",
        "tasks_total",
        "fetch_ok",
        "normalize_ok",
        "failure_count",
        "bytes_ok",
        "tasks_per_minute_wall",
        "mb_per_second_wall",
        "cpu_mean_percent",
        "cpu_max_percent",
        "staging_max_bytes",
        "staging_end_bytes",
    ]
    with (results_dir / "trial_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            resources = summary.get("resources") or {}
            writer.writerow(
                {
                    "trial_name": summary.get("trial_name"),
                    "trial_type": summary.get("trial_type"),
                    "wall_seconds": summary.get("wall_seconds"),
                    "tasks_total": summary.get("tasks_total"),
                    "fetch_ok": summary.get("fetch_ok"),
                    "normalize_ok": summary.get("normalize_ok"),
                    "failure_count": summary.get("failure_count"),
                    "bytes_ok": summary.get("bytes_ok"),
                    "tasks_per_minute_wall": summary.get("tasks_per_minute_wall"),
                    "mb_per_second_wall": summary.get("mb_per_second_wall"),
                    "cpu_mean_percent": resources.get("cpu_mean_percent"),
                    "cpu_max_percent": resources.get("cpu_max_percent"),
                    "staging_max_bytes": resources.get("staging_max_bytes"),
                    "staging_end_bytes": resources.get("staging_end_bytes"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    parser.add_argument("--date", default="2026-06-21")
    parser.add_argument("--cycles", default="0,6,12,18")
    parser.add_argument("--leads", default="0:48:3")
    parser.add_argument("--trials", default=DEFAULT_TRIALS)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    staging_root = Path(args.staging_root)
    ensure_dir(experiment_dir / "results")
    ensure_dir(experiment_dir / "r")
    ensure_dir(experiment_dir / "logs")
    ensure_dir(staging_root)
    trial_names = [item.strip() for item in args.trials.split(",") if item.strip()]
    summaries: list[dict[str, Any]] = []
    run_started = utc_now_iso()
    for trial_name in trial_names:
        summaries.append(run_named_trial(trial_name, args, experiment_dir, staging_root))
        write_aggregate_outputs(experiment_dir, summaries)
    run_completed = utc_now_iso()
    manifest = {
        "script": str(Path(__file__).resolve()),
        "started_at_utc": run_started,
        "completed_at_utc": run_completed,
        "args": vars(args),
        "staging_root": str(staging_root),
        "trial_count": len(summaries),
    }
    (experiment_dir / "results" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        staging_root.rmdir()
    except OSError:
        pass
    print(f"[done] wrote {experiment_dir / 'results' / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
