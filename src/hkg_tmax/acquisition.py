from __future__ import annotations

import csv
import importlib
import json
import os
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .fetch import FetchError, FetchPolicy, infer_extension
from .hashing import sha256_bytes
from .timeutils import require_aware


class AcquisitionError(RuntimeError):
    """Raised when acquisition storage or retrieval cannot be completed safely."""


LOGICAL_DATA_DIRS = (
    "raw",
    "bronze",
    "silver",
    "gold",
    "metadata",
    "manifests",
    "state",
    "logs",
    "quarantine",
)


@dataclass(frozen=True)
class DataRootStatus:
    path: Path
    path_length: int
    exists: bool
    free_bytes: int
    total_bytes: int
    long_path_risk: bool


@dataclass(frozen=True)
class AcquisitionRecord:
    source_id: str
    retrieved_at: datetime
    content_sha256: str
    content_length: int
    content_path: Path
    sidecar_path: Path
    deduplicated: bool
    ledger_path: Path


def resolve_data_root(repo_root: Path) -> Path:
    configured = os.getenv("HKG_TMAX_DATA_ROOT")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()
    return (repo_root / "data").resolve()


def inspect_data_root(repo_root: Path) -> DataRootStatus:
    data_root = resolve_data_root(repo_root)
    anchor = data_root if data_root.exists() else data_root.parent
    usage = os.statvfs(anchor) if hasattr(os, "statvfs") else None
    if usage is not None:
        free_bytes = int(usage.f_bavail * usage.f_frsize)
        total_bytes = int(usage.f_blocks * usage.f_frsize)
    else:
        import shutil

        disk = shutil.disk_usage(anchor)
        free_bytes = int(disk.free)
        total_bytes = int(disk.total)
    return DataRootStatus(
        path=data_root,
        path_length=len(str(data_root)),
        exists=data_root.exists(),
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        long_path_risk=len(str(data_root)) > 80,
    )


def ensure_data_root(repo_root: Path) -> Path:
    data_root = resolve_data_root(repo_root)
    for name in LOGICAL_DATA_DIRS:
        (data_root / name).mkdir(parents=True, exist_ok=True)
    marker = data_root / "metadata" / "DATA_ROOT.json"
    if not marker.exists():
        _atomic_write_bytes(
            marker,
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_at_utc": datetime.now(UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "logical_dirs": list(LOGICAL_DATA_DIRS),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
    return data_root


@contextmanager
def manifest_lock(data_root: Path, name: str, timeout_seconds: float = 30.0) -> Iterator[None]:
    lock_path = data_root / "state" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise AcquisitionError(
                    f"Timed out waiting for acquisition lock: {lock_path}"
                ) from None
            time.sleep(0.1)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_new_bytes(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AcquisitionError(f"Content-addressed object mismatch at {path}")
        return False
    _atomic_write_bytes(path, content)
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pa: Any = importlib.import_module("pyarrow")
        pq: Any = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError:
        return
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd")


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    protected = {"authorization", "cookie", "x-api-key", "api-key"}
    return {
        key: ("<redacted>" if key.lower() in protected else value)
        for key, value in headers.items()
    }


def _iso(dt: datetime) -> str:
    require_aware(dt, "datetime")
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def append_retrieval_ledger(data_root: Path, row: Mapping[str, object]) -> Path:
    ledger_csv = data_root / "manifests" / "retrieval_ledger.csv"
    fieldnames = (
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
    )
    rows = _read_csv(ledger_csv)
    rows.append({field: str(row.get(field, "")) for field in fieldnames})
    _write_csv(ledger_csv, rows, fieldnames)
    _write_parquet(data_root / "manifests" / "retrieval_ledger.parquet", rows)
    return ledger_csv


def _refresh_file_manifest(data_root: Path, ledger_rows: Sequence[Mapping[str, str]]) -> None:
    seen: dict[str, dict[str, str]] = {}
    for row in ledger_rows:
        digest = row.get("content_sha256", "")
        if not digest or digest in seen:
            continue
        seen[digest] = {
            "content_sha256": digest,
            "content_length": row.get("content_length", ""),
            "content_path": row.get("content_path", ""),
            "sidecar_path": row.get("sidecar_path", ""),
            "first_source_id": row.get("source_id", ""),
            "first_retrieved_at": row.get("retrieved_at", ""),
        }
    rows = list(seen.values())
    fieldnames = (
        "content_sha256",
        "content_length",
        "content_path",
        "sidecar_path",
        "first_source_id",
        "first_retrieved_at",
    )
    _write_csv(data_root / "manifests" / "file_manifest.csv", rows, fieldnames)
    _write_parquet(data_root / "manifests" / "file_manifest.parquet", rows)


def _refresh_dataset_lineage(data_root: Path, ledger_rows: Sequence[Mapping[str, str]]) -> None:
    rows = [
        {
            "dataset_id": f"{row.get('source_id', '')}.raw",
            "stage": "raw",
            "source_id": row.get("source_id", ""),
            "content_sha256": row.get("content_sha256", ""),
            "retrieved_at": row.get("retrieved_at", ""),
            "parser_version": "not_parsed",
            "schema_version": "raw_object_v1",
        }
        for row in ledger_rows
        if row.get("content_sha256")
    ]
    fieldnames = (
        "dataset_id",
        "stage",
        "source_id",
        "content_sha256",
        "retrieved_at",
        "parser_version",
        "schema_version",
    )
    _write_csv(data_root / "manifests" / "dataset_lineage.csv", rows, fieldnames)
    _write_parquet(data_root / "manifests" / "dataset_lineage.parquet", rows)


def store_content_addressed_retrieval(
    data_root: Path,
    *,
    source_id: str,
    provider: str,
    content: bytes,
    retrieved_at: datetime,
    extension: str,
    metadata: Mapping[str, object],
) -> AcquisitionRecord:
    require_aware(retrieved_at, "retrieved_at")
    digest = sha256_bytes(content)
    clean_extension = extension.lower().lstrip(".") or "bin"
    if not clean_extension.replace("_", "").isalnum():
        raise AcquisitionError(f"Unsafe extension: {extension!r}")

    object_dir = data_root / "raw" / "objects" / digest[:2]
    content_path = object_dir / f"{digest}.{clean_extension}"
    sidecar_path = object_dir / f"{digest}.metadata.json"
    deduplicated = not _write_new_bytes(content_path, content)
    sidecar = {
        "storage_schema_version": 1,
        "source_id_first_observed": source_id,
        "provider_first_observed": provider,
        "content_sha256": digest,
        "content_length": len(content),
        "content_path": str(content_path),
        "first_retrieved_at": _iso(retrieved_at),
        "metadata": dict(metadata),
    }
    if not sidecar_path.exists():
        _write_new_bytes(
            sidecar_path,
            (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    request_url = str(metadata.get("requested_url", ""))
    retrieved_iso = _iso(retrieved_at)
    ledger_row = {
        "retrieval_id": f"{retrieved_iso}__{source_id}__{digest[:16]}",
        "source_id": source_id,
        "provider": provider,
        "retrieved_at": retrieved_iso,
        "status": "success",
        "http_status": metadata.get("http_status", ""),
        "request_url": request_url,
        "final_url": metadata.get("final_url", ""),
        "etag": metadata.get("etag", ""),
        "last_modified": metadata.get("last_modified", ""),
        "content_sha256": digest,
        "content_length": len(content),
        "content_path": str(content_path),
        "sidecar_path": str(sidecar_path),
        "deduplicated": str(deduplicated).lower(),
        "error": "",
    }
    with manifest_lock(data_root, "manifests"):
        ledger_path = append_retrieval_ledger(data_root, ledger_row)
        ledger_rows = _read_csv(ledger_path)
        _refresh_file_manifest(data_root, ledger_rows)
        _refresh_dataset_lineage(data_root, ledger_rows)

    return AcquisitionRecord(
        source_id=source_id,
        retrieved_at=retrieved_at.astimezone(UTC),
        content_sha256=digest,
        content_length=len(content),
        content_path=content_path,
        sidecar_path=sidecar_path,
        deduplicated=deduplicated,
        ledger_path=ledger_path,
    )


def record_failed_retrieval(
    data_root: Path,
    *,
    source_id: str,
    provider: str,
    retrieved_at: datetime,
    request_url: str,
    http_status: int | str,
    error: str,
) -> Path:
    row = {
        "retrieval_id": f"{_iso(retrieved_at)}__{source_id}__failed",
        "source_id": source_id,
        "provider": provider,
        "retrieved_at": _iso(retrieved_at),
        "status": "failed",
        "http_status": http_status,
        "request_url": request_url,
        "final_url": "",
        "etag": "",
        "last_modified": "",
        "content_sha256": "",
        "content_length": "",
        "content_path": "",
        "sidecar_path": "",
        "deduplicated": "",
        "error": error,
    }
    with manifest_lock(data_root, "manifests"):
        return append_retrieval_ledger(data_root, row)


def fetch_http_to_acquisition(
    data_root: Path,
    *,
    source_id: str,
    provider: str,
    url: str,
    policy: FetchPolicy | None = None,
    request_headers: Mapping[str, str] | None = None,
) -> AcquisitionRecord:
    policy = policy or FetchPolicy()
    if policy.max_attempts < 1:
        raise ValueError("FetchPolicy.max_attempts must be >= 1")
    headers = {"User-Agent": policy.user_agent, **dict(request_headers or {})}
    response: httpx.Response | None = None
    last_error: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            with httpx.Client(
                timeout=policy.timeout_seconds,
                follow_redirects=policy.follow_redirects,
                headers=headers,
            ) as client:
                response = client.get(url)
            break
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < policy.max_attempts and policy.retry_sleep_seconds > 0:
                time.sleep(policy.retry_sleep_seconds)

    retrieved_at = datetime.now(UTC)
    if response is None:
        error = f"Request failed after {policy.max_attempts} attempt(s): {last_error}"
        record_failed_retrieval(
            data_root,
            source_id=source_id,
            provider=provider,
            retrieved_at=retrieved_at,
            request_url=url,
            http_status="",
            error=error,
        )
        raise FetchError(error)

    if response.status_code < 200 or response.status_code >= 300:
        error = f"HTTP {response.status_code}"
        record_failed_retrieval(
            data_root,
            source_id=source_id,
            provider=provider,
            retrieved_at=retrieved_at,
            request_url=url,
            http_status=response.status_code,
            error=error,
        )
        raise FetchError(f"{error} for {source_id} at {url}")
    if not response.content:
        error = "empty payload"
        record_failed_retrieval(
            data_root,
            source_id=source_id,
            provider=provider,
            retrieved_at=retrieved_at,
            request_url=url,
            http_status=response.status_code,
            error=error,
        )
        raise FetchError(f"Empty payload for {source_id} at {url}")
    if len(response.content) > policy.max_bytes:
        error = f"payload exceeds max_bytes: {len(response.content)} > {policy.max_bytes}"
        record_failed_retrieval(
            data_root,
            source_id=source_id,
            provider=provider,
            retrieved_at=retrieved_at,
            request_url=url,
            http_status=response.status_code,
            error=error,
        )
        raise FetchError(error)

    extension = infer_extension(str(response.url), response.headers.get("content-type"))
    metadata: dict[str, object] = {
        "requested_url": url,
        "final_url": str(response.url),
        "request_method": "GET",
        "request_headers": _redact_headers(headers),
        "http_status": response.status_code,
        "response_headers": dict(response.headers),
        "content_type": response.headers.get("content-type"),
        "extension_inferred": extension,
        "etag": response.headers.get("etag", ""),
        "last_modified": response.headers.get("last-modified", ""),
    }
    return store_content_addressed_retrieval(
        data_root,
        source_id=source_id,
        provider=provider,
        content=response.content,
        retrieved_at=retrieved_at,
        extension=extension,
        metadata=metadata,
    )
