from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hashing import sha256_bytes
from .paths import archive_reference_fields, infer_storage_root
from .timeutils import require_aware


class StorageError(RuntimeError):
    """Raised when immutable snapshot storage fails."""


@dataclass(frozen=True)
class RawSnapshot:
    source_id: str
    content_path: Path
    sidecar_path: Path
    sha256: str
    retrieved_at: datetime
    content_length: int


def _safe_source_id(source_id: str) -> str:
    if not source_id or any(part in source_id for part in ("..", "/", "\\")):
        raise StorageError(f"Unsafe source_id: {source_id!r}")
    return source_id


def _atomic_write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise StorageError(f"Refusing to overwrite existing immutable path: {path}")
        return

    # Keep the temp prefix short so atomic writes remain usable on Windows paths.
    fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise StorageError(f"Immutable path race with different bytes: {path}") from None
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def store_raw_bytes(
    raw_root: Path,
    *,
    source_id: str,
    content: bytes,
    retrieved_at: datetime,
    extension: str,
    metadata: dict[str, Any] | None = None,
) -> RawSnapshot:
    source_id = _safe_source_id(source_id)
    require_aware(retrieved_at, "retrieved_at")
    retrieved_utc = retrieved_at.astimezone(UTC)
    digest = sha256_bytes(content)
    clean_extension = extension.lower().lstrip(".") or "bin"
    if not clean_extension.replace("_", "").isalnum():
        raise StorageError(f"Unsafe extension: {extension!r}")

    date_dir = raw_root / source_id / retrieved_utc.strftime("%Y/%m/%d")
    stamp = retrieved_utc.strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{stamp}__{digest[:16]}"
    content_path = date_dir / f"{stem}.{clean_extension}"
    sidecar_path = date_dir / f"{stem}.metadata.json"
    references = archive_reference_fields(
        data_root=infer_storage_root(raw_root),
        content_path=content_path,
        sidecar_path=sidecar_path,
    )

    sidecar: dict[str, Any] = {
        **references,
        "source_id": source_id,
        "retrieved_at": retrieved_utc.isoformat().replace("+00:00", "Z"),
        "content_sha256": digest,
        "content_length": len(content),
    }
    if metadata:
        sidecar["metadata"] = metadata

    _atomic_write_new(content_path, content)
    _atomic_write_new(
        sidecar_path,
        (json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
    return RawSnapshot(
        source_id=source_id,
        content_path=content_path,
        sidecar_path=sidecar_path,
        sha256=digest,
        retrieved_at=retrieved_utc,
        content_length=len(content),
    )
