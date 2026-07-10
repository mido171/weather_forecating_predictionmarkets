"""Shared helpers for the HKG T+24 Experiment Executor.

This module intentionally uses only the Python standard library.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EXPERIMENT_RE = re.compile(r"^(?P<id>\d{4})_(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)$")
COMPLETED_STATUSES = {
    "COMPLETED_PROMOTION_CANDIDATE",
    "COMPLETED_INFORMATION_GAIN_ONLY",
    "COMPLETED_NULL_OR_NEGATIVE",
}
REJECTED_STATUSES = {
    "REJECTED_LEAKAGE",
    "REJECTED_TIMESTAMP",
    "REJECTED_SPECIFICATION",
    "REJECTED_DATA_QUALITY",
    "BLOCKED_MISSING_DATA",
}
ALL_STATUSES = COMPLETED_STATUSES | REJECTED_STATUSES | {"FAILED_RUNTIME"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, exclude_names: Iterable[str] = ()) -> str:
    excluded = set(exclude_names)
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in excluded:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def find_repo_root(start: Path) -> Path:
    """Find a plausible repo root without inventing one."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "experiments").is_dir() and (
            (candidate / "data" / "datasets").exists()
            or (candidate / ".git").exists()
            or (candidate / "AGENTS.md").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not resolve repository root. Pass --repo-root explicitly; "
        "the executor will not guess a path."
    )


def next_experiment_id(experiments_root: Path) -> int:
    highest = 0
    if experiments_root.exists():
        for child in experiments_root.iterdir():
            match = EXPERIMENT_RE.match(child.name)
            if match:
                highest = max(highest, int(match.group("id")))
    return highest + 1


class DirectoryLock:
    """Portable best-effort lock using atomic directory creation."""

    def __init__(self, lock_dir: Path, timeout_seconds: float = 30.0) -> None:
        self.lock_dir = lock_dir
        self.timeout_seconds = timeout_seconds
        self.acquired = False

    def __enter__(self) -> "DirectoryLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.lock_dir.mkdir(parents=False, exist_ok=False)
                owner = {
                    "pid": os.getpid(),
                    "created_at_utc": utc_now(),
                }
                write_json(self.lock_dir / "owner.json", owner)
                self.acquired = True
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not acquire lock: {self.lock_dir}")
                time.sleep(0.15)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired:
            shutil.rmtree(self.lock_dir, ignore_errors=True)
            self.acquired = False
