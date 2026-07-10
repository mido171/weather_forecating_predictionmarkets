"""Shared standard-library helpers for HKG T+24 Research Director scripts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EXPERIMENT_RE = re.compile(r"^(?P<id>\d{4})_(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:  # repo-doctor: allow-unsafe-default - exits at file EOF
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sample_hash(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def normalize_tokens(text: str) -> set[str]:
    stop = {
        "the","and","for","with","from","that","this","into","using","use","a","an",
        "of","to","in","on","is","be","as","or","by","it","at","are","was","were",
    }
    return {
        token for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 2 and token not in stop
    }


def read_text_if_exists(path: Path, max_chars: int = 2_000_000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def resolve_roots(repo_root: Path) -> tuple[Path, Path, Path]:
    repo_root = repo_root.resolve()
    experiments = repo_root / "experiments"
    datasets = repo_root / "data" / "datasets"
    memory = repo_root / ".hkg_t24_research"
    return experiments, datasets, memory
