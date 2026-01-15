"""Artifact persistence helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def compute_dataset_id(csv_path: Path, schema_version: int, feature_config: dict) -> str:
    csv_bytes = csv_path.read_bytes()
    config_json = json.dumps(
        feature_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload = csv_bytes + str(schema_version).encode("utf-8") + config_json
    return sha256_bytes(payload)


def snapshot_to_parquet(df: pd.DataFrame, dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "data.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    return path


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def write_hash_manifest(paths: list[Path], output_path: Path) -> None:
    manifest = {str(path): sha256_file(path) for path in paths}
    write_metadata(output_path, manifest)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
