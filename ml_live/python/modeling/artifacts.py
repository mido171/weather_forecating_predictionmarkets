from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml_live.runtime.paths import artifacts_root


def find_e92_run_dir(root: Path | None = None) -> Path:
    base = root or artifacts_root() / "time_feature_sweep" / "kmia"
    candidates = list(base.glob("**/E92"))
    if not candidates:
        raise FileNotFoundError(f"No E92 run directories found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_feature_list(e92_dir: Path) -> Path:
    direct = e92_dir / "feature_list.json"
    if direct.exists():
        return direct
    base = artifacts_root() / "time_feature_sweep" / "kmia"
    candidates = list(base.glob("**/E92/feature_list.json"))
    if not candidates:
        raise FileNotFoundError(f"No feature_list.json found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_feature_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid feature list: {path}")
    return [str(item) for item in data]


def feature_list_hash(features: list[str]) -> str:
    payload = "\n".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_dataset_path(e92_dir: Path) -> Path:
    dataset_id_path = e92_dir / "dataset_id.txt"
    if dataset_id_path.exists():
        dataset_id = dataset_id_path.read_text(encoding="utf-8").strip()
        if dataset_id:
            resolved = _find_dataset_by_id(dataset_id)
            if resolved is not None:
                return resolved
    return discover_latest_dataset()


def _find_dataset_by_id(dataset_id: str) -> Path | None:
    base = artifacts_root() / "time_feature_sweep" / "kmia"
    cleaned = dataset_id.strip()
    if not cleaned:
        return None
    best_match: Path | None = None
    best_len = 0
    for parquet_path in base.glob("**/datasets/**/data.parquet"):
        dataset_dir = parquet_path.parent
        name = dataset_dir.name
        if not name:
            continue
        if cleaned.startswith(name) or name.startswith(cleaned):
            match_len = min(len(cleaned), len(name))
            if match_len > best_len:
                best_len = match_len
                best_match = parquet_path
    return best_match


def discover_latest_dataset() -> Path:
    base = artifacts_root() / "time_feature_sweep" / "kmia"
    candidates = list(base.glob("**/datasets/**/data.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No datasets found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)
