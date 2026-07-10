from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return REPO_ROOT


def config_path() -> Path:
    return REPO_ROOT / "config" / "live_kmia.yaml"


def models_dir(station_id: str) -> Path:
    return REPO_ROOT / "models" / station_id.lower()


def artifacts_root() -> Path:
    return REPO_ROOT / "artifacts"
