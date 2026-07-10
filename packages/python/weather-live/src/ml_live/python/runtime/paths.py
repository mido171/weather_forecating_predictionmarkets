from __future__ import annotations

import os
from pathlib import Path


def _discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "apps").is_dir()
            and (candidate / "packages").is_dir()
            and (candidate / "projects").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Could not locate the weather-markets repository from {current}")


REPO_ROOT = _discover_repo_root()


def repo_root() -> Path:
    return REPO_ROOT


def config_path() -> Path:
    configured = os.getenv("WEATHER_MARKETS_LIVE_CONFIG", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT / "config" / "examples" / "kmia-live.yaml"
    )


def runtime_root() -> Path:
    configured = os.getenv("WEATHER_MARKETS_RUN_ROOT", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT / "var" / "weather-live"
    )


def models_dir(station_id: str) -> Path:
    return runtime_root() / "models" / station_id.lower()


def artifacts_root() -> Path:
    return runtime_root() / "artifacts"
