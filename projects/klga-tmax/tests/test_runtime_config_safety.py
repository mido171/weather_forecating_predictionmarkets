from __future__ import annotations

import pytest

from klga_tmax.config import ConfigError, load_settings
from klga_tmax.providers.gribstream.config import load_gribstream_settings
from klga_tmax.providers.wunderground.config import load_wunderground_settings


def _clear_runtime_env(monkeypatch) -> None:
    for name in (
        "KLGA_ENV",
        "KLGA_TRADING_MODE",
        "KLGA_ENABLE_LIVE",
        "KLGA_N_JOBS",
        "KLGA_RUN_ROOT",
        "KLGA_ARTIFACT_ROOT",
        "WUNDERGROUND_API_MAX_RETRIES",
        "WUNDERGROUND_API_RATE_LIMIT_PER_MINUTE",
        "WUNDERGROUND_MAX_WORKERS",
        "WUNDERGROUND_CHUNK_DAYS",
        "GRIBSTREAM_SPACING_SECONDS",
        "GRIBSTREAM_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_runtime_defaults_are_offline_and_bounded(monkeypatch, tmp_path) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr("klga_tmax.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("KLGA_RUN_ROOT", str(tmp_path / "runs"))

    settings = load_settings()
    wunderground = load_wunderground_settings()
    gribstream = load_gribstream_settings(require_api_token=False)

    assert settings.trading_mode == "backtest"
    assert settings.n_jobs == 1
    assert settings.artifact_root == tmp_path / "runs" / "artifacts"
    assert wunderground.max_workers == 1
    assert wunderground.rate_limit_per_minute == 30
    assert wunderground.max_retries == 2
    assert wunderground.chunk_days == 7
    assert gribstream.spacing_seconds == 12
    assert gribstream.max_retries == 2


def test_live_mode_requires_exact_acknowledgement(monkeypatch, tmp_path) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr("klga_tmax.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("KLGA_TRADING_MODE", "live")

    with pytest.raises(ConfigError, match="fail-closed"):
        load_settings()


def test_concurrency_above_two_is_rejected(monkeypatch, tmp_path) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr("klga_tmax.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("KLGA_N_JOBS", "3")

    with pytest.raises(ConfigError, match="must be <= 2"):
        load_settings()


def test_gribstream_spacing_below_provider_floor_is_rejected(monkeypatch, tmp_path) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr("klga_tmax.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("GRIBSTREAM_SPACING_SECONDS", "2")

    with pytest.raises(ConfigError, match="must be >= 12"):
        load_gribstream_settings(require_api_token=False)
