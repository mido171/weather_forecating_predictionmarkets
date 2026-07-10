from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .constants import EXIT_CONFIG_ERROR, PROJECT_ROOT


class ConfigError(RuntimeError):
    """Raised when required KLGA runtime configuration is absent or invalid."""

    exit_code = EXIT_CONFIG_ERROR


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    artifact_root: Path
    env: str
    trading_mode: str
    n_jobs: int
    log_level: str


VALID_ENVS = {"local", "paper", "prod"}
VALID_TRADING_MODES = {"backtest", "paper", "live"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


def load_settings(*, require_db: bool = False) -> Settings:
    database_url = os.getenv("KLGA_DB_URL")
    if require_db and not database_url:
        raise ConfigError("KLGA_DB_URL is required for this command")

    env = os.getenv("KLGA_ENV", "local")
    if env not in VALID_ENVS:
        raise ConfigError(f"KLGA_ENV must be one of {sorted(VALID_ENVS)}")

    trading_mode = os.getenv("KLGA_TRADING_MODE", "paper")
    if trading_mode not in VALID_TRADING_MODES:
        raise ConfigError(
            f"KLGA_TRADING_MODE must be one of {sorted(VALID_TRADING_MODES)}"
        )

    artifact_root_raw = os.getenv("KLGA_ARTIFACT_ROOT")
    if not artifact_root_raw:
        if env == "local":
            artifact_root = PROJECT_ROOT / "artifacts" / "klga_tmax"
        else:
            raise ConfigError("KLGA_ARTIFACT_ROOT is required outside local env")
    else:
        artifact_root = Path(artifact_root_raw)

    n_jobs_raw = os.getenv("KLGA_N_JOBS", "1")
    try:
        n_jobs = int(n_jobs_raw)
    except ValueError as exc:
        raise ConfigError("KLGA_N_JOBS must be an integer") from exc
    if n_jobs < 1:
        raise ConfigError("KLGA_N_JOBS must be >= 1")

    log_level = os.getenv("KLGA_LOG_LEVEL", "INFO")
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(f"KLGA_LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}")

    return Settings(
        database_url=database_url,
        artifact_root=artifact_root,
        env=env,
        trading_mode=trading_mode,
        n_jobs=n_jobs,
        log_level=log_level,
    )
