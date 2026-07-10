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


def load_project_env() -> None:
    """Load ignored local defaults without overriding the process environment."""

    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def load_settings(*, require_db: bool = False) -> Settings:
    load_project_env()
    database_url = os.getenv("KLGA_DB_URL")
    if require_db and not database_url:
        raise ConfigError("KLGA_DB_URL is required for this command")

    env = os.getenv("KLGA_ENV", "local")
    if env not in VALID_ENVS:
        raise ConfigError(f"KLGA_ENV must be one of {sorted(VALID_ENVS)}")

    trading_mode = os.getenv("KLGA_TRADING_MODE", "backtest")
    if trading_mode not in VALID_TRADING_MODES:
        raise ConfigError(
            f"KLGA_TRADING_MODE must be one of {sorted(VALID_TRADING_MODES)}"
        )

    if env == "prod" or trading_mode == "live":
        acknowledgement = os.getenv("KLGA_ENABLE_LIVE", "")
        if acknowledgement != "I_UNDERSTAND_LIVE_EXECUTION":
            raise ConfigError(
                "prod/live mode is fail-closed; set KLGA_ENABLE_LIVE to the exact "
                "documented acknowledgement after an operator review"
            )

    run_root = Path(
        os.getenv(
            "KLGA_RUN_ROOT",
            str(Path.home() / ".local" / "share" / "weather-markets" / "klga-tmax"),
        )
    ).expanduser()
    artifact_root = Path(
        os.getenv("KLGA_ARTIFACT_ROOT", str(run_root / "artifacts"))
    ).expanduser()

    n_jobs_raw = os.getenv("KLGA_N_JOBS", "1")
    try:
        n_jobs = int(n_jobs_raw)
    except ValueError as exc:
        raise ConfigError("KLGA_N_JOBS must be an integer") from exc
    if n_jobs < 1:
        raise ConfigError("KLGA_N_JOBS must be >= 1")
    if n_jobs > 2:
        raise ConfigError("KLGA_N_JOBS must be <= 2 without a reviewed code change")

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
