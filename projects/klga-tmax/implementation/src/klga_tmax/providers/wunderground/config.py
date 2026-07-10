from __future__ import annotations

from dataclasses import dataclass
import os

from klga_tmax.config import ConfigError


@dataclass(frozen=True)
class WundergroundSettings:
    base_url: str
    api_key: str | None
    timeout_seconds: int
    max_retries: int
    rate_limit_per_minute: int
    max_workers: int
    chunk_days: int
    intraday_available_lag_minutes: int
    user_agent: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def _int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def load_wunderground_settings(*, require_api_key: bool = False) -> WundergroundSettings:
    api_key = os.getenv("WUNDERGROUND_API_KEY") or os.getenv("WEATHERCOM_API_KEY")
    if require_api_key and not api_key:
        raise ConfigError("WUNDERGROUND_API_KEY or WEATHERCOM_API_KEY is required for live WU fetches")

    return WundergroundSettings(
        base_url=os.getenv("WUNDERGROUND_API_BASE_URL", "https://api.weather.com").rstrip("/"),
        api_key=api_key,
        timeout_seconds=_int_env("WUNDERGROUND_API_TIMEOUT_SECONDS", 30, minimum=1),
        max_retries=_int_env("WUNDERGROUND_API_MAX_RETRIES", 5, minimum=0),
        rate_limit_per_minute=_int_env("WUNDERGROUND_API_RATE_LIMIT_PER_MINUTE", 120, minimum=1),
        max_workers=_int_env("WUNDERGROUND_MAX_WORKERS", 4, minimum=1),
        chunk_days=_int_env("WUNDERGROUND_CHUNK_DAYS", 31, minimum=1),
        intraday_available_lag_minutes=_int_env(
            "WUNDERGROUND_INTRADAY_AVAILABLE_LAG_MINUTES",
            90,
            minimum=0,
        ),
        user_agent=os.getenv(
            "WUNDERGROUND_API_USER_AGENT",
            "klga-tmax/0.1 weathercom-historical-observations",
        ),
    )


def redacted_settings_payload(settings: WundergroundSettings) -> dict[str, object]:
    return {
        "base_url": settings.base_url,
        "api_key_present": settings.has_api_key,
        "api_key_source": (
            "WUNDERGROUND_API_KEY"
            if os.getenv("WUNDERGROUND_API_KEY")
            else ("WEATHERCOM_API_KEY" if os.getenv("WEATHERCOM_API_KEY") else None)
        ),
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "max_workers": settings.max_workers,
        "chunk_days": settings.chunk_days,
        "intraday_available_lag_minutes": settings.intraday_available_lag_minutes,
        "user_agent": settings.user_agent,
    }
