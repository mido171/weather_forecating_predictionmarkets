from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from klga_tmax.config import ConfigError, load_project_env, load_settings
from klga_tmax.constants import PROJECT_ROOT


DEFAULT_GRIBSTREAM_BASE_URL = "https://gribstream.com/api/v2"
@dataclass(frozen=True)
class GribStreamSettings:
    api_token: str | None
    base_url: str
    artifact_root: Path
    timeout_seconds: float
    spacing_seconds: float
    max_retries: int
    user_agent: str


def load_gribstream_settings(*, require_api_token: bool) -> GribStreamSettings:
    load_project_env()
    base_settings = load_settings(require_db=False)
    token = None
    token_file_raw = os.getenv("GRIBSTREAM_API_TOKEN_FILE", "").strip()
    if token_file_raw:
        token_file = Path(token_file_raw).expanduser()
        if not token_file.is_absolute():
            token_file = PROJECT_ROOT / token_file
        if not token_file.is_file():
            raise ConfigError(f"GRIBSTREAM_API_TOKEN_FILE does not exist: {token_file}")
        token = token_file.read_text(encoding="utf-8").strip()
    token = token or os.getenv("GRIBSTREAM_API_TOKEN") or os.getenv("GRIBSTREAM_API_KEY")
    if token:
        token = token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
    if require_api_token and not token:
        raise ConfigError("GRIBSTREAM_API_TOKEN or GRIBSTREAM_API_KEY is required")

    base_url = os.getenv("GRIBSTREAM_BASE_URL", DEFAULT_GRIBSTREAM_BASE_URL).rstrip("/")
    timeout_raw = os.getenv("GRIBSTREAM_TIMEOUT_SECONDS", "90")
    spacing_raw = os.getenv("GRIBSTREAM_SPACING_SECONDS", "12")
    retries_raw = os.getenv("GRIBSTREAM_MAX_RETRIES", "2")
    try:
        timeout_seconds = float(timeout_raw)
        spacing_seconds = float(spacing_raw)
        max_retries = int(retries_raw)
    except ValueError as exc:
        raise ConfigError(
            "GRIBSTREAM_TIMEOUT_SECONDS and GRIBSTREAM_SPACING_SECONDS must be numeric; "
            "GRIBSTREAM_MAX_RETRIES must be an integer"
        ) from exc
    if timeout_seconds <= 0:
        raise ConfigError("GRIBSTREAM_TIMEOUT_SECONDS must be > 0")
    if spacing_seconds < 12:
        raise ConfigError("GRIBSTREAM_SPACING_SECONDS must be >= 12")
    if max_retries < 0 or max_retries > 3:
        raise ConfigError("GRIBSTREAM_MAX_RETRIES must be between 0 and 3")

    artifact_root = Path(
        os.getenv(
            "KLGA_GRIBSTREAM_ARTIFACT_ROOT",
            str(base_settings.artifact_root / "gribstream"),
        )
    )
    return GribStreamSettings(
        api_token=token,
        base_url=base_url,
        artifact_root=artifact_root,
        timeout_seconds=timeout_seconds,
        spacing_seconds=spacing_seconds,
        max_retries=max_retries,
        user_agent=os.getenv("GRIBSTREAM_USER_AGENT", "klga-tmax-gribstream-backfill/1.0"),
    )


def redacted_settings_payload(settings: GribStreamSettings) -> dict[str, object]:
    return {
        "api_token_present": bool(settings.api_token),
        "base_url": settings.base_url,
        "artifact_root": str(settings.artifact_root),
        "timeout_seconds": settings.timeout_seconds,
        "spacing_seconds": settings.spacing_seconds,
        "max_retries": settings.max_retries,
        "user_agent": settings.user_agent,
    }
