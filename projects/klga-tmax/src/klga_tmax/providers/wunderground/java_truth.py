from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import subprocess
from urllib.parse import unquote, urlparse

from klga_tmax.config import Settings
from klga_tmax.constants import PROJECT_ROOT
from klga_tmax.providers.wunderground.config import WundergroundSettings


JAVA_MAIN_CLASS = "com.predictionmarkets.weather.klga.wu.KlgaWundergroundTruthCli"


@dataclass(frozen=True)
class JavaWuResult:
    payload: dict
    stdout: str
    stderr: str


def run_java_wu_truth(
    *,
    settings: Settings,
    wu_settings: WundergroundSettings,
    command: str,
    args: dict[str, object],
) -> JavaWuResult:
    if not settings.database_url:
        raise RuntimeError("KLGA_DB_URL is required")
    extraction_root = _find_extraction_root()
    maven = shutil.which("mvn.cmd") or shutil.which("mvn")
    if not maven:
        raise RuntimeError("Maven executable not found on PATH; required for Java WU fetch runner")

    db_parts = _postgres_jdbc_parts(settings.database_url)
    env = os.environ.copy()
    env["KLGA_WU_JDBC_URL"] = db_parts.jdbc_url
    if db_parts.user:
        env["KLGA_WU_DB_USER"] = db_parts.user
    if db_parts.password:
        env["KLGA_WU_DB_PASSWORD"] = db_parts.password
    if wu_settings.api_key:
        env["KLGA_WU_API_KEY"] = wu_settings.api_key
    env["KLGA_WU_BASE_URL"] = wu_settings.base_url
    env["KLGA_ARTIFACT_ROOT"] = str(settings.artifact_root)

    java_args = ["--command", command]
    for key, value in args.items():
        if value is None:
            continue
        java_args.extend([f"--{key.replace('_', '-')}", str(value)])

    timeout_seconds = int(os.getenv("KLGA_JAVA_TIMEOUT_SECONDS", "900"))
    if timeout_seconds < 30 or timeout_seconds > 1800:
        raise RuntimeError("KLGA_JAVA_TIMEOUT_SECONDS must be between 30 and 1800")

    completed = subprocess.run(
        [
            maven,
            "-q",
            "-pl",
            "apps/ingestion-service",
            "-DskipTests",
            "exec:java",
            f"-Dexec.mainClass={JAVA_MAIN_CLASS}",
            f"-Dexec.args={' '.join(java_args)}",
        ],
        cwd=extraction_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Java WU truth runner failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    payload = _extract_json_payload(completed.stdout)
    return JavaWuResult(payload=payload, stdout=completed.stdout, stderr=completed.stderr)


@dataclass(frozen=True)
class JdbcParts:
    jdbc_url: str
    user: str | None
    password: str | None


def _postgres_jdbc_parts(database_url: str) -> JdbcParts:
    parsed = urlparse(database_url.replace("postgresql+psycopg://", "postgresql://"))
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise RuntimeError("KLGA_DB_URL must be a PostgreSQL URL for Java WU truth runner")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    return JdbcParts(
        jdbc_url=f"jdbc:postgresql://{host}:{port}{path}{query}",
        user=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else None,
    )


def _find_extraction_root() -> Path:
    current = PROJECT_ROOT
    for candidate in (current, *current.parents):
        if (candidate / "apps" / "ingestion-service" / "pom.xml").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the weather-markets root containing apps/ingestion-service/pom.xml"
    )


def _extract_json_payload(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise RuntimeError("Java WU truth runner produced no stdout")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"Java WU truth runner did not print JSON: {text[-1000:]}")
    return json.loads(text[start : end + 1])
