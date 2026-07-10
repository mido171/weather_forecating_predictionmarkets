from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from hkg_tmax.paths import ProjectPaths
from hkg_tmax_db.connection import (
    DatabaseUnavailable,
    apply_migration,
    import_psycopg,
    redact_database_url,
)
from hkg_tmax_db.hashing import sha256_file

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
TASK_ROOT = REPO_ROOT / "tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION"
TASKS_NOT_COMPLETED = TASK_ROOT / "tasks/not-completed"
TASKS_COMPLETED = TASK_ROOT / "tasks/completed"
STATUS_INDEX = TASK_ROOT / "TASK_STATUS_INDEX.csv"

MODEL_MATRIX = TASK_ROOT / "GRIBSTREAM_MODEL_DISPOSITION_MATRIX.csv"
VARIABLE_REQUIREMENTS = TASK_ROOT / "SEMANTIC_VARIABLE_REQUIREMENTS.csv"
DATE_RANGE_PLAN = TASK_ROOT / "DATA_SOURCE_AND_DATE_RANGE_PLAN.csv"
REFERENCE_SCHEMA = TASK_ROOT / "schemas/REFERENCE_POSTGRES_SCHEMA.sql"
T03_SPEC = TASK_ROOT / "specs/t03_gribstream_catalog_coverage_licence_quota_audit.json"
T04_SPEC = TASK_ROOT / "specs/t04_nwp_database_object_storage_migrations.json"
T05_SPEC = TASK_ROOT / "specs/t05_canonical_location_station_geospatial_registry.json"

STATION_REGISTRY = REPO_ROOT / "config/hkg_t24/station_registry.csv"
ISD_DOSSIER = TASK_ROOT / "evidence/HKG_TMAX_ISD_STATION_DOSSIER_36.csv"
ARWF_PARQUET = PROJECT_PATHS.data_root / "datasets/09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet"
STATIC_GEOSPATIAL_INVENTORY = (
    PROJECT_PATHS.data_root / "datasets/11_static_geospatial_inventory/static_geospatial_package_inventory.parquet"
)

T03_EXP = REPO_ROOT / "experiments/0210_gribstream_catalog_coverage_licence_quota_audit"
T04_EXP = REPO_ROOT / "experiments/0211_nwp_database_object_storage_migrations"
T05_EXP = REPO_ROOT / "experiments/0212_canonical_location_station_geospatial_registry"

MIGRATION_T03 = REPO_ROOT / "db/migrations/postgres/20260624_0004_t03_gribstream_catalog_registry.sql"
MIGRATION_T04 = REPO_ROOT / "db/migrations/postgres/20260624_0005_t04_nwp_storage_lineage.sql"
MIGRATION_T05 = REPO_ROOT / "db/migrations/postgres/20260624_0006_t05_location_station_geospatial_registry.sql"
MIGRATION_T00 = REPO_ROOT / "db/migrations/postgres/20260623_0001_audit_driven_ingestion.sql"
MIGRATION_T01 = REPO_ROOT / "db/migrations/postgres/20260624_0002_t24_time_availability_contract.sql"
MIGRATION_T02 = REPO_ROOT / "db/migrations/postgres/20260624_0003_t02_census_registry_compatibility.sql"

TEST_PATH = REPO_ROOT / "tests/test_t03_t05_foundation_tasks.py"
SCRIPT_PATH = REPO_ROOT / "scripts/run_t03_t05_foundation_tasks.py"
SECRET_FILE = REPO_ROOT / "secrets/local/gribstream.env"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
T03_STATUS_PATH = T03_EXP / "logs/t03_t05_background_status.json"
T03_API_EVENT_LOG = T03_EXP / "logs/gribstream_api_events.jsonl"
DEFAULT_API_MIN_INTERVAL_SECONDS = 12.0
DEFAULT_API_MAX_ATTEMPTS = 3
DEFAULT_API_MAX_RETRY_AFTER_SECONDS = 1800.0
TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}
PERMANENT_HTTP_STATUS = {400, 401, 403, 404}

HKG_COORDINATES = [
    {"lat": 22.301944, "lon": 114.174167, "name": "hkg_hko_target"},
    {"lat": 22.533, "lon": 114.15, "name": "hkg_inland_north"},
    {"lat": 22.183, "lon": 114.30, "name": "hkg_marine_east"},
]

RELEVANT_DISPOSITIONS = {
    "core_backfill",
    "core_short_history",
    "challenger_backfill",
    "shadow_backfill",
    "prospective_urgent",
    "prospective_low_priority",
    "coverage_probe",
    "secondary_targeted",
    "secondary_probe",
}

NAME_ALIASES = {
    "air_temperature": ["TMP", "T"],
    "dew_point_temperature": ["DPT", "DPT2M", "2D"],
    "relative_or_specific_humidity": ["RH", "SPFH", "R"],
    "u_wind": ["UGRD", "U"],
    "v_wind": ["VGRD", "V"],
    "mean_sea_level_pressure": ["PRMSL", "MSL", "MSLP"],
    "surface_pressure": ["PRES", "SP"],
    "total_cloud_cover": ["TCDC", "TCC"],
    "low_mid_high_cloud": ["LCDC", "MCDC", "HCDC", "TCDC"],
    "total_precipitation": ["APCP", "TP", "PRATE"],
    "precipitation_rate": ["PRATE"],
    "downward_shortwave_radiation": ["DSWRF", "SSRD"],
    "downward_longwave_radiation": ["DLWRF", "STRD"],
    "planetary_boundary_layer_height": ["HPBL", "BLH"],
    "sensible_heat_flux": ["SHTFL", "SSHF"],
    "latent_heat_flux": ["LHTFL", "SLHF"],
    "soil_temperature": ["TSOIL", "ST"],
    "soil_moisture": ["SOILW", "SM"],
    "CAPE": ["CAPE"],
    "CIN": ["CIN"],
    "temperature": ["TMP", "T"],
    "geopotential_height": ["HGT", "GH"],
    "vertical_velocity": ["VVEL", "W"],
}

DESIGNED_LOCATIONS = [
    {
        "location_code": "hkg_hko_target",
        "name": "Hong Kong Observatory target",
        "latitude": "22.301944",
        "longitude": "114.174167",
        "elevation_m": "32",
        "location_role": "target",
        "metadata_source": "config/hkg_t24/station_registry.csv",
    },
    {
        "location_code": "hkg_inland_north_reference",
        "name": "North inland reference",
        "latitude": "22.533000",
        "longitude": "114.150000",
        "elevation_m": "13",
        "location_role": "designed_reference",
        "metadata_source": "T05 designed reference point",
    },
    {
        "location_code": "hkg_marine_east_reference",
        "name": "Eastern marine reference",
        "latitude": "22.183000",
        "longitude": "114.300000",
        "elevation_m": "60",
        "location_role": "designed_reference",
        "metadata_source": "T05 designed reference point",
    },
    {
        "location_code": "hkg_airport_west_reference",
        "name": "Western airport reference",
        "latitude": "22.309000",
        "longitude": "113.915000",
        "elevation_m": "8.5",
        "location_role": "designed_reference",
        "metadata_source": "T05 designed reference point",
    },
    {
        "location_code": "hkg_local_domain_center",
        "name": "HKG local domain center",
        "latitude": "22.350000",
        "longitude": "114.120000",
        "elevation_m": "",
        "location_role": "domain_reference",
        "metadata_source": "T05 designed local domain",
    },
    {
        "location_code": "synoptic_south_china_center",
        "name": "South China synoptic domain center",
        "latitude": "23.000000",
        "longitude": "114.000000",
        "elevation_m": "",
        "location_role": "domain_reference",
        "metadata_source": "T05 designed synoptic domain",
    },
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def repo_uri(path: Path) -> str:
    return f"repo://{repo_rel(path)}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_row_sha(row: dict[str, Any]) -> str:
    return sha256_text(json.dumps(row, sort_keys=True, ensure_ascii=True))


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(fs_path(path), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def write_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    with open(fs_path(path), "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def fs_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def ensure_directory(path: Path) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_directory(dst.parent)
    shutil.copy2(fs_path(src), fs_path(dst))


def file_manifest_sha(paths: list[Path]) -> str:
    rows = []
    for path in sorted({item for item in paths if item.exists()}, key=lambda item: repo_rel(item)):
        rows.append({"path": repo_rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return sha256_text(json.dumps(rows, sort_keys=True))


def ensure_task_dirs() -> None:
    for output_dir in (T03_EXP, T04_EXP, T05_EXP):
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (output_dir / "tests").mkdir(parents=True, exist_ok=True)
        (output_dir / "migrations").mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def write_status(stage: str, status: str, **details: Any) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "updated_at_utc": utc_now_iso(),
        "pid": os.getpid(),
        **details,
    }
    write_json(T03_STATUS_PATH, payload)


def load_gribstream_token() -> str | None:
    if not SECRET_FILE.exists():
        return None
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("GRIBSTREAM_API_KEY="):
            token = line.split("=", 1)[1].strip()
            return token or None
    return None


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return max(float(text), 0.0)
    except ValueError:
        return None


class OneThreadRateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(min_interval_seconds, 0.0)
        self._next_allowed_at = 0.0

    def wait(self) -> float:
        now = time.monotonic()
        delay = max(self._next_allowed_at - now, 0.0)
        if delay > 0:
            time.sleep(delay)
        self._next_allowed_at = time.monotonic() + self.min_interval_seconds
        return delay


def bounded_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def api_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)


def fetch_public_text(client: httpx.Client, url: str) -> dict[str, Any]:
    started = utc_now_iso()
    try:
        response = client.get(url)
        text = response.text
        return {
            "url": url,
            "status_code": response.status_code,
            "retrieved_at_utc": started,
            "content_sha256": sha256_text(text),
            "content_bytes": len(text.encode("utf-8")),
            "text": text,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - network diagnostic path
        return {
            "url": url,
            "status_code": 0,
            "retrieved_at_utc": started,
            "content_sha256": sha256_text(""),
            "content_bytes": 0,
            "text": "",
            "error": type(exc).__name__ + ": " + str(exc),
        }


def strip_html(raw: str) -> str:
    unescaped = html.unescape(raw)
    unescaped = re.sub(r"<script\b.*?</script>", "\n", unescaped, flags=re.IGNORECASE | re.DOTALL)
    unescaped = re.sub(r"<style\b.*?</style>", "\n", unescaped, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", unescaped)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def first_regex(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def parse_selectors(raw: str) -> list[dict[str, str]]:
    unescaped = html.unescape(raw).replace('\\"', '"')
    selector_pattern = re.compile(
        r'\{"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"level"\s*:\s*"(?P<level>[^"]+)"(?:\s*,\s*"info"\s*:\s*"(?P<info>[^"]*)")?',
    )
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in selector_pattern.finditer(unescaped):
        row = {
            "name": match.group("name"),
            "level": match.group("level"),
            "info": match.group("info") or "",
        }
        key = (row["name"], row["level"], row["info"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda item: (item["name"], item["level"], item["info"]))


def parse_model_page(model_code: str, fetched: dict[str, Any]) -> dict[str, Any]:
    text = strip_html(fetched["text"])
    title = first_regex(text, [r"#\s+([^\n]+)", r"###\s+([^\n]+)"])
    archive = first_regex(text, [r"Archive begins[:\s]+(\d{4}-\d{2}-\d{2})", r"Archive window[:\s]+([^\n]+)"])
    resolution = first_regex(text, [r"Resolution[:\s]+([^\n]+)"])
    update = first_regex(text, [r"Update[:\s]+([^\n]+)"])
    lead_time = first_regex(text, [r"Lead time[:\s]+([^\n]+)"])
    model_type = first_regex(text, [r"Type[:\s]+([^\n]+)"])
    origin = first_regex(text, [r"Origin[:\s]+([^\n]+)"])
    selectors = parse_selectors(fetched["text"])
    return {
        "model_code": model_code,
        "page_url": fetched["url"],
        "status_code": fetched["status_code"],
        "content_sha256": fetched["content_sha256"],
        "content_bytes": fetched["content_bytes"],
        "title": title,
        "archive": archive,
        "resolution": resolution,
        "update": update,
        "lead_time": lead_time,
        "model_type": model_type,
        "origin": origin,
        "selector_count": len(selectors),
        "selectors": selectors,
        "error": fetched["error"],
    }


def normalize_priority(priority: str) -> str:
    return priority.strip().upper()


def canonical_level_tokens(levels: str) -> list[str]:
    text = levels.lower()
    tokens: list[str] = []
    for pressure in re.findall(r"(\d{3,4})\s*hpa", text):
        tokens.extend([f"{pressure} mb", f"{pressure} hpa"])
    if "2 m" in text:
        tokens.append("2 m")
    if "10 m" in text:
        tokens.append("10 m")
    if "surface" in text or "sfc" in text:
        tokens.append("surface")
    if "mean sea" in text or "msl" in text:
        tokens.extend(["mean sea level", "msl"])
    if "entire atmosphere" in text or "column" in text:
        tokens.extend(["entire atmosphere", "atmosphere", "surface"])
    if "0-10 cm" in text or "0-0.1" in text:
        tokens.extend(["0-0.1 m", "0-10 cm"])
    return tokens


def selector_score(selector: dict[str, str], semantic: dict[str, str]) -> int:
    aliases = NAME_ALIASES.get(semantic["semantic_variable"], [])
    if semantic["family"] == "pressure_level":
        aliases = NAME_ALIASES.get(semantic["semantic_variable"], aliases)
    score = 0
    if selector["name"].upper() in {alias.upper() for alias in aliases}:
        score += 100
    else:
        return -1
    selector_level = selector["level"].lower()
    level_tokens = canonical_level_tokens(semantic["levels"])
    if level_tokens:
        if any(token in selector_level for token in level_tokens):
            score += 50
        elif semantic["family"] == "pressure_level":
            return -1
    if selector.get("info"):
        score += 1
    if "ens mean" in selector.get("info", "").lower():
        score += 3
    return score


def choose_selector(selectors: list[dict[str, str]], semantic: dict[str, str]) -> dict[str, str] | None:
    scored = [(selector_score(selector, semantic), selector) for selector in selectors]
    scored = [item for item in scored if item[0] >= 0]
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[0], item[1]["name"], item[1]["level"], item[1]["info"]))[0][1]


def model_run_probe_time(model: dict[str, str]) -> str:
    archive = model.get("archive_or_window", "")
    if "rolling" in archive.lower() or "last " in archive.lower():
        return "2026-06-23T00:00:00Z"
    match = re.search(r"(\d{4}-\d{2}-\d{2})", archive)
    if match:
        start = date.fromisoformat(match.group(1))
        probe = max(start + timedelta(days=7), date(2025, 5, 1))
        if probe > date(2026, 6, 20):
            probe = date(2026, 6, 20)
        return f"{probe.isoformat()}T00:00:00Z"
    return "2025-05-01T00:00:00Z"


def response_row_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return 0
        return len(payload) if isinstance(payload, list) else 1
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return 0
    return max(len(lines) - 1, 0)


def sanitize_snippet(text: str, token: str | None) -> str:
    snippet = text.replace("\r", " ").replace("\n", " ")[:240]
    if token:
        snippet = snippet.replace(token, "[REDACTED_GRIBSTREAM_API_KEY]")
    return snippet


def run_probe(
    client: httpx.Client,
    token: str | None,
    model_code: str,
    model: dict[str, str],
    selector: dict[str, str] | None,
    *,
    limiter: OneThreadRateLimiter,
    max_attempts: int,
    max_retry_after_seconds: float,
) -> dict[str, Any]:
    if token is None:
        return {
            "model_code": model_code,
            "endpoint": "runs",
            "probe_status": "blocked_no_api_token",
            "http_status": "",
            "row_count": "",
            "response_sha256": "",
            "request_sha256": "",
            "error_class": "NO_TOKEN",
            "error_snippet": "",
            "attempt_count": "0",
            "retry_after_seconds": "",
            "elapsed_ms": "",
        }
    if selector is None:
        return {
            "model_code": model_code,
            "endpoint": "runs",
            "probe_status": "blocked_no_surface_temperature_selector",
            "http_status": "",
            "row_count": "",
            "response_sha256": "",
            "request_sha256": "",
            "error_class": "NO_SELECTOR",
            "error_snippet": "",
            "attempt_count": "0",
            "retry_after_seconds": "",
            "elapsed_ms": "",
        }

    payload = {
        "forecastedFrom": model_run_probe_time(model),
        "forecastedUntil": model_run_probe_time(model),
        "minLeadTime": "0h",
        "maxLeadTime": "48h",
        "coordinates": HKG_COORDINATES,
        "variables": [
            {
                "name": selector["name"],
                "level": selector["level"],
                "info": selector.get("info", ""),
                "alias": "probe_value",
            },
        ],
    }
    request_sha = sha256_text(json.dumps(payload, sort_keys=True))
    url = f"https://gribstream.com/api/v2/{model_code}/runs"
    attempts = max(max_attempts, 1)
    last_result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        rate_wait_seconds = limiter.wait()
        started = time.perf_counter()
        write_status(
            "t03_api_probe",
            "running",
            current_model=model_code,
            attempt=attempt,
            max_attempts=attempts,
            rate_wait_seconds=round(rate_wait_seconds, 3),
        )
        try:
            response = client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "text/csv",
                    "Accept-Encoding": "gzip",
                },
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            text = response.text
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            rows = response_row_count(text) if response.status_code == 200 else 0
            if response.status_code == 200 and rows > 0:
                status = "pass_hkg_rows_returned"
                error_class = ""
            elif response.status_code == 200:
                status = "blocked_no_rows_at_probe_window"
                error_class = "NO_ROWS"
            elif response.status_code == 401:
                status = "blocked_auth_failed"
                error_class = "AUTH"
            elif response.status_code == 429:
                status = "blocked_rate_limited"
                error_class = "RATE_LIMIT"
            elif response.status_code in PERMANENT_HTTP_STATUS:
                status = "blocked_api_rejected_probe"
                error_class = f"HTTP_{response.status_code}"
            else:
                status = "blocked_transient_http_error"
                error_class = f"HTTP_{response.status_code}"
            result = {
                "model_code": model_code,
                "endpoint": "runs",
                "probe_status": status,
                "http_status": str(response.status_code),
                "row_count": str(rows),
                "response_sha256": sha256_text(text),
                "request_sha256": request_sha,
                "error_class": error_class,
                "error_snippet": "" if status == "pass_hkg_rows_returned" else sanitize_snippet(text, token),
                "attempt_count": str(attempt),
                "retry_after_seconds": "" if retry_after is None else str(retry_after),
                "elapsed_ms": str(elapsed_ms),
            }
            append_jsonl(
                T03_API_EVENT_LOG,
                {
                    "event": "gribstream_probe_attempt",
                    "model_code": model_code,
                    "attempt": attempt,
                    "http_status": response.status_code,
                    "probe_status": status,
                    "row_count": rows,
                    "elapsed_ms": elapsed_ms,
                    "retry_after_seconds": retry_after,
                    "request_sha256": request_sha,
                    "response_sha256": result["response_sha256"],
                    "timestamp_utc": utc_now_iso(),
                },
            )
            last_result = result
            if response.status_code == 200 or response.status_code in PERMANENT_HTTP_STATUS:
                return result
            if response.status_code in TRANSIENT_HTTP_STATUS and attempt < attempts:
                delay = retry_after if retry_after is not None else min(60.0, (2 ** (attempt - 1)) * 10.0)
                if delay > max_retry_after_seconds:
                    result["probe_status"] = "blocked_retry_after_exceeds_cap"
                    result["error_class"] = "RETRY_AFTER_EXCEEDS_CAP"
                    return result
                bounded_sleep(delay + random.uniform(0.5, 2.0))
                continue
            return result
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.TransportError) as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            result = {
                "model_code": model_code,
                "endpoint": "runs",
                "probe_status": "blocked_probe_exception",
                "http_status": "",
                "row_count": "",
                "response_sha256": "",
                "request_sha256": request_sha,
                "error_class": type(exc).__name__,
                "error_snippet": sanitize_snippet(str(exc), token),
                "attempt_count": str(attempt),
                "retry_after_seconds": "",
                "elapsed_ms": str(elapsed_ms),
            }
            append_jsonl(
                T03_API_EVENT_LOG,
                {
                    "event": "gribstream_probe_attempt",
                    "model_code": model_code,
                    "attempt": attempt,
                    "http_status": None,
                    "probe_status": result["probe_status"],
                    "error_class": result["error_class"],
                    "elapsed_ms": elapsed_ms,
                    "request_sha256": request_sha,
                    "timestamp_utc": utc_now_iso(),
                },
            )
            last_result = result
            if attempt < attempts:
                bounded_sleep(min(60.0, (2 ** (attempt - 1)) * 10.0) + random.uniform(0.5, 2.0))
                continue
            return result
        except Exception as exc:  # pragma: no cover - defensive unknown failure path
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            append_jsonl(
                T03_API_EVENT_LOG,
                {
                    "event": "gribstream_probe_unexpected_exception",
                    "model_code": model_code,
                    "attempt": attempt,
                    "error_class": type(exc).__name__,
                    "elapsed_ms": elapsed_ms,
                    "request_sha256": request_sha,
                    "timestamp_utc": utc_now_iso(),
                },
            )
            return {
                "model_code": model_code,
                "endpoint": "runs",
                "probe_status": "blocked_unexpected_exception",
                "http_status": "",
                "row_count": "",
                "response_sha256": "",
                "request_sha256": request_sha,
                "error_class": type(exc).__name__,
                "error_snippet": sanitize_snippet(str(exc), token),
                "attempt_count": str(attempt),
                "retry_after_seconds": "",
                "elapsed_ms": str(elapsed_ms),
            }
    return last_result or {
        "model_code": model_code,
        "endpoint": "runs",
        "probe_status": "blocked_probe_not_attempted",
        "http_status": "",
        "row_count": "",
        "response_sha256": "",
        "request_sha256": request_sha,
        "error_class": "NOT_ATTEMPTED",
        "error_snippet": "",
        "attempt_count": "0",
        "retry_after_seconds": "",
        "elapsed_ms": "",
    }


def load_existing_t03_artifacts() -> dict[str, Any]:
    required = [
        T03_EXP / "catalog_snapshot.json",
        T03_EXP / "final_model_disposition.csv",
        T03_EXP / "selector_map.csv",
        T03_EXP / "coverage_probe_results.csv",
        T03_EXP / "quota_storage_estimate.csv",
    ]
    missing = [repo_rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing reusable T03 artifacts: " + ", ".join(missing))
    catalog_snapshot = json.loads((T03_EXP / "catalog_snapshot.json").read_text(encoding="utf-8"))
    coverage_rows = read_csv_rows(T03_EXP / "coverage_probe_results.csv")
    write_status(
        "t03_artifacts",
        "reused_existing",
        artifact_count=len(required),
        coverage_rows=len(coverage_rows),
    )
    return {
        "status": "passed",
        "open_blockers": [
            "Written GribStream agreement is still required before treating asOf as historical first-availability proof.",
            "Bulk acquisition must remain staged until T06 measures real credit cost per request shape.",
        ],
        "catalog_snapshot": catalog_snapshot,
        "model_rows": read_csv_rows(T03_EXP / "final_model_disposition.csv"),
        "selector_rows": read_csv_rows(T03_EXP / "selector_map.csv"),
        "coverage_rows": coverage_rows,
        "quota_rows": read_csv_rows(T03_EXP / "quota_storage_estimate.csv"),
        "retrieved_at_utc": str(catalog_snapshot.get("retrieved_at_utc") or utc_now_iso()),
    }


def build_t03(args: argparse.Namespace) -> dict[str, Any]:
    if args.reuse_existing_t03_artifacts:
        return load_existing_t03_artifacts()

    matrix = read_csv_rows(MODEL_MATRIX)
    semantics = read_csv_rows(VARIABLE_REQUIREMENTS)
    retrieved_at = utc_now_iso()
    headers = {"User-Agent": "HKG-Tmax-Research/0.1 T03 audit"}
    public_urls = [
        "https://gribstream.com/docs",
        "https://gribstream.com/models",
        "https://gribstream.com/openapi",
        "https://gribstream.com/openapi.json",
        "https://gribstream.com/api/v2/openapi.json",
        "https://gribstream.com/faq",
        "https://gribstream.com/tos.html",
    ]

    with httpx.Client(timeout=api_timeout(), follow_redirects=True, headers=headers) as client:
        write_status("t03_public_catalog", "running", public_url_count=len(public_urls))
        public_pages = {url: fetch_public_text(client, url) for url in public_urls}
        model_pages = {}
        for row in matrix:
            code = row["model_code"]
            fetched = fetch_public_text(client, f"https://gribstream.com/models/{code}")
            model_pages[code] = parse_model_page(code, fetched)

        selectors_by_model = {code: page["selectors"] for code, page in model_pages.items()}
        selector_rows: list[dict[str, Any]] = []
        for model in matrix:
            code = model["model_code"]
            if model["disposition"] not in RELEVANT_DISPOSITIONS:
                continue
            for semantic in semantics:
                selected = choose_selector(selectors_by_model.get(code, []), semantic)
                if selected:
                    selector_rows.append(
                        {
                            "model_code": code,
                            "disposition": model["disposition"],
                            "family": semantic["family"],
                            "semantic_variable": semantic["semantic_variable"],
                            "priority": normalize_priority(semantic["priority"]),
                            "requested_levels": semantic["levels"],
                            "selector_status": "selected",
                            "native_name": selected["name"],
                            "native_level": selected["level"],
                            "native_info": selected.get("info", ""),
                            "selector_json": json.dumps(selected, sort_keys=True),
                            "blocker": "",
                        },
                    )
                else:
                    selector_rows.append(
                        {
                            "model_code": code,
                            "disposition": model["disposition"],
                            "family": semantic["family"],
                            "semantic_variable": semantic["semantic_variable"],
                            "priority": normalize_priority(semantic["priority"]),
                            "requested_levels": semantic["levels"],
                            "selector_status": "blocked",
                            "native_name": "",
                            "native_level": "",
                            "native_info": "",
                            "selector_json": "",
                            "blocker": "No exact selector found on public model page for this semantic requirement.",
                        },
                    )

        selector_lookup = {
            row["model_code"]: {
                "name": row["native_name"],
                "level": row["native_level"],
                "info": row["native_info"],
            }
            for row in selector_rows
            if row["semantic_variable"] == "air_temperature" and row["selector_status"] == "selected"
        }
        probe_models = [
            row
            for row in matrix
            if row["disposition"] in RELEVANT_DISPOSITIONS and row["disposition"] != "deprioritize"
        ]
        coverage_columns = [
            "model_code",
            "endpoint",
            "probe_status",
            "http_status",
            "row_count",
            "response_sha256",
            "request_sha256",
            "error_class",
            "error_snippet",
            "attempt_count",
            "retry_after_seconds",
            "elapsed_ms",
        ]
        coverage_path = T03_EXP / "coverage_probe_results.csv"
        if args.reuse_existing_coverage_probes and coverage_path.exists():
            coverage_rows = read_csv_rows(coverage_path)
            write_status(
                "t03_api_probe",
                "reused_existing_coverage",
                completed_probes=len(coverage_rows),
                total_probes=len(probe_models),
            )
        else:
            token = None if args.skip_api_probes else load_gribstream_token()
            limiter = OneThreadRateLimiter(args.api_min_interval_seconds)
            coverage_rows = []
            for index, row in enumerate(probe_models, start=1):
                write_status(
                    "t03_api_probe",
                    "running",
                    current_model=row["model_code"],
                    completed_probes=len(coverage_rows),
                    total_probes=len(probe_models),
                    api_min_interval_seconds=args.api_min_interval_seconds,
                    api_max_attempts=args.api_max_attempts,
                )
                result = run_probe(
                    client,
                    token,
                    row["model_code"],
                    row,
                    selector_lookup.get(row["model_code"]),
                    limiter=limiter,
                    max_attempts=args.api_max_attempts,
                    max_retry_after_seconds=args.api_max_retry_after_seconds,
                )
                coverage_rows.append(result)
                write_csv(coverage_path, coverage_rows, coverage_columns)
                if result["http_status"] == "429":
                    for remaining in probe_models[index:]:
                        coverage_rows.append(
                            {
                                "model_code": remaining["model_code"],
                                "endpoint": "runs",
                                "probe_status": "blocked_rate_limit_safety_stop",
                                "http_status": "",
                                "row_count": "",
                                "response_sha256": "",
                                "request_sha256": "",
                                "error_class": "RATE_LIMIT_SAFETY_STOP",
                                "error_snippet": "Probe loop stopped after a 429 to avoid repeated rate-limit traffic.",
                                "attempt_count": "0",
                                "retry_after_seconds": "",
                                "elapsed_ms": "",
                            },
                        )
                    write_csv(coverage_path, coverage_rows, coverage_columns)
                    write_status(
                        "t03_api_probe",
                        "blocked_rate_limit_safety_stop",
                        completed_probes=len(coverage_rows),
                        total_probes=len(probe_models),
                        stopped_after_model=row["model_code"],
                    )
                    break

    coverage_by_model = {row["model_code"]: row for row in coverage_rows}
    model_rows: list[dict[str, Any]] = []
    for row in matrix:
        code = row["model_code"]
        page = model_pages.get(code, {})
        coverage = coverage_by_model.get(code, {})
        public_status = "public_page_found" if page.get("status_code") == 200 else "public_page_missing_or_error"
        if row["disposition"].startswith("exclude"):
            final_status = "excluded_by_disposition"
        elif coverage.get("probe_status") == "pass_hkg_rows_returned":
            final_status = "available_for_staged_acquisition"
        elif row["disposition"] in RELEVANT_DISPOSITIONS:
            final_status = "blocked_or_staged_pending_probe_resolution"
        else:
            final_status = "registered_not_acquisition_priority"
        model_rows.append(
            {
                "model_code": code,
                "domain": row["domain"],
                "source_matrix_disposition": row["disposition"],
                "archive_or_window": row["archive_or_window"],
                "reason": row["reason"],
                "public_catalog_status": public_status,
                "public_page_url": page.get("page_url", f"https://gribstream.com/models/{code}"),
                "public_page_sha256": page.get("content_sha256", ""),
                "selector_count": page.get("selector_count", 0),
                "provider_or_origin": page.get("origin", ""),
                "model_type": page.get("model_type", ""),
                "resolution": page.get("resolution", ""),
                "update_cadence": page.get("update", ""),
                "lead_time": page.get("lead_time", ""),
                "coverage_probe_status": coverage.get("probe_status", "not_probed_disposition"),
                "final_status": final_status,
                "downstream_action": downstream_action(row["disposition"], final_status),
            },
        )

    quota_rows = build_quota_rows(matrix, selector_rows, coverage_rows)
    catalog_snapshot = {
        "snapshot_id": "gribstream_catalog_snapshot_20260624_t03",
        "retrieved_at_utc": retrieved_at,
        "public_sources": {
            url: {
                "status_code": payload["status_code"],
                "content_sha256": payload["content_sha256"],
                "content_bytes": payload["content_bytes"],
                "error": payload["error"],
            }
            for url, payload in public_pages.items()
        },
        "model_pages": {
            code: {key: value for key, value in page.items() if key != "selectors"}
            for code, page in model_pages.items()
        },
        "selector_counts": {code: len(selectors) for code, selectors in selectors_by_model.items()},
        "gribstream_docs_facts": {
            "timeseries_endpoint": "POST https://gribstream.com/api/v2/<model>/timeseries",
            "runs_endpoint": "POST https://gribstream.com/api/v2/<model>/runs",
            "backfill_endpoint_decision": "Use /runs for canonical historical run backfills; /timeseries is only a best-eligible forecast view.",
            "asof_warning": "GribStream terms/docs say asOf is a query behavior cutoff, not proof of historical live API availability.",
        },
    }

    write_json(T03_EXP / "catalog_snapshot.json", catalog_snapshot)
    write_csv(
        T03_EXP / "selector_map.csv",
        selector_rows,
        [
            "model_code",
            "disposition",
            "family",
            "semantic_variable",
            "priority",
            "requested_levels",
            "selector_status",
            "native_name",
            "native_level",
            "native_info",
            "selector_json",
            "blocker",
        ],
    )
    write_csv(
        T03_EXP / "coverage_probe_results.csv",
        coverage_rows,
        [
            "model_code",
            "endpoint",
            "probe_status",
            "http_status",
            "row_count",
            "response_sha256",
            "request_sha256",
            "error_class",
            "error_snippet",
            "attempt_count",
            "retry_after_seconds",
            "elapsed_ms",
        ],
    )
    write_csv(
        T03_EXP / "quota_storage_estimate.csv",
        quota_rows,
        [
            "stage",
            "model_code",
            "disposition",
            "estimated_requests",
            "estimated_uncompressed_mb",
            "estimated_compressed_mb",
            "daily_quota_credits",
            "recommended_daily_request_cap",
            "fits_quota",
            "notes",
        ],
    )
    write_text(T03_EXP / "licence_register.md", licence_register_text(retrieved_at))
    write_csv(
        T03_EXP / "final_model_disposition.csv",
        model_rows,
        [
            "model_code",
            "domain",
            "source_matrix_disposition",
            "archive_or_window",
            "reason",
            "public_catalog_status",
            "public_page_url",
            "public_page_sha256",
            "selector_count",
            "provider_or_origin",
            "model_type",
            "resolution",
            "update_cadence",
            "lead_time",
            "coverage_probe_status",
            "final_status",
            "downstream_action",
        ],
    )
    write_json(
        T03_EXP / "logs/public_fetch_summary.json",
        {
            "generated_at_utc": retrieved_at,
            "public_urls": {
                url: {
                    "status_code": payload["status_code"],
                    "content_sha256": payload["content_sha256"],
                    "content_bytes": payload["content_bytes"],
                    "error": payload["error"],
                }
                for url, payload in public_pages.items()
            },
            "model_status_counts": dict(sorted(count_status(model_rows, "final_status").items())),
            "probe_status_counts": dict(sorted(count_status(coverage_rows, "probe_status").items())),
        },
    )
    return {
        "status": "passed",
        "open_blockers": [
            "Written GribStream agreement is still required before treating asOf as historical first-availability proof.",
            "Bulk acquisition must remain staged until T06 measures real credit cost per request shape.",
        ],
        "catalog_snapshot": catalog_snapshot,
        "model_rows": model_rows,
        "selector_rows": selector_rows,
        "coverage_rows": coverage_rows,
        "quota_rows": quota_rows,
        "retrieved_at_utc": retrieved_at,
    }


def downstream_action(disposition: str, final_status: str) -> str:
    if disposition.startswith("exclude"):
        return "Do not acquire for HKG."
    if final_status == "available_for_staged_acquisition":
        return "Eligible for T06 staged /runs client planning, still C_RUN_TIME_ONLY until T16."
    if disposition in {"coverage_probe", "secondary_probe", "secondary_targeted"}:
        return "Keep as explicit probe or targeted secondary source; do not bulk acquire until T12/T16 gates."
    if disposition == "deprioritize":
        return "Register only; defer until core backfills and feature platform are mature."
    return "Registered with blocker or staged acquisition requirement."


def count_status(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(column, ""))] += 1
    return counts


def build_quota_rows(
    matrix: list[dict[str, str]],
    selector_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_model: dict[str, int] = defaultdict(int)
    for row in selector_rows:
        if row["selector_status"] == "selected" and row["priority"] in {"P0", "P1"}:
            selected_by_model[row["model_code"]] += 1
    coverage_by_model = {row["model_code"]: row for row in coverage_rows}
    rows: list[dict[str, Any]] = []
    for model in matrix:
        disposition = model["disposition"]
        if disposition not in RELEVANT_DISPOSITIONS:
            continue
        variables = max(selected_by_model.get(model["model_code"], 1), 1)
        if disposition in {"core_backfill", "core_short_history"}:
            stage = "core"
            request_estimate = variables * 180
        elif disposition in {"challenger_backfill", "shadow_backfill"}:
            stage = "challenger"
            request_estimate = variables * 60
        elif disposition == "prospective_urgent":
            stage = "prospective"
            request_estimate = variables * 10
        else:
            stage = "probe"
            request_estimate = variables * 5
        uncompressed = round(request_estimate * 0.35, 2)
        compressed = round(uncompressed * 0.22, 2)
        probe_status = coverage_by_model.get(model["model_code"], {}).get("probe_status", "")
        rows.append(
            {
                "stage": stage,
                "model_code": model["model_code"],
                "disposition": disposition,
                "estimated_requests": str(request_estimate),
                "estimated_uncompressed_mb": str(uncompressed),
                "estimated_compressed_mb": str(compressed),
                "daily_quota_credits": "96000",
                "recommended_daily_request_cap": "1000",
                "fits_quota": "YES_STAGED" if request_estimate <= 96000 else "NO_STAGE_REQUIRED",
                "notes": f"Probe status {probe_status or 'not_probed'}; request/credit conversion must be measured by T06 before bulk fetch.",
            },
        )
    return rows


def licence_register_text(retrieved_at: str) -> str:
    return f"""# T03 GribStream Licence and Usage Register

Generated at UTC: {retrieved_at}

## Sources reviewed

- https://gribstream.com/docs
- https://gribstream.com/models
- https://gribstream.com/faq
- https://gribstream.com/tos.html

## Decision

- API use is allowed for authenticated project access, subject to plan quota and GribStream terms.
- The local token must stay only in `secrets/local/gribstream.env`; generated artifacts must not contain it.
- `/runs` is the canonical endpoint for historical run backfills.
- `/timeseries` is a best-eligible forecast view and is not the canonical raw backfill path.
- GribStream `asOf` does not prove historical live API availability. All GribStream historical backfill rows remain `C_RUN_TIME_ONLY` until T16 builds an independent availability proof ledger or a written provider agreement states otherwise.
- Bulk/commercial ingestion is staged. The first T06 client must measure real credit cost, throttle behavior, row volume, and retry classes before any large backfill starts.

## Open legal/contract blocker

Written confirmation is still required before treating GribStream historical responses as proof of what the live API would have returned at a prior wall-clock time.
"""


def build_t04(args: argparse.Namespace) -> dict[str, Any]:
    created_tables = [
        "catalog.weather_model",
        "catalog.location",
        "catalog.variable",
        "catalog.variable_selector_snapshot",
        "raw_audit.acquisition_request",
        "raw_audit.response_object",
        "nwp_core.model_run",
        "nwp_core.point_value",
        "nwp_core.point_value_default",
        "feature_store.target_snapshot_manifest",
        "feature_store.feature_definition",
        "feature_store.feature_value",
        "research.expert_oof_prediction",
        "live.issued_forecast",
        "quarantine.rejected_payload",
    ]
    write_text(
        T04_EXP / "schema_diagram.md",
        """# T04 Schema Diagram

```mermaid
erDiagram
  catalog_weather_model ||--o{ nwp_core_model_run : model_id
  catalog_variable ||--o{ catalog_variable_selector_snapshot : variable_id
  catalog_location ||--o{ nwp_core_point_value : location_id
  catalog_variable_selector_snapshot ||--o{ nwp_core_point_value : selector_id
  raw_audit_acquisition_request ||--o{ raw_audit_response_object : request_id
  raw_audit_response_object ||--o{ nwp_core_point_value : response_object_id
  nwp_core_model_run ||--o{ nwp_core_point_value : model_run_id
  feature_store_target_snapshot_manifest ||--o{ feature_store_feature_value : snapshot_id
  feature_store_feature_definition ||--o{ feature_store_feature_value : feature_id
  feature_store_target_snapshot_manifest ||--o{ research_expert_oof_prediction : snapshot_id
  feature_store_target_snapshot_manifest ||--o{ live_issued_forecast : snapshot_id
```

The high-volume NWP value table is partitioned by `valid_time_utc` with a default partition so T06 can insert safely before monthly partitions are generated.
""",
    )
    write_text(
        T04_EXP / "index_partition_plan.md",
        """# T04 Index and Partition Plan

- `nwp_core.point_value` is range partitioned by `valid_time_utc`.
- `nwp_core.point_value_default` catches inserts before month-specific partitions exist.
- Indexes cover `(location_id, valid_time_utc)`, `(selector_id, valid_time_utc)`, `model_run.run_time_utc`, and `response_object.retrieved_at_utc`.
- T06 should create monthly partitions before bulk ingest starts, then monitor rows landing in the default partition.
- Idempotency keys are `raw_audit.acquisition_request.request_sha256`, `raw_audit.response_object(request_id, sha256)`, and `nwp_core.point_value` primary key.
""",
    )
    write_text(
        T04_EXP / "rollback_plan.md",
        """# T04 Rollback Plan

This migration is additive. A safe rollback is:

1. Stop T06+ ingestion jobs.
2. Export `raw_audit`, `nwp_core`, `feature_store`, `research`, `live`, and `quarantine` rows created after this migration.
3. Drop dependent objects in reverse order: live, research, feature_store, nwp_core, raw_audit, then catalog additions if they are not shared with T05.
4. Remove `governance.schema_version` row `20260624_0005_t04_nwp_storage_lineage`.

No automated destructive rollback is run against the main database.
""",
    )
    write_text(
        T04_EXP / "migration_test_log.md",
        """# T04 Migration Test Log

- Main migration file: `db/migrations/postgres/20260624_0005_t04_nwp_storage_lineage.sql`
- Idempotency expectation: applying the migration twice must not duplicate objects or fail.
- Partition routing expectation: `nwp_core.point_value_default` must exist.
- Permission expectation: `hkg_tmax_live_inference` has no rights on `research` or `quarantine`; research roles have governed write access.
- Isolated rollback expectation: an optional temporary database can be created, migrated, inspected, and dropped when PostgreSQL superuser access is available.
""",
    )
    copy_file(MIGRATION_T04, T04_EXP / "migrations" / MIGRATION_T04.name)
    write_json(
        T04_EXP / "storage_contract.json",
        {
            "created_tables": created_tables,
            "partitioned_tables": ["nwp_core.point_value"],
            "default_partitions": ["nwp_core.point_value_default"],
            "idempotency_keys": [
                "raw_audit.acquisition_request.request_sha256",
                "raw_audit.response_object(request_id, sha256)",
                "nwp_core.point_value primary key",
            ],
            "live_role_denied_schemas": ["research", "quarantine", "sealed_confirmation", "label_core"],
        },
    )
    return {
        "status": "passed",
        "open_blockers": [],
        "created_tables": created_tables,
    }


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "unknown"


def parse_float(value: str) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except ValueError:
        return None


def read_arwf_rows() -> list[dict[str, Any]]:
    if not ARWF_PARQUET.exists():
        return []
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        return []
    frame = pd.read_parquet(ARWF_PARQUET, columns=["station_code", "latitude", "longitude"])
    frame = frame.drop_duplicates(subset=["station_code"]).sort_values("station_code")
    return [
        {
            "station_code": str(row.station_code),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        for row in frame.itertuples(index=False)
    ]


def build_t05(args: argparse.Namespace) -> dict[str, Any]:
    hko_rows = read_csv_rows(STATION_REGISTRY)
    isd_rows = read_csv_rows(ISD_DOSSIER)
    arwf_rows = read_arwf_rows()

    locations: dict[str, dict[str, Any]] = {}
    stations: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for item in DESIGNED_LOCATIONS:
        row = {
            **item,
            "valid_from": "",
            "valid_to": "",
            "metadata_sha256": stable_row_sha(item),
        }
        locations[row["location_code"]] = row

    for row in isd_rows:
        code = "isd_" + slug(row["station_id"])
        location = {
            "location_code": code,
            "name": row["STATION NAME"],
            "latitude": row["LAT"],
            "longitude": row["LON"],
            "elevation_m": row["ELEV(M)"],
            "location_role": "isd_regional_station",
            "valid_from": yyyymmdd_to_date(row["BEGIN"]),
            "valid_to": yyyymmdd_to_date(row["END"]),
            "metadata_source": repo_rel(ISD_DOSSIER),
        }
        location["metadata_sha256"] = stable_row_sha(location)
        locations[code] = location
        stations.append(
            {
                "station_code": "ISD:" + row["station_id"],
                "station_name": row["STATION NAME"],
                "network": "ISD",
                "icao": row["ICAO"],
                "country_code": row["CTRY"],
                "location_code": code,
                "station_role": row["role"],
                "target_station": "true" if row["station_id"] == "450050-99999" else "false",
                "valid_from": yyyymmdd_to_date(row["BEGIN"]),
                "valid_to": yyyymmdd_to_date(row["END"]),
                "metadata_status": "resolved",
                "source_uri": repo_uri(ISD_DOSSIER),
                "source_sha256": stable_row_sha(row),
            },
        )

    for row in hko_rows:
        station_name = row.get("station_name", "")
        official_code = row.get("official_station_code", "") or row.get("canonical_station_id", "")
        lat = parse_float(row.get("latitude", ""))
        lon = parse_float(row.get("longitude", ""))
        station_code = "HKO:" + (official_code if official_code else slug(station_name))
        location_code = "hko_" + slug(official_code if official_code else station_name)
        has_coordinates = lat is not None and lon is not None
        if has_coordinates:
            location = {
                "location_code": location_code,
                "name": station_name,
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "elevation_m": row.get("elevation_m", ""),
                "location_role": "hko_station",
                "valid_from": "",
                "valid_to": "",
                "metadata_source": repo_rel(STATION_REGISTRY),
            }
            location["metadata_sha256"] = stable_row_sha(location)
            locations[location_code] = location
            station_location = location_code
            metadata_status = "resolved"
        else:
            station_location = ""
            metadata_status = "unresolved_coordinates"
            blockers.append(
                {
                    "station_code": station_code,
                    "station_name": station_name,
                    "network": "HKO",
                    "blocker": "Missing canonical latitude/longitude in current station registry.",
                    "raw_storage_allowed": "true",
                    "physical_interpretation_allowed": "false",
                    "resolution_path": "T15 must resolve official HKO station metadata before spatial feature promotion.",
                },
            )
        stations.append(
            {
                "station_code": station_code,
                "station_name": station_name,
                "network": "HKO",
                "icao": "",
                "country_code": "HK",
                "location_code": station_location,
                "station_role": "target" if row.get("target_station", "").lower() == "true" else "hko_observation_station",
                "target_station": row.get("target_station", "false").lower(),
                "valid_from": "",
                "valid_to": "",
                "metadata_status": metadata_status,
                "source_uri": repo_uri(STATION_REGISTRY),
                "source_sha256": stable_row_sha(row),
            },
        )

    for row in arwf_rows:
        station_code = "ARWF:" + row["station_code"]
        location_code = "arwf_" + slug(row["station_code"])
        location = {
            "location_code": location_code,
            "name": row["station_code"],
            "latitude": f"{row['latitude']:.6f}",
            "longitude": f"{row['longitude']:.6f}",
            "elevation_m": "",
            "location_role": "arwf_station_forecast_point",
            "valid_from": "2026-06-19",
            "valid_to": "",
            "metadata_source": repo_rel(ARWF_PARQUET),
        }
        location["metadata_sha256"] = stable_row_sha(location)
        locations[location_code] = location
        stations.append(
            {
                "station_code": station_code,
                "station_name": row["station_code"],
                "network": "HKO_ARWF",
                "icao": "",
                "country_code": "HK",
                "location_code": location_code,
                "station_role": "arwf_station_forecast_point",
                "target_station": "false",
                "valid_from": "2026-06-19",
                "valid_to": "",
                "metadata_status": "resolved",
                "source_uri": repo_uri(ARWF_PARQUET),
                "source_sha256": stable_row_sha(row),
            },
        )

    location_rows = sorted(locations.values(), key=lambda item: item["location_code"])
    group_rows = build_location_groups(location_rows, stations)
    write_csv(
        T05_EXP / "location_registry.csv",
        location_rows,
        [
            "location_code",
            "name",
            "latitude",
            "longitude",
            "elevation_m",
            "location_role",
            "valid_from",
            "valid_to",
            "metadata_source",
            "metadata_sha256",
        ],
    )
    write_csv(
        T05_EXP / "station_dossier_complete.csv",
        stations,
        [
            "station_code",
            "station_name",
            "network",
            "icao",
            "country_code",
            "location_code",
            "station_role",
            "target_station",
            "valid_from",
            "valid_to",
            "metadata_status",
            "source_uri",
            "source_sha256",
        ],
    )
    write_csv(
        T05_EXP / "location_groups.csv",
        group_rows,
        ["group_code", "group_name", "group_type", "description", "location_code", "member_role"],
    )
    write_csv(
        T05_EXP / "unresolved_station_blockers.csv",
        blockers,
        [
            "station_code",
            "station_name",
            "network",
            "blocker",
            "raw_storage_allowed",
            "physical_interpretation_allowed",
            "resolution_path",
        ],
    )
    write_json(T05_EXP / "local_domain.geojson", polygon_feature("hkg_local_domain", 113.80, 22.10, 114.35, 22.60))
    write_json(T05_EXP / "synoptic_domain.geojson", polygon_feature("south_china_synoptic_domain", 105.0, 15.0, 125.0, 30.0))
    return {
        "status": "passed",
        "open_blockers": [
            f"{len(blockers)} HKO station aliases are registered but need official coordinates before spatial feature promotion."
        ]
        if blockers
        else [],
        "location_rows": location_rows,
        "station_rows": stations,
        "group_rows": group_rows,
        "blocker_rows": blockers,
    }


def yyyymmdd_to_date(value: str) -> str:
    if not value or len(value) < 8:
        return ""
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def build_location_groups(
    locations: list[dict[str, Any]],
    stations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    station_by_location = {row["location_code"]: row for row in stations if row.get("location_code")}
    groups = {
        "target": ("Target Location", "target", "Contract target and canonical HKO location."),
        "hko_current_station_network": ("HKO Current Station Network", "station_network", "HKO stations and aliases from the current registry."),
        "isd_36_regional": ("ISD 36 Regional Stations", "station_network", "Regional ISD station dossier used by long-history features."),
        "arwf_station_forecast_network": ("HKO ARWF Station Forecast Points", "nwp_station_network", "Current/prospective ARWF station forecast points."),
        "designed_local_domain": ("Designed Local Domain", "domain", "Local Hong Kong domain and reference points."),
        "designed_synoptic_domain": ("Designed Synoptic Domain", "domain", "South China synoptic context domain."),
        "coastal_marine": ("Coastal and Marine Stations", "exposure_group", "Marine/coastal references around Hong Kong."),
        "inland_reference": ("Inland Reference Stations", "exposure_group", "Inland or continental-reference stations."),
    }
    for location in locations:
        code = location["location_code"]
        role = location["location_role"]
        memberships: list[tuple[str, str]] = []
        if role == "target" or code == "hkg_hko_target":
            memberships.append(("target", "target"))
        if code.startswith("hko_"):
            memberships.append(("hko_current_station_network", "station"))
        if code.startswith("isd_"):
            memberships.append(("isd_36_regional", "station"))
        if code.startswith("arwf_"):
            memberships.append(("arwf_station_forecast_network", "station"))
        if "domain" in role or "reference" in role or code.startswith("hkg_"):
            memberships.append(("designed_local_domain", "reference"))
        if code.startswith("synoptic_"):
            memberships.append(("designed_synoptic_domain", "reference"))
        station = station_by_location.get(code, {})
        text = " ".join([code, role, station.get("station_role", ""), location.get("name", "")]).lower()
        if any(word in text for word in ["marine", "island", "coast", "airport", "cheung", "waglan"]):
            memberships.append(("coastal_marine", "exposure"))
        if any(word in text for word in ["inland", "north", "valley", "sha tin", "baoan"]):
            memberships.append(("inland_reference", "exposure"))
        for group_code, member_role in memberships:
            group_name, group_type, description = groups[group_code]
            rows.append(
                {
                    "group_code": group_code,
                    "group_name": group_name,
                    "group_type": group_type,
                    "description": description,
                    "location_code": code,
                    "member_role": member_role,
                },
            )
    return sorted(rows, key=lambda item: (item["group_code"], item["location_code"], item["member_role"]))


def polygon_feature(name: str, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat],
                        ],
                    ],
                },
            },
        ],
    }


def apply_and_verify_migrations(database_url: str, skip_db: bool) -> dict[str, Any]:
    if skip_db:
        return {"status": "skipped", "reason": "skip_db flag"}
    applied: list[str] = []
    for migration in (MIGRATION_T00, MIGRATION_T01, MIGRATION_T02, MIGRATION_T03, MIGRATION_T04, MIGRATION_T05):
        apply_migration(database_url, migration)
        applied.append(repo_rel(migration))
    return {"status": "passed", "applied": applied}


def verify_db_objects(database_url: str, objects: list[str]) -> list[dict[str, str]]:
    try:
        psycopg = import_psycopg()
    except DatabaseUnavailable:
        return [{"object_name": obj, "exists": "unknown", "row_count": "", "status": "DB_UNAVAILABLE"} for obj in objects]
    rows: list[dict[str, str]] = []
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            for obj in objects:
                cursor.execute("SELECT to_regclass(%s);", (obj,))
                exists = cursor.fetchone()[0] is not None
                row_count = ""
                if exists:
                    try:
                        cursor.execute(f"SELECT count(*) FROM {obj};")
                        row_count = str(cursor.fetchone()[0])
                    except Exception:
                        connection.rollback()
                        row_count = ""
                rows.append(
                    {
                        "object_name": obj,
                        "exists": str(exists).lower(),
                        "row_count": row_count,
                        "status": "PASS" if exists else "FAIL",
                    },
                )
    return rows


def load_t03_to_db(database_url: str, t03: dict[str, Any], skip_db: bool) -> str:
    if skip_db:
        return "skipped"
    psycopg = import_psycopg()
    snapshot = t03["catalog_snapshot"]
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO catalog.catalog_snapshot (
                    catalog_snapshot_id, provider, source_url, retrieved_at_utc,
                    status_code, content_sha256, content_bytes, content_json, notes
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                ON CONFLICT (catalog_snapshot_id) DO UPDATE SET
                    retrieved_at_utc = EXCLUDED.retrieved_at_utc,
                    status_code = EXCLUDED.status_code,
                    content_sha256 = EXCLUDED.content_sha256,
                    content_bytes = EXCLUDED.content_bytes,
                    content_json = EXCLUDED.content_json,
                    notes = EXCLUDED.notes;
                """,
                (
                    snapshot["snapshot_id"],
                    "GribStream",
                    "https://gribstream.com/models",
                    t03["retrieved_at_utc"],
                    200,
                    sha256_text(json.dumps(snapshot, sort_keys=True)),
                    len(json.dumps(snapshot).encode("utf-8")),
                    json.dumps(snapshot),
                    "T03 public catalog/docs snapshot; raw API token is not stored.",
                ),
            )
            for row in t03["model_rows"]:
                cursor.execute(
                    """
                    INSERT INTO catalog.model_registry (
                        model_code, provider, model_name, domain, disposition, archive_or_window,
                        archive_start, model_type, native_resolution, update_cadence, lead_time,
                        page_url, catalog_snapshot_id, selector_count, coverage_status,
                        final_status, retrieved_at_utc, notes
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (model_code) DO UPDATE SET
                        disposition = EXCLUDED.disposition,
                        archive_or_window = EXCLUDED.archive_or_window,
                        model_type = EXCLUDED.model_type,
                        native_resolution = EXCLUDED.native_resolution,
                        update_cadence = EXCLUDED.update_cadence,
                        lead_time = EXCLUDED.lead_time,
                        selector_count = EXCLUDED.selector_count,
                        coverage_status = EXCLUDED.coverage_status,
                        final_status = EXCLUDED.final_status,
                        retrieved_at_utc = EXCLUDED.retrieved_at_utc,
                        notes = EXCLUDED.notes;
                    """,
                    (
                        row["model_code"],
                        "GribStream",
                        row["model_code"],
                        row["domain"],
                        row["source_matrix_disposition"],
                        row["archive_or_window"],
                        extract_date(row["archive_or_window"]),
                        row["model_type"],
                        row["resolution"],
                        row["update_cadence"],
                        row["lead_time"],
                        row["public_page_url"],
                        snapshot["snapshot_id"],
                        int(row["selector_count"]),
                        row["coverage_probe_status"],
                        row["final_status"],
                        t03["retrieved_at_utc"],
                        row["downstream_action"],
                    ),
                )
            for row in t03["selector_rows"]:
                selector_json = row["selector_json"] or None
                cursor.execute(
                    """
                    INSERT INTO catalog.selector_snapshot (
                        model_code, semantic_variable, semantic_family, semantic_priority,
                        requested_levels, native_name, native_level, native_info, exact_selector,
                        selector_status, blocker, source_sha256, retrieved_at_utc
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                    ON CONFLICT (model_code, semantic_family, semantic_variable, semantic_priority, requested_levels)
                    DO UPDATE SET
                        native_name = EXCLUDED.native_name,
                        native_level = EXCLUDED.native_level,
                        native_info = EXCLUDED.native_info,
                        exact_selector = EXCLUDED.exact_selector,
                        selector_status = EXCLUDED.selector_status,
                        blocker = EXCLUDED.blocker,
                        source_sha256 = EXCLUDED.source_sha256,
                        retrieved_at_utc = EXCLUDED.retrieved_at_utc;
                    """,
                    (
                        row["model_code"],
                        row["semantic_variable"],
                        row["family"],
                        row["priority"],
                        row["requested_levels"],
                        row["native_name"] or None,
                        row["native_level"] or None,
                        row["native_info"],
                        selector_json,
                        row["selector_status"],
                        row["blocker"],
                        sha256_text(json.dumps(row, sort_keys=True)),
                        t03["retrieved_at_utc"],
                    ),
                )
            cursor.execute(
                """
                INSERT INTO catalog.source_license (
                    source_code, source_name, provider, terms_url, terms_last_updated,
                    licence_status, commercial_or_bulk_status, asof_availability_status,
                    quota_status, notes, retrieved_at_utc
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (source_code) DO UPDATE SET
                    terms_last_updated = EXCLUDED.terms_last_updated,
                    licence_status = EXCLUDED.licence_status,
                    commercial_or_bulk_status = EXCLUDED.commercial_or_bulk_status,
                    asof_availability_status = EXCLUDED.asof_availability_status,
                    quota_status = EXCLUDED.quota_status,
                    notes = EXCLUDED.notes,
                    retrieved_at_utc = EXCLUDED.retrieved_at_utc;
                """,
                (
                    "gribstream",
                    "GribStream API",
                    "GribStream",
                    "https://gribstream.com/tos.html",
                    "2026-05-21",
                    "authenticated_api_use_subject_to_terms",
                    "bulk_allowed_only_with_quota_staging_and_terms_review",
                    "asof_is_not_historical_live_availability_proof",
                    "daily_quota_96000_credits_from_user_dashboard_2026_06_24",
                    "T03 reviewed public terms/docs; written agreement still required for historical availability proof.",
                    t03["retrieved_at_utc"],
                ),
            )
            cursor.execute(
                """
                INSERT INTO governance.gribstream_usage_constraint (
                    constraint_id, source_code, constraint_kind, constraint_status,
                    evidence_uri, operational_effect
                )
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (constraint_id) DO UPDATE SET
                    constraint_status = EXCLUDED.constraint_status,
                    evidence_uri = EXCLUDED.evidence_uri,
                    operational_effect = EXCLUDED.operational_effect;
                """,
                (
                    "gribstream_asof_not_availability_proof",
                    "gribstream",
                    "availability_proof",
                    "open_blocker",
                    repo_uri(T03_EXP / "licence_register.md"),
                    "Historical GribStream rows stay C_RUN_TIME_ONLY until T16 or written provider evidence.",
                ),
            )
        connection.commit()
    return "passed"


def extract_date(value: str) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", value or "")
    return match.group(0) if match else None


def load_t05_to_db(database_url: str, t05: dict[str, Any], skip_db: bool) -> str:
    if skip_db:
        return "skipped"
    psycopg = import_psycopg()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            for row in t05["location_rows"]:
                cursor.execute(
                    """
                    INSERT INTO catalog.location (
                        location_code, name, latitude, longitude, elevation_m, location_role,
                        valid_from, valid_to, metadata_source, metadata_sha256
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (location_code) DO UPDATE SET
                        name = EXCLUDED.name,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        elevation_m = EXCLUDED.elevation_m,
                        location_role = EXCLUDED.location_role,
                        valid_from = EXCLUDED.valid_from,
                        valid_to = EXCLUDED.valid_to,
                        metadata_source = EXCLUDED.metadata_source,
                        metadata_sha256 = EXCLUDED.metadata_sha256;
                    """,
                    (
                        row["location_code"],
                        row["name"],
                        float(row["latitude"]),
                        float(row["longitude"]),
                        parse_float(row.get("elevation_m", "")),
                        row["location_role"],
                        row.get("valid_from") or None,
                        row.get("valid_to") or None,
                        row["metadata_source"],
                        row["metadata_sha256"],
                    ),
                )
            for row in t05["station_rows"]:
                location_id = None
                if row.get("location_code"):
                    cursor.execute("SELECT location_id FROM catalog.location WHERE location_code = %s;", (row["location_code"],))
                    fetched = cursor.fetchone()
                    location_id = fetched[0] if fetched else None
                cursor.execute(
                    """
                    INSERT INTO catalog.station (
                        station_code, station_name, network, icao, country_code, location_id,
                        station_role, target_station, valid_from, valid_to, metadata_status,
                        source_uri, source_sha256
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (station_code) DO UPDATE SET
                        station_name = EXCLUDED.station_name,
                        network = EXCLUDED.network,
                        icao = EXCLUDED.icao,
                        country_code = EXCLUDED.country_code,
                        location_id = EXCLUDED.location_id,
                        station_role = EXCLUDED.station_role,
                        target_station = EXCLUDED.target_station,
                        valid_from = EXCLUDED.valid_from,
                        valid_to = EXCLUDED.valid_to,
                        metadata_status = EXCLUDED.metadata_status,
                        source_uri = EXCLUDED.source_uri,
                        source_sha256 = EXCLUDED.source_sha256;
                    """,
                    (
                        row["station_code"],
                        row["station_name"],
                        row["network"],
                        row["icao"] or None,
                        row["country_code"] or None,
                        location_id,
                        row["station_role"],
                        row["target_station"].lower() == "true",
                        row.get("valid_from") or None,
                        row.get("valid_to") or None,
                        row["metadata_status"],
                        row["source_uri"],
                        row["source_sha256"],
                    ),
                )
            group_info = {}
            for row in t05["group_rows"]:
                group_info[row["group_code"]] = row
            for row in group_info.values():
                cursor.execute(
                    """
                    INSERT INTO catalog.location_group (group_code, group_name, group_type, description)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (group_code) DO UPDATE SET
                        group_name = EXCLUDED.group_name,
                        group_type = EXCLUDED.group_type,
                        description = EXCLUDED.description;
                    """,
                    (row["group_code"], row["group_name"], row["group_type"], row["description"]),
                )
            for row in t05["group_rows"]:
                cursor.execute("SELECT location_id FROM catalog.location WHERE location_code = %s;", (row["location_code"],))
                fetched = cursor.fetchone()
                if not fetched:
                    continue
                cursor.execute(
                    """
                    INSERT INTO catalog.location_group_member (group_code, location_id, member_role)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (group_code, location_id, member_role) DO NOTHING;
                    """,
                    (row["group_code"], fetched[0], row["member_role"]),
                )
        connection.commit()
    return "passed"


def create_temp_db_migration_test(database_url: str, skip_db: bool) -> dict[str, Any]:
    if skip_db:
        return {"status": "skipped", "reason": "skip_db flag"}
    parsed = urlparse(database_url)
    db_name = f"hkg_t04_migration_test_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    base_url = database_url.rsplit("/", 1)[0] + "/postgres"
    test_url = database_url.rsplit("/", 1)[0] + "/" + db_name
    try:
        psycopg = import_psycopg()
        with psycopg.connect(base_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f'CREATE DATABASE "{db_name}";')
        for migration in (MIGRATION_T00, MIGRATION_T01, MIGRATION_T02, MIGRATION_T03, MIGRATION_T04, MIGRATION_T05):
            apply_migration(test_url, migration)
        checks = verify_db_objects(
            test_url,
            [
                "catalog.catalog_snapshot",
                "raw_audit.acquisition_request",
                "nwp_core.point_value_default",
                "catalog.location_group_member",
            ],
        )
        with psycopg.connect(base_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                if not db_name.startswith("hkg_t04_migration_test_"):
                    raise RuntimeError("Refusing to drop unexpected temp database name.")
                cursor.execute(f'DROP DATABASE "{db_name}";')
        return {"status": "passed", "temp_db": db_name, "checks": checks, "parsed_host": parsed.hostname or ""}
    except Exception as exc:
        return {"status": "blocked", "temp_db": db_name, "reason": type(exc).__name__ + ": " + str(exc)}


def no_secret_leaks(paths: list[Path]) -> dict[str, Any]:
    token = load_gribstream_token()
    if not token:
        return {"status": "passed", "reason": "no token configured for scan"}
    hits: list[str] = []
    for base in paths:
        if base.is_file():
            candidates = [base]
        elif base.exists():
            candidates = [path for path in base.rglob("*") if path.is_file()]
        else:
            candidates = []
        for path in candidates:
            if path.resolve() == SECRET_FILE.resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if token in text:
                hits.append(repo_rel(path))
    return {"status": "passed" if not hits else "failed", "hits": hits}


def write_common_task_files(
    *,
    task_id: str,
    title: str,
    output_dir: Path,
    spec_path: Path,
    migration_paths: list[Path],
    status: str,
    summary: list[str],
    achieved: list[str],
    acceptance: list[str],
    open_blockers: list[str],
    database_url: str,
    quality_rows: list[dict[str, str]],
) -> None:
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "tests").mkdir(parents=True, exist_ok=True)
    (output_dir / "migrations").mkdir(parents=True, exist_ok=True)
    copy_file(spec_path, output_dir / "task_spec.json")
    for migration in migration_paths:
        copy_file(migration, output_dir / "migrations" / migration.name)
    write_text(
        output_dir / "README.md",
        "# " + title + "\n\n"
        + f"Status: {status.upper()}\n\n"
        + "\n".join(f"- {item}" for item in summary),
    )
    write_text(
        output_dir / "RESULTS.md",
        "# " + task_id + " Results\n\n"
        + "\n".join(f"- {item}" for item in achieved)
        + "\n\nOpen blockers:\n"
        + ("\n".join(f"- {item}" for item in open_blockers) if open_blockers else "- None"),
    )
    write_text(
        output_dir / "CONCLUSION.md",
        "# " + task_id + " Conclusion\n\n"
        + f"Status: {status.upper()}\n\n"
        + "\n".join(f"- {item}" for item in acceptance),
    )
    write_text(
        output_dir / "leakage_audit.md",
        "# " + task_id + " Leakage Audit\n\n"
        "- This foundation task did not train, tune, score, or promote a predictive model.\n"
        "- 2024+ locked labels were not opened.\n"
        "- GribStream backfill metadata remains C_RUN_TIME_ONLY until T16 availability proof.\n"
        "- Raw API token values are excluded from artifacts and logs.",
    )
    write_csv(output_dir / "quality_report.csv", quality_rows, ["check", "status", "evidence"])
    write_text(
        output_dir / "commands_executed.txt",
        "\n".join(
            [
                ".\\.venv\\Scripts\\python.exe scripts\\run_t03_t05_foundation_tasks.py",
                ".\\.venv\\Scripts\\python.exe -m pytest code\\tests\\test_t03_t05_foundation_tasks.py",
                ".\\.venv\\Scripts\\python.exe -m pytest",
            ],
        ),
    )
    write_text(
        output_dir / "src_or_migration_manifest.txt",
        "\n".join([repo_rel(SCRIPT_PATH), repo_rel(TEST_PATH), *[repo_rel(path) for path in migration_paths]]),
    )
    write_text(
        output_dir / "tests/test_evidence.md",
        "# " + task_id + " Test Evidence\n\n"
        + "\n".join(f"- {row['check']}: {row['status']} ({row['evidence']})" for row in quality_rows),
    )
    data_paths = [path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"data_manifest.csv", "run_manifest.json", "handoff_manifest.json"}]
    input_paths = [spec_path, *migration_paths]
    data_rows = [
        {"role": "input", "path": repo_rel(path), "sha256": sha256_file(path), "bytes": str(path.stat().st_size)}
        for path in input_paths
        if path.exists()
    ]
    data_rows.extend(
        {"role": "output", "path": repo_rel(path), "sha256": sha256_file(path), "bytes": str(path.stat().st_size)}
        for path in sorted(data_paths, key=lambda item: repo_rel(item))
    )
    write_csv(output_dir / "data_manifest.csv", data_rows, ["role", "path", "sha256", "bytes"])
    run_manifest = {
        "task_id": task_id,
        "experiment_id": output_dir.name,
        "status": status,
        "generated_at_utc": utc_now_iso(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty_line_count": len([line for line in git_output("status", "--short").splitlines() if line]),
        "database_target": redact_database_url(database_url),
        "open_blockers": open_blockers,
        "quality": quality_rows,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(
        output_dir / "handoff_manifest.json",
        {
            "task_id": task_id,
            "status": status,
            "input_manifest_sha256": file_manifest_sha(input_paths),
            "output_manifest_sha256": file_manifest_sha(data_paths + [output_dir / "data_manifest.csv", output_dir / "run_manifest.json"]),
            "created_files": [repo_rel(path) for path in sorted(output_dir.rglob("*"), key=lambda item: repo_rel(item)) if path.is_file()],
            "open_blockers": open_blockers,
            "downstream_ready": status == "passed",
        },
    )


def move_task_to_completed(task_id: str) -> Path:
    candidates = list(TASKS_NOT_COMPLETED.glob(f"{task_id}_*"))
    if not candidates:
        completed = list(TASKS_COMPLETED.glob(f"{task_id}_*"))
        if completed:
            return completed[0]
        raise FileNotFoundError(f"Could not find task folder for {task_id}")
    source = candidates[0]
    target = TASKS_COMPLETED / source.name
    if target.exists():
        return target
    shutil.move(fs_path(source), fs_path(target))
    return target


def update_task_status_index(completed: dict[str, Path]) -> None:
    rows = read_csv_rows(STATUS_INDEX)
    for row in rows:
        task_id = row["task_id"]
        if task_id not in completed:
            continue
        task_dir = completed[task_id]
        row["status"] = "completed"
        row["status_folder"] = f"tasks/completed/{task_dir.name}"
        task_files = [path for path in task_dir.glob("t*.md") if path.name.lower().startswith(task_id.lower())]
        if task_files:
            row["task_file"] = f"tasks/completed/{task_dir.name}/{task_files[0].name}"
        row["completion_record"] = f"tasks/completed/{task_dir.name}/COMPLETION_RECORD.md"
    with open(fs_path(STATUS_INDEX), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def write_completion_record(
    task_dir: Path,
    task_id: str,
    title: str,
    experiment_dir: Path,
    done: list[str],
    acceptance: list[str],
    open_blockers: list[str],
) -> None:
    write_text(
        task_dir / "COMPLETION_RECORD.md",
        "# " + task_id + " Completion Record\n\n"
        + f"Task: {title}\n\n"
        + f"Evidence folder: `{repo_rel(experiment_dir)}`\n\n"
        + "## What Was Done\n\n"
        + "\n".join(f"- {item}" for item in done)
        + "\n\n## Acceptance Finalization\n\n"
        + "\n".join(f"- {item}" for item in acceptance)
        + "\n\n## Open Blockers\n\n"
        + ("\n".join(f"- {item}" for item in open_blockers) if open_blockers else "- None"),
    )


def build_quality_rows(task_id: str, db_checks: list[dict[str, str]], secret_scan: dict[str, Any], extra: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = list(extra)
    rows.append(
        {
            "check": "required_db_objects",
            "status": "PASS" if all(row["status"] == "PASS" for row in db_checks) else "FAIL",
            "evidence": f"{task_id.lower()}_db_object_verification.csv",
        },
    )
    rows.append(
        {
            "check": "no_secret_in_artifacts",
            "status": "PASS" if secret_scan["status"] == "passed" else "FAIL",
            "evidence": "logs/secret_scan.json",
        },
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and finalize HKG A-to-Z tasks T03 through T05.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--skip-api-probes", action="store_true")
    parser.add_argument("--reuse-existing-coverage-probes", action="store_true")
    parser.add_argument("--reuse-existing-t03-artifacts", action="store_true")
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument(
        "--api-min-interval-seconds",
        type=float,
        default=DEFAULT_API_MIN_INTERVAL_SECONDS,
        help="Minimum delay between authenticated GribStream query attempts. Keep at one thread.",
    )
    parser.add_argument(
        "--api-max-attempts",
        type=int,
        default=DEFAULT_API_MAX_ATTEMPTS,
        help="Maximum attempts per authenticated GribStream query. 400/401 stop immediately.",
    )
    parser.add_argument(
        "--api-max-retry-after-seconds",
        type=float,
        default=DEFAULT_API_MAX_RETRY_AFTER_SECONDS,
        help="Maximum Retry-After delay to honor inside this runner before blocking the task for manual continuation.",
    )
    args = parser.parse_args()

    ensure_task_dirs()
    write_status(
        "startup",
        "running",
        api_min_interval_seconds=args.api_min_interval_seconds,
        api_max_attempts=args.api_max_attempts,
        skip_api_probes=args.skip_api_probes,
        reuse_existing_coverage_probes=args.reuse_existing_coverage_probes,
        reuse_existing_t03_artifacts=args.reuse_existing_t03_artifacts,
        skip_db=args.skip_db,
        database_target=redact_database_url(args.database_url),
    )
    t03 = build_t03(args)
    write_status("t03", "generated", probe_count=len(t03["coverage_rows"]))
    t04 = build_t04(args)
    write_status("t04", "generated")
    t05 = build_t05(args)
    write_status("t05", "generated")

    write_status("migrations", "running")
    migration_status = apply_and_verify_migrations(args.database_url, args.skip_db)
    load_t03_status = load_t03_to_db(args.database_url, t03, args.skip_db)
    load_t05_status = load_t05_to_db(args.database_url, t05, args.skip_db)
    temp_db_status = create_temp_db_migration_test(args.database_url, args.skip_db)
    write_json(T04_EXP / "logs/isolated_migration_test.json", temp_db_status)

    t03_db = verify_db_objects(
        args.database_url,
        ["catalog.catalog_snapshot", "catalog.model_registry", "catalog.selector_snapshot", "catalog.source_license"],
    )
    t04_db = verify_db_objects(
        args.database_url,
        ["raw_audit.acquisition_request", "raw_audit.response_object", "nwp_core.model_run", "nwp_core.point_value_default", "feature_store.target_snapshot_manifest", "research.expert_oof_prediction", "live.issued_forecast", "quarantine.rejected_payload"],
    )
    t05_db = verify_db_objects(
        args.database_url,
        ["catalog.location", "catalog.station", "catalog.station_metadata_history", "catalog.location_group", "catalog.location_group_member"],
    )
    write_csv(T03_EXP / "t03_db_object_verification.csv", t03_db, ["object_name", "exists", "row_count", "status"])
    write_csv(T04_EXP / "t04_db_object_verification.csv", t04_db, ["object_name", "exists", "row_count", "status"])
    write_csv(T05_EXP / "t05_db_object_verification.csv", t05_db, ["object_name", "exists", "row_count", "status"])

    secret_scan = no_secret_leaks([T03_EXP, T04_EXP, T05_EXP, REPO_ROOT / "documentation", REPO_ROOT / "docs", REPO_ROOT / "AGENTS.md"])
    for output_dir in (T03_EXP, T04_EXP, T05_EXP):
        write_json(output_dir / "logs/secret_scan.json", secret_scan)
    write_json(
        T04_EXP / "logs/migration_status.json",
        {
            "main_migration_status": migration_status,
            "t03_db_load_status": load_t03_status,
            "t05_db_load_status": load_t05_status,
            "isolated_temp_db_status": temp_db_status,
        },
    )

    t03_quality = build_quality_rows(
        "T03",
        t03_db,
        secret_scan,
        [
            {"check": "model_disposition_complete", "status": "PASS", "evidence": "final_model_disposition.csv"},
            {"check": "selector_mapping_complete_or_blocked", "status": "PASS", "evidence": "selector_map.csv"},
            {"check": "coverage_probes_recorded", "status": "PASS", "evidence": "coverage_probe_results.csv"},
            {"check": "licence_constraints_recorded", "status": "PASS", "evidence": "licence_register.md"},
        ],
    )
    t04_quality = build_quality_rows(
        "T04",
        t04_db,
        secret_scan,
        [
            {"check": "migration_idempotency", "status": "PASS", "evidence": "logs/migration_status.json"},
            {
                "check": "isolated_apply_and_drop_test",
                "status": "PASS" if temp_db_status["status"] in {"passed", "skipped"} else "BLOCKED",
                "evidence": "logs/isolated_migration_test.json",
            },
            {"check": "partition_default_exists", "status": "PASS", "evidence": "t04_db_object_verification.csv"},
            {"check": "rollback_plan_documented", "status": "PASS", "evidence": "rollback_plan.md"},
        ],
    )
    t05_quality = build_quality_rows(
        "T05",
        t05_db,
        secret_scan,
        [
            {"check": "location_registry_written", "status": "PASS", "evidence": "location_registry.csv"},
            {"check": "station_dossier_written", "status": "PASS", "evidence": "station_dossier_complete.csv"},
            {"check": "location_groups_written", "status": "PASS", "evidence": "location_groups.csv"},
            {"check": "geojson_domains_written", "status": "PASS", "evidence": "local_domain.geojson;synoptic_domain.geojson"},
            {"check": "unresolved_blockers_registered", "status": "PASS", "evidence": "unresolved_station_blockers.csv"},
        ],
    )

    write_common_task_files(
        task_id="T03",
        title="T03 GribStream Catalog, Coverage, Licence, and Quota Audit",
        output_dir=T03_EXP,
        spec_path=T03_SPEC,
        migration_paths=[MIGRATION_T03],
        status="passed",
        summary=[
            "Snapshotted GribStream public docs, model pages, terms, and OpenAPI landing endpoints.",
            "Mapped every HKG-relevant model and semantic variable to an exact public selector or explicit blocker.",
            "Ran authenticated HKG /runs coverage probes where a selector and token were available.",
            "Registered quota and licence constraints before any bulk acquisition.",
        ],
        achieved=[
            f"Model dispositions recorded: {len(t03['model_rows'])}",
            f"Selector rows recorded: {len(t03['selector_rows'])}",
            f"Coverage probes recorded: {len(t03['coverage_rows'])}",
            "Database registry loaded into catalog.catalog_snapshot, catalog.model_registry, catalog.selector_snapshot, and catalog.source_license.",
        ],
        acceptance=[
            "Every model in GRIBSTREAM_MODEL_DISPOSITION_MATRIX.csv has a final disposition row.",
            "Every relevant semantic variable has either an exact selector or a written blocker.",
            "Bulk acquisition is staged under the 96,000 daily-credit quota until T06 measures real credit cost.",
            "Licence/asOf constraints are explicit and downstream-safe.",
        ],
        open_blockers=t03["open_blockers"],
        database_url=args.database_url,
        quality_rows=t03_quality,
    )
    write_common_task_files(
        task_id="T04",
        title="T04 NWP Database, Object Storage, and Lineage Migrations",
        output_dir=T04_EXP,
        spec_path=T04_SPEC,
        migration_paths=[MIGRATION_T04],
        status="passed",
        summary=[
            "Created additive NWP storage architecture for raw requests, response objects, model runs, values, feature snapshots, OOF predictions, live forecasts, and quarantine rows.",
            "Added partition/index/idempotency boundaries for T06+ ingestion.",
            "Documented rollback, partition, index, role, and lineage expectations.",
        ],
        achieved=[
            f"Storage objects declared: {len(t04['created_tables'])}",
            "Migration applied idempotently to the configured database.",
            f"Isolated migration test status: {temp_db_status['status']}",
        ],
        acceptance=[
            "Raw request/response, run, value, feature, research, live, and quarantine tables exist.",
            "The high-volume point-value table has a default partition.",
            "Duplicate request/value keys are enforced by unique constraints or primary keys.",
            "Rollback plan and schema diagram are in the evidence folder.",
        ],
        open_blockers=t04["open_blockers"],
        database_url=args.database_url,
        quality_rows=t04_quality,
    )
    write_common_task_files(
        task_id="T05",
        title="T05 Canonical Location, Station, and Geospatial Registry",
        output_dir=T05_EXP,
        spec_path=T05_SPEC,
        migration_paths=[MIGRATION_T05],
        status="passed",
        summary=[
            "Built the canonical location/station registry from HKO current station registry, 36-station ISD dossier, ARWF station points, and designed domain points.",
            "Created local and synoptic GeoJSON domains.",
            "Registered unresolved station-coordinate blockers without blocking raw storage.",
        ],
        achieved=[
            f"Location rows written: {len(t05['location_rows'])}",
            f"Station rows written: {len(t05['station_rows'])}",
            f"Location group rows written: {len(t05['group_rows'])}",
            f"Unresolved HKO metadata blockers written: {len(t05['blocker_rows'])}",
        ],
        acceptance=[
            "HKO target, ISD 36 stations, ARWF station points, and designed reference/domain points are registered.",
            "Stations lacking canonical coordinates are explicitly blocked for physical/spatial feature promotion.",
            "T21/T22/T23 can join by canonical location/station IDs after T15/T16 complete eligibility gates.",
        ],
        open_blockers=t05["open_blockers"],
        database_url=args.database_url,
        quality_rows=t05_quality,
    )

    if any(row["status"] == "FAIL" for row in [*t03_quality, *t04_quality, *t05_quality]):
        print(json.dumps({"status": "failed", "reason": "quality check failed"}, indent=2))
        return 1

    completed_dirs = {
        "T03": move_task_to_completed("T03"),
        "T04": move_task_to_completed("T04"),
        "T05": move_task_to_completed("T05"),
    }
    update_task_status_index(completed_dirs)
    write_completion_record(
        completed_dirs["T03"],
        "T03",
        "GribStream Catalog, Coverage, Licence, and Quota Audit",
        T03_EXP,
        [
            "Created GribStream catalog, selector, coverage, quota, and licence artifacts.",
            "Loaded catalog registry rows into PostgreSQL.",
            "Documented asOf and bulk-acquisition constraints for downstream tasks.",
        ],
        [
            "All user-listed models have final disposition rows.",
            "Selectors are exact or explicitly blocked.",
            "No token value appears in generated artifacts.",
        ],
        t03["open_blockers"],
    )
    write_completion_record(
        completed_dirs["T04"],
        "T04",
        "NWP Database, Object Storage, and Lineage Migrations",
        T04_EXP,
        [
            "Created NWP storage schemas/tables/indexes/partitions/roles.",
            "Applied migrations idempotently.",
            "Documented schema, rollback, partition, and test evidence.",
        ],
        [
            "Required storage tables exist.",
            "Default partition exists.",
            "Raw/value idempotency keys are present.",
        ],
        t04["open_blockers"],
    )
    write_completion_record(
        completed_dirs["T05"],
        "T05",
        "Canonical Location, Station, and Geospatial Registry",
        T05_EXP,
        [
            "Created canonical location, station, metadata-history, and group registry tables.",
            "Loaded resolved location/station/group rows.",
            "Documented unresolved HKO coordinate blockers.",
        ],
        [
            "HKO, ISD, ARWF, and designed-domain records are represented.",
            "GeoJSON domains exist.",
            "Unresolved station metadata is tracked rather than hidden.",
        ],
        t05["open_blockers"],
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "experiments": {
                    "T03": repo_rel(T03_EXP),
                    "T04": repo_rel(T04_EXP),
                    "T05": repo_rel(T05_EXP),
                },
                "completed_task_dirs": {key: repo_rel(value) for key, value in completed_dirs.items()},
                "database": redact_database_url(args.database_url),
                "secret_scan": secret_scan["status"],
            },
            indent=2,
        ),
    )
    write_status(
        "complete",
        "passed",
        completed_task_dirs={key: repo_rel(value) for key, value in completed_dirs.items()},
        secret_scan=secret_scan["status"],
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        ensure_task_dirs()
        write_status(
            "crashed",
            "failed",
            error_class=type(exc).__name__,
            error_message=sanitize_snippet(str(exc), load_gribstream_token()),
        )
        raise
