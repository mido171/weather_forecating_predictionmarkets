from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT_ID = "0007_public_7day_gfs_gefs_himawari_backfill_rehearsal_20260708"
SOURCE_NORMALIZED_DIR = (
    REPO_ROOT
    / "experiments"
    / "hkg_tmax"
    / SOURCE_EXPERIMENT_ID
    / "normalized"
)
EXPERIMENT_ID = "0008_last2_gfs_gefs_radar_structured_delivery_20260708"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / EXPERIMENT_ID
NORMALIZED_DIR = EXPERIMENT_DIR / "normalized"
METADATA_DIR = EXPERIMENT_DIR / "metadata"
LOG_DIR = EXPERIMENT_DIR / "logs"
STAGING_DIR = EXPERIMENT_DIR / "staging"
USER_AGENT = "weather-markets-hkg-last2-gfs-gefs-radar-structured/1.0"

LAST_COMPLETE_UTC_END = date(2026, 7, 7)
LAST_TWO_UTC_DAYS = [date(2026, 7, 6), date(2026, 7, 7)]
UTC_WINDOW_START = datetime(2026, 7, 6, 0, 0, tzinfo=timezone.utc)
UTC_WINDOW_END_EXCLUSIVE = datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc)
HKT_OFFSET = timedelta(hours=8)


def wp(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def ensure_dirs() -> None:
    for path in [EXPERIMENT_DIR, NORMALIZED_DIR, METADATA_DIR, LOG_DIR, STAGING_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(wp(SOURCE_NORMALIZED_DIR / name))


def write_table(df: pd.DataFrame, stem: str) -> dict[str, Any]:
    csv_path = NORMALIZED_DIR / f"{stem}.csv"
    parquet_path = NORMALIZED_DIR / f"{stem}.parquet"
    df.to_csv(wp(csv_path), index=False)
    if not df.empty:
        df.to_parquet(wp(parquet_path), index=False)
    parquet_exists = os.path.exists(wp(parquet_path))
    return {
        "name": stem,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "csv_path": str(csv_path.relative_to(EXPERIMENT_DIR)),
        "csv_bytes": int(os.path.getsize(wp(csv_path))),
        "parquet_path": str(parquet_path.relative_to(EXPERIMENT_DIR)) if parquet_exists else None,
        "parquet_bytes": int(os.path.getsize(wp(parquet_path))) if parquet_exists else 0,
    }


def write_json(path: Path, payload: Any) -> None:
    with open(wp(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def last2_filter(df: pd.DataFrame, column: str = "issue_day_utc") -> pd.DataFrame:
    allowed = {d.isoformat() for d in LAST_TWO_UTC_DAYS}
    return df[df[column].astype(str).isin(allowed)].copy()


def parse_hkt_datetime(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y/%m/%d %H:%M").replace(tzinfo=timezone(HKT_OFFSET))


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def envf_url(start_hkt: datetime, npics: int = 72, interval_hours: float = 0.2) -> str:
    return (
        "https://envf.ust.hk/dataview/hko_radar/current/index.py?"
        f"year__int={start_hkt.year}"
        f"&month__int={start_hkt.month}"
        f"&day__int={start_hkt.day}"
        f"&hour__int={start_hkt.hour}"
        f"&npics__int={npics}"
        f"&interval__float={interval_hours}"
        "&display=Search"
    )


def envf_chunk_starts() -> list[datetime]:
    start_hkt = UTC_WINDOW_START.astimezone(timezone(HKT_OFFSET))
    # ENVF max npics is 72 at 0.2h spacing, i.e. 14.4h coverage per query.
    # The form only accepts an integer start hour, so use overlapping 12h chunks
    # to avoid small gaps at non-hour chunk boundaries.
    chunk = timedelta(hours=12)
    starts: list[datetime] = []
    cursor = start_hkt
    while cursor.astimezone(timezone.utc) < UTC_WINDOW_END_EXCLUSIVE:
        starts.append(cursor)
        cursor += chunk
    return starts


def fetch_envf_manifest() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict[str, Any]] = []
    chunk_logs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for start_hkt in envf_chunk_starts():
        url = envf_url(start_hkt)
        t0 = time.perf_counter()
        try:
            resp = session.get(url, timeout=60)
            elapsed = time.perf_counter() - t0
            resp.raise_for_status()
            html = resp.text
            status = "ok"
            error = None
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            html = ""
            status = "error"
            error = repr(exc)

        chunk_logs.append(
            {
                "chunk_start_hkt": start_hkt.isoformat(),
                "url": url,
                "status": status,
                "elapsed_seconds": elapsed,
                "html_bytes": len(html.encode("utf-8")),
                "error": error,
            }
        )
        if status != "ok":
            continue

        links = re.findall(r'href="([^"]*display_large_image[^"]*)"', html)
        thumbs = re.findall(r'<img[^>]+src="([^"]+)"', html, flags=re.I)
        thumb_by_index = {i: urljoin(url, link) for i, link in enumerate(thumbs)}

        for idx, href in enumerate(links):
            full_href = urljoin(url, href.replace("&amp;", "&"))
            parsed = urlparse(full_href)
            qs = parse_qs(parsed.query)
            dt_values = qs.get("datetime", [])
            imagef_values = qs.get("imagef", [])
            if not dt_values:
                continue
            frame_hkt = parse_hkt_datetime(dt_values[0])
            observed_utc = frame_hkt.astimezone(timezone.utc)
            if not (UTC_WINDOW_START <= observed_utc < UTC_WINDOW_END_EXCLUSIVE):
                continue
            key = observed_utc.isoformat()
            if key in seen:
                continue
            seen.add(key)
            availability_utc = observed_utc + timedelta(minutes=30)
            rows.append(
                {
                    "source": "envf_hkust_hko_radar",
                    "provider": "HKUST ENVF mirror of HKO radar imagery",
                    "product": "hko_radar_image",
                    "frame_time_hkt": frame_hkt.isoformat(),
                    "observed_at_utc": to_utc_iso(observed_utc),
                    "availability_proxy_utc": to_utc_iso(availability_utc),
                    "availability_proxy_method": "historical_display_proxy_observed_at_plus_30m",
                    "native_issue_metadata_status": "not_native_exact_vintage",
                    "envf_query_url": url,
                    "display_large_image_url": full_href,
                    "display_image_url": thumb_by_index.get(idx),
                    "envf_temp_image_path": unquote(imagef_values[0]) if imagef_values else None,
                    "frame_index_in_chunk": idx,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("observed_at_utc").reset_index(drop=True)
    return df, chunk_logs


def rgb_features(content: bytes) -> dict[str, Any]:
    sha = hashlib.sha256(content).hexdigest()
    with Image.open(io.BytesIO(content)) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    sat = maxc - minc
    bright = (r + g + b) / 3.0

    colored = (sat >= 35) & (maxc >= 60)
    dark = bright < 40
    blue = colored & (b > r + 25) & (b > g + 10)
    cyan = colored & (b > r + 20) & (g > r + 20) & (np.abs(g - b) <= 55)
    green = colored & (g > r + 25) & (g > b + 10)
    yellow = colored & (r > b + 35) & (g > b + 35) & (np.abs(r - g) <= 80)
    orange = colored & (r > g + 20) & (g > b + 25)
    red = colored & (r > g + 35) & (r > b + 35)
    purple = colored & (r > g + 20) & (b > g + 20)
    total = int(arr.shape[0] * arr.shape[1])

    def frac(mask: np.ndarray) -> float:
        return float(mask.sum() / total) if total else 0.0

    return {
        "image_sha256": sha,
        "image_bytes": len(content),
        "image_width": int(arr.shape[1]),
        "image_height": int(arr.shape[0]),
        "pixel_count": total,
        "rgb_r_mean": float(r.mean()),
        "rgb_g_mean": float(g.mean()),
        "rgb_b_mean": float(b.mean()),
        "rgb_brightness_mean": float(bright.mean()),
        "rgb_saturation_mean": float(sat.mean()),
        "dark_pixel_fraction": frac(dark),
        "rain_colored_pixel_fraction": frac(colored),
        "rain_blue_fraction": frac(blue),
        "rain_cyan_fraction": frac(cyan),
        "rain_green_fraction": frac(green),
        "rain_yellow_fraction": frac(yellow),
        "rain_orange_fraction": frac(orange),
        "rain_red_fraction": frac(red),
        "rain_purple_fraction": frac(purple),
        "rain_intensity_proxy": float(
            frac(blue) * 1.0
            + frac(cyan) * 1.2
            + frac(green) * 1.7
            + frac(yellow) * 2.5
            + frac(orange) * 3.2
            + frac(red) * 4.0
            + frac(purple) * 4.5
        ),
    }


def fetch_radar_image_features(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    for record in manifest.to_dict(orient="records"):
        url = record.get("display_image_url") or record.get("display_large_image_url")
        t0 = time.perf_counter()
        status = "error"
        error = None
        features: dict[str, Any] = {}
        attempts = 0
        for attempt in range(1, 4):
            attempts = attempt
            try:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                content = resp.content
                features = rgb_features(content)
                status = "ok"
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
                if attempt < 3:
                    time.sleep(1.5 * attempt)
        elapsed = time.perf_counter() - t0

        logs.append(
            {
                "observed_at_utc": record.get("observed_at_utc"),
                "url": url,
                "status": status,
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "error": error,
            }
        )
        row = {**record, "image_fetch_status": status, "image_fetch_error": error, **features}
        rows.append(row)

    return pd.DataFrame(rows), logs


def build_source_issue_glue(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fetch_manifest = tables["model_fetch_manifest_last2"]
    for record in fetch_manifest.to_dict(orient="records"):
        rows.append(
            {
                "issue_key": record.get("item_id"),
                "source": record.get("source"),
                "product": record.get("kind"),
                "issue_day_utc": record.get("issue_day_utc"),
                "issued_at_utc": record.get("issued_at_utc"),
                "observed_at_utc": None,
                "valid_at_utc": record.get("valid_at_utc"),
                "availability_proxy_utc": record.get("availability_proxy_utc"),
                "status": record.get("status"),
                "raw_sha256": record.get("sha256"),
                "raw_bytes": record.get("bytes"),
                "raw_retention_policy": "source_run_raw_retained_in_0007_only; 0008_contains_normalized_only",
                "normalized_tables": "model_station_features_last2;model_bbox_features_last2;model_idx_catalog_last2",
                "source_url": record.get("url"),
            }
        )

    radar_manifest = tables["radar_envf_manifest_frames_last2"]
    for record in radar_manifest.to_dict(orient="records"):
        rows.append(
            {
                "issue_key": f"envf_hko_radar:{record.get('observed_at_utc')}",
                "source": record.get("source"),
                "product": record.get("product"),
                "issue_day_utc": str(record.get("observed_at_utc"))[:10],
                "issued_at_utc": None,
                "observed_at_utc": record.get("observed_at_utc"),
                "valid_at_utc": record.get("observed_at_utc"),
                "availability_proxy_utc": record.get("availability_proxy_utc"),
                "status": "ok",
                "raw_sha256": None,
                "raw_bytes": None,
                "raw_retention_policy": "image_bytes_loaded_in_memory_then_discarded",
                "normalized_tables": "radar_envf_manifest_frames_last2;radar_envf_image_features_last2",
                "source_url": record.get("display_large_image_url"),
            }
        )

    return pd.DataFrame(rows)


def attribute_catalog(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table, df in tables.items():
        for column in df.columns:
            series = df[column]
            non_null = series.dropna()
            sample_values = []
            for value in non_null.astype(str).drop_duplicates().head(5).tolist():
                sample_values.append(value[:500])
            row: dict[str, Any] = {
                "table": table,
                "column": column,
                "dtype": str(series.dtype),
                "row_count": int(len(series)),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "distinct_count": int(non_null.astype(str).nunique()) if len(non_null) else 0,
                "sample_values_json": json.dumps(sample_values, ensure_ascii=True),
            }
            if pd.api.types.is_numeric_dtype(series) and len(non_null):
                row["min"] = float(non_null.min())
                row["max"] = float(non_null.max())
                row["mean"] = float(non_null.mean())
            else:
                row["min"] = None
                row["max"] = None
                row["mean"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def postgres_glue_schema() -> str:
    return """-- High-level Postgres glue schema for raw-purged weather backfills.
-- Store only metadata, pointers, row counts, hashes, and leakage clocks in Postgres.
-- Keep heavy normalized feature/patch tables as partitioned Parquet/Arrow files.

CREATE SCHEMA IF NOT EXISTS weather_backfill;

CREATE TABLE IF NOT EXISTS weather_backfill.source_issue (
    issue_key text PRIMARY KEY,
    source text NOT NULL,
    product text NOT NULL,
    issue_day_utc date,
    issued_at_utc timestamptz,
    observed_at_utc timestamptz,
    valid_at_utc timestamptz,
    availability_proxy_utc timestamptz NOT NULL,
    availability_proxy_method text,
    status text NOT NULL,
    source_url text,
    raw_sha256 text,
    raw_bytes bigint,
    raw_retention_policy text NOT NULL,
    normalized_dataset_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS weather_backfill.normalized_artifact (
    artifact_id text PRIMARY KEY,
    dataset_id text NOT NULL,
    source text NOT NULL,
    product text NOT NULL,
    date_start_utc date,
    date_end_utc date,
    uri text NOT NULL,
    format text NOT NULL,
    row_count bigint NOT NULL,
    column_count integer NOT NULL,
    bytes bigint,
    content_sha256 text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS weather_backfill.artifact_column (
    artifact_id text NOT NULL REFERENCES weather_backfill.normalized_artifact(artifact_id),
    column_name text NOT NULL,
    dtype text NOT NULL,
    non_null_count bigint,
    distinct_count bigint,
    min_text text,
    max_text text,
    sample_values_json jsonb,
    PRIMARY KEY (artifact_id, column_name)
);

CREATE INDEX IF NOT EXISTS ix_weather_source_issue_time
ON weather_backfill.source_issue (source, product, availability_proxy_utc, valid_at_utc);
"""


def grouped_counts(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key, count in df.groupby(columns).size().items():
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(columns, key)}
        row["count"] = int(count)
        rows.append(row)
    return rows


def main() -> int:
    ensure_dirs()
    t0 = time.perf_counter()

    station = last2_filter(read_csv("model_cycle_lead_station_features.csv"))
    bbox = last2_filter(read_csv("model_cycle_lead_bbox_summary_features.csv"))
    idx = last2_filter(read_csv("model_idx_catalog.csv"))
    fetch = last2_filter(read_csv("fetch_manifest.csv"))
    fetch = fetch[fetch["source"].isin(["gfs", "gefs_control"])].copy()

    radar_manifest, radar_chunk_logs = fetch_envf_manifest()
    radar_features, radar_fetch_logs = fetch_radar_image_features(radar_manifest)

    tables: dict[str, pd.DataFrame] = {
        "model_fetch_manifest_last2": fetch,
        "model_idx_catalog_last2": idx,
        "model_station_features_last2": station,
        "model_bbox_features_last2": bbox,
        "radar_envf_manifest_frames_last2": radar_manifest,
        "radar_envf_image_features_last2": radar_features,
    }
    glue = build_source_issue_glue(tables)
    tables["source_issue_glue_last2"] = glue
    attrs = attribute_catalog(tables)
    tables["attribute_catalog_last2"] = attrs

    inventory = []
    for stem, df in tables.items():
        inventory.append(write_table(df, stem))
    inventory_df = pd.DataFrame(inventory)
    write_table(inventory_df, "normalized_file_inventory")

    write_json(METADATA_DIR / "radar_envf_query_logs.json", radar_chunk_logs)
    write_json(METADATA_DIR / "radar_image_fetch_logs.json", radar_fetch_logs)
    with open(wp(METADATA_DIR / "postgres_glue_schema.sql"), "w", encoding="utf-8") as handle:
        handle.write(postgres_glue_schema())

    if STAGING_DIR.exists():
        shutil.rmtree(wp(STAGING_DIR))

    elapsed = time.perf_counter() - t0
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "date_range_utc": {
            "start": UTC_WINDOW_START.isoformat().replace("+00:00", "Z"),
            "end_exclusive": UTC_WINDOW_END_EXCLUSIVE.isoformat().replace("+00:00", "Z"),
            "complete_utc_days": [d.isoformat() for d in LAST_TWO_UTC_DAYS],
        },
        "row_counts": {name: int(len(df)) for name, df in tables.items()},
        "model_sources": {
            "fetch_ok_by_source": grouped_counts(fetch, ["source", "status"]),
            "idx_ok_by_source": grouped_counts(idx, ["source", "status"]),
            "station_rows_by_source": grouped_counts(station, ["source"]),
            "bbox_rows_by_source": grouped_counts(bbox, ["source"]),
        },
        "radar": {
            "source": "HKUST ENVF HKO radar historical display",
            "frame_count": int(len(radar_manifest)),
            "image_feature_ok": int((radar_features.get("image_fetch_status") == "ok").sum()) if not radar_features.empty else 0,
            "cadence_minutes": 12,
            "native_issue_metadata_status": "not_native_exact_vintage",
            "availability_proxy_method": "observed_at_utc + 30m",
        },
        "raw_retention": {
            "experiment_staging_dir_exists_after_run": STAGING_DIR.exists(),
            "raw_policy": "0008 keeps no raw model or radar image payloads; radar image bytes are decoded in memory and discarded",
        },
        "elapsed_seconds": elapsed,
    }
    write_json(METADATA_DIR / "summary.json", summary)

    readme = f"""# {EXPERIMENT_ID}

Two-day structured delivery for GFS, GEFS control, and radar data.

UTC window: `{UTC_WINDOW_START.isoformat().replace("+00:00", "Z")}` to `{UTC_WINDOW_END_EXCLUSIVE.isoformat().replace("+00:00", "Z")}` exclusive.

## Key Outputs

| Output | Rows | Meaning |
|---|---:|---|
| `normalized/model_fetch_manifest_last2.csv` | {len(fetch)} | GFS/GEFS requested object manifest, status, URL, issue/valid/as-of clocks, raw hash/bytes from the 7-day run. |
| `normalized/model_idx_catalog_last2.csv` | {len(idx)} | Full NOMADS GRIB index-level catalog per cycle/lead, including available variables. |
| `normalized/model_station_features_last2.csv` | {len(station)} | HKO point feature rows. |
| `normalized/model_bbox_features_last2.csv` | {len(bbox)} | HKG bounding-box summary feature rows. |
| `normalized/radar_envf_manifest_frames_last2.csv` | {len(radar_manifest)} | Historical ENVF-served HKO radar frame manifest. |
| `normalized/radar_envf_image_features_last2.csv` | {len(radar_features)} | Numeric image-derived radar color/rainfall proxies. |
| `normalized/attribute_catalog_last2.csv` | {len(attrs)} | Column-by-column attribute catalog for every table above. |
| `normalized/source_issue_glue_last2.csv` | {len(glue)} | High-level glue rows suitable for a Postgres registry table. |
| `metadata/postgres_glue_schema.sql` | - | Proposed high-level Postgres schema. |

## Leakage / As-Of Clocks

GFS/GEFS rows retain `issued_at_utc`, `valid_at_utc`, and `availability_proxy_utc` from experiment 0007.

Radar rows are from HKUST ENVF historical display of HKO radar imagery. They have observed image times, not native HKO historical issue metadata, so the delivery marks them `not_native_exact_vintage` and uses `observed_at_utc + 30m` as a conservative availability proxy.

## Raw Retention

This folder intentionally keeps no raw payloads. Radar image bytes are fetched, decoded into numeric features, and discarded in memory. The script removes its staging directory at the end.
"""
    with open(wp(EXPERIMENT_DIR / "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)

    status = """state: COMPLETE_WITH_RADAR_NATIVE_ISSUE_CAVEAT
gate_result: LAST2_STRUCTURED_DELIVERY_DONE
date_start_utc: 2026-07-06
date_end_utc_exclusive: 2026-07-08
raw_payloads_retained_in_this_folder: false
"""
    with open(wp(EXPERIMENT_DIR / "STATUS.yaml"), "w", encoding="utf-8") as handle:
        handle.write(status)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
