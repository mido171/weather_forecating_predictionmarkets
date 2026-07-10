from __future__ import annotations

import bz2
import json
import math
import os
import re
import struct
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import cfgrib
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "cfgrib/eccodes is required. Install with: "
        ".\\.venv\\Scripts\\python.exe -m pip install cfgrib eccodes"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "0005_public_gfs_gefs_himawari_fetch_smoke_20260708"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / EXPERIMENT_ID
NORMALIZED_DIR = EXPERIMENT_DIR / "normalized"

HKG_BBOX = {
    "min_lat": 21.5,
    "max_lat": 23.5,
    "min_lon": 113.0,
    "max_lon": 115.5,
}

TARGET_STATION = {
    "station_id": "hko:HKO",
    "station_name": "Hong Kong Observatory",
    "latitude": 22.301944,
    "longitude": 114.174167,
    "elevation_m": 32,
    "metadata_source": "config/sources/stations_hko.yaml target_station",
}

SOURCE_FILES = {
    "gfs": "raw/gfs/gfs_20260708_00z_f024_hkg_bbox.grib2",
    "gefs_control": "raw/gefs/gefs_control_20260708_00z_f024_hkg_bbox.grib2",
}

VARIABLES = {
    "t2m": {
        "canonical": "temperature_2m",
        "friendly": "2m air temperature",
        "normalized_unit": "degC",
    },
    "d2m": {
        "canonical": "dewpoint_2m",
        "friendly": "2m dewpoint temperature",
        "normalized_unit": "degC",
    },
    "tmax": {
        "canonical": "temperature_2m_max",
        "friendly": "2m forecast-period maximum temperature",
        "normalized_unit": "degC",
    },
    "tmin": {
        "canonical": "temperature_2m_min",
        "friendly": "2m forecast-period minimum temperature",
        "normalized_unit": "degC",
    },
    "u10": {
        "canonical": "wind_u_10m",
        "friendly": "10m U wind component",
        "normalized_unit": "m_s-1",
    },
    "v10": {
        "canonical": "wind_v_10m",
        "friendly": "10m V wind component",
        "normalized_unit": "m_s-1",
    },
    "prmsl": {
        "canonical": "pressure_msl",
        "friendly": "mean sea-level pressure",
        "normalized_unit": "hPa",
    },
}


@dataclass
class GribSourceOutput:
    source: str
    input_path: str
    grid_rows: int
    grid_csv: str
    station_feature_csv: str
    bbox_summary_csv: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def win_path(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(win_path(path)).write_text(text, encoding="utf-8")


def safe_write_json(path: Path, payload: Any) -> None:
    safe_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(win_path(path), index=False)


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(win_path(path), index=False)
    except Exception as exc:
        safe_write_text(
            path.with_suffix(path.suffix + ".skipped.txt"),
            f"Parquet write skipped: {type(exc).__name__}: {exc}\n",
        )


def iso_from_np_datetime(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def hours_from_timedelta(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value / np.timedelta64(1, "h"))
    except Exception:
        return None


def normalize_value(var_name: str, value: float, native_units: str | None) -> tuple[float, str]:
    if var_name in {"t2m", "d2m", "tmax", "tmin"}:
        return float(value) - 273.15, "degC"
    if var_name == "prmsl":
        return float(value) / 100.0, "hPa"
    return float(value), VARIABLES.get(var_name, {}).get("normalized_unit", native_units or "")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def wind_dir_from_degrees(u: float, v: float) -> float:
    # Meteorological direction: direction wind comes from.
    return (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0


def crop_lat_lon(da: Any) -> Any:
    lats = da["latitude"].values
    lon_slice = slice(HKG_BBOX["min_lon"], HKG_BBOX["max_lon"])
    if float(lats[0]) > float(lats[-1]):
        lat_slice = slice(HKG_BBOX["max_lat"], HKG_BBOX["min_lat"])
    else:
        lat_slice = slice(HKG_BBOX["min_lat"], HKG_BBOX["max_lat"])
    return da.sel(latitude=lat_slice, longitude=lon_slice)


def nearest_grid_point(ds: Any) -> tuple[float, float, float]:
    lat_values = np.asarray(ds["latitude"].values, dtype=float)
    lon_values = np.asarray(ds["longitude"].values, dtype=float)
    target_lat = float(TARGET_STATION["latitude"])
    target_lon = float(TARGET_STATION["longitude"])
    lat = float(lat_values[np.argmin(np.abs(lat_values - target_lat))])
    lon = float(lon_values[np.argmin(np.abs(lon_values - target_lon))])
    return lat, lon, haversine_km(target_lat, target_lon, lat, lon)


def load_grib_datasets(path: Path) -> list[Any]:
    return cfgrib.open_datasets(win_path(path), backend_kwargs={"indexpath": ""})


def source_timing_from_da(da: Any) -> dict[str, Any]:
    time_value = da.coords["time"].values if "time" in da.coords else None
    valid_value = da.coords["valid_time"].values if "valid_time" in da.coords else None
    step_value = da.coords["step"].values if "step" in da.coords else None
    return {
        "issued_at_utc": iso_from_np_datetime(time_value),
        "valid_at_utc": iso_from_np_datetime(valid_value),
        "lead_hour": hours_from_timedelta(step_value),
    }


def normalize_grib_source(source: str, rel_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, GribSourceOutput]:
    path = EXPERIMENT_DIR / rel_path
    datasets = load_grib_datasets(path)
    grid_rows: list[dict[str, Any]] = []
    station_values: dict[str, Any] = {
        "source": source,
        "station_id": TARGET_STATION["station_id"],
        "station_name": TARGET_STATION["station_name"],
        "target_latitude": TARGET_STATION["latitude"],
        "target_longitude": TARGET_STATION["longitude"],
    }
    bbox_summary_rows: list[dict[str, Any]] = []
    nearest_done = False

    for ds in datasets:
        if not nearest_done:
            grid_lat, grid_lon, distance_km = nearest_grid_point(ds)
            station_values.update(
                {
                    "nearest_grid_latitude": grid_lat,
                    "nearest_grid_longitude": grid_lon,
                    "nearest_grid_distance_km": distance_km,
                }
            )
            nearest_done = True

        for var_name in ds.data_vars:
            if var_name not in VARIABLES:
                continue

            da = ds[var_name]
            attrs = dict(da.attrs)
            timing = source_timing_from_da(da)
            native_units = attrs.get("GRIB_units")
            meta = VARIABLES[var_name]
            cropped = crop_lat_lon(da)
            lat_values = np.asarray(cropped["latitude"].values, dtype=float)
            lon_values = np.asarray(cropped["longitude"].values, dtype=float)
            values = np.asarray(cropped.values, dtype=float)

            normalized_values = np.empty_like(values, dtype=float)
            for idx, value in np.ndenumerate(values):
                normalized_values[idx] = normalize_value(var_name, float(value), native_units)[0]
            normalized_unit = normalize_value(var_name, float(values.flat[0]), native_units)[1]

            for lat_idx, lat in enumerate(lat_values):
                for lon_idx, lon in enumerate(lon_values):
                    value_native = float(values[lat_idx, lon_idx])
                    value_normalized = float(normalized_values[lat_idx, lon_idx])
                    grid_rows.append(
                        {
                            "source": source,
                            "member": "control" if source == "gefs_control" else "deterministic",
                            "issued_at_utc": timing["issued_at_utc"],
                            "valid_at_utc": timing["valid_at_utc"],
                            "lead_hour": timing["lead_hour"],
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "variable": var_name,
                            "canonical_variable": meta["canonical"],
                            "friendly_name": meta["friendly"],
                            "grib_short_name": attrs.get("GRIB_shortName"),
                            "grib_name": attrs.get("GRIB_name"),
                            "grib_type_of_level": attrs.get("GRIB_typeOfLevel"),
                            "grib_level": attrs.get("GRIB_level"),
                            "grib_step_type": attrs.get("GRIB_stepType"),
                            "native_units": native_units,
                            "value_native": value_native,
                            "normalized_units": normalized_unit,
                            "value_normalized": value_normalized,
                        }
                    )

            bbox_summary_rows.append(
                {
                    "source": source,
                    "issued_at_utc": timing["issued_at_utc"],
                    "valid_at_utc": timing["valid_at_utc"],
                    "lead_hour": timing["lead_hour"],
                    "variable": var_name,
                    "canonical_variable": meta["canonical"],
                    "friendly_name": meta["friendly"],
                    "normalized_units": normalized_unit,
                    "grid_point_count": int(np.isfinite(normalized_values).sum()),
                    "bbox_min": float(np.nanmin(normalized_values)),
                    "bbox_p10": float(np.nanpercentile(normalized_values, 10)),
                    "bbox_mean": float(np.nanmean(normalized_values)),
                    "bbox_median": float(np.nanmedian(normalized_values)),
                    "bbox_p90": float(np.nanpercentile(normalized_values, 90)),
                    "bbox_max": float(np.nanmax(normalized_values)),
                    "bbox_std": float(np.nanstd(normalized_values)),
                }
            )

            grid_lat = station_values["nearest_grid_latitude"]
            grid_lon = station_values["nearest_grid_longitude"]
            point_value = da.sel(latitude=grid_lat, longitude=grid_lon).values
            point_native = float(point_value)
            point_normalized, point_unit = normalize_value(var_name, point_native, native_units)
            station_values["issued_at_utc"] = timing["issued_at_utc"]
            station_values["valid_at_utc"] = timing["valid_at_utc"]
            station_values["lead_hour"] = timing["lead_hour"]
            station_values[f"{meta['canonical']}_{point_unit.replace('-', '').replace(' ', '_')}"] = point_normalized

    if "wind_u_10m_m_s1" in station_values and "wind_v_10m_m_s1" in station_values:
        u = float(station_values["wind_u_10m_m_s1"])
        v = float(station_values["wind_v_10m_m_s1"])
        station_values["wind_speed_10m_m_s1"] = math.hypot(u, v)
        station_values["wind_direction_from_deg"] = wind_dir_from_degrees(u, v)
    if "temperature_2m_degC" in station_values and "dewpoint_2m_degC" in station_values:
        station_values["dewpoint_depression_2m_c"] = (
            float(station_values["temperature_2m_degC"]) - float(station_values["dewpoint_2m_degC"])
        )
    if "temperature_2m_max_degC" in station_values and "temperature_2m_degC" in station_values:
        station_values["tmax_minus_instant_t2m_c"] = (
            float(station_values["temperature_2m_max_degC"]) - float(station_values["temperature_2m_degC"])
        )
    if "temperature_2m_max_degC" in station_values and "temperature_2m_min_degC" in station_values:
        station_values["forecast_period_diurnal_range_c"] = (
            float(station_values["temperature_2m_max_degC"]) - float(station_values["temperature_2m_min_degC"])
        )

    grid_df = pd.DataFrame(grid_rows)
    station_df = pd.DataFrame([station_values])
    bbox_summary_df = pd.DataFrame(bbox_summary_rows)

    source_safe = source.replace("_control", "")
    grid_csv = NORMALIZED_DIR / f"{source_safe}_hkg_bbox_grid_long.csv"
    station_csv = NORMALIZED_DIR / f"{source_safe}_target_station_nearest_features.csv"
    bbox_csv = NORMALIZED_DIR / f"{source_safe}_hkg_bbox_summary_features.csv"
    safe_to_csv(grid_df, grid_csv)
    safe_to_parquet(grid_df, grid_csv.with_suffix(".parquet"))
    safe_to_csv(station_df, station_csv)
    safe_to_csv(bbox_summary_df, bbox_csv)

    output = GribSourceOutput(
        source=source,
        input_path=rel_path,
        grid_rows=len(grid_df),
        grid_csv=str(grid_csv.relative_to(EXPERIMENT_DIR)),
        station_feature_csv=str(station_csv.relative_to(EXPERIMENT_DIR)),
        bbox_summary_csv=str(bbox_csv.relative_to(EXPERIMENT_DIR)),
    )
    return grid_df, station_df, bbox_summary_df, output


def read_c_string(data: bytes, offset: int, length: int) -> str:
    return data[offset : offset + length].split(b"\0", 1)[0].decode("ascii", errors="replace").strip()


def parse_himawari_key_fields(file_name: str) -> dict[str, Any]:
    match = re.match(
        r"HS_(H\d{2})_(\d{8})_(\d{4})_(B\d{2})_(\w+)_(R\d+)_(S(\d{2})(\d{2}))\.DAT",
        file_name,
    )
    if not match:
        return {}
    observed = datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return {
        "satellite_code": match.group(1),
        "observed_at_utc": observed.isoformat().replace("+00:00", "Z"),
        "band": match.group(4),
        "area": match.group(5),
        "resolution_code": match.group(6),
        "segment_code": match.group(7),
        "segment_number_from_name": int(match.group(8)),
        "segment_count_from_name": int(match.group(9)),
    }


def parse_himawari_header() -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = EXPERIMENT_DIR / "artifacts" / "fetch_summary.json"
    fetch_summary = json.loads(Path(win_path(summary_path)).read_text(encoding="utf-8"))
    record = next(item for item in fetch_summary["records"] if item["source"] == "himawari9")
    raw_path = EXPERIMENT_DIR / record["output_path"]

    with bz2.open(win_path(raw_path), "rb") as handle:
        prefix = handle.read(20000)

    blocks: list[dict[str, Any]] = []
    offset = 0
    while offset + 3 <= len(prefix):
        block_no = prefix[offset]
        block_length = struct.unpack_from("<H", prefix, offset + 1)[0]
        if not (1 <= block_no <= 11) or block_length <= 0:
            break
        blocks.append(
            {
                "block_number": int(block_no),
                "offset": int(offset),
                "length_bytes": int(block_length),
            }
        )
        offset += block_length
        if len(blocks) >= 11:
            break

    block_offsets = {item["block_number"]: item["offset"] for item in blocks}
    basic_offset = block_offsets.get(1, 0)
    data_offset = block_offsets.get(2, 282)
    projection_offset = block_offsets.get(3, 332)
    calibration_offset = block_offsets.get(5, 598)
    segment_offset = block_offsets.get(7, 1004)

    file_name = read_c_string(prefix, basic_offset + 114, 128)
    key_fields = parse_himawari_key_fields(file_name)
    bits_per_pixel = struct.unpack_from("<H", prefix, data_offset + 3)[0]
    columns = struct.unpack_from("<H", prefix, data_offset + 5)[0]
    lines = struct.unpack_from("<H", prefix, data_offset + 7)[0]
    band_number = struct.unpack_from("<H", prefix, calibration_offset + 3)[0]
    segment_total = prefix[segment_offset + 3]
    segment_number = prefix[segment_offset + 4]

    payload = {
        "source": "himawari9",
        "raw_payload_path": record["output_path"],
        "url": record["url"],
        "sha256": record["sha256"],
        "compressed_bytes": record["bytes"],
        "retrieved_at_utc": record["retrieved_at_utc"],
        "s3_last_modified_from_fetch_notes": record["notes"],
        "header_blocks": blocks,
        "header_total_bytes_by_block_sum": int(offset),
        "next_bytes_after_header_hex": prefix[offset : offset + 16].hex(),
        "satellite_name": read_c_string(prefix, basic_offset + 6, 16),
        "processing_center": read_c_string(prefix, basic_offset + 22, 16),
        "observation_area": read_c_string(prefix, basic_offset + 38, 4),
        "format_version": read_c_string(prefix, basic_offset + 82, 32),
        "file_name": file_name,
        **key_fields,
        "bits_per_pixel": int(bits_per_pixel),
        "columns": int(columns),
        "lines_in_this_segment": int(lines),
        "pixels_in_this_segment": int(columns) * int(lines),
        "uncompressed_image_bytes_estimate_without_line_overhead": int(columns) * int(lines) * int(bits_per_pixel // 8),
        "band_number_from_calibration_block": int(band_number),
        "segment_number_from_segment_block": int(segment_number),
        "segment_count_from_segment_block": int(segment_total),
        "projection": {
            "sub_satellite_longitude_deg": struct.unpack_from("<d", prefix, projection_offset + 3)[0],
            "satellite_distance_km": struct.unpack_from("<d", prefix, projection_offset + 27)[0],
            "earth_equatorial_radius_km": struct.unpack_from("<d", prefix, projection_offset + 35)[0],
            "earth_polar_radius_km": struct.unpack_from("<d", prefix, projection_offset + 43)[0],
            "earth_flattening": struct.unpack_from("<d", prefix, projection_offset + 51)[0],
        },
        "normalization_status": "metadata_decoded_pixel_values_not_calibrated",
        "pixel_decode_next_step": "Decode HSD line records after the 1523-byte header, apply calibration block coefficients, then remap geostationary x/y to lat/lon for HKG cloud-top/brightness-temperature features.",
    }

    block_df = pd.DataFrame(blocks)
    safe_write_json(NORMALIZED_DIR / "himawari_b13_header_summary.json", payload)
    safe_to_csv(block_df, NORMALIZED_DIR / "himawari_b13_header_blocks.csv")
    safe_write_text(
        NORMALIZED_DIR / "himawari_b13_decompressed_prefix_hex.txt",
        " ".join(f"{byte:02x}" for byte in prefix[:512]) + "\n",
    )
    return payload, block_df


def build_comparison_features(station_df: pd.DataFrame, bbox_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = station_df.set_index("source")
    rows: list[dict[str, Any]] = []
    if {"gfs", "gefs_control"}.issubset(set(wide.index)):
        common_cols = [
            col
            for col in wide.columns
            if pd.api.types.is_numeric_dtype(wide[col])
            and col not in {"target_latitude", "target_longitude", "nearest_grid_latitude", "nearest_grid_longitude"}
        ]
        row = {
            "comparison": "gefs_control_minus_gfs_nearest_hko_gridpoint",
            "issued_at_utc": wide.loc["gfs"].get("issued_at_utc"),
            "valid_at_utc": wide.loc["gfs"].get("valid_at_utc"),
        }
        for col in common_cols:
            row[f"{col}_delta"] = float(wide.loc["gefs_control", col]) - float(wide.loc["gfs", col])
        rows.append(row)
    station_delta_df = pd.DataFrame(rows)

    pivot = bbox_df.pivot_table(
        index=["variable", "canonical_variable", "normalized_units"],
        columns="source",
        values=["bbox_mean", "bbox_min", "bbox_max", "bbox_std"],
        aggfunc="first",
    )
    flat_rows: list[dict[str, Any]] = []
    for idx, row in pivot.iterrows():
        out = {
            "variable": idx[0],
            "canonical_variable": idx[1],
            "normalized_units": idx[2],
        }
        for metric in ["bbox_mean", "bbox_min", "bbox_max", "bbox_std"]:
            gfs_value = row.get((metric, "gfs"), np.nan)
            gefs_value = row.get((metric, "gefs_control"), np.nan)
            out[f"gfs_{metric}"] = float(gfs_value) if pd.notna(gfs_value) else None
            out[f"gefs_control_{metric}"] = float(gefs_value) if pd.notna(gefs_value) else None
            out[f"gefs_control_minus_gfs_{metric}"] = (
                float(gefs_value) - float(gfs_value) if pd.notna(gfs_value) and pd.notna(gefs_value) else None
            )
        flat_rows.append(out)
    bbox_delta_df = pd.DataFrame(flat_rows)
    return station_delta_df, bbox_delta_df


def write_readme(manifest: dict[str, Any], snapshot: dict[str, Any]) -> None:
    station = snapshot["station_nearest_features"]
    rows = []
    for source in ["gfs", "gefs_control"]:
        item = station[source]
        rows.append(
            "| {source} | {issued} | {valid} | {tmax:.2f} | {t2m:.2f} | {d2m:.2f} | {wind:.2f} | {prmsl:.2f} |".format(
                source=source,
                issued=item.get("issued_at_utc", ""),
                valid=item.get("valid_at_utc", ""),
                tmax=item.get("temperature_2m_max_degC", float("nan")),
                t2m=item.get("temperature_2m_degC", float("nan")),
                d2m=item.get("dewpoint_2m_degC", float("nan")),
                wind=item.get("wind_speed_10m_m_s1", float("nan")),
                prmsl=item.get("pressure_msl_hPa", float("nan")),
            )
        )

    readme = f"""# Normalized Public GFS/GEFS/Himawari Smoke Data

Generated: `{manifest["generated_at_utc"]}`

This folder converts the provider-native files under `raw/` into readable HKG-focused artifacts.

## Fastest files to open

| File | Use |
|---|---|
| `feature_snapshot.json` | Compact one-file summary for humans and code. |
| `model_target_station_features.csv` | One row per model source at the nearest grid point to the canonical HKO target station. |
| `model_source_comparison_features.csv` | Direct GEFS-control minus GFS deltas. |
| `gfs_hkg_bbox_grid_long.csv` / `gefs_hkg_bbox_grid_long.csv` | Long-form cropped HKG grid values. |
| `himawari_b13_header_summary.json` | Decoded Himawari Standard Data metadata for the B13 segment. |

## Nearest-HKO Model Features

| Source | issuedAt UTC | validAt UTC | tmax C | t2m C | dewpoint C | wind10 m/s | MSLP hPa |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Scope

- GFS/GEFS are decoded from GRIB2 with `cfgrib/eccodes`, cropped to `{HKG_BBOX["min_lat"]}-{HKG_BBOX["max_lat"]}N`, `{HKG_BBOX["min_lon"]}-{HKG_BBOX["max_lon"]}E`.
- The target station is `{TARGET_STATION["station_name"]}` from `{TARGET_STATION["metadata_source"]}` at `{TARGET_STATION["latitude"]}`, `{TARGET_STATION["longitude"]}`.
- Temperatures are normalized from Kelvin to Celsius; pressure from Pa to hPa; wind remains m/s.
- Himawari is decoded to Standard Data header metadata. Pixel-value calibration/remapping is explicitly marked as not completed here.
"""
    safe_write_text(NORMALIZED_DIR / "README.md", readme)


def update_experiment_docs(manifest: dict[str, Any]) -> None:
    note = f"""## Normalized Outputs

Normalized, readable outputs were generated at `{manifest["generated_at_utc"]}` under `normalized/`.

- `normalized/README.md` is the human entrypoint.
- `normalized/model_target_station_features.csv` gives nearest-HKO model features.
- `normalized/model_source_comparison_features.csv` gives GEFS-control minus GFS deltas.
- `normalized/hkg_bbox_grid_long_all_sources.csv` gives all cropped model grid rows in long form.
- `normalized/himawari_b13_header_summary.json` decodes Himawari B13 metadata and records that pixel calibration is still separate work.
"""
    for name in ["README.md", "RESULTS.md"]:
        path = EXPERIMENT_DIR / name
        existing = Path(win_path(path)).read_text(encoding="utf-8")
        marker = "## Normalized Outputs"
        if marker in existing:
            existing = existing[: existing.index(marker)].rstrip()
        safe_write_text(path, existing.rstrip() + "\n\n" + note)


def main() -> int:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    all_grid: list[pd.DataFrame] = []
    all_station: list[pd.DataFrame] = []
    all_bbox: list[pd.DataFrame] = []
    source_outputs: list[GribSourceOutput] = []

    for source, rel_path in SOURCE_FILES.items():
        grid_df, station_df, bbox_df, source_output = normalize_grib_source(source, rel_path)
        all_grid.append(grid_df)
        all_station.append(station_df)
        all_bbox.append(bbox_df)
        source_outputs.append(source_output)

    grid_all_df = pd.concat(all_grid, ignore_index=True)
    station_all_df = pd.concat(all_station, ignore_index=True)
    bbox_all_df = pd.concat(all_bbox, ignore_index=True)

    safe_to_csv(grid_all_df, NORMALIZED_DIR / "hkg_bbox_grid_long_all_sources.csv")
    safe_to_parquet(grid_all_df, NORMALIZED_DIR / "hkg_bbox_grid_long_all_sources.parquet")
    safe_to_csv(station_all_df, NORMALIZED_DIR / "model_target_station_features.csv")
    safe_to_csv(bbox_all_df, NORMALIZED_DIR / "model_hkg_bbox_summary_features.csv")

    station_delta_df, bbox_delta_df = build_comparison_features(station_all_df, bbox_all_df)
    safe_to_csv(station_delta_df, NORMALIZED_DIR / "model_source_comparison_features.csv")
    safe_to_csv(bbox_delta_df, NORMALIZED_DIR / "model_bbox_comparison_features.csv")

    himawari_summary, _block_df = parse_himawari_header()

    snapshot = {
        "generated_at_utc": utc_now_iso(),
        "target_station": TARGET_STATION,
        "hkg_bbox": HKG_BBOX,
        "station_nearest_features": {
            row["source"]: {
                key: (None if pd.isna(value) else value)
                for key, value in row.items()
            }
            for row in station_all_df.to_dict(orient="records")
        },
        "source_comparison": station_delta_df.to_dict(orient="records"),
        "bbox_comparison": bbox_delta_df.to_dict(orient="records"),
        "himawari9": himawari_summary,
    }
    safe_write_json(NORMALIZED_DIR / "feature_snapshot.json", snapshot)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now_iso(),
        "normalization_status": "complete_for_gfs_gefs_model_fields_partial_metadata_for_himawari",
        "decoder": {
            "grib": "cfgrib/eccodes",
            "himawari": "custom Standard Data header parser; pixel values not calibrated in this smoke",
        },
        "source_outputs": [asdict(item) for item in source_outputs],
        "combined_outputs": {
            "grid_all_csv": "normalized/hkg_bbox_grid_long_all_sources.csv",
            "grid_all_parquet": "normalized/hkg_bbox_grid_long_all_sources.parquet",
            "station_features_csv": "normalized/model_target_station_features.csv",
            "bbox_summary_csv": "normalized/model_hkg_bbox_summary_features.csv",
            "station_source_comparison_csv": "normalized/model_source_comparison_features.csv",
            "bbox_source_comparison_csv": "normalized/model_bbox_comparison_features.csv",
            "snapshot_json": "normalized/feature_snapshot.json",
            "himawari_header_summary_json": "normalized/himawari_b13_header_summary.json",
        },
        "row_counts": {
            "grid_all": int(len(grid_all_df)),
            "station_features": int(len(station_all_df)),
            "bbox_summary": int(len(bbox_all_df)),
            "station_source_comparison": int(len(station_delta_df)),
            "bbox_source_comparison": int(len(bbox_delta_df)),
        },
    }
    safe_write_json(NORMALIZED_DIR / "normalization_manifest.json", manifest)
    write_readme(manifest, snapshot)
    update_experiment_docs(manifest)
    print(NORMALIZED_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
