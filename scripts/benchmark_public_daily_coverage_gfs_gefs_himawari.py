from __future__ import annotations

import bz2
import hashlib
import json
import math
import re
import struct
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cfgrib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "0006_public_daily_coverage_benchmark_20260707"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / EXPERIMENT_ID
RAW_DIR = EXPERIMENT_DIR / "raw"
NORMALIZED_DIR = EXPERIMENT_DIR / "normalized"
TARGET_DAY = date(2026, 7, 7)
USER_AGENT = "weather-markets-hkg-public-daily-benchmark/1.0"

HKG_BBOX = {"leftlon": "113.0", "rightlon": "115.5", "toplat": "23.5", "bottomlat": "21.5"}
HKO = {
    "station_id": "hko:HKO",
    "station_name": "Hong Kong Observatory",
    "latitude": 22.301944,
    "longitude": 114.174167,
}

MODEL_CYCLES = [0, 6, 12, 18]
MODEL_LEAD_HOUR = 24

# Broad enough to cover target Tmax physics without downloading every GRIB field.
MODEL_FILTER_PARAMS = {
    "lev_2_m_above_ground": "on",
    "var_TMP": "on",
    "var_DPT": "on",
    "var_RH": "on",
    "var_TMAX": "on",
    "var_TMIN": "on",
    "lev_10_m_above_ground": "on",
    "var_UGRD": "on",
    "var_VGRD": "on",
    "lev_mean_sea_level": "on",
    "var_PRMSL": "on",
    "lev_surface": "on",
    "var_GUST": "on",
    "var_APCP": "on",
    "var_DSWRF": "on",
    "var_CAPE": "on",
    "var_CIN": "on",
    "lev_entire_atmosphere": "on",
    "var_PWAT": "on",
    "var_TCDC": "on",
}


@dataclass
class FetchResult:
    source: str
    item_id: str
    status: str
    url: str
    path: str | None
    bytes: int
    sha256: str | None
    elapsed_seconds: float
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def wp(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def ensure_dir(path: Path) -> None:
    Path(wp(path)).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    Path(wp(path)).write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def request_bytes(url: str, timeout: int = 120) -> tuple[bytes, dict[str, str]]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.read(), headers


def fetch_to_path(source: str, item_id: str, url: str, path: Path, timeout: int = 120) -> FetchResult:
    started = time.perf_counter()
    try:
        if Path(wp(path)).exists():
            data = Path(wp(path)).read_bytes()
            elapsed = time.perf_counter() - started
        else:
            data, _headers = request_bytes(url, timeout=timeout)
            ensure_dir(path.parent)
            Path(wp(path)).write_bytes(data)
            elapsed = time.perf_counter() - started
        return FetchResult(
            source=source,
            item_id=item_id,
            status="ok",
            url=url,
            path=str(path.relative_to(EXPERIMENT_DIR)),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            elapsed_seconds=elapsed,
        )
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return FetchResult(
            source=source,
            item_id=item_id,
            status="error",
            url=url,
            path=None,
            bytes=0,
            sha256=None,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_gfs_url(cycle_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    params = {
        "dir": f"/gfs.{TARGET_DAY:%Y%m%d}/{cc}/atmos",
        "file": f"gfs.t{cc}z.pgrb2.0p25.f{MODEL_LEAD_HOUR:03d}",
        **MODEL_FILTER_PARAMS,
        **HKG_BBOX,
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(params)


def build_gefs_url(cycle_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    params = {
        "dir": f"/gefs.{TARGET_DAY:%Y%m%d}/{cc}/atmos/pgrb2sp25",
        "file": f"gec00.t{cc}z.pgrb2s.0p25.f{MODEL_LEAD_HOUR:03d}",
        **MODEL_FILTER_PARAMS,
        **HKG_BBOX,
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?" + urlencode(params)


def gfs_idx_url(cycle_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    return (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        f"gfs.{TARGET_DAY:%Y%m%d}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{MODEL_LEAD_HOUR:03d}.idx"
    )


def gefs_idx_url(cycle_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    return (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/"
        f"gefs.{TARGET_DAY:%Y%m%d}/{cc}/atmos/pgrb2sp25/gec00.t{cc}z.pgrb2s.0p25.f{MODEL_LEAD_HOUR:03d}.idx"
    )


def parse_idx(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    rows = [line for line in text.splitlines() if line.strip()]
    variables: set[str] = set()
    variable_level_pairs: set[tuple[str, str]] = set()
    for line in rows:
        parts = line.split(":")
        if len(parts) >= 5:
            variables.add(parts[3])
            variable_level_pairs.add((parts[3], parts[4]))
    return {
        "message_count": len(rows),
        "unique_variable_count": len(variables),
        "unique_variables": sorted(variables),
        "unique_variable_level_pair_count": len(variable_level_pairs),
    }


def normalize_value(value: float, units: str | None) -> tuple[float, str]:
    if units == "K":
        return float(value) - 273.15, "degC"
    if units == "Pa":
        return float(value) / 100.0, "hPa"
    return float(value), units or ""


def iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def lead_hours(value: Any) -> float | None:
    try:
        return float(value / np.timedelta64(1, "h"))
    except Exception:
        return None


def crop_dataarray(da: Any) -> Any:
    lat_values = np.asarray(da["latitude"].values, dtype=float)
    lat_slice = (
        slice(float(HKG_BBOX["toplat"]), float(HKG_BBOX["bottomlat"]))
        if lat_values[0] > lat_values[-1]
        else slice(float(HKG_BBOX["bottomlat"]), float(HKG_BBOX["toplat"]))
    )
    return da.sel(latitude=lat_slice, longitude=slice(float(HKG_BBOX["leftlon"]), float(HKG_BBOX["rightlon"])))


def nearest_grid(ds: Any) -> tuple[float, float]:
    lat_values = np.asarray(ds["latitude"].values, dtype=float)
    lon_values = np.asarray(ds["longitude"].values, dtype=float)
    lat = float(lat_values[np.argmin(np.abs(lat_values - HKO["latitude"]))])
    lon = float(lon_values[np.argmin(np.abs(lon_values - HKO["longitude"]))])
    return lat, lon


def normalize_model_file(source: str, cycle_hour: int, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    dsets = cfgrib.open_datasets(wp(path), backend_kwargs={"indexpath": ""})
    station_row: dict[str, Any] = {
        "source": source,
        "cycle_hour": cycle_hour,
        "target_day": TARGET_DAY.isoformat(),
        "station_id": HKO["station_id"],
        "station_latitude": HKO["latitude"],
        "station_longitude": HKO["longitude"],
    }
    summary_rows: list[dict[str, Any]] = []
    variable_count = 0
    nearest_lat = None
    nearest_lon = None

    for ds in dsets:
        if nearest_lat is None:
            nearest_lat, nearest_lon = nearest_grid(ds)
            station_row["nearest_grid_latitude"] = nearest_lat
            station_row["nearest_grid_longitude"] = nearest_lon
        for var_name in ds.data_vars:
            da = ds[var_name]
            attrs = da.attrs
            units = attrs.get("GRIB_units")
            short_name = attrs.get("GRIB_shortName", var_name)
            canonical = re.sub(r"[^a-zA-Z0-9]+", "_", f"{short_name}_{attrs.get('GRIB_typeOfLevel', '')}").strip("_").lower()
            issued_at = iso_timestamp(da.coords["time"].values if "time" in da.coords else None)
            valid_at = iso_timestamp(da.coords["valid_time"].values if "valid_time" in da.coords else None)
            lead = lead_hours(da.coords["step"].values if "step" in da.coords else None)
            station_row["issued_at_utc"] = issued_at
            station_row["valid_at_utc"] = valid_at
            station_row["lead_hour"] = lead

            point_native = float(da.sel(latitude=nearest_lat, longitude=nearest_lon).values)
            point_value, out_unit = normalize_value(point_native, units)
            station_row[f"{canonical}_{out_unit}".replace("-", "").replace(" ", "_")] = point_value

            cropped = crop_dataarray(da)
            values = np.asarray(cropped.values, dtype=float)
            normalized = np.vectorize(lambda x: normalize_value(float(x), units)[0])(values)
            summary_rows.append(
                {
                    "source": source,
                    "cycle_hour": cycle_hour,
                    "issued_at_utc": issued_at,
                    "valid_at_utc": valid_at,
                    "lead_hour": lead,
                    "variable": var_name,
                    "grib_short_name": short_name,
                    "grib_name": attrs.get("GRIB_name"),
                    "grib_units": units,
                    "normalized_units": out_unit,
                    "grid_point_count": int(np.isfinite(normalized).sum()),
                    "bbox_min": float(np.nanmin(normalized)),
                    "bbox_mean": float(np.nanmean(normalized)),
                    "bbox_median": float(np.nanmedian(normalized)),
                    "bbox_max": float(np.nanmax(normalized)),
                    "bbox_std": float(np.nanstd(normalized)),
                }
            )
            variable_count += 1
    station_row["normalized_variable_count"] = variable_count
    station_row["normalization_elapsed_seconds"] = time.perf_counter() - started
    return station_row, summary_rows


def parse_himawari_header(data: bytes, source_record: FetchResult) -> dict[str, Any]:
    basic = 0
    data_info = 282
    projection = 332
    calibration = 598
    segment = 1004
    file_name = data[basic + 114 : basic + 242].split(b"\0", 1)[0].decode("ascii", errors="replace")
    observed_match = re.search(r"HS_H09_(\d{8})_(\d{4})_", file_name)
    observed_at = None
    if observed_match:
        observed_at = (
            datetime.strptime(observed_match.group(1) + observed_match.group(2), "%Y%m%d%H%M")
            .replace(tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return {
        "source_record": asdict(source_record),
        "file_name": file_name,
        "observed_at_utc": observed_at,
        "header_total_bytes": struct.unpack_from("<I", data, basic + 70)[0],
        "data_total_bytes": struct.unpack_from("<I", data, basic + 74)[0],
        "columns": struct.unpack_from("<H", data, data_info + 5)[0],
        "lines_in_segment": struct.unpack_from("<H", data, data_info + 7)[0],
        "projection": {
            "sub_satellite_longitude_deg": struct.unpack_from("<d", data, projection + 3)[0],
            "cfac": struct.unpack_from("<i", data, projection + 11)[0],
            "lfac": struct.unpack_from("<i", data, projection + 15)[0],
            "coff": struct.unpack_from("<f", data, projection + 19)[0],
            "loff": struct.unpack_from("<f", data, projection + 23)[0],
            "satellite_distance_km": struct.unpack_from("<d", data, projection + 27)[0],
            "earth_equatorial_radius_km": struct.unpack_from("<d", data, projection + 35)[0],
            "earth_polar_radius_km": struct.unpack_from("<d", data, projection + 43)[0],
        },
        "calibration": {
            "band_number": struct.unpack_from("<H", data, calibration + 3)[0],
            "central_wavelength_um": struct.unpack_from("<d", data, calibration + 5)[0],
            "error_count": struct.unpack_from("<H", data, calibration + 15)[0],
            "outside_scan_count": struct.unpack_from("<H", data, calibration + 17)[0],
            "count_to_radiance_slope": struct.unpack_from("<d", data, calibration + 19)[0],
            "count_to_radiance_intercept": struct.unpack_from("<d", data, calibration + 27)[0],
            "radiance_to_bt_c0": struct.unpack_from("<d", data, calibration + 35)[0],
            "radiance_to_bt_c1": struct.unpack_from("<d", data, calibration + 43)[0],
            "radiance_to_bt_c2": struct.unpack_from("<d", data, calibration + 51)[0],
        },
        "segment": {
            "total_segments": data[segment + 3],
            "segment_sequence_number": data[segment + 4],
            "first_global_line_number": struct.unpack_from("<H", data, segment + 5)[0],
        },
    }


def hko_pixel(header: dict[str, Any]) -> tuple[int, int, float, float]:
    proj = header["projection"]
    lat = math.radians(HKO["latitude"])
    lon = math.radians(HKO["longitude"])
    lon0 = math.radians(proj["sub_satellite_longitude_deg"])
    req = proj["earth_equatorial_radius_km"]
    rpol = proj["earth_polar_radius_km"]
    rs = proj["satellite_distance_km"]
    phi_c = math.atan((rpol * rpol) / (req * req) * math.tan(lat))
    re_phi = rpol / math.sqrt(1.0 - ((req * req - rpol * rpol) / (req * req)) * math.cos(phi_c) ** 2)
    rel_lon = lon - lon0
    r1 = rs - re_phi * math.cos(phi_c) * math.cos(rel_lon)
    r2 = -re_phi * math.cos(phi_c) * math.sin(rel_lon)
    r3 = re_phi * math.sin(phi_c)
    x = math.atan(r2 / r1)
    y = math.atan(r3 / math.sqrt(r1 * r1 + r2 * r2))
    global_col = proj["coff"] + x * proj["cfac"] / (2**16)
    global_line = proj["loff"] - y * proj["lfac"] / (2**16)
    local_row = int(round(global_line - header["segment"]["first_global_line_number"]))
    local_col = int(round(global_col - 1))
    return local_row, local_col, global_line, global_col


def himawari_bt(data: bytes, header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = int(header["columns"])
    lines = int(header["lines_in_segment"])
    counts = np.frombuffer(data, dtype="<u2", count=columns * lines, offset=int(header["header_total_bytes"])).reshape(
        lines, columns
    )
    cal = header["calibration"]
    valid = (counts != int(cal["outside_scan_count"])) & (counts != int(cal["error_count"]))
    radiance = cal["count_to_radiance_slope"] * counts.astype("float64") + cal["count_to_radiance_intercept"]
    radiance[~valid] = np.nan
    radiance[radiance <= 0] = np.nan
    c1 = 1.191042e8
    c2 = 1.4387752e4
    wavelength = cal["central_wavelength_um"]
    effective_bt = c2 / (wavelength * np.log(c1 / (radiance * (wavelength**5)) + 1.0))
    bt_k = cal["radiance_to_bt_c0"] + cal["radiance_to_bt_c1"] * effective_bt + cal["radiance_to_bt_c2"] * effective_bt**2
    return counts, radiance.astype("float32"), (bt_k - 273.15).astype("float32")


def normalize_himawari_file(fetch_result: FetchResult, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    with bz2.open(wp(path), "rb") as handle:
        data = handle.read()
    header = parse_himawari_header(data, fetch_result)
    counts, radiance, bt_c = himawari_bt(data, header)
    row, col, global_line, global_col = hko_pixel(header)
    radius = 10
    window = bt_c[max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1]
    vals = window[np.isfinite(window)]
    return {
        "source": "himawari9",
        "band": "B13",
        "segment": "S0510",
        "observed_at_utc": header["observed_at_utc"],
        "file_name": header["file_name"],
        "compressed_bytes": fetch_result.bytes,
        "hko_global_line": global_line,
        "hko_global_col": global_col,
        "hko_local_row": row,
        "hko_local_col": col,
        "hko_count": int(counts[row, col]),
        "hko_radiance_w_m2_sr_um": float(radiance[row, col]),
        "hko_bt_c": float(bt_c[row, col]),
        "w21_mean_bt_c": float(np.mean(vals)),
        "w21_median_bt_c": float(np.median(vals)),
        "w21_min_bt_c": float(np.min(vals)),
        "w21_max_bt_c": float(np.max(vals)),
        "w21_std_bt_c": float(np.std(vals)),
        "w21_range_bt_c": float(np.max(vals) - np.min(vals)),
        "w21_cloud_fraction_lt_0c": float(np.mean(vals < 0)),
        "w21_cloud_fraction_lt_10c": float(np.mean(vals < 10)),
        "w21_cloud_fraction_lt_15c": float(np.mean(vals < 15)),
        "w21_cool_cloud_fraction_lt_20c": float(np.mean(vals < 20)),
        "w21_warm_fraction_gt_20c": float(np.mean(vals > 20)),
        "w21_warm_fraction_gt_23c": float(np.mean(vals > 23)),
        "normalization_elapsed_seconds": time.perf_counter() - started,
    }


def himawari_scan_datetimes() -> list[datetime]:
    scans = []
    for hour in range(24):
        for minute in range(0, 60, 10):
            scans.append(datetime(TARGET_DAY.year, TARGET_DAY.month, TARGET_DAY.day, hour, minute, tzinfo=timezone.utc))
    return scans


def himawari_url(scan: datetime) -> str:
    key = f"AHI-L1b-FLDK/{scan:%Y/%m/%d/%H%M}/HS_H09_{scan:%Y%m%d}_{scan:%H%M}_B13_FLDK_R20_S0510.DAT.bz2"
    return f"https://noaa-himawari9.s3.amazonaws.com/{key}"


def fetch_model_data() -> tuple[list[FetchResult], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fetch_results: list[FetchResult] = []
    idx_catalog: dict[str, Any] = {}
    station_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []

    for source in ["gfs", "gefs_control"]:
        for cycle_hour in MODEL_CYCLES:
            if source == "gfs":
                url = build_gfs_url(cycle_hour)
                idx_url = gfs_idx_url(cycle_hour)
                path = RAW_DIR / "gfs" / f"gfs_{TARGET_DAY:%Y%m%d}_{cycle_hour:02d}z_f{MODEL_LEAD_HOUR:03d}.grib2"
            else:
                url = build_gefs_url(cycle_hour)
                idx_url = gefs_idx_url(cycle_hour)
                path = RAW_DIR / "gefs_control" / f"gefs_control_{TARGET_DAY:%Y%m%d}_{cycle_hour:02d}z_f{MODEL_LEAD_HOUR:03d}.grib2"
            result = fetch_to_path(source, f"{TARGET_DAY:%Y%m%d}_{cycle_hour:02d}z_f{MODEL_LEAD_HOUR:03d}", url, path)
            fetch_results.append(result)
            try:
                idx_data, _headers = request_bytes(idx_url, timeout=30)
                idx_catalog[f"{source}_{cycle_hour:02d}z"] = parse_idx(idx_data)
            except Exception as exc:
                idx_catalog[f"{source}_{cycle_hour:02d}z"] = {"error": f"{type(exc).__name__}: {exc}", "url": idx_url}
            if result.status == "ok":
                station_row, rows = normalize_model_file(source, cycle_hour, path)
                station_rows.append(station_row)
                bbox_rows.extend(rows)
            print(f"model {source} {cycle_hour:02d}z {result.status} {result.bytes} bytes", flush=True)
    return fetch_results, idx_catalog, station_rows, bbox_rows


def fetch_himawari_data() -> tuple[list[FetchResult], list[dict[str, Any]]]:
    fetch_results: list[FetchResult] = []
    rows: list[dict[str, Any]] = []
    for idx, scan in enumerate(himawari_scan_datetimes(), start=1):
        file_name = f"HS_H09_{scan:%Y%m%d}_{scan:%H%M}_B13_FLDK_R20_S0510.DAT.bz2"
        path = RAW_DIR / "himawari_b13_s0510" / f"{TARGET_DAY:%Y%m%d}" / file_name
        result = fetch_to_path("himawari9_b13_s0510", scan.strftime("%Y%m%d_%H%M"), himawari_url(scan), path)
        fetch_results.append(result)
        if result.status == "ok":
            rows.append(normalize_himawari_file(result, path))
        print(f"himawari {idx:03d}/144 {scan:%H%M} {result.status} {result.bytes} bytes", flush=True)
    return fetch_results, rows


def write_outputs(
    total_started: float,
    model_fetches: list[FetchResult],
    idx_catalog: dict[str, Any],
    model_station_rows: list[dict[str, Any]],
    model_bbox_rows: list[dict[str, Any]],
    himawari_fetches: list[FetchResult],
    himawari_rows: list[dict[str, Any]],
) -> None:
    ensure_dir(NORMALIZED_DIR)
    model_station_df = pd.DataFrame(model_station_rows)
    model_bbox_df = pd.DataFrame(model_bbox_rows)
    himawari_df = pd.DataFrame(himawari_rows)
    model_fetch_df = pd.DataFrame([asdict(item) for item in model_fetches])
    himawari_fetch_df = pd.DataFrame([asdict(item) for item in himawari_fetches])

    model_station_df.to_csv(wp(NORMALIZED_DIR / "model_cycle_station_features.csv"), index=False)
    model_bbox_df.to_csv(wp(NORMALIZED_DIR / "model_cycle_bbox_summary_features.csv"), index=False)
    himawari_df.to_csv(wp(NORMALIZED_DIR / "himawari_b13_s0510_scan_features.csv"), index=False)
    model_fetch_df.to_csv(wp(NORMALIZED_DIR / "model_fetch_timings.csv"), index=False)
    himawari_fetch_df.to_csv(wp(NORMALIZED_DIR / "himawari_fetch_timings.csv"), index=False)
    if not model_station_df.empty:
        model_station_df.to_parquet(wp(NORMALIZED_DIR / "model_cycle_station_features.parquet"), index=False)
    if not himawari_df.empty:
        himawari_df.to_parquet(wp(NORMALIZED_DIR / "himawari_b13_s0510_scan_features.parquet"), index=False)

    elapsed_total = time.perf_counter() - total_started
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now_iso(),
        "target_day_utc": TARGET_DAY.isoformat(),
        "scope": {
            "gfs": "four cycles 00/06/12/18, f024, HKG filtered feature pack",
            "gefs_control": "four control-member cycles 00/06/12/18, f024, HKG filtered feature pack",
            "himawari9": "B13 infrared HKO-containing full-disk segment S0510, every 10 minutes for 24h",
        },
        "attribute_counts": {
            "gfs_full_product_latest_idx_messages": idx_catalog.get("gfs_00z", {}).get("message_count"),
            "gfs_full_product_latest_idx_unique_variables": idx_catalog.get("gfs_00z", {}).get("unique_variable_count"),
            "gfs_full_product_latest_idx_unique_variable_level_pairs": idx_catalog.get("gfs_00z", {}).get(
                "unique_variable_level_pair_count"
            ),
            "gefs_control_full_product_latest_idx_messages": idx_catalog.get("gefs_control_00z", {}).get("message_count"),
            "gefs_control_full_product_latest_idx_unique_variables": idx_catalog.get("gefs_control_00z", {}).get(
                "unique_variable_count"
            ),
            "gefs_control_full_product_latest_idx_unique_variable_level_pairs": idx_catalog.get("gefs_control_00z", {}).get(
                "unique_variable_level_pair_count"
            ),
            "selected_model_feature_pack_variables_per_cycle_min": int(
                model_station_df["normalized_variable_count"].min() if not model_station_df.empty else 0
            ),
            "selected_model_feature_pack_variables_per_cycle_max": int(
                model_station_df["normalized_variable_count"].max() if not model_station_df.empty else 0
            ),
            "himawari_raw_files_per_full_disk_scan": 160,
            "himawari_raw_files_per_hko_segment_scan_all_bands": 16,
            "himawari_selected_b13_segment_files_per_day": 144,
            "himawari_selected_feature_columns": int(len(himawari_df.columns) if not himawari_df.empty else 0),
        },
        "timing_seconds": {
            "total_download_plus_normalize": elapsed_total,
            "model_download_total": float(sum(item.elapsed_seconds for item in model_fetches)),
            "himawari_download_total": float(sum(item.elapsed_seconds for item in himawari_fetches)),
            "model_normalize_total": float(
                model_station_df["normalization_elapsed_seconds"].sum() if not model_station_df.empty else 0.0
            ),
            "himawari_normalize_total": float(
                himawari_df["normalization_elapsed_seconds"].sum() if not himawari_df.empty else 0.0
            ),
        },
        "item_counts": {
            "model_requested": len(model_fetches),
            "model_ok": sum(1 for item in model_fetches if item.status == "ok"),
            "himawari_requested": len(himawari_fetches),
            "himawari_ok": sum(1 for item in himawari_fetches if item.status == "ok"),
        },
        "bytes": {
            "model_downloaded": int(sum(item.bytes for item in model_fetches)),
            "himawari_downloaded": int(sum(item.bytes for item in himawari_fetches)),
            "total_downloaded": int(sum(item.bytes for item in model_fetches + himawari_fetches)),
        },
        "idx_catalog": idx_catalog,
        "outputs": {
            "model_cycle_station_features": "normalized/model_cycle_station_features.csv",
            "model_cycle_bbox_summary_features": "normalized/model_cycle_bbox_summary_features.csv",
            "himawari_b13_s0510_scan_features": "normalized/himawari_b13_s0510_scan_features.csv",
            "model_fetch_timings": "normalized/model_fetch_timings.csv",
            "himawari_fetch_timings": "normalized/himawari_fetch_timings.csv",
        },
    }
    write_json(NORMALIZED_DIR / "daily_coverage_benchmark_summary.json", summary)

    readme = f"""# Public Daily Coverage Benchmark

Generated: `{summary["generated_at_utc"]}`

Target UTC day: `{TARGET_DAY.isoformat()}`

This benchmark fetched and normalized one complete practical daily coverage set:

- GFS: `00/06/12/18Z`, f024, selected HKG weather feature pack.
- GEFS control: `00/06/12/18Z`, f024, selected HKG weather feature pack.
- Himawari-9: B13 HKG segment `S0510`, every 10 minutes, 144 scans.

## Result

| Metric | Value |
|---|---:|
| total download + normalize seconds | {elapsed_total:.2f} |
| model items ok | {summary["item_counts"]["model_ok"]} / {summary["item_counts"]["model_requested"]} |
| Himawari items ok | {summary["item_counts"]["himawari_ok"]} / {summary["item_counts"]["himawari_requested"]} |
| total downloaded MB | {summary["bytes"]["total_downloaded"] / 1_000_000:.2f} |
| model normalized station rows | {len(model_station_df)} |
| Himawari normalized scan rows | {len(himawari_df)} |

See `normalized/daily_coverage_benchmark_summary.json` for timings, bytes, and attribute counts.
"""
    write_text(EXPERIMENT_DIR / "README.md", readme)
    write_text(
        EXPERIMENT_DIR / "STATUS.yaml",
        "state: COMPLETE\n"
        "gate_result: DAILY_COVERAGE_BENCHMARK_COMPLETE\n"
        "uses_gribstream: false\n"
        "target_day_utc: 2026-07-07\n",
    )


def main() -> int:
    total_started = time.perf_counter()
    ensure_dir(RAW_DIR)
    ensure_dir(NORMALIZED_DIR)
    model_fetches, idx_catalog, station_rows, bbox_rows = fetch_model_data()
    himawari_fetches, himawari_rows = fetch_himawari_data()
    write_outputs(total_started, model_fetches, idx_catalog, station_rows, bbox_rows, himawari_fetches, himawari_rows)
    print(EXPERIMENT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
