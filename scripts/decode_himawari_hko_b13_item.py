from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import math
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "0005_public_gfs_gefs_himawari_fetch_smoke_20260708"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / EXPERIMENT_ID
RAW_DIR = EXPERIMENT_DIR / "raw" / "himawari_hko_candidate"
OUT_DIR = EXPERIMENT_DIR / "normalized" / "himawari_hko_b13_s0510_item"

SEGMENT = "0510"
FILE_NAME = f"HS_H09_20260708_0620_B13_FLDK_R20_S{SEGMENT}.DAT.bz2"
URL = f"https://noaa-himawari9.s3.amazonaws.com/AHI-L1b-FLDK/2026/07/08/0620/{FILE_NAME}"

HKO = {
    "station_id": "hko:HKO",
    "station_name": "Hong Kong Observatory",
    "latitude": 22.301944,
    "longitude": 114.174167,
    "metadata_source": "config/stations_hko.yaml target_station",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def wp(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def write_text(path: Path, text: str) -> None:
    Path(wp(path.parent)).mkdir(parents=True, exist_ok=True)
    Path(wp(path)).write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def read_c_string(data: bytes, offset: int, length: int) -> str:
    return data[offset : offset + length].split(b"\0", 1)[0].decode("ascii", errors="replace").strip()


def fetch_if_missing(path: Path) -> dict[str, Any]:
    Path(wp(path.parent)).mkdir(parents=True, exist_ok=True)
    if not Path(wp(path)).exists():
        req = Request(URL, headers={"User-Agent": "weather-markets-hkg-himawari-item/1.0"})
        with urlopen(req, timeout=120) as response:
            payload = response.read()
            headers = {key.lower(): value for key, value in response.headers.items()}
        Path(wp(path)).write_bytes(payload)
        fetched = True
    else:
        payload = Path(wp(path)).read_bytes()
        headers = {}
        fetched = False
    return {
        "url": URL,
        "path": str(path.relative_to(EXPERIMENT_DIR)),
        "fetched_now": fetched,
        "compressed_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_type": headers.get("content-type"),
    }


def parse_file_name(file_name: str) -> dict[str, Any]:
    match = re.match(
        r"HS_(H\d{2})_(\d{8})_(\d{4})_(B\d{2})_(\w+)_(R\d+)_(S(\d{2})(\d{2}))\.DAT",
        file_name.replace(".bz2", ""),
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
        "segment_number": int(match.group(8)),
        "segment_count": int(match.group(9)),
    }


def mjd_to_iso(mjd: float) -> str | None:
    if not np.isfinite(mjd):
        return None
    # MJD day zero is 1858-11-17 00:00 UTC.
    ts = pd.Timestamp("1858-11-17T00:00:00Z") + pd.to_timedelta(mjd, unit="D")
    return ts.isoformat().replace("+00:00", "Z")


def parse_header(data: bytes, source_record: dict[str, Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    offset = 0
    while offset + 3 <= len(data):
        block_no = data[offset]
        block_length = struct.unpack_from("<H", data, offset + 1)[0]
        if not (1 <= block_no <= 11) or block_length <= 0:
            break
        blocks.append({"block_number": int(block_no), "offset": int(offset), "length_bytes": int(block_length)})
        offset += block_length
        if len(blocks) == 11:
            break

    basic = 0
    data_info = 282
    projection = 332
    calibration = 598
    segment = 1004
    observation_time = 1132

    hsd_file_name = read_c_string(data, basic + 114, 128)
    calibration_values = {
        "band_number": struct.unpack_from("<H", data, calibration + 3)[0],
        "central_wavelength_um": struct.unpack_from("<d", data, calibration + 5)[0],
        "valid_bits_per_pixel": struct.unpack_from("<H", data, calibration + 13)[0],
        "error_count": struct.unpack_from("<H", data, calibration + 15)[0],
        "outside_scan_count": struct.unpack_from("<H", data, calibration + 17)[0],
        "count_to_radiance_slope": struct.unpack_from("<d", data, calibration + 19)[0],
        "count_to_radiance_intercept": struct.unpack_from("<d", data, calibration + 27)[0],
        "radiance_to_bt_c0": struct.unpack_from("<d", data, calibration + 35)[0],
        "radiance_to_bt_c1": struct.unpack_from("<d", data, calibration + 43)[0],
        "radiance_to_bt_c2": struct.unpack_from("<d", data, calibration + 51)[0],
        "bt_to_radiance_C0": struct.unpack_from("<d", data, calibration + 59)[0],
        "bt_to_radiance_C1": struct.unpack_from("<d", data, calibration + 67)[0],
        "bt_to_radiance_C2": struct.unpack_from("<d", data, calibration + 75)[0],
        "speed_of_light_m_s": struct.unpack_from("<d", data, calibration + 83)[0],
        "planck_constant_j_s": struct.unpack_from("<d", data, calibration + 91)[0],
        "boltzmann_constant_j_k": struct.unpack_from("<d", data, calibration + 99)[0],
    }

    number_of_observation_times = struct.unpack_from("<H", data, observation_time + 3)[0]
    observation_times: list[dict[str, Any]] = []
    cursor = observation_time + 5
    for _ in range(number_of_observation_times):
        line_number = struct.unpack_from("<H", data, cursor)[0]
        mjd = struct.unpack_from("<d", data, cursor + 2)[0]
        observation_times.append({"line_number": int(line_number), "mjd": mjd, "utc": mjd_to_iso(mjd)})
        cursor += 10

    header = {
        "source": "himawari9",
        "source_record": source_record,
        "hsd_file_name": hsd_file_name,
        **parse_file_name(hsd_file_name),
        "satellite_name": read_c_string(data, basic + 6, 16),
        "processing_center": read_c_string(data, basic + 22, 16),
        "observation_area": read_c_string(data, basic + 38, 4),
        "observation_timeline_hhmm": struct.unpack_from("<H", data, basic + 44)[0],
        "observation_start_mjd": struct.unpack_from("<d", data, basic + 46)[0],
        "observation_start_utc": mjd_to_iso(struct.unpack_from("<d", data, basic + 46)[0]),
        "observation_end_mjd": struct.unpack_from("<d", data, basic + 54)[0],
        "observation_end_utc": mjd_to_iso(struct.unpack_from("<d", data, basic + 54)[0]),
        "file_creation_mjd": struct.unpack_from("<d", data, basic + 62)[0],
        "file_creation_utc": mjd_to_iso(struct.unpack_from("<d", data, basic + 62)[0]),
        "header_total_bytes": struct.unpack_from("<I", data, basic + 70)[0],
        "data_total_bytes": struct.unpack_from("<I", data, basic + 74)[0],
        "format_version": read_c_string(data, basic + 82, 32),
        "header_blocks": blocks,
        "bits_per_pixel": struct.unpack_from("<H", data, data_info + 3)[0],
        "columns": struct.unpack_from("<H", data, data_info + 5)[0],
        "lines_in_segment": struct.unpack_from("<H", data, data_info + 7)[0],
        "data_block_compression_flag": data[data_info + 9],
        "projection": {
            "sub_satellite_longitude_deg": struct.unpack_from("<d", data, projection + 3)[0],
            "cfac": struct.unpack_from("<i", data, projection + 11)[0],
            "lfac": struct.unpack_from("<i", data, projection + 15)[0],
            "coff": struct.unpack_from("<f", data, projection + 19)[0],
            "loff": struct.unpack_from("<f", data, projection + 23)[0],
            "satellite_distance_km": struct.unpack_from("<d", data, projection + 27)[0],
            "earth_equatorial_radius_km": struct.unpack_from("<d", data, projection + 35)[0],
            "earth_polar_radius_km": struct.unpack_from("<d", data, projection + 43)[0],
            "earth_flattening": struct.unpack_from("<d", data, projection + 51)[0],
            "rpol2_over_req2": struct.unpack_from("<d", data, projection + 59)[0],
            "req2_over_rpol2": struct.unpack_from("<d", data, projection + 67)[0],
            "sd_coefficient": struct.unpack_from("<d", data, projection + 75)[0],
        },
        "calibration": calibration_values,
        "segment": {
            "total_segments": data[segment + 3],
            "segment_sequence_number": data[segment + 4],
            "first_global_line_number": struct.unpack_from("<H", data, segment + 5)[0],
        },
        "observation_times": observation_times,
    }
    header["segment"]["last_global_line_number"] = (
        header["segment"]["first_global_line_number"] + header["lines_in_segment"] - 1
    )
    return header


def hko_pixel_from_projection(header: dict[str, Any]) -> dict[str, Any]:
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
    global_column = proj["coff"] + x * proj["cfac"] / (2**16)
    # For the HSD image array, northern latitudes are above the disk center.
    global_line = proj["loff"] - y * proj["lfac"] / (2**16)
    local_row = int(round(global_line - header["segment"]["first_global_line_number"]))
    local_col = int(round(global_column - 1))
    return {
        "target_station": HKO,
        "global_line_float": global_line,
        "global_column_float": global_column,
        "local_row_0based": local_row,
        "local_col_0based": local_col,
        "inside_this_segment": 0 <= local_row < header["lines_in_segment"] and 0 <= local_col < header["columns"],
    }


def decode_pixels(data: bytes, header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = int(header["columns"])
    lines = int(header["lines_in_segment"])
    offset = int(header["header_total_bytes"])
    counts = np.frombuffer(data, dtype="<u2", count=columns * lines, offset=offset).reshape(lines, columns)
    cal = header["calibration"]
    error_count = int(cal["error_count"])
    outside_count = int(cal["outside_scan_count"])
    valid = (counts != error_count) & (counts != outside_count)
    radiance = cal["count_to_radiance_slope"] * counts.astype("float64") + cal["count_to_radiance_intercept"]
    radiance[~valid] = np.nan
    radiance[radiance <= 0] = np.nan
    wavelength_um = cal["central_wavelength_um"]
    c1 = 1.191042e8
    c2 = 1.4387752e4
    effective_bt = c2 / (wavelength_um * np.log(c1 / (radiance * (wavelength_um**5)) + 1.0))
    bt_k = (
        cal["radiance_to_bt_c0"]
        + cal["radiance_to_bt_c1"] * effective_bt
        + cal["radiance_to_bt_c2"] * effective_bt * effective_bt
    )
    bt_c = bt_k - 273.15
    quality_code = np.zeros_like(counts, dtype="uint8")
    quality_code[counts == outside_count] = 1
    quality_code[counts == error_count] = 2
    return counts, radiance.astype("float32"), bt_c.astype("float32"), quality_code


def write_pixel_outputs(
    header: dict[str, Any],
    counts: np.ndarray,
    radiance: np.ndarray,
    bt_c: np.ndarray,
    quality_code: np.ndarray,
    hko_pixel: dict[str, Any],
) -> dict[str, Any]:
    Path(wp(OUT_DIR)).mkdir(parents=True, exist_ok=True)
    rows, cols = counts.shape
    local_row = np.repeat(np.arange(rows, dtype=np.uint16), cols)
    local_col = np.tile(np.arange(cols, dtype=np.uint16), rows)
    first_global_line = int(header["segment"]["first_global_line_number"])
    pixel_df = pd.DataFrame(
        {
            "local_row_0based": local_row,
            "local_col_0based": local_col,
            "global_line_1based": local_row.astype(np.uint16) + first_global_line,
            "global_col_1based": local_col.astype(np.uint16) + 1,
            "count_uint16": counts.reshape(-1),
            "quality_code": quality_code.reshape(-1),
            "radiance_w_m2_sr_um": radiance.reshape(-1),
            "brightness_temp_c": bt_c.reshape(-1),
        }
    )

    parquet_path = OUT_DIR / "hko_b13_s0510_all_pixels.parquet"
    csv_gz_path = OUT_DIR / "hko_b13_s0510_all_pixels.csv.gz"
    count_matrix_path = OUT_DIR / "hko_b13_s0510_count_matrix_uint16.npy"
    bt_matrix_path = OUT_DIR / "hko_b13_s0510_brightness_temp_c_matrix_float32.npy"
    sample_path = OUT_DIR / "hko_b13_s0510_all_pixels_first_5000_rows.csv"

    pixel_df.to_parquet(wp(parquet_path), index=False)
    with gzip.open(wp(csv_gz_path), "wt", encoding="utf-8", newline="") as handle:
        pixel_df.to_csv(handle, index=False)
    np.save(wp(count_matrix_path), counts)
    np.save(wp(bt_matrix_path), bt_c)
    pixel_df.head(5000).to_csv(wp(sample_path), index=False)

    hko_row = int(hko_pixel["local_row_0based"])
    hko_col = int(hko_pixel["local_col_0based"])
    window_radius = 10
    r0 = max(0, hko_row - window_radius)
    r1 = min(rows, hko_row + window_radius + 1)
    c0 = max(0, hko_col - window_radius)
    c1 = min(cols, hko_col + window_radius + 1)
    window = pixel_df[
        (pixel_df["local_row_0based"] >= r0)
        & (pixel_df["local_row_0based"] < r1)
        & (pixel_df["local_col_0based"] >= c0)
        & (pixel_df["local_col_0based"] < c1)
    ].copy()
    window["is_projected_hko_pixel"] = (
        (window["local_row_0based"] == hko_row) & (window["local_col_0based"] == hko_col)
    )
    window_path = OUT_DIR / "hko_b13_s0510_hko_21x21_window.csv"
    window.to_csv(wp(window_path), index=False)

    valid_bt = bt_c[np.isfinite(bt_c)]
    bins = np.arange(math.floor(float(np.nanmin(valid_bt))), math.ceil(float(np.nanmax(valid_bt))) + 1, 1.0)
    hist_counts, edges = np.histogram(valid_bt, bins=bins)
    hist_df = pd.DataFrame(
        {
            "brightness_temp_c_bin_left": edges[:-1],
            "brightness_temp_c_bin_right": edges[1:],
            "pixel_count": hist_counts,
        }
    )
    hist_path = OUT_DIR / "hko_b13_s0510_brightness_temp_c_histogram.csv"
    hist_df.to_csv(wp(hist_path), index=False)

    return {
        "all_pixels_parquet": str(parquet_path.relative_to(EXPERIMENT_DIR)),
        "all_pixels_csv_gz": str(csv_gz_path.relative_to(EXPERIMENT_DIR)),
        "all_pixels_first_5000_csv": str(sample_path.relative_to(EXPERIMENT_DIR)),
        "count_matrix_npy": str(count_matrix_path.relative_to(EXPERIMENT_DIR)),
        "brightness_temp_c_matrix_npy": str(bt_matrix_path.relative_to(EXPERIMENT_DIR)),
        "hko_21x21_window_csv": str(window_path.relative_to(EXPERIMENT_DIR)),
        "histogram_csv": str(hist_path.relative_to(EXPERIMENT_DIR)),
    }


def build_summary(
    header: dict[str, Any],
    counts: np.ndarray,
    radiance: np.ndarray,
    bt_c: np.ndarray,
    quality_code: np.ndarray,
    hko_pixel: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    hko_row = int(hko_pixel["local_row_0based"])
    hko_col = int(hko_pixel["local_col_0based"])
    radius = 10
    window = bt_c[
        max(0, hko_row - radius) : min(bt_c.shape[0], hko_row + radius + 1),
        max(0, hko_col - radius) : min(bt_c.shape[1], hko_col + radius + 1),
    ]
    valid_bt = bt_c[np.isfinite(bt_c)]
    summary = {
        "generated_at_utc": utc_now_iso(),
        "purpose": "One complete readable Himawari HKG item: B13 HSD segment containing the projected HKO pixel.",
        "header": header,
        "hko_projected_pixel": {
            **hko_pixel,
            "count_uint16": int(counts[hko_row, hko_col]),
            "quality_code": int(quality_code[hko_row, hko_col]),
            "radiance_w_m2_sr_um": float(radiance[hko_row, hko_col]),
            "brightness_temp_c": float(bt_c[hko_row, hko_col]),
        },
        "segment_pixel_summary": {
            "rows": int(counts.shape[0]),
            "columns": int(counts.shape[1]),
            "pixel_count": int(counts.size),
            "valid_pixel_count": int(np.sum(quality_code == 0)),
            "outside_scan_pixel_count": int(np.sum(quality_code == 1)),
            "error_pixel_count": int(np.sum(quality_code == 2)),
            "count_min_valid": int(counts[quality_code == 0].min()),
            "count_median_valid": float(np.median(counts[quality_code == 0])),
            "count_max_valid": int(counts[quality_code == 0].max()),
            "brightness_temp_c_min": float(np.nanmin(valid_bt)),
            "brightness_temp_c_p05": float(np.nanpercentile(valid_bt, 5)),
            "brightness_temp_c_median": float(np.nanmedian(valid_bt)),
            "brightness_temp_c_mean": float(np.nanmean(valid_bt)),
            "brightness_temp_c_p95": float(np.nanpercentile(valid_bt, 95)),
            "brightness_temp_c_max": float(np.nanmax(valid_bt)),
        },
        "hko_21x21_window_summary": {
            "window_radius_pixels": radius,
            "pixel_count": int(np.isfinite(window).sum()),
            "brightness_temp_c_min": float(np.nanmin(window)),
            "brightness_temp_c_mean": float(np.nanmean(window)),
            "brightness_temp_c_median": float(np.nanmedian(window)),
            "brightness_temp_c_max": float(np.nanmax(window)),
        },
        "outputs": outputs,
    }
    return summary


def write_header_tables(header: dict[str, Any]) -> None:
    pd.DataFrame(header["header_blocks"]).to_csv(wp(OUT_DIR / "hko_b13_s0510_header_blocks.csv"), index=False)
    write_json(OUT_DIR / "hko_b13_s0510_header_full.json", header)
    write_json(OUT_DIR / "hko_b13_s0510_calibration.json", header["calibration"])


def write_readme(summary: dict[str, Any]) -> None:
    hko = summary["hko_projected_pixel"]
    seg = summary["segment_pixel_summary"]
    win = summary["hko_21x21_window_summary"]
    readme = f"""# Himawari HKG B13 S0510 Item

Generated: `{summary["generated_at_utc"]}`

This is one decoded Himawari-9 item for HKG inspection: B13 infrared full-disk segment `S0510`, observed at `{summary["header"]["observed_at_utc"]}`. It contains the projected Hong Kong Observatory pixel.

## Fastest Files

| File | What it contains |
|---|---|
| `hko_b13_s0510_item_summary.json` | Header, calibration, HKO pixel, segment stats, and output inventory. |
| `hko_b13_s0510_hko_21x21_window.csv` | Readable 441-pixel local window centered on HKO. |
| `hko_b13_s0510_all_pixels_first_5000_rows.csv` | First rows of the full pixel table for quick viewing. |
| `hko_b13_s0510_all_pixels.parquet` | Full decoded 3,025,000-pixel table. |
| `hko_b13_s0510_all_pixels.csv.gz` | Full decoded 3,025,000-pixel table as compressed CSV. |
| `hko_b13_s0510_header_full.json` | Complete parsed HSD header fields. |
| `hko_b13_s0510_calibration.json` | Count-to-radiance and radiance-to-brightness-temperature coefficients. |

## HKO Pixel

| Field | Value |
|---|---:|
| global line | {hko["global_line_float"]:.3f} |
| global column | {hko["global_column_float"]:.3f} |
| local row | {hko["local_row_0based"]} |
| local column | {hko["local_col_0based"]} |
| count | {hko["count_uint16"]} |
| radiance | {hko["radiance_w_m2_sr_um"]:.6f} |
| B13 brightness temp C | {hko["brightness_temp_c"]:.3f} |

## Segment Summary

| Metric | Value |
|---|---:|
| pixels | {seg["pixel_count"]} |
| valid pixels | {seg["valid_pixel_count"]} |
| outside-scan pixels | {seg["outside_scan_pixel_count"]} |
| brightness temp C min | {seg["brightness_temp_c_min"]:.3f} |
| brightness temp C median | {seg["brightness_temp_c_median"]:.3f} |
| brightness temp C max | {seg["brightness_temp_c_max"]:.3f} |
| HKO 21x21 mean C | {win["brightness_temp_c_mean"]:.3f} |

`quality_code`: `0 = valid`, `1 = outside scan`, `2 = error`.
"""
    write_text(OUT_DIR / "README.md", readme)


def update_status() -> None:
    status_path = EXPERIMENT_DIR / "STATUS.yaml"
    text = Path(wp(status_path)).read_text(encoding="utf-8")
    line = "himawari_hko_item_result: HKO_B13_S0510_FULL_PIXEL_TABLE_WRITTEN"
    if line not in text:
        write_text(status_path, text.rstrip() + "\n" + line + "\n")


def main() -> int:
    raw_path = RAW_DIR / FILE_NAME
    source_record = fetch_if_missing(raw_path)
    with bz2.open(wp(raw_path), "rb") as handle:
        data = handle.read()
    header = parse_header(data, source_record)
    hko_pixel = hko_pixel_from_projection(header)
    if not hko_pixel["inside_this_segment"]:
        raise RuntimeError(f"HKO projected pixel is not inside {FILE_NAME}: {hko_pixel}")
    counts, radiance, bt_c, quality_code = decode_pixels(data, header)
    Path(wp(OUT_DIR)).mkdir(parents=True, exist_ok=True)
    write_header_tables(header)
    outputs = write_pixel_outputs(header, counts, radiance, bt_c, quality_code, hko_pixel)
    summary = build_summary(header, counts, radiance, bt_c, quality_code, hko_pixel, outputs)
    write_json(OUT_DIR / "hko_b13_s0510_item_summary.json", summary)
    write_readme(summary)
    update_status()
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
