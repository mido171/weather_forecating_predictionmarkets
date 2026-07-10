from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from app.extract.point_extract import (
    VariableSpec,
    convert_metrics,
    ensure_dir,
    extract_grid_metrics,
    file_manifest,
    forecast_hour_for_index,
    isoformat_utc,
    iter_time_slices,
    normalize_dataset,
    open_grib_dataset,
    parse_cycle_time_utc,
    response_template,
    valid_time_for_index,
)


BASE_URL = "https://noaa-gefs-retrospective.s3.amazonaws.com"

FILE_SPECS: tuple[tuple[str, dict[str, Any] | None, VariableSpec], ...] = (
    ("tmp_2m", {"typeOfLevel": "heightAboveGround", "level": 2}, VariableSpec("temp_2m_f", ("tmp_2m",), conversion="kelvin_to_f")),
    ("tmax_2m", {"typeOfLevel": "heightAboveGround", "level": 2}, VariableSpec("tmax_2m_f", ("tmax_2m",), conversion="kelvin_to_f")),
    ("apcp_sfc", {"typeOfLevel": "surface"}, VariableSpec("qpf_in", ("apcp_sfc",), conversion="mm_to_in")),
    ("tcdc_eatm", {"typeOfLevel": "entireAtmosphere"}, VariableSpec("cloud_cover_pct", ("tcdc_eatm",))),
    ("ugrd_hgt", {"typeOfLevel": "heightAboveGround", "level": 10}, VariableSpec("wind_u_10m_ms", ("ugrd_hgt",))),
    ("vgrd_hgt", {"typeOfLevel": "heightAboveGround", "level": 10}, VariableSpec("wind_v_10m_ms", ("vgrd_hgt",))),
    ("pres_sfc", {"typeOfLevel": "surface"}, VariableSpec("pressure_hpa", ("pres_sfc",), conversion="pa_to_hpa")),
)


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=180.0) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return destination


def run(request: dict) -> dict:
    request = dict(request)
    request.setdefault("model_name", "gefs_reforecast")
    request.setdefault("max_forecast_hours", 48)
    members = request.get("members") or ["c00", "p01", "p02", "p03"]
    cycle_time = parse_cycle_time_utc(str(request["cycle_time_utc"]))
    cycle_time_iso = isoformat_utc(cycle_time)
    cache_root = ensure_dir(request.get("cache_dir") or "ingestion-service/data/tmp/gefs_reforecast")
    lat = float(request["lat"])
    lon = float(request["lon"])
    max_forecast_hours = int(request["max_forecast_hours"])
    cycle_token = cycle_time.strftime("%Y%m%d%H")
    response = response_template(request, "gefs_reforecast")
    touched_by_path: dict[str, dict[str, Any]] = {}

    for member in [str(item) for item in members]:
        for file_stub, filter_by_keys, spec in FILE_SPECS:
            key = f"GEFSv12/reforecast/{cycle_time.year}/{cycle_token}/{member}/Days:1-10/{file_stub}_{cycle_token}_{member}.grib2"
            url = f"{BASE_URL}/{key}"
            local_path = cache_root / str(cycle_time.year) / cycle_token / member / f"{file_stub}_{cycle_token}_{member}.grib2"
            try:
                if not local_path.exists():
                    _download(url, local_path)
                dataset = open_grib_dataset(local_path, filter_by_keys=filter_by_keys)
            except Exception as exc:  # pragma: no cover - network/model dependent
                response["warnings"].append(
                    {"member": member, "file_stub": file_stub, "message": str(exc), "url": url}
                )
                continue
            data_array = normalize_dataset(dataset)
            for index, slice_da in iter_time_slices(dataset, data_array):
                valid_time_utc = valid_time_for_index(dataset, cycle_time, index)
                forecast_hour = forecast_hour_for_index(dataset, cycle_time, valid_time_utc, index)
                if forecast_hour > max_forecast_hours:
                    continue
                metrics = convert_metrics(extract_grid_metrics(dataset, slice_da, lat, lon), spec.conversion)
                variable_name = f"{member}_{spec.variable_name}"
                response["records"].append(
                    {
                        "model_name": request["model_name"],
                        "cycle_time_utc": cycle_time_iso,
                        "valid_time_utc": isoformat_utc(valid_time_utc),
                        "forecast_hour": int(forecast_hour),
                        "variable_name": variable_name,
                        "nearest_value": metrics.nearest_value,
                        "bilinear_value": metrics.bilinear_value,
                        "nbr_mean": metrics.nbr_mean,
                        "nbr_min": metrics.nbr_min,
                        "nbr_max": metrics.nbr_max,
                        "nbr_std": metrics.nbr_std,
                        "grid_source_lat": metrics.grid_source_lat,
                        "grid_source_lon": metrics.grid_source_lon,
                        "grid_distance_km": metrics.grid_distance_km,
                        "interpolation_method": metrics.interpolation_method,
                        "source_identifier": f"gefs_reforecast::{cycle_time_iso}::{member}::{file_stub}::f{int(forecast_hour):03d}",
                        "request_url_or_bucket_key": url,
                        "issue_time_utc": cycle_time_iso,
                    }
                )
            touched_by_path[str(local_path)] = file_manifest(
                local_path=local_path,
                remote_key=url,
                cycle_time_utc=cycle_time_iso,
                forecast_hour=None,
                domain_name="GEFSv12/reforecast",
            )

    response["touched_objects"] = list(touched_by_path.values())
    if not response["records"]:
        response["status"] = "NO_DATA"
    return response
