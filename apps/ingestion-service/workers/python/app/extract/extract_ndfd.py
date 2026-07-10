from __future__ import annotations

from pathlib import Path

import httpx

from app.extract.point_extract import (
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
    request.setdefault("model_name", "ndfd_historical")
    cycle_time = parse_cycle_time_utc(str(request["cycle_time_utc"]))
    cycle_time_iso = isoformat_utc(cycle_time)
    cache_root = ensure_dir(request.get("cache_dir") or "ingestion-service/data/tmp/ndfd_historical")
    lat = float(request["lat"])
    lon = float(request["lon"])
    files = request.get("files") or []
    response = response_template(request, "ndfd")
    touched_by_path: dict[str, dict] = {}

    for file_request in files:
        url = str(file_request["url"])
        variable_name = str(file_request["variable_name"])
        conversion = str(file_request.get("conversion", "identity"))
        filter_by_keys = file_request.get("filter_by_keys")
        local_name = str(file_request.get("local_name") or Path(url).name)
        local_path = cache_root / local_name
        try:
            if not local_path.exists():
                _download(url, local_path)
            dataset = open_grib_dataset(local_path, filter_by_keys=filter_by_keys)
        except Exception as exc:  # pragma: no cover - network/model dependent
            response["warnings"].append({"url": url, "variable_name": variable_name, "message": str(exc)})
            continue
        data_array = normalize_dataset(dataset)
        for index, slice_da in iter_time_slices(dataset, data_array):
            valid_time_utc = valid_time_for_index(dataset, cycle_time, index)
            forecast_hour = forecast_hour_for_index(dataset, cycle_time, valid_time_utc, index)
            metrics = convert_metrics(extract_grid_metrics(dataset, slice_da, lat, lon), conversion)
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
                    "source_identifier": f"ndfd::{cycle_time_iso}::{variable_name}::f{int(forecast_hour):03d}",
                    "request_url_or_bucket_key": url,
                    "issue_time_utc": cycle_time_iso,
                }
            )
        touched_by_path[str(local_path)] = file_manifest(
            local_path=local_path,
            remote_key=url,
            cycle_time_utc=cycle_time_iso,
            forecast_hour=None,
            domain_name=str(file_request.get("domain_name", "ndfd")),
        )

    response["touched_objects"] = list(touched_by_path.values())
    if not response["records"]:
        response["status"] = "NO_DATA"
    return response
