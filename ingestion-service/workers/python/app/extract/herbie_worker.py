from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from herbie import Herbie

from app.extract.point_extract import (
    VariableSpec,
    best_search_result,
    convert_metrics,
    ensure_dir,
    extract_grid_metrics,
    file_manifest,
    forecast_hour_for_index,
    isoformat_utc,
    iter_time_slices,
    normalize_dataset,
    parse_cycle_time_utc,
    response_template,
    valid_time_for_index,
)


@dataclass(frozen=True)
class HerbieModelConfig:
    model: str
    product: str
    default_forecast_hours: tuple[int, ...]
    priority: tuple[str, ...] = ("aws", "nomads")
    default_members: tuple[str, ...] = ()


def _pick_dataset(candidate: Any, preferred_var_names: tuple[str, ...]) -> Any:
    if isinstance(candidate, list):
        return best_search_result(candidate, preferred_var_names)
    return candidate


def _members(request: dict[str, Any], config: HerbieModelConfig) -> list[str | None]:
    members = request.get("members")
    if isinstance(members, list) and members:
        return [str(item) for item in members]
    member = request.get("member")
    if member:
        return [str(member)]
    if config.default_members:
        return list(config.default_members)
    return [None]


def _forecast_hours(request: dict[str, Any], config: HerbieModelConfig) -> list[int]:
    raw = request.get("forecast_hours")
    if isinstance(raw, list) and raw:
        return sorted({int(item) for item in raw})
    max_forecast_hour = int(request.get("max_forecast_hours", max(config.default_forecast_hours, default=0)))
    return [hour for hour in config.default_forecast_hours if hour <= max_forecast_hour]


def _search_patterns(spec: VariableSpec) -> tuple[str, ...]:
    return spec.search_patterns


def extract_with_herbie(
    request: dict[str, Any],
    service: str,
    config: HerbieModelConfig,
    variable_specs: tuple[VariableSpec, ...],
) -> dict[str, Any]:
    response = response_template(request, service)
    cycle_time = parse_cycle_time_utc(str(request["cycle_time_utc"]))
    cycle_time_iso = isoformat_utc(cycle_time)
    forecast_hours = _forecast_hours(request, config)
    members = _members(request, config)
    cache_dir = ensure_dir(request.get("cache_dir") or "ingestion-service/data/tmp/herbie_worker_cache")
    lat = float(request["lat"])
    lon = float(request["lon"])

    touched_by_path: dict[str, dict[str, Any]] = {}

    for member in members:
        for forecast_hour in forecast_hours:
            kwargs: dict[str, Any] = {
                "model": config.model,
                "product": config.product,
                "fxx": int(forecast_hour),
                "save_dir": str(cache_dir),
                "verbose": False,
                "priority": list(config.priority),
            }
            if member is not None:
                kwargs["member"] = member
            herbie = Herbie(cycle_time.replace(tzinfo=None), **kwargs)
            for spec in variable_specs:
                dataset = None
                local_path = None
                last_error = None
                for pattern in _search_patterns(spec):
                    try:
                        candidate = herbie.xarray(search=pattern, remove_grib=False)
                        dataset = _pick_dataset(candidate, tuple(filter(None, [spec.short_name])))
                        local_path = herbie.get_localFilePath(search=pattern)
                        break
                    except Exception as exc:  # pragma: no cover - network/model dependent
                        last_error = exc
                        continue
                if dataset is None:
                    response["warnings"].append(
                        {
                            "forecast_hour": int(forecast_hour),
                            "member": member,
                            "variable_name": spec.variable_name,
                            "message": str(last_error) if last_error else "variable search returned no dataset",
                        }
                    )
                    continue
                da = normalize_dataset(dataset, spec.short_name)
                for index, slice_da in iter_time_slices(dataset, da):
                    valid_time_utc = valid_time_for_index(dataset, cycle_time, index)
                    derived_forecast_hour = forecast_hour_for_index(dataset, cycle_time, valid_time_utc, index)
                    metrics = convert_metrics(extract_grid_metrics(dataset, slice_da, lat, lon), spec.conversion)
                    variable_name = spec.variable_name if member in (None, "", "mean") else f"{member}_{spec.variable_name}"
                    response["records"].append(
                        {
                            "model_name": request.get("model_name", service),
                            "cycle_time_utc": cycle_time_iso,
                            "valid_time_utc": isoformat_utc(valid_time_utc),
                            "forecast_hour": int(derived_forecast_hour),
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
                            "source_identifier": f"{service}::{cycle_time_iso}::f{int(derived_forecast_hour):03d}::{variable_name}",
                            "request_url_or_bucket_key": herbie.grib,
                            "issue_time_utc": cycle_time_iso,
                        }
                    )
                if local_path is not None:
                    touched_by_path[str(local_path)] = file_manifest(
                        local_path=local_path,
                        remote_key=herbie.grib,
                        cycle_time_utc=cycle_time_iso,
                        forecast_hour=int(forecast_hour),
                        domain_name=config.product,
                    )

    response["touched_objects"] = list(touched_by_path.values())
    if not response["records"]:
        response["status"] = "NO_DATA"
    return response
