from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import xarray as xr


UTC = timezone.utc


@dataclass(frozen=True)
class VariableSpec:
    variable_name: str
    search_patterns: tuple[str, ...]
    conversion: str = "identity"
    short_name: str | None = None


@dataclass(frozen=True)
class GridPointMetrics:
    nearest_value: float | None
    bilinear_value: float | None
    nbr_mean: float | None
    nbr_min: float | None
    nbr_max: float | None
    nbr_std: float | None
    grid_source_lat: float | None
    grid_source_lon: float | None
    grid_distance_km: float | None
    interpolation_method: str


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_cycle_time_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def lon_to_360(lon: float) -> float:
    return lon + 360.0 if lon < 0.0 else lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(max(1.0e-12, 1.0 - a)))


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_dataset(candidate: Any, data_var_hint: str | None = None) -> xr.Dataset:
    if isinstance(candidate, xr.Dataset):
        return candidate
    if isinstance(candidate, list):
        datasets = [item for item in candidate if isinstance(item, xr.Dataset)]
        if not datasets:
            raise ValueError("No xarray datasets were returned")
        if data_var_hint:
            for item in datasets:
                if data_var_hint in item.data_vars:
                    return item
        return datasets[0]
    raise TypeError(f"Unsupported dataset payload type: {type(candidate)!r}")


def _data_array(ds: xr.Dataset, preferred_name: str | None = None) -> xr.DataArray:
    if preferred_name and preferred_name in ds.data_vars:
        return ds[preferred_name]
    for name in ds.data_vars:
        return ds[name]
    raise ValueError("Dataset has no data variables")


def _prepare_target_lon(target_lon: float, lon_values: np.ndarray) -> float:
    if np.nanmax(lon_values) > 180.0 and target_lon < 0.0:
        return lon_to_360(target_lon)
    if np.nanmin(lon_values) < 0.0 and target_lon > 180.0:
        return ((target_lon + 180.0) % 360.0) - 180.0
    return target_lon


def _nearest_four_indices_curvilinear(lat2d: np.ndarray, lon2d: np.ndarray, target_lat: float, target_lon: float) -> list[tuple[int, int]]:
    distances: list[tuple[float, tuple[int, int]]] = []
    for row in range(lat2d.shape[0]):
        for col in range(lat2d.shape[1]):
            lat = float(lat2d[row, col])
            lon = float(lon2d[row, col])
            if math.isnan(lat) or math.isnan(lon):
                continue
            distances.append((haversine_km(lat, lon, target_lat, target_lon), (row, col)))
    distances.sort(key=lambda item: item[0])
    return [item[1] for item in distances[:4]]


def _closest_pair(values: np.ndarray, target: float) -> tuple[int, int]:
    arr = np.asarray(values, dtype=float)
    ascending = arr[0] <= arr[-1]
    if not ascending:
        arr = arr[::-1]
    idx = int(np.searchsorted(arr, target, side="left"))
    left = max(0, min(idx - 1, len(arr) - 1))
    right = max(0, min(idx, len(arr) - 1))
    if not ascending:
        left = len(values) - 1 - left
        right = len(values) - 1 - right
    if left == right:
        if right < len(values) - 1:
            right += 1
        elif left > 0:
            left -= 1
    return min(left, right), max(left, right)


def _bilinear_rectilinear(
    values: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[float | None, list[float], tuple[float, float] | None]:
    lat_lo, lat_hi = _closest_pair(lat_values, target_lat)
    lon_lo, lon_hi = _closest_pair(lon_values, target_lon)
    corners = [
        (lat_lo, lon_lo),
        (lat_lo, lon_hi),
        (lat_hi, lon_lo),
        (lat_hi, lon_hi),
    ]
    corner_values: list[float] = []
    for row, col in corners:
        corner = float(values[row, col])
        if math.isnan(corner):
            return None, [], None
        corner_values.append(corner)
    y1 = float(lat_values[lat_lo])
    y2 = float(lat_values[lat_hi])
    x1 = float(lon_values[lon_lo])
    x2 = float(lon_values[lon_hi])
    if abs(y2 - y1) < 1.0e-9 or abs(x2 - x1) < 1.0e-9:
        return float(sum(corner_values) / len(corner_values)), corner_values, (y1, x1)
    wy = (target_lat - y1) / (y2 - y1)
    wx = (target_lon - x1) / (x2 - x1)
    bilinear = (
        corner_values[0] * (1.0 - wx) * (1.0 - wy)
        + corner_values[1] * wx * (1.0 - wy)
        + corner_values[2] * (1.0 - wx) * wy
        + corner_values[3] * wx * wy
    )
    return float(bilinear), corner_values, (float(lat_values[lat_lo]), float(lon_values[lon_lo]))


def _stats(values: Iterable[float]) -> tuple[float | None, float | None, float | None, float | None]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None, None, None
    mean = float(sum(vals) / len(vals))
    variance = float(sum((v - mean) ** 2 for v in vals) / len(vals))
    return mean, min(vals), max(vals), math.sqrt(variance)


def extract_grid_metrics(ds: xr.Dataset, da: xr.DataArray, target_lat: float, target_lon: float) -> GridPointMetrics:
    lat_values = np.asarray(ds["latitude"].values, dtype=float)
    lon_values = np.asarray(ds["longitude"].values, dtype=float)
    target_lon_adjusted = _prepare_target_lon(target_lon, lon_values)

    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lat_idx = int(np.argmin(np.abs(lat_values - target_lat)))
        lon_idx = int(np.argmin(np.abs(lon_values - target_lon_adjusted)))
        nearest_value = float(np.asarray(da.values)[lat_idx, lon_idx])
        nearest_lat = float(lat_values[lat_idx])
        nearest_lon = float(lon_values[lon_idx])
        bilinear_value, corner_values, _ = _bilinear_rectilinear(
            np.asarray(da.values, dtype=float), lat_values, lon_values, target_lat, target_lon_adjusted
        )
        nbr_mean, nbr_min, nbr_max, nbr_std = _stats(corner_values)
        return GridPointMetrics(
            nearest_value=None if math.isnan(nearest_value) else nearest_value,
            bilinear_value=bilinear_value,
            nbr_mean=nbr_mean,
            nbr_min=nbr_min,
            nbr_max=nbr_max,
            nbr_std=nbr_std,
            grid_source_lat=nearest_lat,
            grid_source_lon=nearest_lon,
            grid_distance_km=haversine_km(nearest_lat, nearest_lon, target_lat, target_lon_adjusted),
            interpolation_method="bilinear_rectilinear",
        )

    lat2d = lat_values
    lon2d = lon_values
    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError("Unsupported latitude/longitude coordinate layout")
    indices = _nearest_four_indices_curvilinear(lat2d, lon2d, target_lat, target_lon_adjusted)
    values: list[float] = []
    nearest_lat = None
    nearest_lon = None
    nearest_value = None
    nearest_distance = None
    for rank, (row, col) in enumerate(indices):
        value = float(np.asarray(da.values)[row, col])
        if math.isnan(value):
            continue
        lat = float(lat2d[row, col])
        lon = float(lon2d[row, col])
        distance = haversine_km(lat, lon, target_lat, target_lon_adjusted)
        values.append(value)
        if rank == 0:
            nearest_lat = lat
            nearest_lon = lon
            nearest_value = value
            nearest_distance = distance
    nbr_mean, nbr_min, nbr_max, nbr_std = _stats(values)
    weights = [1.0 / max(0.1, haversine_km(float(lat2d[row, col]), float(lon2d[row, col]), target_lat, target_lon_adjusted))
               for row, col in indices[: len(values)]]
    bilinear = None
    if values and weights:
      bilinear = float(sum(v * w for v, w in zip(values, weights)) / sum(weights))
    return GridPointMetrics(
        nearest_value=nearest_value,
        bilinear_value=bilinear,
        nbr_mean=nbr_mean,
        nbr_min=nbr_min,
        nbr_max=nbr_max,
        nbr_std=nbr_std,
        grid_source_lat=nearest_lat,
        grid_source_lon=nearest_lon,
        grid_distance_km=nearest_distance,
        interpolation_method="inverse_distance_4closest",
    )


def apply_conversion(value: float | None, conversion: str) -> float | None:
    if value is None:
        return None
    if conversion == "kelvin_to_f":
        return (value - 273.15) * 9.0 / 5.0 + 32.0
    if conversion == "mps_to_kt":
        return value * 1.9438444924406
    if conversion == "pa_to_hpa":
        return value / 100.0
    if conversion == "mm_to_in":
        return value / 25.4
    return value


def apply_spread_conversion(value: float | None, conversion: str) -> float | None:
    if value is None:
        return None
    if conversion == "kelvin_to_f":
        return value * 9.0 / 5.0
    if conversion == "mps_to_kt":
        return value * 1.9438444924406
    if conversion == "pa_to_hpa":
        return value / 100.0
    if conversion == "mm_to_in":
        return value / 25.4
    return value


def convert_metrics(metrics: GridPointMetrics, conversion: str) -> GridPointMetrics:
    return GridPointMetrics(
        nearest_value=apply_conversion(metrics.nearest_value, conversion),
        bilinear_value=apply_conversion(metrics.bilinear_value, conversion),
        nbr_mean=apply_conversion(metrics.nbr_mean, conversion),
        nbr_min=apply_conversion(metrics.nbr_min, conversion),
        nbr_max=apply_conversion(metrics.nbr_max, conversion),
        nbr_std=apply_spread_conversion(metrics.nbr_std, conversion),
        grid_source_lat=metrics.grid_source_lat,
        grid_source_lon=metrics.grid_source_lon,
        grid_distance_km=metrics.grid_distance_km,
        interpolation_method=metrics.interpolation_method,
    )


def valid_time_for_index(ds: xr.Dataset, cycle_time_utc: datetime, index: int | None = None) -> datetime:
    if "valid_time" in ds.coords:
        raw = np.asarray(ds["valid_time"].values)
        value = raw if raw.ndim == 0 else raw[index or 0]
        epoch_ns = int(np.datetime64(value, "ns").astype("int64"))
        return datetime.fromtimestamp(epoch_ns / 1_000_000_000.0, tz=UTC)
    if "step" in ds.coords:
        step_values = np.asarray(ds["step"].values)
        step_value = step_values if step_values.ndim == 0 else step_values[index or 0]
        if isinstance(step_value, np.timedelta64):
            seconds = int(step_value / np.timedelta64(1, "s"))
            return cycle_time_utc + timedelta(seconds=seconds)
    return cycle_time_utc


def forecast_hour_for_index(ds: xr.Dataset, cycle_time_utc: datetime, valid_time_utc: datetime, index: int | None = None) -> int:
    if "step" in ds.coords:
        step_values = np.asarray(ds["step"].values)
        step_value = step_values if step_values.ndim == 0 else step_values[index or 0]
        if isinstance(step_value, np.timedelta64):
            return int(step_value / np.timedelta64(1, "h"))
    return int(round((valid_time_utc - cycle_time_utc).total_seconds() / 3600.0))


def normalize_dataset(ds: xr.Dataset, preferred_var_name: str | None = None) -> xr.DataArray:
    da = _data_array(ds, preferred_var_name)
    squeeze_dims = [dim for dim in da.dims if dim not in {"latitude", "longitude", "y", "x", "step", "valid_time"}]
    if squeeze_dims:
        da = da.squeeze(squeeze_dims, drop=True)
    return da


def iter_time_slices(ds: xr.Dataset, da: xr.DataArray) -> list[tuple[int | None, xr.DataArray]]:
    for dim in ("valid_time", "step"):
        if dim in da.dims:
            return [(idx, da.isel({dim: idx})) for idx in range(da.sizes[dim])]
    return [(None, da)]


def file_manifest(local_path: str | Path, remote_key: str, cycle_time_utc: str, forecast_hour: int | None, domain_name: str) -> dict[str, Any]:
    path = Path(local_path)
    return {
        "request_url_or_bucket_key": remote_key,
        "local_path": str(path),
        "content_length": path.stat().st_size if path.exists() else None,
        "checksum_sha256": sha256_file(path) if path.exists() else None,
        "cycle_time_utc": cycle_time_utc,
        "forecast_hour": forecast_hour,
        "domain_name": domain_name,
        "payload_encoding": "application/octet-stream",
    }


def open_grib_dataset(local_path: str | Path, filter_by_keys: dict[str, Any] | None = None) -> xr.Dataset:
    backend_kwargs: dict[str, Any] = {"indexpath": ""}
    if filter_by_keys:
        backend_kwargs["filter_by_keys"] = filter_by_keys
    return xr.open_dataset(local_path, engine="cfgrib", backend_kwargs=backend_kwargs)


def best_search_result(search_results: Sequence[xr.Dataset], preferred_var_names: Sequence[str]) -> xr.Dataset:
    for preferred in preferred_var_names:
        for dataset in search_results:
            if preferred in dataset.data_vars:
                return dataset
    for dataset in search_results:
        if dataset.data_vars:
            return dataset
    raise ValueError("No data variables found in search results")


def cleanup_temp_file(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
    idx_path = Path(f"{path}.idx")
    try:
        idx_path.unlink(missing_ok=True)
    except OSError:
        pass


def response_template(request: dict[str, Any], service: str) -> dict[str, Any]:
    return {
        "status": "SUCCESS",
        "service": service,
        "request": request,
        "records": [],
        "touched_objects": [],
        "warnings": [],
    }
