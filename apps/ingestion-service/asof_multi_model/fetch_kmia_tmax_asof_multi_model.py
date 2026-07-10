#!/usr/bin/env python3
"""
fetch_kmia_tmax_asof_multi_model.py

Fetch KMIA "as-of" daily Tmax forecasts for a target local date T
using model initialization at (T-1) 12Z (and (T-1) 09Z for RAP).

Key design choice: for models that do NOT provide enough lead time in early years
(e.g., RAP in 2017 with ~18-hour forecasts), the script will NOT output a bogus
"daily Tmax". Instead, it enforces a minimum "daytime coverage" gate and returns
NaN + status.

Models included by default:
- GFS (routes to NCEI for older dates via Herbie)
- GEFS ensemble MEAN and SPREAD (member="mean" and member="spr")
Optional:
- HRRR
- RAP

Dependencies (conda-forge recommended):
  conda create -n asof_fetch python=3.11 -y
  conda activate asof_fetch
  conda install -c conda-forge herbie-data cfgrib eccodes xarray pandas numpy scikit-learn tqdm -y

Example:
  python fetch_kmia_tmax_asof_multi_model.py --start 2017-03-01 --end 2017-03-10 --out kmia_asof_tmax_20170301_20170310.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from zoneinfo import ZoneInfo

try:
    from herbie import Herbie
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import herbie. Install dependencies via conda-forge:\n"
        "  conda install -c conda-forge herbie-data cfgrib eccodes xarray pandas numpy scikit-learn tqdm\n"
        f"Original error: {e}"
    )

UTC = timezone.utc


# -----------------------------
# Helpers
# -----------------------------

def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange_inclusive(start: date, end: date) -> List[date]:
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def lon_to_360(lon_deg: float) -> float:
    return lon_deg + 360.0 if lon_deg < 0 else lon_deg


def kelvin_to_fahrenheit(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def kelvin_spread_to_fahrenheit(k_spread: float) -> float:
    return k_spread * 9.0 / 5.0


def local_day_window_utc(target_date_local: date, tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(target_date_local, time(0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def required_daytime_end_utc(target_date_local: date, tz_name: str, required_end_local_hour: int) -> datetime:
    tz = ZoneInfo(tz_name)
    req_local = datetime.combine(target_date_local, time(required_end_local_hour, 0), tzinfo=tz)
    return req_local.astimezone(UTC)


def asof_naive_utc_for_target(target_date_local: date, run_hour_utc: int) -> datetime:
    init_date = target_date_local - timedelta(days=1)
    return datetime(init_date.year, init_date.month, init_date.day, run_hour_utc, 0, 0)


def select_fxx_for_window(
    asof_naive_utc: datetime,
    window_start_utc: datetime,
    window_end_utc: datetime,
    schedule_hours: List[int],
) -> List[int]:
    selected: List[int] = []
    ws = window_start_utc.replace(tzinfo=None)
    we = window_end_utc.replace(tzinfo=None)
    for fxx in schedule_hours:
        vt = asof_naive_utc + timedelta(hours=int(fxx))
        if ws <= vt < we:
            selected.append(int(fxx))
    return selected


def schedule_step(max_fxx: int, step: int) -> List[int]:
    return list(range(0, max_fxx + 1, step))


def _coerce_dataset(ds_obj: Any):
    if isinstance(ds_obj, list):
        for item in ds_obj:
            if hasattr(item, "data_vars") and len(item.data_vars) > 0:
                return item
        return ds_obj[0]
    return ds_obj


# -----------------------------
# Model config
# -----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    herbie_model: str
    product: Optional[str]
    run_hour_utc: int
    fxx_step_hours: int
    max_fxx_hours: int
    herbie_member: Optional[str] = None
    is_spread: bool = False
    search_str: str = ":TMP:2 m"
    use_lon_360: bool = False
    priority: Optional[List[str]] = None


def default_model_specs(include_hrrr: bool, include_rap: bool) -> List[ModelSpec]:
    specs: List[ModelSpec] = [
        ModelSpec(
            name="gfs",
            herbie_model="gfs",
            product="0.5-degree",
            run_hour_utc=12,
            fxx_step_hours=3,
            max_fxx_hours=60,
            herbie_member=None,
            is_spread=False,
            search_str=":TMP:2 m",
            use_lon_360=True,
        ),
        ModelSpec(
            name="gefs_mean",
            herbie_model="gefs",
            product="atmos.5",
            run_hour_utc=12,
            fxx_step_hours=3,
            max_fxx_hours=60,
            herbie_member="mean",
            is_spread=False,
            search_str=":TMP:2 m",
            use_lon_360=True,
        ),
        ModelSpec(
            name="gefs_spread",
            herbie_model="gefs",
            product="atmos.5",
            run_hour_utc=12,
            fxx_step_hours=3,
            max_fxx_hours=60,
            herbie_member="spr",
            is_spread=True,
            search_str=":TMP:2 m",
            use_lon_360=True,
        ),
    ]

    if include_hrrr:
        specs.append(
            ModelSpec(
                name="hrrr",
                herbie_model="hrrr",
                product="sfc",
                run_hour_utc=12,
                fxx_step_hours=1,
                max_fxx_hours=48,
                herbie_member=None,
                is_spread=False,
                search_str=":TMP:2 m",
                use_lon_360=False,
            )
        )

    if include_rap:
        specs.append(
            ModelSpec(
                name="rap",
                herbie_model="rap",
                product="awp130pgrb",
                run_hour_utc=9,
                fxx_step_hours=1,
                max_fxx_hours=51,
                herbie_member=None,
                is_spread=False,
                search_str=":TMP:2 m",
                use_lon_360=False,
            )
        )

    return specs


# -----------------------------
# Fetch logic
# -----------------------------
@dataclass
class PointSpec:
    station_id: str
    lat: float
    lon: float
    tz_name: str


@dataclass
class FetchOneResult:
    fxx: int
    valid_time_utc: datetime
    value_k: float
    point_grid_distance_km: Optional[float]


def fetch_one_point_value(
    *,
    spec: ModelSpec,
    asof_naive_utc: datetime,
    fxx: int,
    point: PointSpec,
    cache_dir: Path,
    remove_grib: bool,
) -> FetchOneResult:
    kwargs: Dict[str, Any] = {}
    if spec.priority is not None:
        kwargs["priority"] = spec.priority

    logging.debug(
        "Fetch start | model=%s fxx=%s asof=%s product=%s member=%s",
        spec.name,
        fxx,
        asof_naive_utc,
        spec.product,
        spec.herbie_member,
    )
    H = Herbie(
        asof_naive_utc,
        model=spec.herbie_model,
        product=spec.product,
        fxx=int(fxx),
        member=spec.herbie_member,
        save_dir=str(cache_dir),
        verbose=False,
        **kwargs,
    )

    def _open_subset() -> Any:
        return H.xarray(search=spec.search_str, remove_grib=bool(remove_grib))

    try:
        ds = _open_subset()
        ds = _coerce_dataset(ds)
    except Exception as e:
        msg = str(e)
        if any(
            s in msg
            for s in [
                "index file",
                "Index file",
                "No index file",
                "Cant open index file",
                "403 Client Error",
                "404 Client Error",
            ]
        ):
            logging.warning(
                "IDX unavailable for %s fxx=%s asof=%s; falling back to full GRIB. Error: %s",
                spec.name,
                fxx,
                asof_naive_utc,
                msg.splitlines()[0] if msg else "unknown",
            )
            ds = H.xarray(
                search=None,
                backend_kwargs={"filter_by_keys": {"shortName": "2t"}},
                remove_grib=bool(remove_grib),
            )
            ds = _coerce_dataset(ds)
        else:
            raise

    lon_val = lon_to_360(point.lon) if spec.use_lon_360 else point.lon

    points_df = pd.DataFrame(
        {"longitude": [lon_val], "latitude": [point.lat]},
        index=[point.station_id],
    )

    picked = ds.herbie.pick_points(points_df, method="nearest", tree_name=f"{spec.name}_balltree")

    if len(picked.data_vars) < 1:
        raise RuntimeError(f"No data_vars after pick_points for {spec.name} fxx={fxx}")

    var_name = list(picked.data_vars)[0]
    val = float(np.asarray(picked[var_name].values).squeeze())

    if "valid_time" in ds.coords:
        vt_raw = ds["valid_time"].values
        vt = pd.to_datetime(vt_raw).to_pydatetime()
        if vt.tzinfo is None:
            vt = vt.replace(tzinfo=UTC)
        else:
            vt = vt.astimezone(UTC)
    else:
        vt = (asof_naive_utc + timedelta(hours=int(fxx))).replace(tzinfo=UTC)

    dist_km = None
    if "point_grid_distance" in picked.coords:
        try:
            dist_km = float(np.asarray(picked["point_grid_distance"].values).squeeze())
        except Exception:
            dist_km = None

    logging.debug(
        "Fetch ok | model=%s fxx=%s valid_time_utc=%s value_k=%.3f",
        spec.name,
        fxx,
        vt,
        val,
    )

    return FetchOneResult(
        fxx=int(fxx),
        valid_time_utc=vt,
        value_k=val,
        point_grid_distance_km=dist_km,
    )


def compute_daily_tmax_for_model(
    *,
    spec: ModelSpec,
    target_date_local: date,
    point: PointSpec,
    cache_dir: Path,
    max_workers: int,
    remove_grib: bool,
    min_required_points: int,
    required_end_local_hour: int,
    value_field_suffix: str,
) -> Dict[str, Any]:
    asof_naive = asof_naive_utc_for_target(target_date_local, spec.run_hour_utc)
    asof_utc_str = asof_naive.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")

    window_start_utc, window_end_utc = local_day_window_utc(target_date_local, point.tz_name)
    required_end_utc = required_daytime_end_utc(target_date_local, point.tz_name, required_end_local_hour)

    schedule = schedule_step(spec.max_fxx_hours, spec.fxx_step_hours)
    selected_fxx = select_fxx_for_window(asof_naive, window_start_utc, window_end_utc, schedule)

    logging.info(
        "Model start | model=%s asof=%s window_utc=[%s, %s) required_end_utc=%s selected_fxx_count=%d",
        spec.name,
        asof_utc_str,
        window_start_utc.isoformat().replace("+00:00", "Z"),
        window_end_utc.isoformat().replace("+00:00", "Z"),
        required_end_utc.isoformat().replace("+00:00", "Z"),
        len(selected_fxx),
    )
    logging.debug("Model %s selected_fxx=%s", spec.name, selected_fxx)

    out_prefix = spec.name
    value_key = f"{out_prefix}_{value_field_suffix}"

    base: Dict[str, Any] = {
        f"{out_prefix}_asof_utc": asof_utc_str,
        value_key: np.nan,
        f"{out_prefix}_n_valid_times_used": 0,
        f"{out_prefix}_valid_times_utc_used": "",
        f"{out_prefix}_missing_fxx": "",
        f"{out_prefix}_status": "missing_all",
        f"{out_prefix}_point_grid_distance_km": np.nan,
    }

    if len(selected_fxx) == 0:
        base[f"{out_prefix}_status"] = "no_fxx_in_window"
        return base

    results: List[FetchOneResult] = []
    missing: List[int] = []

    def _task(f: int) -> FetchOneResult:
        return fetch_one_point_value(
            spec=spec,
            asof_naive_utc=asof_naive,
            fxx=f,
            point=point,
            cache_dir=cache_dir,
            remove_grib=remove_grib,
        )

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_fxx = {ex.submit(_task, f): f for f in selected_fxx}
        for fut in cf.as_completed(fut_to_fxx):
            f = fut_to_fxx[fut]
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                logging.debug("Failed %s fxx=%s asof=%s: %s", spec.name, f, asof_naive, str(e).splitlines()[0])
                missing.append(int(f))

    results.sort(key=lambda r: r.valid_time_utc)
    n_used = len(results)

    base[f"{out_prefix}_n_valid_times_used"] = int(n_used)
    base[f"{out_prefix}_missing_fxx"] = "|".join(str(x) for x in sorted(set(missing)))

    if n_used == 0:
        base[f"{out_prefix}_status"] = "not_found_or_download_failed"
        logging.info("Model done | model=%s status=%s n_used=%d", spec.name, base[f"{out_prefix}_status"], n_used)
        return base

    max_vt = max(r.valid_time_utc for r in results)
    if max_vt < required_end_utc:
        base[f"{out_prefix}_status"] = "insufficient_daytime_coverage"
        base[f"{out_prefix}_valid_times_utc_used"] = "|".join(
            r.valid_time_utc.isoformat().replace("+00:00", "Z") for r in results
        )
        dist_vals = [r.point_grid_distance_km for r in results if r.point_grid_distance_km is not None]
        base[f"{out_prefix}_point_grid_distance_km"] = float(np.median(dist_vals)) if dist_vals else np.nan
        base[value_key] = np.nan
        logging.info("Model done | model=%s status=%s n_used=%d", spec.name, base[f"{out_prefix}_status"], n_used)
        return base

    if n_used < min_required_points:
        base[f"{out_prefix}_status"] = "insufficient_points"
        base[f"{out_prefix}_valid_times_utc_used"] = "|".join(
            r.valid_time_utc.isoformat().replace("+00:00", "Z") for r in results
        )
        dist_vals = [r.point_grid_distance_km for r in results if r.point_grid_distance_km is not None]
        base[f"{out_prefix}_point_grid_distance_km"] = float(np.median(dist_vals)) if dist_vals else np.nan
        base[value_key] = np.nan
        logging.info("Model done | model=%s status=%s n_used=%d", spec.name, base[f"{out_prefix}_status"], n_used)
        return base

    if spec.is_spread:
        vals_f = [kelvin_spread_to_fahrenheit(r.value_k) for r in results]
    else:
        vals_f = [kelvin_to_fahrenheit(r.value_k) for r in results]

    base[value_key] = float(np.max(vals_f))
    base[f"{out_prefix}_status"] = "ok"
    base[f"{out_prefix}_valid_times_utc_used"] = "|".join(
        r.valid_time_utc.isoformat().replace("+00:00", "Z") for r in results
    )
    dist_vals = [r.point_grid_distance_km for r in results if r.point_grid_distance_km is not None]
    base[f"{out_prefix}_point_grid_distance_km"] = float(np.median(dist_vals)) if dist_vals else np.nan
    logging.info("Model done | model=%s status=%s n_used=%d", spec.name, base[f"{out_prefix}_status"], n_used)
    return base


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch KMIA as-of daily Tmax forecasts for multiple models (GFS, GEFS mean/spread; optional HRRR/RAP)."
    )
    p.add_argument("--start", type=str, default="2017-03-01", help="Start target_date_local YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, default="2017-03-10", help="End target_date_local YYYY-MM-DD (inclusive)")
    p.add_argument("--station-id", type=str, default="KMIA")
    p.add_argument("--lat", type=float, default=25.79536)
    p.add_argument("--lon", type=float, default=-80.29012)
    p.add_argument("--tz", type=str, default="America/New_York")

    p.add_argument("--include-hrrr", action="store_true", help="Also attempt HRRR")
    p.add_argument("--include-rap", action="store_true", help="Also attempt RAP at 09Z")

    p.add_argument("--cache-dir", type=str, default="./asof_model_cache", help="Directory for Herbie downloads/cache")
    p.add_argument("--out", type=str, default="./kmia_asof_tmax_20170301_20170310.csv", help="Output CSV path")

    p.add_argument("--max-workers", type=int, default=6, help="Threads used per model/day")
    p.add_argument("--remove-grib", action="store_true", help="Remove subset GRIB after reading into xarray")

    p.add_argument("--min-required-points", type=int, default=6, help="Minimum valid times required to accept Tmax")
    p.add_argument(
        "--required-end-local-hour",
        type=int,
        default=18,
        help="Require valid times reaching at least this local hour to accept Tmax",
    )

    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start = parse_ymd(args.start)
    end = parse_ymd(args.end)
    targets = daterange_inclusive(start, end)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    point = PointSpec(
        station_id=str(args.station_id),
        lat=float(args.lat),
        lon=float(args.lon),
        tz_name=str(args.tz),
    )

    specs = default_model_specs(include_hrrr=bool(args.include_hrrr), include_rap=bool(args.include_rap))

    rows: List[Dict[str, Any]] = []

    for t in tqdm(targets, desc="Target days"):
        logging.info("Target day start | %s", t.isoformat())
        row: Dict[str, Any] = {
            "station_id": point.station_id,
            "target_date_local": t.isoformat(),
            "tz": point.tz_name,
        }
        for spec in specs:
            value_suffix = "max_f" if spec.name == "gefs_spread" else "tmax_f"
            model_out = compute_daily_tmax_for_model(
                spec=spec,
                target_date_local=t,
                point=point,
                cache_dir=cache_dir,
                max_workers=int(args.max_workers),
                remove_grib=bool(args.remove_grib),
                min_required_points=int(args.min_required_points),
                required_end_local_hour=int(args.required_end_local_hour),
                value_field_suffix=value_suffix,
            )
            row.update(model_out)

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    display_cols = [
        "target_date_local",
        "gfs_tmax_f",
        "gefs_mean_tmax_f",
        "gefs_spread_max_f",
    ]
    for c in ["hrrr_tmax_f", "rap_tmax_f"]:
        if c in df.columns:
            display_cols.append(c)

    print("\n=== DAILY AS-OF TMAX (F) SUMMARY ===")
    print(df[display_cols].to_string(index=False))

    print(f"\nWrote {len(df)} rows to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
