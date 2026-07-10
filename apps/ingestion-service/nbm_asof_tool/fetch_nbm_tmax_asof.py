#!/usr/bin/env python3
"""
fetch_nbm_tmax_asof.py

Fetch KMIA (Miami International) NBM "as-of" daily max temperature forecasts.

As-of policy:
  For target local date T, use NBM initialization at (T-1) 12:00Z only.
  Within the local day T (America/New_York), collect all available NBM valid times
  from that run, extract 2m temperature at KMIA, convert to Fahrenheit,
  then take the maximum.

Output:
  CSV with one row per target date.

Dependencies (recommended via conda-forge):
  conda create -n nbm_fetch python=3.11 -y
  conda activate nbm_fetch
  conda install -c conda-forge herbie-data cfgrib eccodes xarray pandas numpy scikit-learn tqdm -y
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

try:
    from herbie import Herbie
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import herbie. Install dependencies via conda-forge:\n"
        "  conda install -c conda-forge herbie-data cfgrib eccodes xarray pandas numpy scikit-learn tqdm\n"
        f"Original import error: {e}"
    )


UTC = timezone.utc


def nbm_forecast_hours_schedule() -> List[int]:
    """
    NBM core GRIB2 forecast hours schedule (integers), per NCO naming rules:
      1-36 hourly
      39-240 every 3 hours
      246-384 every 6 hours
      396-2640 every 12 hours
    """
    hours: List[int] = []
    hours.extend(range(1, 37, 1))
    hours.extend(range(39, 241, 3))
    hours.extend(range(246, 385, 6))
    hours.extend(range(396, 2641, 12))
    return sorted(set(hours))


def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange_inclusive(start: date, end: date) -> List[date]:
    if end < start:
        raise ValueError(f"--end {end} is before --start {start}")
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def local_day_window_utc(target_date_local: date, tz_name: str) -> Tuple[datetime, datetime]:
    """
    Returns [start_utc, end_utc) corresponding to local midnight->midnight for target_date_local.
    """
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(target_date_local, time(0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def asof_utc_for_target(target_date_local: date, asof_hour_utc: int) -> datetime:
    """
    asof_utc = (T - 1 day) at asof_hour_utc:00Z
    """
    init_date = target_date_local - timedelta(days=1)
    return datetime(init_date.year, init_date.month, init_date.day, asof_hour_utc, 0, 0, tzinfo=UTC)


def select_fxx_for_window(
    *,
    asof_utc: datetime,
    window_start_utc: datetime,
    window_end_utc: datetime,
    schedule_hours: List[int],
) -> List[int]:
    """
    Select forecast hours fxx such that valid_time = asof_utc + fxx hours is within [window_start_utc, window_end_utc).
    """
    selected: List[int] = []
    for fxx in schedule_hours:
        vt = asof_utc + timedelta(hours=int(fxx))
        if window_start_utc <= vt < window_end_utc:
            selected.append(int(fxx))
    return selected


def kelvin_to_fahrenheit(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


@dataclass
class PointSpec:
    station_id: str
    latitude: float
    longitude: float
    tz_name: str


@dataclass
class FetchResult:
    fxx: int
    valid_time_utc: datetime
    temp_f: float
    point_grid_distance_km: Optional[float]


def fetch_one_fxx_t2m(
    *,
    asof_utc: datetime,
    fxx: int,
    product: str,
    point: PointSpec,
    save_dir: Path,
    tree_name: str,
    remove_grib: bool,
    verbose_herbie: bool = False,
) -> FetchResult:
    """
    Fetch NBM 2m temperature for a single forecast hour and extract nearest-point value.
    """
    # Herbie expects a naive datetime (UTC). Keep our UTC-aware datetime elsewhere.
    asof_naive = asof_utc.replace(tzinfo=None)
    H = Herbie(
        asof_naive,
        model="nbm",
        product=product,
        fxx=int(fxx),
        save_dir=str(save_dir),
        verbose=bool(verbose_herbie),
        priority=["aws", "nomads"],
    )

    def _coerce_dataset(ds_obj):
        if isinstance(ds_obj, list):
            for item in ds_obj:
                if hasattr(item, "data_vars") and len(item.data_vars) > 0:
                    return item
            return ds_obj[0]
        return ds_obj

    try:
        ds = H.xarray(search=":TMP:2 m", remove_grib=bool(remove_grib))
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
                "IDX unavailable for fxx=%s asof=%s; falling back to full GRIB download. Error: %s",
                fxx,
                asof_utc,
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

    points_df = pd.DataFrame(
        {"longitude": [point.longitude], "latitude": [point.latitude]},
        index=[point.station_id],
    )

    picked = ds.herbie.pick_points(points_df, method="nearest", tree_name=tree_name)

    if len(picked.data_vars) < 1:
        raise RuntimeError("No data_vars in picked dataset after subsetting :TMP:2 m")

    var_name = list(picked.data_vars)[0]
    t_k = float(picked[var_name].values)
    t_f = kelvin_to_fahrenheit(t_k)

    if "valid_time" in ds.coords:
        vt_raw = ds["valid_time"].values
        vt = pd.to_datetime(vt_raw).to_pydatetime()
        if vt.tzinfo is None:
            vt = vt.replace(tzinfo=UTC)
        else:
            vt = vt.astimezone(UTC)
    else:
        vt = asof_utc + timedelta(hours=int(fxx))

    dist_km = None
    if "point_grid_distance" in picked.coords:
        try:
            dist_km = float(picked["point_grid_distance"].values)
        except Exception:
            dist_km = None

    return FetchResult(
        fxx=int(fxx),
        valid_time_utc=vt,
        temp_f=t_f,
        point_grid_distance_km=dist_km,
    )


def compute_daily_tmax_for_target(
    *,
    target_date_local: date,
    asof_hour_utc: int,
    product: str,
    point: PointSpec,
    schedule_hours: List[int],
    save_dir: Path,
    max_workers: int,
    remove_grib: bool,
    min_required_points_for_value: int = 6,
    ok_threshold_points: int = 18,
    tree_name: str = "nbm_co_balltree",
) -> dict:
    """
    Compute daily max temperature forecast (F) for target_date_local using as-of (T-1 12Z).
    Returns a dict row for output CSV.
    """
    asof_utc = asof_utc_for_target(target_date_local, asof_hour_utc)
    window_start_utc, window_end_utc = local_day_window_utc(target_date_local, point.tz_name)

    selected_fxx = select_fxx_for_window(
        asof_utc=asof_utc,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        schedule_hours=schedule_hours,
    )

    logging.info(
        "Target %s | asof=%s | window_utc=[%s, %s) | selected_fxx=%s",
        target_date_local.isoformat(),
        asof_utc.isoformat().replace("+00:00", "Z"),
        window_start_utc.isoformat().replace("+00:00", "Z"),
        window_end_utc.isoformat().replace("+00:00", "Z"),
        selected_fxx,
    )

    results: List[FetchResult] = []
    missing_fxx: List[int] = []

    def _task(f: int) -> FetchResult:
        return fetch_one_fxx_t2m(
            asof_utc=asof_utc,
            fxx=f,
            product=product,
            point=point,
            save_dir=save_dir,
            tree_name=tree_name,
            remove_grib=remove_grib,
            verbose_herbie=False,
        )

    if len(selected_fxx) == 0:
        return {
            "station_id": point.station_id,
            "target_date_local": target_date_local.isoformat(),
            "asof_utc": asof_utc.isoformat().replace("+00:00", "Z"),
            "nbm_tmax_f": np.nan,
            "n_valid_times_used": 0,
            "valid_times_utc_used": "",
            "missing_fxx": "",
            "status": "missing_all",
            "point_grid_distance_km": np.nan,
        }

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_fxx = {ex.submit(_task, f): f for f in selected_fxx}
        for fut in cf.as_completed(fut_to_fxx):
            f = fut_to_fxx[fut]
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                logging.warning("Failed fxx=%s for target=%s asof=%s: %s", f, target_date_local, asof_utc, e)
                missing_fxx.append(int(f))

    results.sort(key=lambda r: r.valid_time_utc)

    n_used = len(results)
    if n_used < min_required_points_for_value:
        status = "insufficient_coverage" if n_used > 0 else "missing_all"
        nbm_tmax_f = np.nan
    else:
        nbm_tmax_f = float(max(r.temp_f for r in results))
        status = "ok" if n_used >= ok_threshold_points else "partial"

    valid_times = "|".join(r.valid_time_utc.isoformat().replace("+00:00", "Z") for r in results)
    missing_fxx_str = "|".join(str(x) for x in sorted(set(missing_fxx)))

    dist_vals = [r.point_grid_distance_km for r in results if r.point_grid_distance_km is not None]
    dist_km = float(np.median(dist_vals)) if dist_vals else np.nan

    return {
        "station_id": point.station_id,
        "target_date_local": target_date_local.isoformat(),
        "asof_utc": asof_utc.isoformat().replace("+00:00", "Z"),
        "nbm_tmax_f": nbm_tmax_f,
        "n_valid_times_used": int(n_used),
        "valid_times_utc_used": valid_times,
        "missing_fxx": missing_fxx_str,
        "status": status,
        "point_grid_distance_km": dist_km,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch KMIA NBM as-of daily max temp (F) from AWS using Herbie (subset by :TMP:2 m)."
    )
    p.add_argument("--start", type=str, default="2026-07-01", help="Start target_date_local YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, default="2026-07-10", help="End target_date_local YYYY-MM-DD (inclusive)")
    p.add_argument("--asof-hour", type=int, default=12, help="As-of initialization hour in UTC (default 12)")
    p.add_argument("--station-id", type=str, default="KMIA")
    p.add_argument("--lat", type=float, default=25.79536)
    p.add_argument("--lon", type=float, default=-80.29012)
    p.add_argument("--tz", type=str, default="America/New_York")
    p.add_argument("--product", type=str, default="co", help="NBM product (default co for CONUS)")
    p.add_argument("--cache-dir", type=str, default="./nbm_cache", help="Directory where Herbie saves data")
    p.add_argument(
        "--out",
        type=str,
        default="./nbm_kmia_tmax_asof_20260701_20260710.csv",
        help="Output CSV path",
    )
    p.add_argument("--max-workers", type=int, default=1, help="Threads per target day")
    p.add_argument("--remove-grib", action="store_true", help="Remove subset GRIB files after reading into xarray")
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
    schedule_hours = nbm_forecast_hours_schedule()

    save_dir = Path(args.cache_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    point = PointSpec(
        station_id=args.station_id,
        latitude=float(args.lat),
        longitude=float(args.lon),
        tz_name=args.tz,
    )

    rows: List[dict] = []
    for t in targets:
        row = compute_daily_tmax_for_target(
            target_date_local=t,
            asof_hour_utc=int(args.asof_hour),
            product=str(args.product),
            point=point,
            schedule_hours=schedule_hours,
            save_dir=save_dir,
            max_workers=int(args.max_workers),
            remove_grib=bool(args.remove_grib),
        )
        rows.append(row)

    cols = [
        "station_id",
        "target_date_local",
        "asof_utc",
        "nbm_tmax_f",
        "n_valid_times_used",
        "valid_times_utc_used",
        "missing_fxx",
        "status",
        "point_grid_distance_km",
    ]
    df = pd.DataFrame(rows, columns=cols)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logging.info("Wrote %d rows to %s", len(df), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
