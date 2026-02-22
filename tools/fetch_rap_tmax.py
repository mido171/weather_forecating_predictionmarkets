#!/usr/bin/env python3
"""
Compute RAP130 daily Tmax for target local days using strict as-of semantics.

Source order: local -> aws -> thredds -> has (configurable)
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path so `rap` can be imported when running via tools\*.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from rap.rap_sources import RapSourceResolver
from tools.rap_has_downloader import main as has_downloader_main


def _isoz(d: dt.datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _local_day_window_utc(target_date_local: dt.date, tz_name: str) -> Tuple[dt.datetime, dt.datetime]:
    tz = ZoneInfo(tz_name)
    start_local = dt.datetime(
        target_date_local.year,
        target_date_local.month,
        target_date_local.day,
        0,
        0,
        0,
        tzinfo=tz,
    )
    end_local = start_local + dt.timedelta(days=1)
    return start_local.astimezone(dt.timezone.utc), end_local.astimezone(dt.timezone.utc)


def _asof_run_utc(target_date_local: dt.date, asof_hour: int) -> dt.datetime:
    run_date = target_date_local - dt.timedelta(days=1)
    return dt.datetime(run_date.year, run_date.month, run_date.day, asof_hour, 0, 0, tzinfo=dt.timezone.utc)


def _build_expected_fhrs(
    run_utc: dt.datetime, window_start_utc: dt.datetime, window_end_utc: dt.datetime, step_hours: int
) -> List[int]:
    fhrs: List[int] = []
    vt = window_start_utc
    while vt < window_end_utc:
        delta = vt - run_utc
        fhr = int(round(delta.total_seconds() / 3600.0))
        if fhr >= 0:
            fhrs.append(fhr)
        vt += dt.timedelta(hours=step_hours)
    return fhrs


def _extract_t2m_k(path: Path, lat: float, lon: float) -> float:
    import xarray as xr

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2}, "indexpath": ""},
    )
    try:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise RuntimeError("No data variables in GRIB dataset.")

        chosen = None
        for v in data_vars:
            attrs = ds[v].attrs or {}
            cfname = str(attrs.get("GRIB_cfName", "")).lower()
            short = str(attrs.get("GRIB_shortName", "")).lower()
            name = str(attrs.get("GRIB_name", "")).lower()
            if cfname == "air_temperature":
                chosen = v
                break
            if "temperature" in name and short in ("t", "2t"):
                chosen = v
                break
            if v.lower() in ("t", "t2m", "tmp", "temperature"):
                chosen = v
                break
        if chosen is None:
            chosen = data_vars[0]

        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise RuntimeError("Dataset missing latitude/longitude coordinates.")

        lats = ds["latitude"].values
        lons = ds["longitude"].values
        lon_use = lon
        try:
            if np.nanmax(lons) > 180.0 and lon_use < 0.0:
                lon_use = lon_use + 360.0
        except Exception:
            pass

        dist2 = (lats - lat) ** 2 + (lons - lon_use) ** 2
        iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)
        vals = ds[chosen].values
        if vals.ndim == 2:
            temp_k = float(vals[iy, ix])
        elif vals.ndim == 3:
            temp_k = float(vals[0, iy, ix])
        else:
            raise RuntimeError(f"Unexpected variable dimensionality: {vals.ndim}")
        return temp_k
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _k_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, help="Target local day start YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="Target local day end YYYY-MM-DD")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--lat", type=float, default=25.7959)
    ap.add_argument("--lon", type=float, default=-80.2870)
    ap.add_argument("--asof-hour", type=int, default=9)
    ap.add_argument("--model-step-hours", type=int, default=1)
    ap.add_argument("--cache-dir", default="data/rap130")
    ap.add_argument(
        "--prefer-sources",
        default="local,comet,aws,thredds,has",
        help="Comma-separated source order",
    )
    ap.add_argument("--enable-comet", action="store_true", help="Enable COMET/UCAR archive tier")
    ap.add_argument("--disable-comet", action="store_true", help="Disable COMET/UCAR archive tier")
    ap.add_argument("--comet-base-url", default="http://soostrc.comet.ucar.edu/data/grib/rap/")
    ap.add_argument("--comet-product", default="hybrid")
    ap.add_argument("--auto-has", action="store_true", help="If HAS data missing, attempt to download by HAS ID")
    ap.add_argument("--has-id", default="", help="HAS order ID (required for auto-has)")
    ap.add_argument("--has-email", default="", help="HAS order email (required for auto-has)")
    ap.add_argument("--has-transport", default="ftp", choices=["ftp", "http"])
    ap.add_argument("--coverage-min-points", type=int, default=12)
    ap.add_argument("--dump-values", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    index_db = cache_dir / "index" / "rap130_index.sqlite"

    enable_comet = args.enable_comet and not args.disable_comet
    resolver = RapSourceResolver(
        cache_dir=cache_dir,
        index_db=index_db,
        prefer_sources=tuple(s.strip() for s in args.prefer_sources.split(",") if s.strip()),
        enable_comet=enable_comet,
        comet_base_url=args.comet_base_url,
        comet_product=args.comet_product,
    )

    rows: List[Dict[str, str]] = []
    d = start
    while d <= end:
        window_start, window_end = _local_day_window_utc(d, args.tz)
        run_utc = _asof_run_utc(d, args.asof_hour)
        expected_fhrs = _build_expected_fhrs(run_utc, window_start, window_end, args.model_step_hours)

        temps_f: List[float] = []
        used_fhrs: List[int] = []
        source_counts: Dict[str, int] = {"local": 0, "comet": 0, "aws": 0, "thredds": 0, "has": 0}
        missing_fhrs: List[int] = []

        for fhr in expected_fhrs:
            path, src = resolver.resolve(run_utc, fhr)
            if path is None or src is None:
                missing_fhrs.append(fhr)
                continue
            temp_k = _extract_t2m_k(Path(path), args.lat, args.lon)
            temp_f = _k_to_f(temp_k)
            temps_f.append(temp_f)
            used_fhrs.append(fhr)
            source_counts[src] = source_counts.get(src, 0) + 1
            if args.dump_values:
                vt = run_utc + dt.timedelta(hours=fhr)
                print(f"{d} fhr={fhr:03d} valid={_isoz(vt)} temp_f={temp_f:.6f} source={src}")

        # If auto-has is enabled and we are missing everything, try HAS download if ID provided.
        if args.auto_has and not temps_f and args.has_id and args.has_email:
            # invoke downloader (idempotent)
            has_args = [
                "--has-id", args.has_id,
                "--email", args.has_email,
                "--cache-dir", str(cache_dir / "has"),
                "--transport", args.has_transport,
                "--resume",
            ]
            try:
                _ = has_downloader_main.__call__  # type: ignore[attr-defined]
            except Exception:
                # execute via subprocess-like call
                pass

            try:
                import sys as _sys
                _sys.argv = ["rap_has_downloader.py"] + has_args
                has_downloader_main()
            except SystemExit:
                pass

            # Rebuild index if needed
            # User should run rap_file_index.py in practice; here we just re-resolve.
            temps_f = []
            used_fhrs = []
            source_counts = {"local": 0, "comet": 0, "aws": 0, "thredds": 0, "has": 0}
            missing_fhrs = []
            for fhr in expected_fhrs:
                path, src = resolver.resolve(run_utc, fhr)
                if path is None or src is None:
                    missing_fhrs.append(fhr)
                    continue
                temp_k = _extract_t2m_k(Path(path), args.lat, args.lon)
                temp_f = _k_to_f(temp_k)
                temps_f.append(temp_f)
                used_fhrs.append(fhr)
                source_counts[src] = source_counts.get(src, 0) + 1

        expected_points = len(expected_fhrs)
        n_points = len(temps_f)
        coverage_ratio = (n_points / expected_points) if expected_points > 0 else 0.0

        if n_points == 0:
            status = "missing_all"
        elif n_points < args.coverage_min_points:
            status = "insufficient_coverage"
        elif n_points < expected_points:
            status = "partial"
        else:
            status = "ok"

        tmax_f = max(temps_f) if temps_f else math.nan
        notes = []
        notes.append("used 09Z extended cycle" if args.asof_hour == 9 else "used 12Z standard cycle")
        if missing_fhrs:
            notes.append(f"missing_fhr={','.join(str(x) for x in missing_fhrs)}")

        row = {
            "target_date_local": d.isoformat(),
            "run_time_utc": _isoz(run_utc),
            "lat": f"{args.lat:.4f}",
            "lon": f"{args.lon:.4f}",
            "tmax_f": f"{tmax_f:.6f}" if not math.isnan(tmax_f) else "",
            "n_points_used": str(n_points),
            "expected_points": str(expected_points),
            "coverage_ratio": f"{coverage_ratio:.3f}",
            "status": status,
            "source_breakdown": ";".join(f"{k}={v}" for k, v in source_counts.items()),
            "notes": " | ".join(notes),
        }
        rows.append(row)
        d += dt.timedelta(days=1)

    df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
