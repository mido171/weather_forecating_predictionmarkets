from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


UTC = timezone.utc


def utc_now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _detect_columns(path: Path) -> tuple[str, str]:
    sample = pd.read_csv(path, nrows=5)
    cols = [c.strip() for c in sample.columns]
    time_candidates = [
        "valid(UTC)",
        "valid",
        "timestamp",
        "ts",
        "time",
        "datetime",
        "date_time",
    ]
    temp_candidates = ["tmpf", "temp_f", "temperature", "airtemp", "temp"]
    time_col = next((c for c in cols if c in time_candidates), None)
    temp_col = next((c for c in cols if c in temp_candidates), None)
    if time_col is None:
        for c in cols:
            if "valid" in c.lower() or "time" in c.lower() or "date" in c.lower():
                time_col = c
                break
    if temp_col is None:
        for c in cols:
            if "tmp" in c.lower() or "temp" in c.lower():
                temp_col = c
                break
    if time_col is None or temp_col is None:
        raise ValueError(f"Could not detect time/temp columns in {path}")
    return time_col, temp_col


def _load_minute_files(path: Path) -> Iterable[Path]:
    return sorted(path.glob("*.csv"))


def _build_profile(series: pd.Series, bins: np.ndarray) -> np.ndarray:
    if series.empty:
        return np.full(len(bins), np.nan)
    series = series.groupby(level=0).median()
    series = series.reindex(bins)
    series = series.interpolate(method="linear", limit_direction="both")
    series = series.ffill().bfill()
    return series.to_numpy(dtype=float)


def _dct_matrix(n: int, k: int) -> np.ndarray:
    n_idx = np.arange(n, dtype=float)
    mat = np.zeros((n, k), dtype=float)
    for i in range(k):
        if i == 0:
            scale = np.sqrt(1.0 / n)
        else:
            scale = np.sqrt(2.0 / n)
        mat[:, i] = scale * np.cos(np.pi / n * (n_idx + 0.5) * i)
    return mat


def _compute_dct_features(
    profiles: np.ndarray,
    mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if profiles.size == 0:
        return np.empty((0, mat.shape[1])), np.empty((0,))
    filled = np.where(np.isnan(profiles), np.nanmedian(profiles, axis=1, keepdims=True), profiles)
    coeffs = filled @ mat
    energy_total = np.sum(filled * filled, axis=1)
    energy_low = np.sum(coeffs * coeffs, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        energy_hi = np.where(energy_total > 0, np.maximum(energy_total - energy_low, 0.0) / energy_total, np.nan)
    return coeffs, energy_hi


def _compute_minute_profiles(
    minute_dir: Path,
    *,
    tz_name: str,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone = ZoneInfo(tz_name)
    local_start = start_date - timedelta(days=1)
    local_end = end_date - timedelta(days=1)
    utc_start = datetime.combine(start_date - timedelta(days=1), time(0, 0), tzinfo=UTC)
    utc_end = datetime.combine(end_date, time(23, 59), tzinfo=UTC)

    local_rows: list[pd.DataFrame] = []
    utc_rows: list[pd.DataFrame] = []
    max_ts_local: dict[date, datetime] = {}
    max_ts_utc_early: dict[date, datetime] = {}

    files = list(_load_minute_files(minute_dir))
    if not files:
        raise FileNotFoundError(f"No minute CSVs found in {minute_dir}")

    time_col, temp_col = _detect_columns(files[0])

    for path in files:
        df = pd.read_csv(path, usecols=[time_col, temp_col], low_memory=False)
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df = df.dropna(subset=[time_col, temp_col])
        if df.empty:
            continue
        df = df[(df[time_col] >= utc_start) & (df[time_col] <= utc_end)]
        if df.empty:
            continue

        ts = df[time_col]
        df["utc_date"] = ts.dt.date
        df["utc_minute"] = ts.dt.hour * 60 + ts.dt.minute
        df["utc_bin"] = (df["utc_minute"] // 5) * 5

        local_dt = ts.dt.tz_convert(zone)
        df["local_date"] = local_dt.dt.date
        df["local_minute"] = local_dt.dt.hour * 60 + local_dt.dt.minute
        df["local_bin"] = (df["local_minute"] // 5) * 5

        df_local = df[(df["local_date"] >= local_start) & (df["local_date"] <= local_end)]
        if not df_local.empty:
            local_rows.append(df_local[["local_date", "local_bin", temp_col, time_col]].rename(columns={temp_col: "temp"}))
            max_local = df_local.groupby("local_date")[time_col].max()
            for d, ts_max in max_local.items():
                prev = max_ts_local.get(d)
                if prev is None or ts_max > prev:
                    max_ts_local[d] = ts_max

        df_utc = df[(df["utc_date"] >= start_date) & (df["utc_date"] <= end_date)]
        if not df_utc.empty:
            utc_rows.append(df_utc[["utc_date", "utc_bin", temp_col, time_col]].rename(columns={temp_col: "temp"}))
            early = df_utc[df_utc["utc_minute"] <= 360]
            if not early.empty:
                max_early = early.groupby("utc_date")[time_col].max()
                for d, ts_max in max_early.items():
                    prev = max_ts_utc_early.get(d)
                    if prev is None or ts_max > prev:
                        max_ts_utc_early[d] = ts_max

    if not local_rows and not utc_rows:
        raise ValueError("No minute rows found for requested range.")

    local_df = pd.concat(local_rows, ignore_index=True) if local_rows else pd.DataFrame()
    utc_df = pd.concat(utc_rows, ignore_index=True) if utc_rows else pd.DataFrame()

    local_bins = np.arange(0, 1440, 5)
    utc_bins = np.arange(0, 360, 5)

    local_profiles: dict[date, np.ndarray] = {}
    local_stats: dict[date, dict[str, float]] = {}
    if not local_df.empty:
        local_df = local_df.groupby(["local_date", "local_bin"], as_index=False)["temp"].median()
        for local_date, group in local_df.groupby("local_date"):
            series = group.set_index("local_bin")["temp"]
            profile = _build_profile(series, local_bins)
            local_profiles[local_date] = profile
            if np.all(np.isnan(profile)):
                local_stats[local_date] = {
                    "tmax": np.nan,
                    "tmin": np.nan,
                    "tmax_time": np.nan,
                    "range": np.nan,
                    "max_drop_30": np.nan,
                    "drop_cnt_15_19": np.nan,
                }
                continue
            tmax = float(np.nanmax(profile))
            tmin = float(np.nanmin(profile))
            tmax_time = float(local_bins[np.nanargmax(profile)]) if np.isfinite(tmax) else np.nan
            diff30 = profile[:-6] - profile[6:]
            max_drop_30 = float(np.nanmax(diff30)) if diff30.size else np.nan
            start_idx = int(15 * 60 / 5)
            end_idx = int(19 * 60 / 5)
            window = profile[start_idx:end_idx + 1]
            drop_cnt = 0.0
            if window.size >= 7:
                diff30_w = window[:-6] - window[6:]
                drop_cnt = float(np.sum(diff30_w >= 2.0))
            local_stats[local_date] = {
                "tmax": tmax,
                "tmin": tmin,
                "tmax_time": tmax_time,
                "range": tmax - tmin if np.isfinite(tmax) and np.isfinite(tmin) else np.nan,
                "max_drop_30": max_drop_30,
                "drop_cnt_15_19": drop_cnt,
            }

    utc_profiles: dict[date, np.ndarray] = {}
    if not utc_df.empty:
        utc_df = utc_df.groupby(["utc_date", "utc_bin"], as_index=False)["temp"].median()
        for utc_date, group in utc_df.groupby("utc_date"):
            series = group.set_index("utc_bin")["temp"]
            profile = _build_profile(series, utc_bins)
            utc_profiles[utc_date] = profile

    target_dates = pd.date_range(start_date, end_date, freq="D").date
    rows: list[dict[str, Any]] = []
    for target_date in target_dates:
        local_date = target_date - timedelta(days=1)
        prof_local = local_profiles.get(local_date)
        prof_utc = utc_profiles.get(target_date)
        if prof_local is None:
            prof_local = np.full(len(local_bins), np.nan)
        if prof_utc is None:
            prof_utc = np.full(len(utc_bins), np.nan)

        local_med = float(np.nanmedian(prof_local)) if not np.all(np.isnan(prof_local)) else np.nan
        prof_local_centered = prof_local - local_med
        prof_local_anchored = prof_local - prof_local[-1] if np.isfinite(prof_local[-1]) else prof_local
        utc_med = float(np.nanmedian(prof_utc)) if not np.all(np.isnan(prof_utc)) else np.nan
        prof_utc_centered = prof_utc - utc_med

        stats = local_stats.get(local_date, {})
        row: dict[str, Any] = {
            "target_date_local": target_date,
            "iem_range_day0": local_stats.get(target_date, {}).get("range", np.nan),
            "minute_max_ts_utc_tminus1": max_ts_local.get(local_date),
            "minute_max_ts_utc_early": max_ts_utc_early.get(target_date),
        }
        row.update(
            {
                "iem_tminus1_tmax": stats.get("tmax", np.nan),
                "iem_tminus1_tmin": stats.get("tmin", np.nan),
                "iem_tminus1_tmax_time_min": stats.get("tmax_time", np.nan),
                "iem_tminus1_range": stats.get("range", np.nan),
                "iem_tminus1_max_drop_30": stats.get("max_drop_30", np.nan),
                "iem_tminus1_drop_cnt_15_19": stats.get("drop_cnt_15_19", np.nan),
            }
        )
        for i, val in enumerate(prof_local):
            row[f"iem_tminus1_profile_{local_bins[i]:04d}"] = float(val) if np.isfinite(val) else np.nan
        for i, val in enumerate(prof_local_centered):
            row[f"iem_tminus1_profile_centered_{local_bins[i]:04d}"] = float(val) if np.isfinite(val) else np.nan
        for i, val in enumerate(prof_local_anchored):
            row[f"iem_tminus1_profile_anchored_{local_bins[i]:04d}"] = float(val) if np.isfinite(val) else np.nan
        for i, val in enumerate(prof_utc):
            row[f"iem_utc00_06_profile_{utc_bins[i]:04d}"] = float(val) if np.isfinite(val) else np.nan
        for i, val in enumerate(prof_utc_centered):
            row[f"iem_utc00_06_profile_centered_{utc_bins[i]:04d}"] = float(val) if np.isfinite(val) else np.nan
        rows.append(row)

    profile_df = pd.DataFrame(rows)
    return profile_df, pd.DataFrame(
        {
            "target_date_local": list(target_dates),
            "minute_max_ts_utc_tminus1": [max_ts_local.get(d - timedelta(days=1)) for d in target_dates],
            "minute_max_ts_utc_early": [max_ts_utc_early.get(d) for d in target_dates],
        }
    )


def _load_mos_rows(
    engine_url: str,
    station_id: str,
    models: list[str],
    var_codes: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    engine = create_engine(engine_url, pool_pre_ping=True)
    model_placeholders = ", ".join([f":m{i}" for i in range(len(models))])
    var_placeholders = ", ".join([f":v{i}" for i in range(len(var_codes))])
    sql = f"""
        SELECT id, station_id, model, variable_code, target_date_local,
               asof_utc, runtime_utc, retrieved_at_utc,
               value_mean, value_max, value_min
        FROM mos_daily_value
        WHERE station_id = :station_id
          AND model IN ({model_placeholders})
          AND variable_code IN ({var_placeholders})
          AND target_date_local BETWEEN :start_date AND :end_date
    """
    params: dict[str, Any] = {
        "station_id": station_id,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    params.update({f"m{i}": m for i, m in enumerate(models)})
    params.update({f"v{i}": v for i, v in enumerate(var_codes)})
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True, errors="coerce")
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True, errors="coerce")
    df["model"] = df["model"].astype(str).str.lower()
    df["variable_code"] = df["variable_code"].astype(str).str.lower()
    return df


def _select_latest_by_bucket(
    mos_df: pd.DataFrame,
    target_dates: list[date],
    buckets: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    cal = pd.DataFrame({"target_date_local": target_dates})
    cal["decision_utc"] = pd.to_datetime(cal["target_date_local"]).dt.tz_localize(UTC) + pd.Timedelta(hours=6)
    max_asof = pd.Series(index=target_dates, data=pd.NaT, dtype="datetime64[ns, UTC]")
    out = pd.DataFrame({"target_date_local": target_dates})
    if mos_df.empty:
        return out, max_asof
    for hours in buckets:
        merged = mos_df.merge(cal, on="target_date_local", how="left")
        merged["cutoff"] = merged["decision_utc"] - pd.Timedelta(hours=hours)
        eligible = merged[(merged["asof_utc"] <= merged["cutoff"]) & (merged["runtime_utc"] <= merged["asof_utc"])]
        if eligible.empty:
            continue
        eligible = eligible.sort_values(
            ["target_date_local", "model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id"]
        )
        latest = eligible.groupby(["target_date_local", "model", "variable_code"], as_index=False).tail(1)
        latest_max = latest.groupby("target_date_local")["asof_utc"].max()
        max_asof = max_asof.combine(latest_max, func=lambda a, b: b if pd.isna(a) else max(a, b))

        use_max = latest["variable_code"].isin(["p12", "q12", "cig"])
        val = latest["value_mean"].copy()
        val = val.where(~use_max, latest["value_max"])
        val = val.where(~val.isna(), latest["value_max"])
        val = val.where(~val.isna(), latest["value_min"])
        rest = latest.assign(value=val)
        tmp = latest[latest["variable_code"] == "tmp"]
        tmp_max = tmp.assign(variable_code="tmp_max", value=tmp["value_max"])
        tmp_min = tmp.assign(variable_code="tmp_min", value=tmp["value_min"])
        latest = pd.concat([rest[rest["variable_code"] != "tmp"], tmp_max, tmp_min], ignore_index=True)
        pivot = latest.pivot_table(
            index="target_date_local",
            columns=["model", "variable_code"],
            values="value",
            aggfunc="first",
        )
        pivot.columns = [f"mos_{var}_{model}_b{hours}" for (model, var) in pivot.columns]
        pivot = pivot.reset_index()
        out = out.merge(pivot, on="target_date_local", how="left")
    return out, max_asof


def _blend(a: pd.Series, b: pd.Series) -> pd.Series:
    blend = 0.5 * (a + b)
    blend = blend.where(~a.isna(), b)
    blend = blend.where(~b.isna(), a)
    return blend


def _aggregate_bucket_features(df: pd.DataFrame, buckets: list[int]) -> pd.DataFrame:
    df = df.copy()
    for b in buckets:
        suffix = f"_b{b}"
        def col(name: str) -> pd.Series:
            return pd.to_numeric(df.get(f"mos_{name}{suffix}"), errors="coerce")

        tmp_max = _blend(col("tmp_max_gfs"), col("tmp_max_nam"))
        tmp_min = _blend(col("tmp_min_gfs"), col("tmp_min_nam"))
        dpt_mean = _blend(col("dpt_gfs"), col("dpt_nam"))
        wsp_mean = _blend(col("wsp_gfs"), col("wsp_nam"))
        wdr_mean = _blend(col("wdr_gfs"), col("wdr_nam"))
        cig_min = _blend(col("cig_gfs"), col("cig_nam"))
        p12_max = _blend(col("p12_gfs"), col("p12_nam"))
        q12_max = _blend(col("q12_gfs"), col("q12_nam"))

        df[f"v6_tmp_max_mean_models_b{b}"] = tmp_max
        df[f"v6_tmp_min_mean_models_b{b}"] = tmp_min
        df[f"v6_tmp_range_mean_models_b{b}"] = tmp_max - tmp_min
        df[f"v6_dpt_mean_models_b{b}"] = dpt_mean
        df[f"v6_dd_models_b{b}"] = tmp_max - dpt_mean
        df[f"v6_cig_min_b{b}"] = cig_min
        df[f"v6_p12_max_b{b}"] = p12_max
        df[f"v6_q12_max_b{b}"] = q12_max
        df[f"v6_wsp_mean_b{b}"] = wsp_mean
        wdr_rad = np.deg2rad(wdr_mean)
        df[f"v6_u_b{b}"] = -wsp_mean * np.sin(wdr_rad)
        df[f"v6_v_b{b}"] = -wsp_mean * np.cos(wdr_rad)

        if b in (0, 24):
            gfs_tmp_max = col("tmp_max_gfs")
            nam_tmp_max = col("tmp_max_nam")
            gfs_tmp_min = col("tmp_min_gfs")
            nam_tmp_min = col("tmp_min_nam")
            gfs_cig = col("cig_gfs")
            nam_cig = col("cig_nam")
            df[f"v6_disc_tmp_max_b{b}"] = gfs_tmp_max - nam_tmp_max
            df[f"v6_abs_disc_tmp_max_b{b}"] = (gfs_tmp_max - nam_tmp_max).abs()
            df[f"v6_disc_tmp_min_b{b}"] = gfs_tmp_min - nam_tmp_min
            df[f"v6_abs_disc_tmp_min_b{b}"] = (gfs_tmp_min - nam_tmp_min).abs()
            df[f"v6_disc_cig_b{b}"] = gfs_cig - nam_cig
            df[f"v6_abs_disc_cig_b{b}"] = (gfs_cig - nam_cig).abs()
    return df


def _add_revision_features(df: pd.DataFrame, buckets: list[int]) -> pd.DataFrame:
    df = df.copy()
    pairs = [(0, 12), (0, 24), (12, 24), (24, 48)]
    base_vars = [
        "tmp_max_mean_models",
        "tmp_min_mean_models",
        "tmp_range_mean_models",
        "dpt_mean_models",
        "dd_models",
        "cig_min",
        "p12_max",
        "q12_max",
        "wsp_mean",
        "u",
        "v",
    ]
    hours = [0, 6, 12, 24, 36, 48]
    xs = np.array(hours, dtype=float)
    for var in base_vars:
        for a, b in pairs:
            ca = f"v6_{var}_b{a}"
            cb = f"v6_{var}_b{b}"
            if ca in df.columns and cb in df.columns:
                df[f"v6_abs_rev_{var}_b{a}_b{b}"] = (df[ca] - df[cb]).abs()
        # trend slope
        vals = np.vstack(
            [pd.to_numeric(df.get(f"v6_{var}_b{h}", np.nan), errors="coerce") for h in hours]
        ).T
        slope = np.full(len(df), np.nan)
        for i in range(len(df)):
            y = vals[i]
            mask = np.isfinite(y)
            if mask.sum() < 3:
                continue
            try:
                slope[i] = float(np.polyfit(xs[mask], y[mask], 1)[0])
            except Exception:
                slope[i] = np.nan
        df[f"v6_trend_{var}"] = slope
    return df


def build_feature_store(
    base_feature_store: Path,
    minute_dir: Path,
    db_url: str,
    *,
    station_id: str = "KMIA",
    tz_name: str = "America/New_York",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(base_feature_store)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    start_date = df["target_date_local"].min()
    end_date = df["target_date_local"].max()

    profiles, minute_audit = _compute_minute_profiles(
        minute_dir,
        tz_name=tz_name,
        start_date=start_date,
        end_date=end_date,
    )

    profiles["target_date_local"] = pd.to_datetime(profiles["target_date_local"]).dt.date
    df = df.merge(profiles, on="target_date_local", how="left")

    # DCT features
    local_bins = np.arange(0, 1440, 5)
    utc_bins = np.arange(0, 360, 5)
    local_cols = [f"iem_tminus1_profile_{b:04d}" for b in local_bins]
    utc_cols = [f"iem_utc00_06_profile_{b:04d}" for b in utc_bins]
    local_mat = df[local_cols].to_numpy(dtype=float)
    utc_mat = df[utc_cols].to_numpy(dtype=float)
    dct_local = _dct_matrix(len(local_bins), 20)
    dct_utc = _dct_matrix(len(utc_bins), 10)
    local_coeffs, local_hi = _compute_dct_features(local_mat, dct_local)
    utc_coeffs, utc_hi = _compute_dct_features(utc_mat, dct_utc)
    for i in range(local_coeffs.shape[1]):
        df[f"iem_tminus1_dct_{i:02d}"] = local_coeffs[:, i]
    df["iem_tminus1_dct_energy_hi"] = local_hi
    for i in range(utc_coeffs.shape[1]):
        df[f"iem_utc00_06_dct_{i:02d}"] = utc_coeffs[:, i]
    df["iem_utc00_06_dct_energy_hi"] = utc_hi

    # Multi-day lags using condensed features
    df = df.sort_values("target_date_local").reset_index(drop=True)
    for lag in [2, 3]:
        shift = lag - 1
        for col in [
            "iem_tmax_t1",
            "iem_tmin_t1",
            "iem_range_t1",
            "tmax_time_min_t1",
            "max_drop_30_t1",
            "drop_cnt_15_19_t1",
        ]:
            if col in df.columns:
                df[col.replace("_t1", f"_t{lag}")] = pd.to_numeric(df[col], errors="coerce").shift(shift)
    df["delta_tmax_1d"] = pd.to_numeric(df.get("iem_tmax_t1"), errors="coerce") - pd.to_numeric(
        df.get("iem_tmax_t2"), errors="coerce"
    )
    df["delta_range_1d"] = pd.to_numeric(df.get("iem_range_t1"), errors="coerce") - pd.to_numeric(
        df.get("iem_range_t2"), errors="coerce"
    )
    # 3-day trend
    tmax_stack = np.vstack(
        [
            pd.to_numeric(df.get("iem_tmax_t3"), errors="coerce"),
            pd.to_numeric(df.get("iem_tmax_t2"), errors="coerce"),
            pd.to_numeric(df.get("iem_tmax_t1"), errors="coerce"),
        ]
    ).T
    trend = np.full(len(df), np.nan)
    xs = np.array([0, 1, 2], dtype=float)
    for i in range(len(df)):
        y = tmax_stack[i]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        trend[i] = float(np.polyfit(xs[mask], y[mask], 1)[0])
    df["trend_tmax_3d"] = trend

    # MOS revisions and disagreement
    mos_vars = ["tmp", "dpt", "wdr", "wsp", "p12", "q12", "cig"]
    mos_models = ["gfs", "nam"]
    mos_df = _load_mos_rows(db_url, station_id, [m.upper() for m in mos_models], mos_vars, start_date, end_date)
    target_dates = df["target_date_local"].tolist()
    buckets = [0, 6, 12, 24, 36, 48]
    mos_wide, mos_max_asof = _select_latest_by_bucket(mos_df, target_dates, buckets)
    mos_wide = _aggregate_bucket_features(mos_wide, buckets)
    mos_wide = _add_revision_features(mos_wide, buckets)
    df = df.merge(mos_wide, on="target_date_local", how="left")

    # MOS vs observed mismatch
    df["obs06z_minus_mos_tmpmin"] = pd.to_numeric(df.get("T06_adj"), errors="coerce") - pd.to_numeric(
        df.get("feat_tmp_min_mean_models"), errors="coerce"
    )
    df["obs06z_minus_mos_tmpmax"] = pd.to_numeric(df.get("T06_adj"), errors="coerce") - pd.to_numeric(
        df.get("feat_tmp_max_mean_models"), errors="coerce"
    )
    df["obs00z_minus_mos_tmpmin"] = pd.to_numeric(df.get("T00"), errors="coerce") - pd.to_numeric(
        df.get("feat_tmp_min_mean_models"), errors="coerce"
    )

    audit = {
        "minute_max_ts_utc_tminus1_max": str(minute_audit["minute_max_ts_utc_tminus1"].max()),
        "minute_max_ts_utc_early_max": str(minute_audit["minute_max_ts_utc_early"].max()),
        "mos_max_asof_used_max": str(mos_max_asof.max()),
    }
    df["mos_max_asof_used"] = mos_max_asof.values
    return df, audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V6 feature store with full minute profiles + MOS revisions.")
    parser.add_argument(
        "--base-feature-store",
        default="artifacts/experiments/winners/E37_V5_MINUTE_CONDENSED_V1/feature_store_e37_minute_condensed.parquet",
    )
    parser.add_argument("--minute-dir", default="data/iem_minute_data/MIA/tmpf/UTC/yearly")
    parser.add_argument("--db-url", default="mysql+pymysql://root:root@localhost:3306/weather_predictionmarkets")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/experiments") / f"V6_SUITE_{utc_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, audit = build_feature_store(
        Path(args.base_feature_store),
        Path(args.minute_dir),
        args.db_url,
    )

    feature_store_path = out_dir / "feature_store_v6.parquet"
    df.to_parquet(feature_store_path, index=False)

    (out_dir / "feature_store_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "feature_store_columns.json").write_text(
        json.dumps({"columns": df.columns.tolist()}, indent=2), encoding="utf-8"
    )
    print(f"Wrote V6 feature store to {feature_store_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
