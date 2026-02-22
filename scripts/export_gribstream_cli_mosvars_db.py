from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ml_live.db.mysql import MysqlConfig, MySqlStore


MOS_VARIABLE_CODES = [
    "p06",
    "p12",
    "q06",
    "q12",
    "t06",
    "t06_1",
    "t06_2",
    "tmp",
    "vis",
    "wdr",
    "wsp",
    "cig",
    "dpt",
]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _mysql_from_env() -> MysqlConfig:
    return MysqlConfig(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DB", "weather_predictionmarkets"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
    )


def _fetch_min_max_dates(store: MySqlStore, station_id: str, table: str) -> tuple[date | None, date | None]:
    sql = f"""
        SELECT MIN(target_date_local) AS min_date, MAX(target_date_local) AS max_date
        FROM {table}
        WHERE station_id = :station_id
    """
    with store.engine.begin() as conn:
        row = conn.execute(text(sql), {"station_id": station_id}).mappings().first()
    if not row:
        return None, None
    return row["min_date"], row["max_date"]


def _resolve_date_range(store: MySqlStore, station_id: str) -> tuple[date, date]:
    grib_min, grib_max = _fetch_min_max_dates(store, station_id, "gribstream_daily_feature")
    cli_min, cli_max = _fetch_min_max_dates(store, station_id, "cli_daily")
    if not grib_min or not grib_max:
        raise ValueError(f"No gribstream_daily_feature rows found for {station_id}")
    if not cli_min or not cli_max:
        raise ValueError(f"No cli_daily rows found for {station_id}")
    start_date = max(grib_min, cli_min)
    end_date = min(grib_max, cli_max)
    if start_date > end_date:
        raise ValueError(
            f"No overlapping date range for {station_id}: "
            f"grib={grib_min}..{grib_max}, cli={cli_min}..{cli_max}"
        )
    return start_date, end_date


def _expected_asof(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc) - timedelta(days=1)


def _build_dataset(
    store: MySqlStore,
    station_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    asof_cap = _expected_asof(end_date)

    grib_df = store.fetch_gribstream_daily_feature_history(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
        asof_utc=asof_cap,
    )
    if grib_df.empty:
        raise ValueError(f"No gribstream rows found for {station_id} in range {start_date}..{end_date}")

    mos_nx = store.fetch_mos_n_x_history(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
        asof_utc=asof_cap,
    )
    mos_vars = store.fetch_mos_variable_history(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
        variable_codes=MOS_VARIABLE_CODES,
    )
    # Some stations (e.g., non-ICAO "city" stations like KNYC) may not have MOS rows.
    # Ensure we still have a mergeable frame so downstream joins don't crash.
    if mos_vars.empty or "asof_utc" not in mos_vars.columns:
        mos_vars = pd.DataFrame(columns=["station_id", "target_date_local", "asof_utc"])
    cli_df = store.fetch_cli_truth_history(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
    )

    merged = grib_df.merge(
        mos_nx,
        on=["station_id", "target_date_local", "asof_utc"],
        how="left",
    )
    merged = merged.merge(
        mos_vars,
        on=["station_id", "target_date_local", "asof_utc"],
        how="left",
    )
    merged = merged.merge(
        cli_df,
        on=["station_id", "target_date_local"],
        how="left",
    )
    merged = merged.dropna(subset=["actual_tmax_f"]).reset_index(drop=True)

    # Ensure all expected MOS variable columns exist, even if the station has no MOS rows.
    expected_mos_cols = []
    for model in ["gfs", "nam"]:
        for var in MOS_VARIABLE_CODES:
            for suffix in ["min", "max", "mean", "median", "count"]:
                expected_mos_cols.append(f"mos_{model}_{var}_{suffix}")
    for col in expected_mos_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "station_id",
        "target_date_local",
        "asof_utc",
        "nbm_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
        "gefsatmosmean_tmax_f",
        "gefsatmos_tmp_spread_f",
        "gfs_n_x_max",
        "nam_n_x_max",
    ]
    mos_cols = [c for c in df.columns if c.startswith("mos_")]
    mos_cols = sorted(mos_cols)
    remaining = [c for c in df.columns if c not in base and c not in mos_cols]
    ordered = base + mos_cols + [c for c in remaining if c not in ("actual_tmax_f",)] + ["actual_tmax_f"]
    seen = set()
    final = []
    for col in ordered:
        if col in df.columns and col not in seen:
            final.append(col)
            seen.add(col)
    # Append any stragglers.
    for col in df.columns:
        if col not in seen:
            final.append(col)
            seen.add(col)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Export gribstream+MOS+CLI dataset for ML training.")
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    cfg = _mysql_from_env()
    store = MySqlStore(cfg)

    if args.start_date and args.end_date:
        start_date = _parse_date(args.start_date)
        end_date = _parse_date(args.end_date)
        if start_date is None or end_date is None:
            raise ValueError("Invalid start-date/end-date. Use YYYY-MM-DD.")
    else:
        start_date, end_date = _resolve_date_range(store, args.station_id)

    df = _build_dataset(store, args.station_id, start_date, end_date)
    if df.empty:
        raise ValueError("No rows after merges; check source data coverage.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_columns(df)
    df = df[ordered]
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
