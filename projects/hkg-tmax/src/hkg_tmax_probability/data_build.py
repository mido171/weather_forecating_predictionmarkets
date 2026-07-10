"""PostgreSQL-backed modeling table build for HKG Tmax probability buckets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS, bucket_index, bucket_key
from hkg_tmax_probability.forecast_selection import CutoffProfile, build_revision_features, select_latest_eligible_forecasts

DEFAULT_DATABASE_URL = os.environ.get("HKG_TMAX_DATABASE_URL", "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research")


def query_frame(database_url: str, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with psycopg.connect(database_url) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_targets(database_url: str = DEFAULT_DATABASE_URL) -> pd.DataFrame:
    sql = """
    select local_date::date as target_date,
           target_tmax_c::numeric::float8 as target_tmax_c,
           target_station,
           target_source_id,
           retrieved_at_utc,
           quality_status,
           'canonical_core' as target_table
    from label_core.hko_daily_tmax
    where local_date >= date '2000-01-01'
      and target_tmax_c is not null
    union all
    select local_date::date as target_date,
           target_tmax_c::numeric::float8 as target_tmax_c,
           target_station,
           target_source_id,
           retrieved_at_utc,
           quality_status,
           'sealed_confirmation' as target_table
    from sealed_confirmation.hko_daily_tmax
    where local_date >= date '2024-01-01'
      and target_tmax_c is not null
    """
    frame = query_frame(database_url, sql)
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame = frame.sort_values(["target_date", "target_table"]).drop_duplicates("target_date", keep="last")
    frame["target_tmax_1dp"] = frame["target_tmax_c"].map(lambda value: float(round(value, 1)))
    frame["bucket_key"] = frame["target_tmax_c"].map(bucket_key)
    frame["bucket_index"] = frame["target_tmax_c"].map(bucket_index)
    frame["target_year"] = frame["target_date"].dt.year
    frame["target_month"] = frame["target_date"].dt.month
    frame["target_dayofyear"] = frame["target_date"].dt.dayofyear
    frame["season"] = frame["target_month"].map(month_to_season)
    return frame.reset_index(drop=True)


def load_strict_info_gov_forecasts(database_url: str = DEFAULT_DATABASE_URL) -> pd.DataFrame:
    sql = """
    select bulletin_id,
           source,
           source_url,
           product_type,
           index_date::date,
           snapshot_at_hkt,
           snapshot_at_utc,
           issue_at_hkt,
           issue_at_utc,
           target_date::date,
           target_issue_lead_days,
           forecast_min_c::float8,
           forecast_max_c::float8,
           forecast_range_c::float8,
           forecast_midpoint_c::float8,
           row_quality_status,
           temperature_text,
           raw_sha256,
           raw_path,
           source_archive_path,
           source_archive_mtime_utc,
           ingested_at_utc
    from public.hko_historical_forecasts_2000_2026
    where source = 'info_gov'
      and product_type = 'local'
      and row_quality_status = 'usable_local_minmax'
      and target_issue_lead_days = 1
      and target_date is not null
      and issue_at_utc is not null
      and forecast_min_c is not null
      and forecast_max_c is not null
    """
    frame = query_frame(database_url, sql)
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    for column in ["snapshot_at_utc", "issue_at_utc", "source_archive_mtime_utc", "ingested_at_utc"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["forecast_range_c"] = frame["forecast_range_c"].fillna(frame["forecast_max_c"] - frame["forecast_min_c"])
    frame["forecast_midpoint_c"] = frame["forecast_midpoint_c"].fillna((frame["forecast_max_c"] + frame["forecast_min_c"]) / 2.0)
    return frame.reset_index(drop=True)


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def assign_split_label(target_date: pd.Timestamp) -> str:
    date_value = pd.Timestamp(target_date)
    if pd.Timestamp("2011-01-01") <= date_value <= pd.Timestamp("2013-12-31"):
        return "fold1_validation"
    if pd.Timestamp("2014-01-01") <= date_value <= pd.Timestamp("2016-12-31"):
        return "fold2_validation"
    if pd.Timestamp("2017-01-01") <= date_value <= pd.Timestamp("2019-12-31"):
        return "fold3_validation"
    if pd.Timestamp("2020-01-01") <= date_value <= pd.Timestamp("2021-12-31"):
        return "fold4_validation"
    if pd.Timestamp("2022-01-01") <= date_value <= pd.Timestamp("2023-12-31"):
        return "presealed_holdout"
    if pd.Timestamp("2024-01-01") <= date_value <= pd.Timestamp("2026-05-31"):
        return "sealed_confirmation"
    return "training_pool"


def official_max_bin(value: float) -> str:
    rounded = int(round(float(value)))
    if rounded <= 24:
        return "24_or_below"
    if rounded >= 34:
        return "34_or_higher"
    return str(rounded)


def build_modeling_table(config: dict[str, Any], database_url: str = DEFAULT_DATABASE_URL) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    targets = load_targets(database_url)
    forecasts = load_strict_info_gov_forecasts(database_url)
    profiles = [
        CutoffProfile(name=item["name"], hkt_time=item["hkt_time"], primary=bool(item.get("primary", False)))
        for item in config["cutoffs"]
    ]
    selected, eligible = select_latest_eligible_forecasts(forecasts, targets["target_date"], profiles)
    revisions = build_revision_features(eligible)
    if selected.empty:
        raise RuntimeError("No strict eligible official forecasts were selected before configured cutoffs.")

    selected_slim = selected[
        [
            "target_date",
            "cutoff_profile",
            "cutoff_hkt",
            "cutoff_at_utc",
            "is_primary_cutoff",
            "bulletin_id",
            "source",
            "source_url",
            "product_type",
            "index_date",
            "snapshot_at_utc",
            "issue_at_utc",
            "forecast_min_c",
            "forecast_max_c",
            "forecast_range_c",
            "forecast_midpoint_c",
            "row_quality_status",
            "raw_sha256",
            "raw_path",
            "source_archive_path",
            "ingested_at_utc",
            "selected_rank",
            "eligible_revision_count",
        ]
    ].copy()
    modeling = targets.merge(selected_slim, on="target_date", how="inner").merge(
        revisions, on=["target_date", "cutoff_profile"], how="left", suffixes=("", "_revision")
    )
    modeling["residual_c"] = (modeling["target_tmax_c"] - modeling["forecast_max_c"]).round(1)
    modeling["residual_tenths"] = np.rint(modeling["residual_c"] * 10).astype(int)
    modeling["forecast_max_tenths"] = np.rint(modeling["forecast_max_c"] * 10).astype(int)
    modeling["forecast_range_c"] = modeling["forecast_range_c"].fillna(modeling["forecast_max_c"] - modeling["forecast_min_c"])
    modeling["issue_hour_hkt"] = pd.to_datetime(modeling["issue_at_utc"], utc=True).dt.tz_convert("Asia/Hong_Kong").dt.hour
    modeling["official_max_bin"] = modeling["forecast_max_c"].map(official_max_bin)
    modeling["official_max_round"] = np.rint(modeling["forecast_max_c"]).astype(int)
    modeling["split_label"] = modeling["target_date"].map(assign_split_label)
    modeling["row_identity"] = modeling["target_date"].dt.strftime("%Y-%m-%d") + "|" + modeling["cutoff_profile"]
    modeling = modeling.sort_values(["target_date", "cutoff_profile"]).reset_index(drop=True)

    audit = {
        "targets_loaded": int(len(targets)),
        "strict_forecasts_loaded": int(len(forecasts)),
        "eligible_revision_rows": int(len(eligible)),
        "selected_forecast_rows": int(len(selected_slim)),
        "modeling_rows": int(len(modeling)),
        "primary_modeling_rows": int(modeling["is_primary_cutoff"].sum()),
        "bucket_rule_examples": {
            "24.9": bucket_key("24.9"),
            "25.0": bucket_key("25.0"),
            "31.9": bucket_key("31.9"),
            "32.0": bucket_key("32.0"),
            "34.0": bucket_key("34.0"),
        },
        "rows_by_split": modeling.groupby(["split_label", "cutoff_profile"]).size().reset_index(name="rows").to_dict(orient="records"),
        "bucket_keys": list(BUCKET_KEYS),
    }
    return modeling, selected_slim.reset_index(drop=True), eligible.reset_index(drop=True), audit


def modeling_table_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [{"name": column, "dtype": str(dtype)} for column, dtype in frame.dtypes.items()],
        "row_count": int(len(frame)),
        "primary_key": ["target_date", "cutoff_profile"],
        "notes": "Rows are one target date by cutoff profile. Primary leaderboard uses is_primary_cutoff=true.",
    }


def write_modeling_artifacts(
    output_dir: Path,
    modeling: pd.DataFrame,
    selected: pd.DataFrame,
    eligible: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    modeling.to_parquet(output_dir / "modeling_table.parquet", index=False)
    selected.to_parquet(output_dir / "selected_forecast_rows.parquet", index=False)
    eligible.to_parquet(output_dir / "eligible_forecast_revision_rows.parquet", index=False)
    (output_dir / "modeling_table_schema.json").write_text(json.dumps(modeling_table_schema(modeling), indent=2, default=str), encoding="utf-8")
    (output_dir / "row_count_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    modeling.groupby(["split_label", "cutoff_profile"]).size().reset_index(name="rows").to_csv(
        output_dir / "row_counts_by_split_cutoff.csv", index=False
    )
    modeling.groupby(["target_year", "target_month", "cutoff_profile"]).size().reset_index(name="rows").to_csv(
        output_dir / "row_counts_by_month_cutoff.csv", index=False
    )
    source_audit = (
        selected.groupby(["source", "product_type", "row_quality_status", "target_issue_lead_days" if "target_issue_lead_days" in selected.columns else "source"])
        .size()
        .reset_index(name="rows")
        if "target_issue_lead_days" in selected.columns
        else selected.groupby(["source", "product_type", "row_quality_status"]).size().reset_index(name="rows")
    )
    source_audit.to_csv(output_dir / "source_eligibility_audit.csv", index=False)
