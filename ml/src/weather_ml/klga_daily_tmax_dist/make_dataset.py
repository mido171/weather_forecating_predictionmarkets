from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import time

import json
import pandas as pd

from .config import BANNED_OBS_COLUMNS, PipelineConfig
from .db import create_engine_from_url, ensure_required_indexes, fetch_daily_max, fetch_observations
from .features import build_daily_prior_frame, build_feature_rows, prepare_station_series
from .logging_utils import format_duration
from .timegrid import make_calendar_grid


@dataclass(frozen=True)
class DatasetBuildResult:
    feature_store_path: Path
    integrity_path: Path
    rows: int
    dates: int
    created_indexes: list[str]


def _compute_missing_rates(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if df.empty:
        return out
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = float(pd.to_numeric(df[c], errors="coerce").isna().mean())
    return out


def build_feature_store(
    *,
    cfg: PipelineConfig,
    mysql_url: str | None = None,
    output_root: Path | None = None,
    logger: logging.Logger | None = None,
    log_every_rows: int = 2000,
    log_every_seconds: float = 20.0,
) -> DatasetBuildResult:
    active_logger = logger or logging.getLogger(__name__)
    t0 = time.perf_counter()
    active_logger.info("DATASET_BUILD_START station=%s", cfg.target_station_id)

    engine = create_engine_from_url(mysql_url)
    created_indexes = ensure_required_indexes(engine)
    active_logger.info("DATASET_INDEX_CHECK created_indexes=%s", created_indexes)

    split = cfg.split
    daily_df = fetch_daily_max(
        engine,
        request_location_id=cfg.target_station_id,
        start_date=split.train_start,
        end_date=split.test_end,
    )
    if daily_df.empty:
        raise ValueError("No daily max rows found for KLGA in configured split range.")
    active_logger.info(
        "DATASET_DAILY_LOADED rows=%d date_min=%s date_max=%s",
        len(daily_df),
        daily_df["target_date_local"].min(),
        daily_df["target_date_local"].max(),
    )

    station_zoneid_values = set(daily_df["station_zoneid"].dropna().astype(str))
    if station_zoneid_values and station_zoneid_values != {"America/New_York"}:
        raise ValueError(f"Unexpected station_zoneid values: {sorted(station_zoneid_values)}")

    date_list = sorted(set(daily_df["target_date_local"]))
    calendar_df = make_calendar_grid(date_list, tz=cfg.local_zone)
    if calendar_df.empty:
        raise ValueError("Cutoff calendar grid is empty.")
    active_logger.info(
        "DATASET_CALENDAR_BUILT rows=%d dates=%d",
        len(calendar_df),
        calendar_df["target_date_local"].nunique(),
    )

    start_obs_utc = pd.Timestamp(calendar_df["midnight_utc"].min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    end_obs_utc = pd.Timestamp(calendar_df["cutoff_utc"].max()).tz_convert("UTC")

    obs_df = fetch_observations(
        engine,
        request_location_ids=cfg.all_station_ids,
        start_utc=start_obs_utc.to_pydatetime(),
        end_utc=end_obs_utc.to_pydatetime(),
    )
    if obs_df.empty:
        raise ValueError("No observation rows returned for configured stations/date range.")
    active_logger.info(
        "DATASET_OBS_LOADED rows=%d min_utc=%s max_utc=%s",
        len(obs_df),
        obs_df["valid_time_utc"].min(),
        obs_df["valid_time_utc"].max(),
    )

    station_series = prepare_station_series(obs_df, station_ids=cfg.all_station_ids)
    daily_prior_df = build_daily_prior_frame(daily_df)
    feature_df, audit = build_feature_rows(
        calendar_df=calendar_df,
        station_series=station_series,
        daily_truth_df=daily_df,
        daily_prior_df=daily_prior_df,
        cfg=cfg,
        logger=active_logger,
        log_every_rows=log_every_rows,
        log_every_seconds=log_every_seconds,
    )

    for banned in BANNED_OBS_COLUMNS:
        if banned in feature_df.columns:
            raise AssertionError(f"Banned column leaked into features: {banned}")

    out_root = output_root or cfg.output_root
    feature_dir = out_root / "feature_store"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_store_path = feature_dir / "klga_feature_store.parquet"
    feature_df.to_parquet(feature_store_path, index=False)
    active_logger.info("DATASET_FEATURE_STORE_WRITTEN path=%s rows=%d", feature_store_path, len(feature_df))

    integrity_payload: dict[str, Any] = {
        "rows": int(len(feature_df)),
        "dates": int(feature_df["target_date_local"].nunique()),
        "date_min": str(feature_df["target_date_local"].min()),
        "date_max": str(feature_df["target_date_local"].max()),
        "created_indexes": created_indexes,
        "calendar_rows": int(len(calendar_df)),
        "calendar_unique_dates": int(calendar_df["target_date_local"].nunique()),
        "obs_rows_loaded": int(len(obs_df)),
        "obs_time_min_utc": str(obs_df["valid_time_utc"].min()),
        "obs_time_max_utc": str(obs_df["valid_time_utc"].max()),
        "daily_rows_loaded": int(len(daily_df)),
        "audit": audit,
        "missing_rate_numeric": _compute_missing_rates(feature_df),
    }
    integrity_path = feature_dir / "klga_feature_store_integrity.json"
    integrity_path.write_text(json.dumps(integrity_payload, indent=2, sort_keys=True), encoding="utf-8")
    active_logger.info("DATASET_INTEGRITY_WRITTEN path=%s", integrity_path)
    active_logger.info(
        "DATASET_BUILD_DONE elapsed=%s",
        format_duration(time.perf_counter() - t0),
    )

    return DatasetBuildResult(
        feature_store_path=feature_store_path,
        integrity_path=integrity_path,
        rows=int(len(feature_df)),
        dates=int(feature_df["target_date_local"].nunique()),
        created_indexes=created_indexes,
    )
