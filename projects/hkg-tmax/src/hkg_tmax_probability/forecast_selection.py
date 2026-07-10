"""Point-in-time official forecast selection for HKG Tmax probability models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HKT = ZoneInfo("Asia/Hong_Kong")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class CutoffProfile:
    name: str
    hkt_time: str
    primary: bool = False

    @property
    def time_object(self) -> time:
        hour, minute = self.hkt_time.split(":")
        return time(int(hour), int(minute), tzinfo=HKT)


def target_cutoff_utc(target_date: date | pd.Timestamp, cutoff_hkt: str) -> pd.Timestamp:
    """Return target-date T-1 cutoff timestamp in UTC."""
    ts = pd.Timestamp(target_date).date()
    hour, minute = [int(part) for part in cutoff_hkt.split(":")]
    local_dt = datetime.combine(ts - timedelta(days=1), time(hour=hour, minute=minute), tzinfo=HKT)
    return pd.Timestamp(local_dt.astimezone(UTC))


def build_cutoff_frame(target_dates: pd.Series, profiles: list[CutoffProfile]) -> pd.DataFrame:
    rows = []
    for target_date_value in pd.to_datetime(target_dates).dt.date.unique():
        for profile in profiles:
            rows.append(
                {
                    "target_date": pd.Timestamp(target_date_value),
                    "cutoff_profile": profile.name,
                    "cutoff_hkt": profile.hkt_time,
                    "cutoff_at_utc": target_cutoff_utc(target_date_value, profile.hkt_time),
                    "is_primary_cutoff": bool(profile.primary),
                }
            )
    return pd.DataFrame(rows)


def _deterministic_sort_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "target_date",
        "cutoff_profile",
        "issue_at_utc",
        "snapshot_at_utc",
        "ingested_at_utc",
        "source_archive_mtime_utc",
        "raw_sha256",
        "bulletin_id",
    ]
    return [column for column in candidates if column in frame.columns]


def select_latest_eligible_forecasts(
    forecasts: pd.DataFrame,
    target_dates: pd.Series,
    profiles: list[CutoffProfile],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select latest eligible official forecast before each cutoff.

    Returns selected rows and all eligible revision rows.  Sorting is stable and
    deterministic; the last row after issue/snapshot/ingest/archive/hash/id sort
    is the selected anchor.
    """
    if forecasts.empty:
        return pd.DataFrame(), pd.DataFrame()
    working = forecasts.copy()
    working["target_date"] = pd.to_datetime(working["target_date"])
    for column in ["issue_at_utc", "snapshot_at_utc", "ingested_at_utc", "source_archive_mtime_utc"]:
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], utc=True, errors="coerce")

    cutoffs = build_cutoff_frame(target_dates, profiles)
    merged = cutoffs.merge(working, on="target_date", how="left", suffixes=("", "_forecast"))
    merged = merged[merged["issue_at_utc"].notna() & (merged["issue_at_utc"] <= merged["cutoff_at_utc"])].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    for column in ["raw_sha256", "bulletin_id"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna("")

    sort_columns = _deterministic_sort_columns(merged)
    merged = merged.sort_values(sort_columns, kind="mergesort")
    merged["revision_rank"] = merged.groupby(["target_date", "cutoff_profile"], sort=False).cumcount() + 1
    merged["eligible_revision_count"] = merged.groupby(["target_date", "cutoff_profile"])["revision_rank"].transform("max")
    selected = merged.groupby(["target_date", "cutoff_profile"], sort=False).tail(1).copy()
    selected["selected_rank"] = selected["revision_rank"]
    return selected.reset_index(drop=True), merged.reset_index(drop=True)


def build_revision_features(eligible_rows: pd.DataFrame) -> pd.DataFrame:
    if eligible_rows.empty:
        return pd.DataFrame()
    frame = eligible_rows.copy()
    frame = frame.sort_values(["target_date", "cutoff_profile", "issue_at_utc", "snapshot_at_utc"], kind="mergesort")
    grouped = frame.groupby(["target_date", "cutoff_profile"], sort=False)
    first = grouped.first(numeric_only=False)
    last = grouped.last(numeric_only=False)
    agg = grouped.agg(
        revision_count=("forecast_max_c", "size"),
        forecast_max_min_path=("forecast_max_c", "min"),
        forecast_max_max_path=("forecast_max_c", "max"),
        forecast_max_mean_path=("forecast_max_c", "mean"),
        forecast_max_std_path=("forecast_max_c", "std"),
        forecast_min_min_path=("forecast_min_c", "min"),
        forecast_min_max_path=("forecast_min_c", "max"),
        forecast_range_mean_path=("forecast_range_c", "mean"),
        first_issue_at_utc=("issue_at_utc", "first"),
        latest_issue_at_utc=("issue_at_utc", "last"),
    )
    agg["first_forecast_max_c"] = first["forecast_max_c"]
    agg["latest_forecast_max_c"] = last["forecast_max_c"]
    agg["first_forecast_min_c"] = first["forecast_min_c"]
    agg["latest_forecast_min_c"] = last["forecast_min_c"]
    agg["forecast_max_revision_c"] = agg["latest_forecast_max_c"] - agg["first_forecast_max_c"]
    agg["forecast_min_revision_c"] = agg["latest_forecast_min_c"] - agg["first_forecast_min_c"]
    agg["forecast_max_path_width_c"] = agg["forecast_max_max_path"] - agg["forecast_max_min_path"]
    agg["forecast_max_std_path"] = agg["forecast_max_std_path"].fillna(0.0)

    def slope(values: pd.Series) -> float:
        clean = values.astype(float).to_numpy()
        if len(clean) < 2:
            return 0.0
        x = np.arange(len(clean), dtype=float)
        return float(np.polyfit(x, clean, 1)[0])

    slopes = grouped["forecast_max_c"].apply(slope).rename("forecast_max_path_slope_c_per_revision")
    agg = agg.join(slopes)
    agg["revision_direction"] = np.select(
        [agg["forecast_max_revision_c"] > 0, agg["forecast_max_revision_c"] < 0],
        ["up", "down"],
        default="flat",
    )
    return agg.reset_index()
