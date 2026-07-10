from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    HEADLINE_END,
    RSS_FORECAST_PATH,
    require_no_confirmation_dates,
    select_latest_pre_cutoff_forecast,
)

DEFAULT_ARCHIVE_DB = Path(r"C:\hko_press_2000_2026\metadata\archive.sqlite3")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "0000_research_state_and_data_contract"
    / "hko_official_backfill_monitor"
    / "artifacts"
)
DEFAULT_TARGET_START = pd.Timestamp("2000-01-02")
DEFAULT_TARGET_END = HEADLINE_END


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_sql_frame(db_path: Path, query: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing archive DB: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return pd.read_sql(query, connection)


def read_archive_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    return {
        "candidates": read_sql_frame(
            db_path,
            """
            SELECT source, index_date, title, product_type, url, discovered_at_utc
            FROM candidates
            WHERE source='info_gov'
            """,
        ),
        "retrievals": read_sql_frame(
            db_path,
            """
            SELECT source, url, attempted_at_utc, status_code, error, content_sha256, raw_path
            FROM retrievals
            WHERE source IN ('info_gov_index', 'info_gov_bulletin')
            """,
        ),
        "bulletins": read_sql_frame(
            db_path,
            """
            SELECT bulletin_id, source, source_url, product_type, title, index_date, issue_at_hkt,
                   raw_sha256, raw_path, target_date, forecast_min_c, forecast_max_c, parse_status
            FROM bulletins
            WHERE source='info_gov'
            """,
        ),
        "forecast_days": read_sql_frame(
            db_path,
            """
            SELECT bulletin_id, source, source_url, product_type, issue_at_hkt, target_date,
                   forecast_min_c, forecast_max_c, rh_min_pct, rh_max_pct, wind_text, weather_text,
                   raw_sha256
            FROM forecast_days
            WHERE source='info_gov'
            """,
        ),
    }


def normalize_date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def date_segments(dates: pd.Series) -> pd.DataFrame:
    unique_dates = sorted(set(normalize_date_series(dates).dropna()))
    if not unique_dates:
        return pd.DataFrame(columns=["segment_id", "first_date", "last_date", "observed_days"])

    rows: list[dict[str, object]] = []
    segment_id = 1
    start = previous = unique_dates[0]
    for current in unique_dates[1:]:
        if (current - previous).days == 1:
            previous = current
            continue
        rows.append(
            {
                "segment_id": segment_id,
                "first_date": start.date().isoformat(),
                "last_date": previous.date().isoformat(),
                "observed_days": int((previous - start).days + 1),
            }
        )
        segment_id += 1
        start = previous = current
    rows.append(
        {
            "segment_id": segment_id,
            "first_date": start.date().isoformat(),
            "last_date": previous.date().isoformat(),
            "observed_days": int((previous - start).days + 1),
        }
    )
    return pd.DataFrame(rows)


def gap_rows_from_dates(dates: pd.Series, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    expected = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    observed = set(normalize_date_series(dates).dropna())
    missing = pd.Series([value for value in expected if value not in observed], dtype="datetime64[ns]")
    if missing.empty:
        return pd.DataFrame(columns=["gap_id", "missing_start", "missing_end", "missing_days"])

    segments = date_segments(missing)
    return segments.rename(
        columns={"segment_id": "gap_id", "first_date": "missing_start", "last_date": "missing_end", "observed_days": "missing_days"}
    )


def retrieval_status_summary(retrievals: pd.DataFrame) -> pd.DataFrame:
    if retrievals.empty:
        return pd.DataFrame(columns=["source", "status_code", "retrieval_rows", "error_rows", "last_attempted_at_utc"])
    frame = retrievals.copy()
    frame["status_code"] = pd.to_numeric(frame["status_code"], errors="coerce").astype("Int64")
    frame["has_error"] = frame["error"].fillna("").astype(str).ne("")
    return (
        frame.groupby(["source", "status_code"], dropna=False, observed=True)
        .agg(
            retrieval_rows=("url", "count"),
            unique_urls=("url", "nunique"),
            error_rows=("has_error", "sum"),
            last_attempted_at_utc=("attempted_at_utc", "max"),
        )
        .reset_index()
        .sort_values(["source", "status_code"])
    )


def candidate_raw_coverage(candidates: pd.DataFrame, retrievals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = candidates.copy()
    cand["index_date"] = normalize_date_series(cand["index_date"])
    cand["index_year"] = cand["index_date"].dt.year

    bulletin_retrievals = retrievals[retrievals["source"].eq("info_gov_bulletin")].copy()
    bulletin_retrievals["status_code"] = pd.to_numeric(bulletin_retrievals["status_code"], errors="coerce")
    success = bulletin_retrievals[
        bulletin_retrievals["status_code"].between(200, 299, inclusive="both")
        & bulletin_retrievals["raw_path"].notna()
        & bulletin_retrievals["raw_path"].astype(str).ne("")
    ]
    bulletin_retrievals["error_text"] = bulletin_retrievals["error"].fillna("").astype(str)
    attempts = (
        bulletin_retrievals.groupby("url", observed=True)
        .agg(
            retrieval_attempts=("url", "count"),
            last_attempted_at_utc=("attempted_at_utc", "max"),
            max_status_code=("status_code", "max"),
            sample_error=("error_text", lambda values: next((value for value in values if value), "")),
        )
        .reset_index()
    )
    status = cand.merge(attempts, on="url", how="left")
    status["has_successful_raw_detail"] = status["url"].isin(set(success["url"]))
    status["retrieval_attempts"] = pd.to_numeric(status["retrieval_attempts"], errors="coerce").fillna(0).astype(int)

    coverage = (
        status.groupby(["index_year", "product_type"], observed=True)
        .agg(
            candidate_urls=("url", "nunique"),
            successful_raw_urls=("has_successful_raw_detail", "sum"),
            attempted_urls=("retrieval_attempts", lambda s: int((s > 0).sum())),
        )
        .reset_index()
        .sort_values(["index_year", "product_type"])
    )
    coverage["missing_success_urls"] = coverage["candidate_urls"] - coverage["successful_raw_urls"]
    coverage["raw_success_coverage_pct"] = np.where(
        coverage["candidate_urls"] > 0,
        coverage["successful_raw_urls"] / coverage["candidate_urls"],
        np.nan,
    )

    missing = status[~status["has_successful_raw_detail"]].copy()
    missing["index_date"] = missing["index_date"].dt.date.astype(str)
    missing = missing[
        [
            "index_date",
            "index_year",
            "product_type",
            "title",
            "url",
            "retrieval_attempts",
            "last_attempted_at_utc",
            "max_status_code",
            "sample_error",
        ]
    ].sort_values(["index_date", "url"])
    return coverage.reset_index(drop=True), missing.reset_index(drop=True)


def scoreable_archive_forecast_days(forecast_days: pd.DataFrame) -> pd.DataFrame:
    if forecast_days.empty:
        return forecast_days.copy()
    out = forecast_days.copy()
    out["issue_at_utc"] = pd.to_datetime(out["issue_at_hkt"], errors="coerce", utc=True)
    out["target_date"] = normalize_date_series(out["target_date"])
    out["forecast_max_c"] = pd.to_numeric(out["forecast_max_c"], errors="coerce")
    out["forecast_min_c"] = pd.to_numeric(out["forecast_min_c"], errors="coerce")
    valid_max = out["forecast_max_c"].between(-5.0, 45.0, inclusive="both")
    valid_min = out["forecast_min_c"].isna() | out["forecast_min_c"].between(-5.0, 45.0, inclusive="both")
    ordered_range = out["forecast_min_c"].isna() | (out["forecast_min_c"] <= out["forecast_max_c"])
    issue_date_hkt = out["issue_at_utc"].dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None).dt.normalize()
    out["target_issue_lead_days"] = (out["target_date"] - issue_date_hkt).dt.days
    out["scoreable_row_valid"] = (
        valid_max
        & valid_min
        & ordered_range
        & out["issue_at_utc"].notna()
        & out["target_date"].notna()
        & out["target_issue_lead_days"].between(-1, 15, inclusive="both")
    )
    return out


def selected_press_from_archive(forecast_days: pd.DataFrame) -> pd.DataFrame:
    days = scoreable_archive_forecast_days(forecast_days)
    days = days[days["scoreable_row_valid"]].copy()
    if days.empty:
        return pd.DataFrame()
    selected = select_latest_pre_cutoff_forecast(
        days,
        target_col="target_date",
        issue_col="issue_at_hkt",
        max_col="forecast_max_c",
        min_col="forecast_min_c",
        source_name="hko_press_archive_live_db",
    )
    selected["forecast_source_family"] = "press_archive"
    return selected


def selected_rss_forecasts() -> pd.DataFrame:
    if not RSS_FORECAST_PATH.exists():
        return pd.DataFrame()
    rss = pd.read_parquet(RSS_FORECAST_PATH)
    selected = select_latest_pre_cutoff_forecast(
        rss,
        target_col="forecast_date",
        issue_col="available_at_hkt",
        max_col="forecast_max_temperature_c",
        min_col="forecast_min_temperature_c",
        source_name="hko_rss",
    )
    selected["forecast_source_family"] = "rss_archive"
    return selected


def target_coverage(selected: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    expected_days = int((end.normalize() - start.normalize()).days + 1)
    if selected.empty:
        return {
            "expected_target_days": expected_days,
            "observed_target_days": 0,
            "missing_target_days": expected_days,
            "first_observed_target_date": "",
            "last_observed_target_date": "",
            "segment_count": 0,
            "complete": False,
        }

    target_dates = normalize_date_series(selected["target_date"])
    in_window = target_dates[(target_dates >= start.normalize()) & (target_dates <= end.normalize())]
    observed = int(in_window.nunique())
    segments = date_segments(in_window)
    missing = expected_days - observed
    return {
        "expected_target_days": expected_days,
        "observed_target_days": observed,
        "missing_target_days": int(missing),
        "first_observed_target_date": "" if in_window.empty else in_window.min().date().isoformat(),
        "last_observed_target_date": "" if in_window.empty else in_window.max().date().isoformat(),
        "segment_count": int(len(segments)),
        "complete": bool(missing == 0 and len(segments) == 1),
    }


def source_target_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "forecast_source_family",
                "target_days",
                "first_target_date",
                "last_target_date",
                "latest_issue_utc",
            ]
        )
    frame = selected.copy()
    frame["target_date"] = normalize_date_series(frame["target_date"])
    frame["issue_utc"] = pd.to_datetime(frame["issue_utc"], errors="coerce", utc=True)
    return (
        frame.groupby("forecast_source_family", observed=True)
        .agg(
            target_days=("target_date", "nunique"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
            latest_issue_utc=("issue_utc", "max"),
        )
        .reset_index()
        .assign(
            first_target_date=lambda df: df["first_target_date"].dt.date.astype(str),
            last_target_date=lambda df: df["last_target_date"].dt.date.astype(str),
            latest_issue_utc=lambda df: df["latest_issue_utc"].astype(str),
        )
    )


def build_report(db_path: Path, *, target_start: pd.Timestamp, target_end: pd.Timestamp) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    tables = read_archive_tables(db_path)
    candidates = tables["candidates"]
    retrievals = tables["retrievals"]
    bulletins = tables["bulletins"]
    forecast_days = tables["forecast_days"]

    raw_coverage, missing_raw = candidate_raw_coverage(candidates, retrievals)
    status_summary = retrieval_status_summary(retrievals)
    scoreable_days = scoreable_archive_forecast_days(forecast_days)
    press_selected = selected_press_from_archive(forecast_days)
    rss_selected = selected_rss_forecasts()
    selected_frames = [frame for frame in (press_selected, rss_selected) if not frame.empty]
    combined_selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    combined_selected = combined_selected[combined_selected["target_date"] < CONFIRMATION_START].copy() if not combined_selected.empty else combined_selected
    require_no_confirmation_dates(
        combined_selected["target_date"] if not combined_selected.empty else [],
        context="hko official backfill monitor",
    )

    missing_scored = gap_rows_from_dates(
        combined_selected["target_date"] if not combined_selected.empty else pd.Series(dtype="datetime64[ns]"),
        start=target_start,
        end=target_end,
    )
    source_summary = source_target_summary(combined_selected)
    scoreable = scoreable_days[scoreable_days.get("scoreable_row_valid", pd.Series(dtype=bool))].copy()

    candidate_by_type = (
        candidates.assign(index_date=normalize_date_series(candidates["index_date"]))
        .groupby("product_type", observed=True)
        .agg(candidate_urls=("url", "nunique"), first_index_date=("index_date", "min"), last_index_date=("index_date", "max"))
        .reset_index()
        .assign(
            first_index_date=lambda df: df["first_index_date"].dt.date.astype(str),
            last_index_date=lambda df: df["last_index_date"].dt.date.astype(str),
        )
    )
    bulletin_by_type = (
        bulletins.assign(
            index_date=normalize_date_series(bulletins["index_date"]),
            issue_at_hkt_dt=pd.to_datetime(bulletins["issue_at_hkt"], errors="coerce"),
        )
        .groupby("product_type", observed=True)
        .agg(
            bulletins=("bulletin_id", "nunique"),
            first_index_date=("index_date", "min"),
            last_index_date=("index_date", "max"),
            first_issue_at_hkt=("issue_at_hkt_dt", "min"),
            last_issue_at_hkt=("issue_at_hkt_dt", "max"),
        )
        .reset_index()
        .assign(
            first_index_date=lambda df: df["first_index_date"].dt.date.astype(str),
            last_index_date=lambda df: df["last_index_date"].dt.date.astype(str),
            first_issue_at_hkt=lambda df: df["first_issue_at_hkt"].astype(str),
            last_issue_at_hkt=lambda df: df["last_issue_at_hkt"].astype(str),
        )
    )

    combined_coverage = target_coverage(combined_selected, start=target_start, end=target_end)
    press_coverage = target_coverage(press_selected, start=target_start, end=target_end)
    rss_coverage = target_coverage(rss_selected, start=target_start, end=target_end)

    summary: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "archive_db": str(db_path),
        "target_window_start": target_start.date().isoformat(),
        "target_window_end": target_end.date().isoformat(),
        "candidate_urls": int(candidates["url"].nunique()) if not candidates.empty else 0,
        "bulletins": int(bulletins["bulletin_id"].nunique()) if not bulletins.empty else 0,
        "forecast_day_rows": int(len(forecast_days)),
        "scoreable_archive_forecast_day_rows": int(scoreable["scoreable_row_valid"].sum()) if not scoreable.empty else 0,
        "retrieval_status": status_summary.to_dict("records"),
        "candidate_raw_missing_urls": int(len(missing_raw)),
        "candidate_raw_missing_years": sorted(int(year) for year in missing_raw["index_year"].dropna().unique()) if not missing_raw.empty else [],
        "candidate_raw_coverage_by_product": candidate_by_type.to_dict("records"),
        "bulletin_coverage_by_product": bulletin_by_type.to_dict("records"),
        "press_selected_target_coverage": press_coverage,
        "rss_selected_target_coverage": rss_coverage,
        "combined_selected_target_coverage": combined_coverage,
        "completion_status": "complete_no_gap" if combined_coverage["complete"] else "incomplete_gap_remaining",
        "first_missing_scored_gap": missing_scored.iloc[0].to_dict() if not missing_scored.empty else None,
        "largest_missing_scored_gap": (
            missing_scored.sort_values("missing_days", ascending=False).iloc[0].to_dict()
            if not missing_scored.empty
            else None
        ),
        "last_press_selected_issue_utc": ""
        if press_selected.empty
        else str(pd.to_datetime(press_selected["issue_utc"], errors="coerce", utc=True).max()),
        "last_press_selected_target_date": ""
        if press_selected.empty
        else str(normalize_date_series(press_selected["target_date"]).max().date()),
    }
    frames = {
        "raw_coverage": raw_coverage,
        "missing_raw": missing_raw,
        "retrieval_status": status_summary,
        "press_selected_segments": date_segments(press_selected["target_date"] if not press_selected.empty else pd.Series(dtype="datetime64[ns]")),
        "rss_selected_segments": date_segments(rss_selected["target_date"] if not rss_selected.empty else pd.Series(dtype="datetime64[ns]")),
        "combined_selected_segments": date_segments(combined_selected["target_date"] if not combined_selected.empty else pd.Series(dtype="datetime64[ns]")),
        "missing_scored_gaps": missing_scored,
        "source_target_summary": source_summary,
    }
    return summary, frames


def run(db_path: Path, *, output_dir: Path, target_start: pd.Timestamp, target_end: pd.Timestamp) -> dict[str, Any]:
    summary, frames = build_report(db_path, target_start=target_start, target_end=target_end)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        write_csv(output_dir / f"{name}.csv", frame)
    write_json(output_dir / "monitor_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only monitor for the HKO official forecast backfill.")
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-start", default=DEFAULT_TARGET_START.date().isoformat())
    parser.add_argument("--target-end", default=DEFAULT_TARGET_END.date().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.archive_db,
        output_dir=args.output_dir,
        target_start=pd.Timestamp(args.target_start),
        target_end=pd.Timestamp(args.target_end),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
