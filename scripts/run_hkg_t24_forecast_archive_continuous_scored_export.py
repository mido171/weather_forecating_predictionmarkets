from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_hko_press_archive_offline import (  # noqa: E402
    DEFAULT_ARCHIVE_DB,
)
from scripts.export_hko_press_archive_offline import (  # noqa: E402
    export as export_press_archive,
)
from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    DATASETS_ROOT,
    HEADLINE_END,
    PRESS_FORECAST_EXPORT_PATH,
    RESEARCH_ROOT,
    RSS_FORECAST_PATH,
    hkt_cutoff_utc_for_target_dates,
    load_features,
    markdown_table,
    require_no_confirmation_dates,
    score_prediction_frame,
    select_latest_pre_cutoff_forecast,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_press_archive_raw_detail_gap_audit import (  # noqa: E402
    DEFAULT_RAW_ROOT,
    build_year_product_coverage,
    count_raw_html_files_by_year,
    load_archive_frames,
)

FOLDER_NAME = "0044_forecast_archive_continuous_scored_export"
OUTPUT_DIR = DATASETS_ROOT / "05_hko_historical_rss_forecasts"
PRESS_EXPORT_MANIFEST_PATH = OUTPUT_DIR / "hko_press_archive_offline_export_manifest.json"
SCORED_EXPORT_PATH = OUTPUT_DIR / "hko_official_t15_scored_pre2024.parquet"
SCORED_EXPORT_CSV_PATH = OUTPUT_DIR / "hko_official_t15_scored_pre2024.csv"
SCORED_EXPORT_MANIFEST_PATH = OUTPUT_DIR / "hko_official_t15_scored_pre2024_manifest.json"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def normalize_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def season_from_month(value: object) -> str:
    try:
        month = int(value)
    except (TypeError, ValueError):
        return ""
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    return ""


def contiguous_date_segments(dates: pd.Series) -> pd.DataFrame:
    unique_dates = sorted(set(normalize_dates(dates).dropna()))
    if not unique_dates:
        return pd.DataFrame(columns=["segment_id", "first_date", "last_date", "observed_days"])

    rows: list[dict[str, object]] = []
    start = previous = unique_dates[0]
    segment_id = 1
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


def gap_rows_from_segments(segments: pd.DataFrame) -> pd.DataFrame:
    if len(segments) <= 1:
        return pd.DataFrame(columns=["gap_id", "missing_start", "missing_end", "missing_days"])

    rows: list[dict[str, object]] = []
    for gap_id, (left, right) in enumerate(
        zip(segments.iloc[:-1].itertuples(index=False), segments.iloc[1:].itertuples(index=False), strict=False),
        start=1,
    ):
        left_end = pd.Timestamp(left.last_date)
        right_start = pd.Timestamp(right.first_date)
        missing_start = left_end + pd.Timedelta(days=1)
        missing_end = right_start - pd.Timedelta(days=1)
        rows.append(
            {
                "gap_id": gap_id,
                "missing_start": missing_start.date().isoformat(),
                "missing_end": missing_end.date().isoformat(),
                "missing_days": int((missing_end - missing_start).days + 1),
            }
        )
    return pd.DataFrame(rows)


def continuity_summary(frame: pd.DataFrame, *, source_col: str = "forecast_source_family") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "forecast_source_family",
                "observed_target_days",
                "first_target_date",
                "last_target_date",
                "span_days",
                "missing_days_inside_span",
                "continuity_ratio",
                "segment_count",
                "longest_segment_days",
            ]
        )
    for source, group in frame.groupby(source_col, dropna=False, observed=True):
        dates = normalize_dates(group["target_date"]).dropna()
        segments = contiguous_date_segments(dates)
        if dates.empty:
            rows.append(
                {
                    "forecast_source_family": str(source),
                    "observed_target_days": 0,
                    "first_target_date": "",
                    "last_target_date": "",
                    "span_days": 0,
                    "missing_days_inside_span": 0,
                    "continuity_ratio": math.nan,
                    "segment_count": 0,
                    "longest_segment_days": 0,
                }
            )
            continue
        first = dates.min()
        last = dates.max()
        observed = int(dates.nunique())
        span_days = int((last - first).days + 1)
        rows.append(
            {
                "forecast_source_family": str(source),
                "observed_target_days": observed,
                "first_target_date": first.date().isoformat(),
                "last_target_date": last.date().isoformat(),
                "span_days": span_days,
                "missing_days_inside_span": int(span_days - observed),
                "continuity_ratio": float(observed / span_days) if span_days else math.nan,
                "segment_count": int(len(segments)),
                "longest_segment_days": int(segments["observed_days"].max()) if not segments.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("forecast_source_family").reset_index(drop=True)


def read_archive_summary(db_path: Path, raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates, retrievals, forecast_days = load_archive_frames(db_path)
    raw_files = count_raw_html_files_by_year(raw_root)
    coverage, missing, no_raw = build_year_product_coverage(candidates, retrievals, forecast_days)
    return coverage, missing, no_raw, raw_files


def selected_press_forecasts() -> pd.DataFrame:
    if not PRESS_FORECAST_EXPORT_PATH.exists():
        return pd.DataFrame()
    press = pd.read_parquet(PRESS_FORECAST_EXPORT_PATH)
    selected = select_latest_pre_cutoff_forecast(
        press,
        target_col="target_date",
        issue_col="issue_at_hkt",
        max_col="forecast_max_c",
        min_col="forecast_min_c",
        source_name="hko_press_archive",
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


def build_selected_forecast_inventory() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    press = selected_press_forecasts()
    rss = selected_rss_forecasts()
    frames = [frame for frame in (press, rss) if not frame.empty]
    if not frames:
        return pd.DataFrame(), press, rss
    selected = pd.concat(frames, ignore_index=True)
    selected["target_date"] = normalize_dates(selected["target_date"])
    selected["cutoff_utc"] = hkt_cutoff_utc_for_target_dates(selected["target_date"])
    selected["issue_to_cutoff_hours"] = (
        (selected["cutoff_utc"] - selected["issue_utc"]).dt.total_seconds() / 3600.0
    )
    selected["forecast_range_c"] = selected["forecast_max_c"] - selected["forecast_min_c"]
    selected["forecast_midpoint_c"] = np.where(
        selected["forecast_min_c"].notna(),
        (selected["forecast_min_c"] + selected["forecast_max_c"]) / 2.0,
        np.nan,
    )
    return selected.sort_values(["forecast_source_family", "target_date"]).reset_index(drop=True), press, rss


def build_scored_export(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    scoreable = selected[selected["target_date"] <= HEADLINE_END].copy()
    require_no_confirmation_dates(scoreable["target_date"], context="0044 scored forecast export")
    features = load_features().copy()
    keep_cols = ["target_date", "target_tmax_c", "month"]
    if "season" in features.columns:
        keep_cols.append("season")
    features = features[keep_cols].copy()
    if "season" not in features.columns:
        features["season"] = features["month"].map(season_from_month)
    scored = scoreable.merge(features, on="target_date", how="inner")
    scored["official_error_c"] = scored["forecast_max_c"] - scored["target_tmax_c"]
    scored["official_abs_error_c"] = scored["official_error_c"].abs()
    scored["official_midpoint_error_c"] = scored["forecast_midpoint_c"] - scored["target_tmax_c"]
    return scored.sort_values(["forecast_source_family", "target_date"]).reset_index(drop=True)


def source_scoreboard(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scored.empty:
        return pd.DataFrame()
    for source, group in scored.groupby("forecast_source_family", dropna=False, observed=True):
        rows.append({"forecast_source_family": str(source), **score_prediction_frame(group, "forecast_max_c")})
    total = score_prediction_frame(scored, "forecast_max_c")
    rows.append({"forecast_source_family": "all_sources_rowwise", **total})
    return pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)


def year_coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    grouped = (
        coverage.groupby("index_year", observed=True)
        .agg(
            candidate_count=("candidate_count", "sum"),
            raw_url_count=("raw_url_count", "sum"),
            parsed_forecast_day_rows=("parsed_forecast_day_rows", "sum"),
            scoreable_rows=("scoreable_rows", "sum"),
        )
        .reset_index()
    )
    grouped["raw_detail_status"] = np.where(
        grouped["candidate_count"].eq(0),
        "no_candidates",
        np.where(grouped["raw_url_count"].eq(0), "no_raw_detail", "has_raw_detail"),
    )
    grouped["raw_detail_coverage_pct"] = np.where(
        grouped["candidate_count"] > 0,
        grouped["raw_url_count"] / grouped["candidate_count"],
        math.nan,
    )
    return grouped.sort_values("index_year").reset_index(drop=True)


def write_dataset_scored_export(scored: pd.DataFrame, manifest: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not scored.empty:
        scored.to_parquet(SCORED_EXPORT_PATH, index=False)
        write_csv(SCORED_EXPORT_CSV_PATH, scored)
    write_json(SCORED_EXPORT_MANIFEST_PATH, manifest)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Forecast Archive Continuous Scored Export Promotion\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_forecast_archive_continuous_scored_export.py`:

- `{FOLDER_NAME}`: refreshed HKO press forecast export, selected latest-pre-cutoff official forecasts, pre-2024 scored export, and explicit remaining continuity blockers.

| Metric | Value |
|---|---:|
| Press selected pre-cutoff rows | {manifest['press_selected_rows']} |
| RSS selected pre-cutoff rows | {manifest['rss_selected_rows']} |
| Scored pre-2024 rows | {manifest['scored_rows']} |
| Scored date range | {manifest['scored_first_target_date']} to {manifest['scored_last_target_date']} |
| Combined continuity ratio | {manifest['combined_continuity_ratio']} |
| Continuous 2000-2023 status | {manifest['continuous_2000_2023_status']} |
| Zero raw-detail candidate years | {manifest['zero_raw_detail_years']} |

Leakage contract: the scored export excludes target dates on or after `{manifest['confirmation_start']}`.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object], scoreboard: pd.DataFrame) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# HKG Tmax Milestones\n"
    section_marker = "\n## Forecast Archive Continuous Scored Export Promotion\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""

    score_table = markdown_table(scoreboard, max_rows=10)
    zero_years = manifest["zero_raw_detail_years"]
    zero_years_text = ", ".join(str(year) for year in zero_years) if zero_years else "none"
    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_forecast_archive_continuous_scored_export.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Refreshed press export | `{manifest['press_export_scoreable_rows']}` scoreable forecast-day rows; target dates `{manifest['press_export_first_target_date']}` to `{manifest['press_export_last_target_date']}` | Promoted available raw detail |
| Latest pre-cutoff selected forecasts | press `{manifest['press_selected_rows']}` rows, RSS `{manifest['rss_selected_rows']}` rows | Scored pre-2024 only |
| Combined scored frame | `{manifest['scored_rows']}` rows, target dates `{manifest['scored_first_target_date']}` to `{manifest['scored_last_target_date']}` | Leakage guard PASS |
| Continuity | combined continuity ratio `{manifest['combined_continuity_ratio']}`; zero raw-detail years `{manifest['zero_raw_detail_years']}` | Not continuous through 2023 |
| Archive decision | `{manifest['continuous_2000_2023_status']}` | Remaining acquisition blocker explicit |

Scoreboard on the refreshed selected official forecast frame:

{score_table}

Interpretation: `0044` successfully promotes every currently parsed press forecast row from the local archive into the repo-local scored export, through press target date `{manifest['press_export_last_target_date']}`. It still does not solve the full 2000-2023 historical forecast continuity problem. The archive has candidate links through 2026, yet raw detail HTML is still absent for candidate years `{zero_years_text}`, so the official forecast frame remains non-contiguous. The correct next move is to finish/download the missing raw detail HTML where lawful and then rerun the official-anchor chain from the base correction ladder forward.
"""
    if suffix:
        if next_marker in suffix:
            before_next, _after_next = suffix.split(next_marker, 1)
            next_task = f"""{next_marker}

Implement the next forecast-archive replay gate: finish the lawful raw-detail acquisition for the remaining HKO info.gov.hk forecast candidate years, rerun `0044` after raw detail coverage changes, then rerun or replace the stale official-router chain so downstream champion screens use the expanded scored frame rather than the older partial frame.
"""
            suffix = before_next.rstrip() + next_task
        section += suffix
    write_text(path, section)


def write_outputs(
    *,
    selected: pd.DataFrame,
    press_selected: pd.DataFrame,
    rss_selected: pd.DataFrame,
    scored: pd.DataFrame,
    coverage: pd.DataFrame,
    missing: pd.DataFrame,
    no_raw: pd.DataFrame,
    raw_files: pd.DataFrame,
    press_export_summary: dict[str, Any],
    generated_at_utc: str,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    scored_segments = contiguous_date_segments(scored["target_date"] if not scored.empty else pd.Series(dtype="datetime64[ns]"))
    scored_gaps = gap_rows_from_segments(scored_segments)
    selected_summary = continuity_summary(selected)
    scored_summary = continuity_summary(scored)
    archive_years = year_coverage_summary(coverage)
    scoreboard = source_scoreboard(scored)

    write_csv(artifacts / "press_selected.csv", press_selected)
    write_csv(artifacts / "rss_selected.csv", rss_selected)
    write_csv(artifacts / "all_selected.csv", selected)
    write_csv(artifacts / "scored_pre2024.csv", scored)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "selected_continuity.csv", selected_summary)
    write_csv(artifacts / "scored_continuity.csv", scored_summary)
    write_csv(artifacts / "segments.csv", scored_segments)
    write_csv(artifacts / "gaps.csv", scored_gaps)
    write_csv(artifacts / "archive_year_product.csv", coverage)
    write_csv(artifacts / "archive_missing_raw.csv", missing)
    write_csv(artifacts / "archive_zero_raw.csv", no_raw)
    write_csv(artifacts / "archive_raw_files.csv", raw_files)
    write_csv(artifacts / "archive_year.csv", archive_years)

    combined_summary = continuity_summary(
        scored.assign(forecast_source_family="combined") if not scored.empty else scored
    )
    combined = combined_summary.iloc[0].to_dict() if not combined_summary.empty else {}
    zero_years = sorted(
        int(year)
        for year in archive_years.loc[archive_years["raw_detail_status"].eq("no_raw_detail"), "index_year"]
        .dropna()
        .unique()
    )
    status = "passed_continuous" if int(combined.get("segment_count", 0) or 0) <= 1 else "failed_not_continuous"
    manifest: dict[str, object] = {
        "generated_at_utc": generated_at_utc,
        "folder": FOLDER_NAME,
        "press_forecast_export_path": str(PRESS_FORECAST_EXPORT_PATH),
        "rss_forecast_path": str(RSS_FORECAST_PATH),
        "scored_export_path": str(SCORED_EXPORT_PATH),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "press_export_scoreable_rows": int(press_export_summary.get("scoreable_temperature_forecast_day_rows", 0)),
        "press_export_first_target_date": str(press_export_summary.get("first_target_date", "")),
        "press_export_last_target_date": str(press_export_summary.get("last_target_date", "")),
        "press_export_raw_detail_years": list(press_export_summary.get("years_with_any_raw_detail", [])),
        "press_selected_rows": int(len(press_selected)),
        "rss_selected_rows": int(len(rss_selected)),
        "selected_inventory_rows": int(len(selected)),
        "scored_rows": int(len(scored)),
        "scored_first_target_date": "" if scored.empty else _date_text(scored["target_date"].min()),
        "scored_last_target_date": "" if scored.empty else _date_text(scored["target_date"].max()),
        "combined_continuity_ratio": None
        if not combined
        else float(combined.get("continuity_ratio", math.nan)),
        "combined_segment_count": int(combined.get("segment_count", 0) or 0),
        "combined_missing_days_inside_span": int(combined.get("missing_days_inside_span", 0) or 0),
        "continuous_2000_2023_status": status,
        "zero_raw_detail_years": zero_years,
        "raw_file_years": sorted(
            int(year) for year in raw_files["raw_file_year"].dropna().unique()
        )
        if not raw_files.empty
        else [],
        "scoreboard": scoreboard.to_dict("records"),
    }
    raw_detail_years = manifest["press_export_raw_detail_years"]
    raw_detail_years_text = ", ".join(str(year) for year in raw_detail_years) if raw_detail_years else "none"
    zero_years_text = ", ".join(str(year) for year in zero_years) if zero_years else "none"
    write_json(artifacts / "promotion_summary.json", manifest)
    write_json(RESEARCH_ROOT / "forecast_archive_continuous_scored_export_manifest.json", manifest)
    write_dataset_scored_export(scored, manifest)

    best_score_text = markdown_table(scoreboard, max_rows=10)
    readme = f"""# Forecast Archive Continuous Scored Export Promotion

Generated: `{generated_at_utc}`

## Purpose

`0043` decided that more router tuning should pause until the official forecast archive is refreshed. This folder implements that decision. It refreshes the repo-local HKO press forecast export from the current SQLite archive, selects the latest official forecast available no later than `T-1 15:00 HKT`, joins only pre-2024 target labels, and writes a scored export that downstream experiments can use without touching the sealed 2024+ confirmation period.

## What Changed

- Press forecast export refreshed from the local archive database.
- Currently available parsed press rows from raw-detail years `{raw_detail_years_text}` are now promoted into the repo-local dataset.
- The scored pre-2024 official forecast table is written to `{SCORED_EXPORT_PATH}`.
- Remaining continuity gaps are explicit rather than hidden.

## Leakage Contract

- Scored target labels stop at `{HEADLINE_END.date()}`.
- Any target date on or after `{CONFIRMATION_START.date()}` is rejected before scoring.
- Forecast selection uses only issue times at or before the operational cutoff.
- The command does not download internet data; it only promotes raw/parsed local archive state.

## Main Result

| Metric | Value |
|---|---:|
| Press export scoreable forecast-day rows | {manifest['press_export_scoreable_rows']} |
| Press export target range | {manifest['press_export_first_target_date']} to {manifest['press_export_last_target_date']} |
| Press selected latest-pre-cutoff rows | {manifest['press_selected_rows']} |
| RSS selected latest-pre-cutoff rows | {manifest['rss_selected_rows']} |
| Scored pre-2024 rows | {manifest['scored_rows']} |
| Scored date range | {manifest['scored_first_target_date']} to {manifest['scored_last_target_date']} |
| Combined continuity ratio | {manifest['combined_continuity_ratio']} |
| Combined segment count | {manifest['combined_segment_count']} |
| Missing days inside scored span | {manifest['combined_missing_days_inside_span']} |
| Continuous 2000-2023 status | {manifest['continuous_2000_2023_status']} |

## Scoreboard

{best_score_text}

## Coverage By Source

{markdown_table(scored_summary, max_rows=10)}

## Combined Scored Segments

{markdown_table(scored_segments, max_rows=20)}

## Largest Gaps

{markdown_table(scored_gaps.sort_values('missing_days', ascending=False) if not scored_gaps.empty else scored_gaps, max_rows=20)}

## Raw Archive Coverage

{markdown_table(archive_years, max_rows=40)}

## Interpretation

This promotion is useful but not the final continuous archive. It unlocks the parsed press forecast history that is currently present in SQLite, through press target date `{manifest['press_export_last_target_date']}`. It also proves the remaining problem: candidate years `{zero_years_text}` still lack raw detail HTML in the local immutable archive. Indexed candidate links are not enough; without raw detail pages, no parser can recover the issued temperature forecast text.

The correct next step is therefore a raw-detail backfill or an explicit provider/blocker package for the remaining zero-raw-detail candidate years. After raw detail coverage changes, rerun this command, then rerun or replace the stale official-anchor router chain on the expanded scored frame.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    update_milestones(manifest, scoreboard)
    return manifest


def run(
    *,
    archive_db: Path = DEFAULT_ARCHIVE_DB,
    raw_root: Path = DEFAULT_RAW_ROOT,
    skip_press_refresh: bool = False,
) -> dict[str, object]:
    generated_at = now_utc()
    press_export_summary: dict[str, Any] = {}
    if not skip_press_refresh:
        press_export_summary = export_press_archive(archive_db, mode="parsed-db")
    elif PRESS_EXPORT_MANIFEST_PATH.exists():
        press_export_summary = json.loads(
            PRESS_EXPORT_MANIFEST_PATH.read_text(encoding="utf-8")
        )
    selected, press_selected, rss_selected = build_selected_forecast_inventory()
    scored = build_scored_export(selected)
    require_no_confirmation_dates(scored["target_date"] if not scored.empty else [], context="0044 final scored export")
    coverage, missing, no_raw, raw_files = read_archive_summary(archive_db, raw_root)
    return write_outputs(
        selected=selected,
        press_selected=press_selected,
        rss_selected=rss_selected,
        scored=scored,
        coverage=coverage,
        missing=missing,
        no_raw=no_raw,
        raw_files=raw_files,
        press_export_summary=press_export_summary,
        generated_at_utc=generated_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote HKO forecast archive into scored pre-2024 export.")
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--skip-press-refresh",
        action="store_true",
        help="Use existing repo-local press export instead of refreshing from SQLite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        archive_db=args.archive_db,
        raw_root=args.raw_root,
        skip_press_refresh=args.skip_press_refresh,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
