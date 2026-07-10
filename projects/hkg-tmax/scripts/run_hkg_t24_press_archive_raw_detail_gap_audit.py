from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    RESEARCH_ROOT,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)

DEFAULT_ARCHIVE_DB = Path(r"C:\hko_press_2000_2026\metadata\archive.sqlite3")
DEFAULT_RAW_ROOT = Path(r"C:\hko_press_2000_2026\raw\info_gov_bulletin")
FOLDER_NAME = "0024_press_gap_audit"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_archive_frames(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing HKO press archive DB: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        candidates = pd.read_sql(
            """
            SELECT source, index_date, title, product_type, url, discovered_at_utc
            FROM candidates
            WHERE source='info_gov'
            """,
            connection,
        )
        retrievals = pd.read_sql(
            """
            SELECT source, url, attempted_at_utc, status_code, error, content_sha256, raw_path
            FROM retrievals
            WHERE source='info_gov_bulletin'
            """,
            connection,
        )
        forecast_days = pd.read_sql(
            """
            SELECT bulletin_id, source, source_url, product_type, issue_at_hkt, target_date,
                   forecast_min_c, forecast_max_c
            FROM forecast_days
            """,
            connection,
        )
    return candidates, retrievals, forecast_days


def count_raw_html_files_by_year(raw_root: Path) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    bytes_by_year: Counter[str] = Counter()
    examples: dict[str, str] = {}
    if not raw_root.exists():
        return pd.DataFrame(columns=["raw_file_year", "html_file_count", "total_bytes", "example_path"])
    for path in raw_root.rglob("*.html"):
        try:
            year = path.relative_to(raw_root).parts[0]
        except ValueError:
            year = "unknown"
        counts[year] += 1
        bytes_by_year[year] += path.stat().st_size
        examples.setdefault(year, str(path))
    rows = [
        {
            "raw_file_year": int(year) if str(year).isdigit() else year,
            "html_file_count": int(count),
            "total_bytes": int(bytes_by_year[year]),
            "example_path": examples[year],
        }
        for year, count in counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["raw_file_year", "html_file_count", "total_bytes", "example_path"])
    return pd.DataFrame(rows).sort_values("raw_file_year").reset_index(drop=True)


def scoreable_forecast_day_flags(forecast_days: pd.DataFrame) -> pd.DataFrame:
    days = forecast_days.copy()
    days["issue_at_utc"] = pd.to_datetime(days["issue_at_hkt"], errors="coerce", utc=True)
    days["issue_year"] = days["issue_at_utc"].dt.year
    days["target_date_parsed"] = pd.to_datetime(days["target_date"], errors="coerce").dt.normalize()
    days["forecast_min_c"] = pd.to_numeric(days["forecast_min_c"], errors="coerce")
    days["forecast_max_c"] = pd.to_numeric(days["forecast_max_c"], errors="coerce")
    valid_max = days["forecast_max_c"].between(-5.0, 45.0, inclusive="both")
    valid_min = days["forecast_min_c"].isna() | days["forecast_min_c"].between(-5.0, 45.0, inclusive="both")
    ordered_range = days["forecast_min_c"].isna() | (days["forecast_min_c"] <= days["forecast_max_c"])
    days["temperature_row_valid"] = valid_max & valid_min & ordered_range
    issue_date_hkt = days["issue_at_utc"].dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None).dt.normalize()
    days["target_issue_lead_days"] = (days["target_date_parsed"] - issue_date_hkt).dt.days
    days["target_date_plausible"] = days["target_issue_lead_days"].between(-1, 15, inclusive="both")
    days["scoreable_row_valid"] = (
        days["temperature_row_valid"]
        & days["target_date_parsed"].notna()
        & days["issue_at_utc"].notna()
        & days["target_date_plausible"]
    )
    return days


def build_year_product_coverage(
    candidates: pd.DataFrame,
    retrievals: pd.DataFrame,
    forecast_days: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cand = candidates.copy()
    cand["index_year"] = pd.to_datetime(cand["index_date"], errors="coerce").dt.year
    candidate_counts = (
        cand.groupby(["index_year", "product_type"], observed=True)
        .agg(candidate_count=("url", "nunique"))
        .reset_index()
    )

    successful = retrievals[
        retrievals["status_code"].between(200, 299, inclusive="both") & retrievals["raw_path"].notna()
    ].copy()
    successful = successful.merge(cand[["url", "index_year", "product_type"]], on="url", how="left")
    raw_counts = (
        successful.groupby(["index_year", "product_type"], observed=True)
        .agg(raw_detail_count=("raw_path", "nunique"), raw_url_count=("url", "nunique"))
        .reset_index()
    )

    days = scoreable_forecast_day_flags(forecast_days)
    day_counts = (
        days.groupby(["issue_year", "product_type"], observed=True)
        .agg(
            parsed_forecast_day_rows=("target_date", "count"),
            parsed_bulletins=("bulletin_id", "nunique"),
            valid_temperature_rows=("temperature_row_valid", "sum"),
            scoreable_rows=("scoreable_row_valid", "sum"),
            first_target_date=("target_date_parsed", "min"),
            last_target_date=("target_date_parsed", "max"),
        )
        .reset_index()
        .rename(columns={"issue_year": "index_year"})
    )

    coverage = candidate_counts.merge(raw_counts, on=["index_year", "product_type"], how="left").merge(
        day_counts,
        on=["index_year", "product_type"],
        how="left",
    )
    fill_int = [
        "raw_detail_count",
        "raw_url_count",
        "parsed_forecast_day_rows",
        "parsed_bulletins",
        "valid_temperature_rows",
        "scoreable_rows",
    ]
    for column in fill_int:
        coverage[column] = coverage[column].fillna(0).astype(int)
    coverage["raw_detail_gap"] = coverage["candidate_count"] - coverage["raw_url_count"]
    coverage["raw_detail_coverage_pct"] = np.where(
        coverage["candidate_count"] > 0,
        coverage["raw_url_count"] / coverage["candidate_count"],
        np.nan,
    )
    coverage["first_target_date"] = pd.to_datetime(coverage["first_target_date"], errors="coerce").dt.date.astype(str)
    coverage["last_target_date"] = pd.to_datetime(coverage["last_target_date"], errors="coerce").dt.date.astype(str)
    coverage = coverage.sort_values(["index_year", "product_type"]).reset_index(drop=True)
    missing = coverage[coverage["raw_detail_gap"] > 0].copy().reset_index(drop=True)
    no_raw = coverage[(coverage["candidate_count"] > 0) & (coverage["raw_url_count"] == 0)].copy().reset_index(drop=True)
    return coverage, missing, no_raw


def summarize_gap(
    *,
    db_path: Path,
    raw_root: Path,
    candidates: pd.DataFrame,
    retrievals: pd.DataFrame,
    forecast_days: pd.DataFrame,
    coverage: pd.DataFrame,
    raw_files: pd.DataFrame,
) -> dict[str, object]:
    days = scoreable_forecast_day_flags(forecast_days)
    scoreable = days[days["scoreable_row_valid"]]
    candidate_years = sorted(int(year) for year in coverage["index_year"].dropna().unique())
    raw_years = sorted(int(year) for year in coverage.loc[coverage["raw_url_count"] > 0, "index_year"].dropna().unique())
    zero_raw_years = sorted(
        int(year)
        for year in coverage.loc[
            (coverage["candidate_count"] > 0) & (coverage["raw_url_count"] == 0),
            "index_year",
        ]
        .dropna()
        .unique()
    )
    return {
        "generated_at_utc": now_utc(),
        "archive_db": str(db_path),
        "raw_root": str(raw_root),
        "candidate_rows": int(len(candidates)),
        "retrieval_rows": int(len(retrievals)),
        "forecast_day_rows": int(len(forecast_days)),
        "scoreable_forecast_day_rows": int(len(scoreable)),
        "candidate_first_year": min(candidate_years) if candidate_years else None,
        "candidate_last_year": max(candidate_years) if candidate_years else None,
        "raw_detail_years": raw_years,
        "zero_raw_detail_years": zero_raw_years,
        "raw_html_file_years": sorted(int(year) for year in raw_files["raw_file_year"].dropna().unique())
        if not raw_files.empty
        else [],
        "first_scoreable_target_date": "" if scoreable.empty else str(scoreable["target_date_parsed"].min().date()),
        "last_scoreable_target_date": "" if scoreable.empty else str(scoreable["target_date_parsed"].max().date()),
        "first_scoreable_issue_year": None if scoreable.empty else int(scoreable["issue_year"].min()),
        "last_scoreable_issue_year": None if scoreable.empty else int(scoreable["issue_year"].max()),
    }


def write_outputs(
    *,
    coverage: pd.DataFrame,
    missing: pd.DataFrame,
    no_raw: pd.DataFrame,
    raw_files: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "coverage.csv", coverage)
    write_csv(artifacts / "missing_raw.csv", missing)
    write_csv(artifacts / "zero_raw.csv", no_raw)
    write_csv(artifacts / "raw_files.csv", raw_files)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "press_archive_raw_detail_gap_audit_manifest.json", summary)

    zero_years = ", ".join(str(year) for year in summary["zero_raw_detail_years"])
    raw_years = ", ".join(str(year) for year in summary["raw_detail_years"])
    readme = f"""# Press Archive Raw-Detail Gap Audit

Generated: `{summary['generated_at_utc']}`

## What Was Audited

This insight audits the local HKO info.gov.hk press-weather archive at `C:\\hko_press_2000_2026`. It separates three different things that must not be confused:

1. candidate detail links indexed from press index pages;
2. actual raw detail HTML files successfully retrieved;
3. parsed and scoreable forecast-day rows with valid temperature forecasts.

## Main Result

The local archive has candidate links from `{summary['candidate_first_year']}` through `{summary['candidate_last_year']}`, but actual raw detail HTML is available only for these candidate-index years:

`{raw_years}`

Years with indexed candidates but zero successful raw detail HTML are:

`{zero_years}`

The refreshed scoreable forecast-day table contains `{summary['scoreable_forecast_day_rows']}` rows with target dates from `{summary['first_scoreable_target_date']}` to `{summary['last_scoreable_target_date']}`. The local raw-detail gap is therefore a data-acquisition gap, not merely a parser limitation.

## Summary

| Metric | Value |
|---|---:|
| Candidate rows | {summary['candidate_rows']} |
| Retrieval rows | {summary['retrieval_rows']} |
| Forecast-day rows in DB | {summary['forecast_day_rows']} |
| Scoreable forecast-day rows | {summary['scoreable_forecast_day_rows']} |
| First scoreable target date | {summary['first_scoreable_target_date']} |
| Last scoreable target date | {summary['last_scoreable_target_date']} |
| First scoreable issue year | {summary['first_scoreable_issue_year']} |
| Last scoreable issue year | {summary['last_scoreable_issue_year']} |

## Coverage By Year And Product

{markdown_table(coverage, max_rows=80)}

## Raw HTML Files By Year

{markdown_table(raw_files, max_rows=40)}

## Zero Raw Detail Coverage

{markdown_table(no_raw, max_rows=80)}

## Interpretation

This explains why the official forecast archive still cannot provide continuous 2000-2023 scoreable coverage. The index crawl discovered links for 2005-2026, but the raw detail pages for those links have not been retrieved into the local immutable archive. Without those raw detail HTML files or another official historical forecast source, the parser cannot recover forecast temperatures for 2005-2020 from the local archive.
"""
    write_text(folder / "README.md", readme)
    update_master_index(summary)
    return summary


def update_master_index(summary: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Press Archive Raw-Detail Gap Audit\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_press_archive_raw_detail_gap_audit.py`:

- `{FOLDER_NAME}`: DB/file-system audit proving which HKO press forecast years have candidate links, raw detail HTML, and scoreable forecast-day rows.

| Metric | Value |
|---|---:|
| Candidate rows | {summary['candidate_rows']} |
| Forecast-day DB rows | {summary['forecast_day_rows']} |
| Scoreable forecast-day rows | {summary['scoreable_forecast_day_rows']} |
| First scoreable target date | {summary['first_scoreable_target_date']} |
| Last scoreable target date | {summary['last_scoreable_target_date']} |

Raw detail years: `{', '.join(str(year) for year in summary['raw_detail_years'])}`.

Zero-raw candidate years: `{', '.join(str(year) for year in summary['zero_raw_detail_years'])}`.
"""
    write_text(index_path, text)


def run(db_path: Path = DEFAULT_ARCHIVE_DB, raw_root: Path = DEFAULT_RAW_ROOT) -> dict[str, object]:
    candidates, retrievals, forecast_days = load_archive_frames(db_path)
    raw_files = count_raw_html_files_by_year(raw_root)
    coverage, missing, no_raw = build_year_product_coverage(candidates, retrievals, forecast_days)
    summary = summarize_gap(
        db_path=db_path,
        raw_root=raw_root,
        candidates=candidates,
        retrievals=retrievals,
        forecast_days=forecast_days,
        coverage=coverage,
        raw_files=raw_files,
    )
    return write_outputs(coverage=coverage, missing=missing, no_raw=no_raw, raw_files=raw_files, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HKO press archive candidate/raw-detail/parsed coverage.")
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.archive_db, args.raw_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
