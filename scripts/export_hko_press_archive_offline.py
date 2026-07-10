from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_MODULE_DIR = REPO_ROOT / "scripts" / "hko_forecast_archive_downloader_rss"
if str(ARCHIVE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_MODULE_DIR))

from hko_archive import parse_bulletin  # noqa: E402

DEFAULT_ARCHIVE_DB = Path(r"C:\hko_press_2000_2026\metadata\archive.sqlite3")
OUTPUT_DIR = REPO_ROOT / "data" / "datasets" / "05_hko_historical_rss_forecasts"
RESEARCH_DIR = REPO_ROOT / "experiments" / "0006_press_archive_offline_export"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).copy()
    columns = [str(col) for col in clipped.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in clipped.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    return "\n".join(lines)


def read_archive_tables(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing HKO press archive database: {db_path}")

    import sqlite3

    with sqlite3.connect(db_path) as connection:
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
            SELECT
                r.id AS retrieval_id,
                r.source AS retrieval_source,
                r.url,
                r.attempted_at_utc,
                r.status_code,
                r.error,
                r.content_sha256,
                r.raw_path
            FROM retrievals r
            WHERE r.source='info_gov_bulletin'
              AND r.status_code BETWEEN 200 AND 299
              AND r.raw_path IS NOT NULL
            ORDER BY r.id
            """,
            connection,
        )
    candidate_meta = candidates[["url", "index_date", "title", "product_type"]].rename(
        columns={"product_type": "candidate_product_type"}
    )
    retrievals = retrievals.merge(candidate_meta, on="url", how="left")
    retrievals = retrievals.sort_values(["index_date", "url", "retrieval_id"], na_position="last").reset_index(drop=True)
    return candidates, retrievals


def read_preparsed_tables(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        bulletins = pd.read_sql(
            """
            SELECT
                b.*,
                c.title AS candidate_title,
                c.product_type AS candidate_product_type,
                c.index_date AS candidate_index_date,
                NULL AS retrieval_id,
                NULL AS attempted_at_utc
            FROM bulletins b
            LEFT JOIN candidates c ON c.url = b.source_url
            ORDER BY b.issue_at_hkt, b.source_url
            """,
            connection,
        )
        forecast_days = pd.read_sql(
            """
            SELECT
                f.*,
                b.raw_path,
                c.title AS candidate_title,
                c.product_type AS candidate_product_type,
                c.index_date AS candidate_index_date,
                NULL AS retrieval_id,
                NULL AS attempted_at_utc
            FROM forecast_days f
            LEFT JOIN bulletins b ON b.bulletin_id = f.bulletin_id
            LEFT JOIN candidates c ON c.url = f.source_url
            ORDER BY f.target_date, f.issue_at_hkt, f.source_url
            """,
            connection,
        )
    return bulletins, forecast_days


def parse_date_or_none(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_retrieved_bulletins(retrievals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bulletin_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    seen_raw_paths: set[str] = set()

    for row in retrievals.itertuples(index=False):
        raw_path = Path(str(row.raw_path))
        raw_path_key = str(raw_path)
        if raw_path_key in seen_raw_paths:
            continue
        seen_raw_paths.add(raw_path_key)

        base = {
            "retrieval_id": int(row.retrieval_id),
            "source_url": str(row.url),
            "raw_path": raw_path_key,
            "candidate_index_date": str(row.index_date or ""),
            "candidate_title": str(row.title or ""),
            "candidate_product_type": str(row.candidate_product_type or ""),
            "attempted_at_utc": str(row.attempted_at_utc or ""),
        }
        if not raw_path.exists():
            failure_rows.append({**base, "failure": "raw_path_missing"})
            continue

        try:
            content = raw_path.read_bytes()
            digest = str(row.content_sha256 or "") or sha256_file(raw_path)
            bulletin, days = parse_bulletin(
                source="info_gov",
                source_url=str(row.url),
                content=content,
                raw_path=raw_path_key,
                raw_sha256=digest,
                title_hint=str(row.title or ""),
                index_date=parse_date_or_none(row.index_date),
                snapshot_at_hkt=None,
            )
        except Exception as exc:  # noqa: BLE001 - parser failures are exported for audit.
            failure_rows.append({**base, "failure": type(exc).__name__, "message": str(exc)})
            continue

        bulletin_rows.append({**asdict(bulletin), **base})
        for forecast_day in days:
            day_rows.append({**asdict(forecast_day), **base})

    return pd.DataFrame(bulletin_rows), pd.DataFrame(day_rows), pd.DataFrame(failure_rows)


def normalize_forecast_days(days: pd.DataFrame, *, exported_at_utc: str) -> pd.DataFrame:
    if days.empty:
        return pd.DataFrame()

    out = days.copy()
    out["issue_at_utc"] = pd.to_datetime(out["issue_at_hkt"], errors="coerce", utc=True)
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    out["forecast_min_c"] = pd.to_numeric(out["forecast_min_c"], errors="coerce")
    out["forecast_max_c"] = pd.to_numeric(out["forecast_max_c"], errors="coerce")
    out["rh_min_pct"] = pd.to_numeric(out["rh_min_pct"], errors="coerce")
    out["rh_max_pct"] = pd.to_numeric(out["rh_max_pct"], errors="coerce")

    valid_max = out["forecast_max_c"].between(-5.0, 45.0, inclusive="both")
    valid_min = out["forecast_min_c"].isna() | out["forecast_min_c"].between(-5.0, 45.0, inclusive="both")
    ordered_range = out["forecast_min_c"].isna() | (out["forecast_min_c"] <= out["forecast_max_c"])
    out["temperature_row_valid"] = valid_max & valid_min & ordered_range
    issue_date_hkt = out["issue_at_utc"].dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None).dt.normalize()
    out["target_issue_lead_days"] = (out["target_date"] - issue_date_hkt).dt.days
    out["target_date_plausible"] = out["target_issue_lead_days"].between(-1, 15, inclusive="both")
    out["scoreable_row_valid"] = (
        out["temperature_row_valid"]
        & out["issue_at_utc"].notna()
        & out["target_date"].notna()
        & out["target_date_plausible"]
    )
    out["available_at_hkt"] = out["issue_at_hkt"]
    out["available_at_utc"] = out["issue_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["availability_tier"] = "historical_official_press_archive_replay"
    out["operational_input_allowed"] = True
    out["source_time_policy"] = "official_bulletin_issue_time"
    out["parser"] = "export_hko_press_archive_offline.parse_bulletin"
    out["source_id"] = "hko_info_gov_press_weather_forecast_archive"
    out["exported_at_utc"] = exported_at_utc

    sort_cols = ["target_date", "issue_at_utc", "product_type", "source_url"]
    out = out.sort_values(sort_cols).drop_duplicates(["bulletin_id", "target_date"], keep="last")
    column_order = [
        "source_id",
        "bulletin_id",
        "source",
        "source_url",
        "product_type",
        "issue_at_hkt",
        "issue_at_utc",
        "available_at_hkt",
        "available_at_utc",
        "target_date",
        "forecast_min_c",
        "forecast_max_c",
        "temperature_row_valid",
        "target_issue_lead_days",
        "target_date_plausible",
        "scoreable_row_valid",
        "rh_min_pct",
        "rh_max_pct",
        "wind_text",
        "weather_text",
        "raw_sha256",
        "raw_path",
        "candidate_index_date",
        "candidate_title",
        "candidate_product_type",
        "retrieval_id",
        "attempted_at_utc",
        "availability_tier",
        "operational_input_allowed",
        "source_time_policy",
        "parser",
        "exported_at_utc",
    ]
    return out[[col for col in column_order if col in out.columns]].reset_index(drop=True)


def normalize_bulletins(bulletins: pd.DataFrame, *, exported_at_utc: str) -> pd.DataFrame:
    if bulletins.empty:
        return pd.DataFrame()
    out = bulletins.copy()
    out["issue_at_utc"] = pd.to_datetime(out["issue_at_hkt"], errors="coerce", utc=True)
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    out["exported_at_utc"] = exported_at_utc
    out = out.sort_values(["issue_at_utc", "source_url", "raw_sha256"]).drop_duplicates("bulletin_id", keep="last")
    return out.reset_index(drop=True)


def build_coverage(candidates: pd.DataFrame, retrievals: pd.DataFrame, forecast_days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = candidates.copy()
    cand["index_year"] = pd.to_datetime(cand["index_date"], errors="coerce").dt.year
    candidate_counts = (
        cand.groupby(["index_year", "product_type"], observed=True)
        .agg(candidate_count=("url", "nunique"))
        .reset_index()
    )

    ret = retrievals.copy()
    ret["index_year"] = pd.to_datetime(ret["index_date"], errors="coerce").dt.year
    retrieval_counts = (
        ret.groupby(["index_year", "candidate_product_type"], observed=True)
        .agg(raw_detail_count=("raw_path", "nunique"))
        .reset_index()
        .rename(columns={"candidate_product_type": "product_type"})
    )

    if forecast_days.empty:
        parsed_counts = pd.DataFrame(
            columns=[
                "index_year",
                "product_type",
                "parsed_bulletin_count",
                "forecast_day_count",
                "valid_temperature_day_count",
                "scoreable_temperature_day_count",
            ]
        )
    else:
        days = forecast_days.copy()
        days["index_year"] = pd.to_datetime(days["candidate_index_date"], errors="coerce").dt.year
        parsed_counts = (
            days.groupby(["index_year", "candidate_product_type"], observed=True)
            .agg(
                parsed_bulletin_count=("bulletin_id", "nunique"),
                forecast_day_count=("target_date", "count"),
                valid_temperature_day_count=("temperature_row_valid", "sum"),
                scoreable_temperature_day_count=("scoreable_row_valid", "sum"),
            )
            .reset_index()
            .rename(columns={"candidate_product_type": "product_type"})
        )

    coverage = candidate_counts.merge(retrieval_counts, on=["index_year", "product_type"], how="left").merge(
        parsed_counts,
        on=["index_year", "product_type"],
        how="left",
    )
    for col in (
        "raw_detail_count",
        "parsed_bulletin_count",
        "forecast_day_count",
        "valid_temperature_day_count",
        "scoreable_temperature_day_count",
    ):
        coverage[col] = coverage[col].fillna(0).astype(int)
    coverage["raw_detail_gap"] = coverage["candidate_count"] - coverage["raw_detail_count"]
    coverage["raw_detail_coverage_pct"] = np.where(
        coverage["candidate_count"] > 0,
        coverage["raw_detail_count"] / coverage["candidate_count"],
        np.nan,
    )
    missing = coverage[coverage["raw_detail_gap"] > 0].sort_values(["index_year", "product_type"]).reset_index(drop=True)
    return coverage.sort_values(["index_year", "product_type"]).reset_index(drop=True), missing


def summarize_export(
    db_path: Path,
    candidates: pd.DataFrame,
    retrievals: pd.DataFrame,
    bulletins: pd.DataFrame,
    forecast_days: pd.DataFrame,
    failures: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, object]:
    valid_days = forecast_days[forecast_days["scoreable_row_valid"]] if not forecast_days.empty else pd.DataFrame()
    return {
        "generated_at_utc": now_utc(),
        "archive_db": str(db_path),
        "candidate_rows": int(len(candidates)),
        "successful_raw_detail_rows": int(len(retrievals)),
        "parsed_bulletins": int(len(bulletins)),
        "parse_failures": int(len(failures)),
        "forecast_day_rows": int(len(forecast_days)),
        "scoreable_temperature_forecast_day_rows": int(len(valid_days)),
        "valid_temperature_forecast_day_rows": int(forecast_days["temperature_row_valid"].sum()) if not forecast_days.empty else 0,
        "first_issue_at_hkt": "" if forecast_days.empty else str(forecast_days["issue_at_hkt"].min()),
        "last_issue_at_hkt": "" if forecast_days.empty else str(forecast_days["issue_at_hkt"].max()),
        "first_target_date": "" if valid_days.empty else str(pd.to_datetime(valid_days["target_date"]).min().date()),
        "last_target_date": "" if valid_days.empty else str(pd.to_datetime(valid_days["target_date"]).max().date()),
        "candidate_first_index_date": "" if candidates.empty else str(candidates["index_date"].min()),
        "candidate_last_index_date": "" if candidates.empty else str(candidates["index_date"].max()),
        "years_with_any_raw_detail": sorted(int(year) for year in coverage.loc[coverage["raw_detail_count"] > 0, "index_year"].dropna().unique()),
        "years_with_candidates": sorted(int(year) for year in coverage["index_year"].dropna().unique()),
    }


def write_dataset_outputs(
    bulletins: pd.DataFrame,
    forecast_days: pd.DataFrame,
    failures: pd.DataFrame,
    coverage: pd.DataFrame,
    missing: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not bulletins.empty:
        bulletins.to_parquet(OUTPUT_DIR / "hko_press_archive_bulletins_offline.parquet", index=False)
    if not forecast_days.empty:
        forecast_days.to_parquet(OUTPUT_DIR / "hko_press_archive_forecast_days.parquet", index=False)
        forecast_days[forecast_days["scoreable_row_valid"]].to_parquet(
            OUTPUT_DIR / "hko_press_archive_temperature_forecast_days.parquet",
            index=False,
        )
        write_csv(OUTPUT_DIR / "hko_press_archive_temperature_forecast_days.csv", forecast_days[forecast_days["scoreable_row_valid"]])
    write_csv(OUTPUT_DIR / "hko_press_archive_parse_failures.csv", failures)
    write_csv(OUTPUT_DIR / "hko_press_archive_candidate_detail_coverage.csv", coverage)
    write_csv(OUTPUT_DIR / "hko_press_archive_missing_detail_coverage.csv", missing)
    write_json(OUTPUT_DIR / "hko_press_archive_offline_export_manifest.json", summary)


def write_research_outputs(
    coverage: pd.DataFrame,
    missing: pd.DataFrame,
    summary: dict[str, object],
    failures: pd.DataFrame,
) -> None:
    artifacts = RESEARCH_DIR / "artifacts"
    write_csv(artifacts / "candidate_detail_coverage_by_year_product.csv", coverage)
    write_csv(artifacts / "missing_detail_coverage_by_year_product.csv", missing)
    write_csv(artifacts / "parse_failures.csv", failures)
    write_json(artifacts / "offline_export_summary.json", summary)

    raw_years = ", ".join(str(year) for year in summary["years_with_any_raw_detail"])
    text = f"""# Press Archive Offline Export

Generated: `{summary['generated_at_utc']}`

## What This Does

This insight converts the local HKO info.gov.hk press-weather raw HTML archive into repo-local normalized forecast-day tables. It does not fetch the internet. It reads successful `info_gov_bulletin` raw files already present under `C:\\hko_press_2000_2026`, reparses them, and exports immutable research tables under `data/datasets/05_hko_historical_rss_forecasts`.

## Export Summary

| Metric | Value |
|---|---:|
| Candidate rows indexed | {summary['candidate_rows']} |
| Successful raw detail rows available | {summary['successful_raw_detail_rows']} |
| Parsed bulletins | {summary['parsed_bulletins']} |
| Forecast-day rows | {summary['forecast_day_rows']} |
| Valid temperature forecast-day rows | {summary['valid_temperature_forecast_day_rows']} |
| Scoreable temperature forecast-day rows | {summary['scoreable_temperature_forecast_day_rows']} |
| Parse failures | {summary['parse_failures']} |
| First target date | {summary['first_target_date']} |
| Last target date | {summary['last_target_date']} |
| Candidate index span | {summary['candidate_first_index_date']} to {summary['candidate_last_index_date']} |
| Years with raw detail HTML | {raw_years} |

## Coverage By Year And Product

{markdown_table(coverage, max_rows=40)}

## Missing Detail Coverage

{markdown_table(missing, max_rows=40)}

## Interpretation

The archive is indexed through 2026, but only years with raw detail HTML can be converted into scoreable forecast rows. Candidate index pages are not enough for modelling because they do not contain the forecast temperature text. This export therefore makes the available official forecast history usable while making the remaining raw-detail gap explicit.
"""
    write_text(RESEARCH_DIR / "README.md", text)


def export(db_path: Path, *, mode: str) -> dict[str, object]:
    exported_at = now_utc()
    candidates, retrievals = read_archive_tables(db_path)
    if mode == "raw-reparse":
        bulletins_raw, forecast_days_raw, failures = parse_retrieved_bulletins(retrievals)
    else:
        bulletins_raw, forecast_days_raw = read_preparsed_tables(db_path)
        failures = pd.DataFrame()
    bulletins = normalize_bulletins(bulletins_raw, exported_at_utc=exported_at)
    forecast_days = normalize_forecast_days(forecast_days_raw, exported_at_utc=exported_at)
    coverage, missing = build_coverage(candidates, retrievals, forecast_days)
    summary = summarize_export(db_path, candidates, retrievals, bulletins, forecast_days, failures, coverage)
    summary["exported_at_utc"] = exported_at
    summary["export_mode"] = mode
    write_dataset_outputs(bulletins, forecast_days, failures, coverage, missing, summary)
    write_research_outputs(coverage, missing, summary, failures)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline export of locally archived HKO press forecasts.")
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument(
        "--mode",
        choices=["parsed-db", "raw-reparse"],
        default="parsed-db",
        help="Use existing SQLite parser tables by default; raw-reparse reparses each raw HTML file and is slower.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export(args.archive_db, mode=args.mode)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
