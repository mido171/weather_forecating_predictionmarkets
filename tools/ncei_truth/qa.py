from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _daterange_days(start_date: date, end_date: date) -> pd.Series:
    idx = pd.date_range(start_date, end_date, freq="D")
    return pd.Series(idx.date)


def build_station_qa_report(
    *,
    station_id: str,
    station_df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    if station_df.empty:
        return {
            "station_id": station_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "rows": 0,
            "message": "No rows for station in enriched data.",
        }

    df = station_df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
    df = df[df["target_date_local"].notna()].copy()
    df.sort_values("target_date_local", inplace=True)

    expected_days = _daterange_days(start_date, end_date)
    have_days = pd.Series(df["target_date_local"].unique())
    missing_days = expected_days[~expected_days.isin(have_days)].astype(str).tolist()

    by_year_expected = expected_days.groupby(pd.to_datetime(expected_days).dt.year).size()
    by_year_actual = df.groupby(pd.to_datetime(df["target_date_local"]).dt.year).size()
    yearly = []
    for year in range(start_date.year, end_date.year + 1):
        exp = int(by_year_expected.get(year, 0))
        act = int(by_year_actual.get(year, 0))
        yearly.append(
            {
                "year": year,
                "expected_days": exp,
                "actual_rows": act,
                "missing_days": int(max(0, exp - act)),
            }
        )

    missing_tmax_dates = (
        df[pd.to_numeric(df["tmax_f"], errors="coerce").isna()]["target_date_local"]
        .dropna()
        .astype(str)
        .tolist()
    )

    qf = df["attribute_quality_flag"].fillna("").astype(str).str.strip()
    sf = df["attribute_source_flag"].fillna("").astype(str).str.strip()
    obs_time = df["attribute_obs_time_hhmm"].fillna("").astype(str).str.strip()

    quality_flag_counts = qf.value_counts(dropna=False).to_dict()
    source_flag_counts = sf.value_counts(dropna=False).to_dict()
    obs_time_present_pct = float((obs_time != "").mean()) if len(obs_time) > 0 else 0.0

    dup_count = int(
        df.groupby(["station_id", "target_date_local"]).size().gt(1).sum()
    )
    return {
        "station_id": station_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "rows": int(len(df)),
        "min_date": str(df["target_date_local"].min()),
        "max_date": str(df["target_date_local"].max()),
        "yearly_expected_vs_actual": yearly,
        "missing_dates": missing_days,
        "missing_tmax_dates": missing_tmax_dates,
        "quality_flag_counts": {str(k): int(v) for k, v in quality_flag_counts.items()},
        "source_flag_counts": {str(k): int(v) for k, v in source_flag_counts.items()},
        "obs_time_present_pct": obs_time_present_pct,
        "duplicate_station_date_rows": dup_count,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def write_station_qa_markdown(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# QA Report: {report.get('station_id', '')}",
        "",
        f"- Start date: `{report.get('start_date')}`",
        f"- End date: `{report.get('end_date')}`",
        f"- Rows: `{report.get('rows')}`",
        f"- Min date: `{report.get('min_date')}`",
        f"- Max date: `{report.get('max_date')}`",
        f"- Duplicate station/date rows: `{report.get('duplicate_station_date_rows')}`",
        f"- Obs-time present pct: `{report.get('obs_time_present_pct')}`",
        "",
        "## Yearly Expected vs Actual",
        "",
        "| year | expected_days | actual_rows | missing_days |",
        "|---:|---:|---:|---:|",
    ]
    for row in report.get("yearly_expected_vs_actual", []):
        lines.append(
            f"| {row['year']} | {row['expected_days']} | {row['actual_rows']} | {row['missing_days']} |"
        )
    lines += ["", "## Quality Flag Counts", ""]
    qfc = report.get("quality_flag_counts", {})
    if qfc:
        lines += [f"- `{k}`: {v}" for k, v in sorted(qfc.items(), key=lambda kv: kv[0])]
    else:
        lines += ["- (none)"]
    lines += ["", "## Source Flag Counts", ""]
    sfc = report.get("source_flag_counts", {})
    if sfc:
        lines += [f"- `{k}`: {v}" for k, v in sorted(sfc.items(), key=lambda kv: kv[0])]
    else:
        lines += ["- (none)"]
    missing = report.get("missing_dates", [])
    lines += ["", "## Missing Dates", ""]
    if missing:
        lines += [f"- `{d}`" for d in missing]
    else:
        lines += ["- (none)"]

    out_path.write_text("\n".join(lines), encoding="utf-8")

