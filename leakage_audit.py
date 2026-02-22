from __future__ import annotations

import argparse
import json
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage audit for E37 minute-condensed features.")
    parser.add_argument("--feature-store", required=True, help="Merged feature store parquet")
    parser.add_argument("--minute-features", required=True, help="Minute features parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--tz", default="America/New_York")
    args = parser.parse_args()

    tz = ZoneInfo(args.tz)
    store = pd.read_parquet(args.feature_store)
    minute = pd.read_parquet(args.minute_features)

    store["target_date_local"] = pd.to_datetime(store["target_date_local"]).dt.date
    minute["target_date_local"] = pd.to_datetime(minute["target_date_local"]).dt.date

    df = store.merge(minute, on="target_date_local", how="left", suffixes=("", "_minute"))

    decision_utc = pd.to_datetime(df["target_date_local"]).dt.tz_localize(timezone.utc) + pd.Timedelta(
        hours=6
    )
    max_ts_used = pd.to_datetime(df["max_minute_ts_used_utc"], utc=True, errors="coerce")
    max_ts_t1 = pd.to_datetime(df["max_ts_utc_t1"], utc=True, errors="coerce")
    max_ts_early = pd.to_datetime(df["max_ts_utc_early"], utc=True, errors="coerce")

    # Leakage checks (ignore missing timestamps)
    valid_cutoff = max_ts_used.notna() & decision_utc.notna()
    check_cutoff = (max_ts_used <= decision_utc) & valid_cutoff
    missing_cutoff = (~valid_cutoff).sum()

    valid_t1 = max_ts_t1.notna()
    t1_local_date = max_ts_t1.dt.tz_convert(tz).dt.date
    expected_t1 = (pd.to_datetime(df["target_date_local"]) - pd.Timedelta(days=1)).dt.date
    check_t1_local = (t1_local_date == expected_t1) & valid_t1
    missing_t1 = (~valid_t1).sum()

    valid_early = max_ts_early.notna()
    check_early_utc_day = (max_ts_early.dt.date == pd.to_datetime(df["target_date_local"]).dt.date) & valid_early
    check_early_utc_time = ((max_ts_early.dt.hour * 60 + max_ts_early.dt.minute) <= 360) & valid_early
    missing_early = (~valid_early).sum()

    # Translator consistency
    y = pd.to_numeric(df.get("y_actual_tmax_f"), errors="coerce")
    iem_tmax_t1 = pd.to_numeric(df.get("iem_tmax_t1"), errors="coerce")
    diff_lag1 = pd.to_numeric(df.get("diff_lag1"), errors="coerce")
    diff_calc = y.shift(1) - iem_tmax_t1
    valid_diff = diff_lag1.notna() & diff_calc.notna()
    diff_err = diff_lag1 - diff_calc
    diff_match = (diff_err.abs() <= 1e-6) & valid_diff
    missing_diff = (~valid_diff).sum()

    # EWMA recompute check (by target_date_local)
    df_by_date = (
        minute[["target_date_local", "diff_lag1", "diff_ewma_30"]]
        .drop_duplicates(subset=["target_date_local"])
        .sort_values("target_date_local")
        .reset_index(drop=True)
    )
    diff_series = pd.to_numeric(df_by_date.get("diff_lag1"), errors="coerce")
    alpha = 1 - np.exp(np.log(0.5) / 30.0)
    ewma = []
    prev = np.nan
    for val in diff_series:
        if np.isnan(prev):
            prev = val
        else:
            prev = alpha * val + (1 - alpha) * prev
        ewma.append(prev)
    ewma = pd.Series(ewma, index=df_by_date.index)
    diff_ewma_30 = pd.to_numeric(df_by_date.get("diff_ewma_30"), errors="coerce")
    ewma_err = (diff_ewma_30 - ewma).abs()
    valid_ewma = diff_ewma_30.notna() & ewma.notna()

    audit = {
        "rows": int(len(df)),
        "cutoff_pass": int(check_cutoff.sum()),
        "cutoff_fail": int((valid_cutoff & (~check_cutoff)).sum()),
        "cutoff_missing": int(missing_cutoff),
        "t1_local_pass": int(check_t1_local.sum()),
        "t1_local_fail": int((valid_t1 & (~check_t1_local)).sum()),
        "t1_local_missing": int(missing_t1),
        "early_utc_day_pass": int(check_early_utc_day.sum()),
        "early_utc_day_fail": int((valid_early & (~check_early_utc_day)).sum()),
        "early_utc_day_missing": int(missing_early),
        "early_utc_time_pass": int(check_early_utc_time.sum()),
        "early_utc_time_fail": int((valid_early & (~check_early_utc_time)).sum()),
        "early_utc_time_missing": int(missing_early),
        "diff_lag1_match_pass": int(diff_match.sum()),
        "diff_lag1_match_fail": int((valid_diff & (~diff_match)).sum()),
        "diff_lag1_missing": int(missing_diff),
        "ewma_max_abs_err": float(np.nanmax(ewma_err[valid_ewma])),
        "max_delta_cutoff": str((max_ts_used[valid_cutoff] - decision_utc[valid_cutoff]).max()),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "leakage_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    md_lines = []
    md_lines.append("# Leakage Audit")
    md_lines.append("")
    md_lines.append(f"Rows checked: {audit['rows']}")
    md_lines.append("")
    md_lines.append("## Cutoff checks")
    md_lines.append(
        f"- max_ts_used <= decision_utc: pass {audit['cutoff_pass']} / fail {audit['cutoff_fail']} / missing {audit['cutoff_missing']}"
    )
    md_lines.append("")
    md_lines.append("## T-1 local-day checks")
    md_lines.append(
        f"- max_ts_utc_t1 local date == T-1: pass {audit['t1_local_pass']} / fail {audit['t1_local_fail']} / missing {audit['t1_local_missing']}"
    )
    md_lines.append("")
    md_lines.append("## Early window checks")
    md_lines.append(
        f"- max_ts_utc_early UTC date == T: pass {audit['early_utc_day_pass']} / fail {audit['early_utc_day_fail']} / missing {audit['early_utc_day_missing']}"
    )
    md_lines.append(
        f"- max_ts_utc_early time <= 06:00Z: pass {audit['early_utc_time_pass']} / fail {audit['early_utc_time_fail']} / missing {audit['early_utc_time_missing']}"
    )
    md_lines.append("")
    md_lines.append("## Translator consistency")
    md_lines.append(
        f"- diff_lag1 == y(T-1) - iem_tmax_t1: pass {audit['diff_lag1_match_pass']} / fail {audit['diff_lag1_match_fail']} / missing {audit['diff_lag1_missing']}"
    )
    md_lines.append(f"- diff_ewma_30 max abs error vs recompute: {audit['ewma_max_abs_err']:.6g}")
    md_lines.append("")
    md_lines.append(f"Max(max_ts_used - decision_utc): {audit['max_delta_cutoff']}")

    (out_dir / "leakage_audit.md").write_text("\n".join(md_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
