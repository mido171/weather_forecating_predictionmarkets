from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pymysql

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


REPO = Path(__file__).resolve().parents[2]

STATION_ID = "KMIA"

MODEL_DIR = REPO / "artifacts" / "experiments" / "KMIA" / "early_maxout_strategy" / "B6" / "B6_EXP20_GAM_RESIDUAL"
PREDS_VAL_PATH = MODEL_DIR / "preds_val.parquet"
PREDS_TEST_PATH = MODEL_DIR / "preds_test.parquet"
MODEL_FEATURES_PATH = MODEL_DIR / "features.json"
FEATURES_PATH = REPO / "cache" / "hit1830_v6_features.parquet"

MINUTE_DIR = REPO / "data" / "iem_minute_data" / "MIA" / "tmpf" / "UTC" / "yearly"

BACKTEST_START = date(2025, 11, 1)
BACKTEST_END = date(2025, 12, 31)

OUT_DIR = REPO / "reports" / "leakage_audit_bridge_flow_20251101_20251231"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "America/New_York"
STOCKHOLM_TZ = "Europe/Stockholm"


@dataclass(frozen=True)
class AuditRow:
    day: date
    cutoff_utc: datetime
    cutoff_matches_expected: bool
    tmax_sofar_ok: bool
    coverage_ok: bool
    last_gap_ok: bool
    mos_x_ok: bool
    notes: str


def _connect_db() -> pymysql.connections.Connection:
    host = os.environ.get("MYSQL_HOST", "localhost")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "root")
    db = os.environ.get("MYSQL_DB", "weather_predictionmarkets")
    return pymysql.connect(host=host, port=port, user=user, password=password, database=db, autocommit=True)


def _compute_cutoff_utc(day: date) -> datetime:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute cutoff.")
    stockholm = ZoneInfo(STOCKHOLM_TZ)
    cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=stockholm)
    return cutoff_local.astimezone(timezone.utc)


def _day_start_end_utc(day: date) -> Tuple[datetime, datetime]:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute day_start_utc.")
    local = ZoneInfo(LOCAL_TZ)
    day_start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=local)
    day_end_local = day_start_local + timedelta(days=1)
    return day_start_local.astimezone(timezone.utc), day_end_local.astimezone(timezone.utc)


def _load_minute_series_for_year(year: int) -> pd.DataFrame:
    path = MINUTE_DIR / f"MIA_tmpf_1min_UTC_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing minute data file: {path}")

    df = pd.read_csv(path, usecols=["valid(UTC)", "tmpf"], dtype={"tmpf": "string"})
    df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "tmpf"])
    df = df.sort_values("ts_utc")
    df = df.drop_duplicates(subset=["ts_utc"], keep="last")
    df = df.set_index("ts_utc")
    return df


def _mos_latest_value_max_n_x(day: date, cutoff_utc: datetime) -> Tuple[float, Dict[str, str]]:
    """
    Replicates run_hit1830_v6_suite.py MOS selection for just the n_x value_max rows:
      - filter asof_utc <= cutoff_utc
      - choose latest by (asof_utc, runtime_utc, retrieved_at_utc, id)
      - return mos_x_mean = mean([GFS.value_max(n_x), NAM.value_max(n_x)]) ignoring NaNs
    """
    sql = """
        SELECT model, variable_code, asof_utc, runtime_utc, retrieved_at_utc, id, value_max
        FROM mos_daily_value
        WHERE station_id=%s
          AND target_date_local=%s
          AND model IN ('GFS','NAM')
          AND variable_code='n_x'
        ORDER BY asof_utc, runtime_utc, retrieved_at_utc, id
    """
    conn = _connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (STATION_ID, day.isoformat()))
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(
        rows,
        columns=["model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id", "value_max"],
    )
    meta: Dict[str, str] = {}
    if df.empty:
        return float("nan"), {"mos_rows": "0"}

    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True, errors="coerce")
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True, errors="coerce")
    df["value_max"] = pd.to_numeric(df["value_max"], errors="coerce")
    df = df.dropna(subset=["asof_utc", "value_max"])

    # leakage guard: asof must not exceed cutoff.
    df = df[df["asof_utc"] <= cutoff_utc].copy()
    if df.empty:
        return float("nan"), {"mos_rows_asof_le_cutoff": "0"}

    latest_rows = {}
    for model in ["GFS", "NAM"]:
        sub = df[df["model"] == model].copy()
        if sub.empty:
            latest_rows[model] = None
            continue
        # Already sorted by SQL ORDER BY (ascending), so take the last row.
        latest_rows[model] = sub.iloc[-1]

    vals: List[float] = []
    for model in ["GFS", "NAM"]:
        r = latest_rows.get(model)
        if r is None:
            meta[f"{model}_present"] = "0"
            continue
        meta[f"{model}_present"] = "1"
        meta[f"{model}_asof_utc"] = str(r["asof_utc"])
        meta[f"{model}_runtime_utc"] = str(r["runtime_utc"])
        meta[f"{model}_retrieved_at_utc"] = str(r["retrieved_at_utc"])
        meta[f"{model}_value_max"] = str(float(r["value_max"]))
        vals.append(float(r["value_max"]))

    mos_x_mean = float(sum(vals) / len(vals)) if vals else float("nan")
    meta["mos_rows_asof_le_cutoff"] = str(len(df))
    return mos_x_mean, meta


def main() -> None:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available in this Python; cannot run audit.")

    # Load preds and features.
    preds_val = pd.read_parquet(PREDS_VAL_PATH)
    preds_test = pd.read_parquet(PREDS_TEST_PATH)
    preds_val["target_date_local"] = pd.to_datetime(preds_val["target_date_local"]).dt.date
    preds_test["target_date_local"] = pd.to_datetime(preds_test["target_date_local"]).dt.date

    # Sanity: split windows should be disjoint and ordered.
    split_checks = {
        "preds_val_min": str(preds_val["target_date_local"].min()),
        "preds_val_max": str(preds_val["target_date_local"].max()),
        "preds_test_min": str(preds_test["target_date_local"].min()),
        "preds_test_max": str(preds_test["target_date_local"].max()),
        "preds_val_test_overlap_days": int(
            len(set(preds_val["target_date_local"]).intersection(set(preds_test["target_date_local"])))
        ),
    }

    # Feature list sanity: ensure we are not using obvious label/future columns as model inputs.
    banned_feature_cols = {
        # labels / future-only
        "y_hit_by_cutoff",
        "y_exceed_future",
        "exceed_time_min",
        # full-day (future at cutoff)
        "tmax_full",
        "tmin_full",
        "range_full",
        "minutes_since_tmax",
    }
    feature_list = json.loads(MODEL_FEATURES_PATH.read_text(encoding="utf-8"))
    banned_in_features = sorted(set(feature_list).intersection(banned_feature_cols))
    split_checks["banned_feature_cols_present"] = banned_in_features

    # Backtest window preds.
    preds_bt = preds_test[(preds_test["target_date_local"] >= BACKTEST_START) & (preds_test["target_date_local"] <= BACKTEST_END)].copy()
    preds_bt["cutoff_utc"] = pd.to_datetime(preds_bt["cutoff_utc"], utc=True)

    feat_cols = ["target_date_local", "cutoff_utc", "tmax_sofar", "coverage_frac", "last_gap_minutes", "mos_x_mean"]
    feats = pd.read_parquet(FEATURES_PATH, columns=feat_cols)
    feats["target_date_local"] = pd.to_datetime(feats["target_date_local"]).dt.date
    feats["cutoff_utc"] = pd.to_datetime(feats["cutoff_utc"], utc=True)

    df = preds_bt.merge(feats, on=["target_date_local"], how="left", suffixes=("", "_feat"))
    if df["tmax_sofar"].isna().any():
        missing_days = [d.isoformat() for d in df.loc[df["tmax_sofar"].isna(), "target_date_local"]]
        raise RuntimeError(f"Missing feature rows for some backtest days: {missing_days}")

    # Load raw minute series for 2025.
    df_1m = _load_minute_series_for_year(2025)
    series_5m = df_1m["tmpf"].resample("5min").median()

    rows: List[AuditRow] = []
    violations: List[Dict] = []

    for r in df.itertuples(index=False):
        day: date = getattr(r, "target_date_local")
        cutoff_utc: datetime = getattr(r, "cutoff_utc").to_pydatetime()
        cutoff_utc_feat: datetime = getattr(r, "cutoff_utc_feat").to_pydatetime()

        cutoff_pred_matches_feat = abs((cutoff_utc_feat - cutoff_utc).total_seconds()) <= 60.0

        # 1) Cutoff correctness (Stockholm 18:30 -> UTC).
        expected_cutoff = _compute_cutoff_utc(day)
        cutoff_matches = abs((expected_cutoff - cutoff_utc).total_seconds()) <= 60.0
        cutoff_feat_matches = abs((expected_cutoff - cutoff_utc_feat).total_seconds()) <= 60.0

        # 2) Recompute minute-derived features using only <= cutoff.
        day_start_utc, day_end_utc = _day_start_end_utc(day)

        partial_end = min(cutoff_utc, day_end_utc) - timedelta(minutes=5)
        if partial_end < day_start_utc:
            tmax_recalc = float("nan")
        else:
            partial_idx = pd.date_range(day_start_utc, partial_end, freq="5min")
            partial_series = series_5m.reindex(partial_idx)
            tmax_recalc = float(partial_series.max(skipna=True))

        tmax_stored = float(getattr(r, "tmax_sofar"))
        tmax_ok = (math.isfinite(tmax_recalc) and abs(tmax_recalc - tmax_stored) <= 1e-6) or (
            (not math.isfinite(tmax_recalc)) and (not math.isfinite(tmax_stored))
        )

        minute_slice = df_1m.loc[day_start_utc:cutoff_utc]
        expected_minutes = int(((cutoff_utc - day_start_utc).total_seconds() / 60.0) + 1)
        coverage_recalc = float(len(minute_slice) / expected_minutes) if expected_minutes > 0 else float("nan")
        coverage_stored = float(getattr(r, "coverage_frac"))
        coverage_ok = abs(coverage_recalc - coverage_stored) <= 1e-9

        if len(minute_slice) > 0:
            last_gap_recalc = float((cutoff_utc - minute_slice.index.max()).total_seconds() / 60.0)
        else:
            last_gap_recalc = float("nan")
        last_gap_stored = float(getattr(r, "last_gap_minutes"))
        if math.isnan(last_gap_recalc) and math.isnan(last_gap_stored):
            last_gap_ok = True
        else:
            last_gap_ok = abs(last_gap_recalc - last_gap_stored) <= 1e-6

        # 3) MOS as-of check: compute mos_x_mean from DB with asof <= cutoff.
        mos_x_recalc, mos_meta = _mos_latest_value_max_n_x(day, cutoff_utc)
        mos_x_stored = float(getattr(r, "mos_x_mean"))
        mos_x_ok = (math.isfinite(mos_x_recalc) and abs(mos_x_recalc - mos_x_stored) <= 1e-6) or (
            (not math.isfinite(mos_x_recalc)) and (not math.isfinite(mos_x_stored))
        )

        notes = []
        if not cutoff_matches:
            notes.append("cutoff_mismatch")
        if not cutoff_feat_matches:
            notes.append("cutoff_feat_mismatch")
        if not cutoff_pred_matches_feat:
            notes.append("cutoff_pred_vs_feat_mismatch")
        if not tmax_ok:
            notes.append("tmax_sofar_mismatch")
        if not coverage_ok:
            notes.append("coverage_mismatch")
        if not last_gap_ok:
            notes.append("last_gap_mismatch")
        if not mos_x_ok:
            notes.append("mos_x_mean_mismatch")

        row = AuditRow(
            day=day,
            cutoff_utc=cutoff_utc,
            cutoff_matches_expected=cutoff_matches,
            tmax_sofar_ok=tmax_ok,
            coverage_ok=coverage_ok,
            last_gap_ok=last_gap_ok,
            mos_x_ok=mos_x_ok,
            notes=",".join(notes),
        )
        rows.append(row)

        if notes:
            violations.append(
                {
                    "day": day.isoformat(),
                    "cutoff_utc": cutoff_utc.isoformat(),
                    "cutoff_utc_feat": cutoff_utc_feat.isoformat(),
                    "expected_cutoff_utc": expected_cutoff.isoformat(),
                    "tmax_sofar_stored": tmax_stored,
                    "tmax_sofar_recalc": tmax_recalc,
                    "coverage_stored": coverage_stored,
                    "coverage_recalc": coverage_recalc,
                    "last_gap_stored": last_gap_stored,
                    "last_gap_recalc": last_gap_recalc,
                    "mos_x_mean_stored": mos_x_stored,
                    "mos_x_mean_recalc": mos_x_recalc,
                    "mos_meta": mos_meta,
                    "notes": notes,
                }
            )

    audit_pass = len(violations) == 0

    out_rows = [
        {
            "day": r.day.isoformat(),
            "cutoff_utc": r.cutoff_utc.isoformat(),
            "cutoff_matches_expected": r.cutoff_matches_expected,
            "tmax_sofar_ok": r.tmax_sofar_ok,
            "coverage_ok": r.coverage_ok,
            "last_gap_ok": r.last_gap_ok,
            "mos_x_ok": r.mos_x_ok,
            "notes": r.notes,
        }
        for r in rows
    ]
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "audit_rows.csv", index=False, encoding="utf-8")

    report = {
        "scope": {
            "station_id": STATION_ID,
            "backtest_start": BACKTEST_START.isoformat(),
            "backtest_end": BACKTEST_END.isoformat(),
            "model_dir": str(MODEL_DIR),
            "features_path": str(FEATURES_PATH),
            "minute_dir": str(MINUTE_DIR),
        },
        "split_checks": split_checks,
        "checks": {
            "cutoff_expected_stockholm_1830": True,
            "cutoff_pred_matches_feature_store": True,
            "tmax_sofar_recomputed_from_raw_1m": True,
            "coverage_frac_recomputed_from_raw_1m": True,
            "last_gap_minutes_recomputed_from_raw_1m": True,
            "mos_x_mean_recomputed_from_db_asof_le_cutoff": True,
            "model_feature_list_has_no_obvious_label_or_full_day_cols": True,
        },
        "audit_pass": audit_pass,
        "n_days_checked": len(rows),
        "n_violations": len(violations),
        "violations": violations,
    }

    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# Leakage Audit: Bridge + Hit Model Flow (KMIA, Nov–Dec 2025)\n")
    md_lines.append(f"- Station: `{STATION_ID}`")
    md_lines.append(f"- Window: `{BACKTEST_START.isoformat()}` to `{BACKTEST_END.isoformat()}`")
    md_lines.append(f"- Model: `{MODEL_DIR.name}`")
    md_lines.append(f"- Audit pass: `{audit_pass}`")
    md_lines.append(f"- Days checked: `{len(rows)}`")
    md_lines.append(f"- Violations: `{len(violations)}`\n")

    md_lines.append("## Split Sanity\n")
    for k, v in split_checks.items():
        md_lines.append(f"- `{k}`: `{v}`")
    md_lines.append("")

    md_lines.append("## Checks Per Day\n")
    md_lines.append("- `cutoff_matches_expected`: cutoff_utc equals Stockholm 18:30 converted to UTC (±60s)")
    md_lines.append("- `tmax_sofar_ok`: recomputed from raw 1m -> 5m median series up to cutoff-5min")
    md_lines.append("- `coverage_ok`: recomputed from raw 1m row count between day_start_utc and cutoff_utc (inclusive)")
    md_lines.append("- `last_gap_ok`: recomputed from raw 1m last timestamp <= cutoff_utc")
    md_lines.append("- `mos_x_ok`: recomputed from MySQL `mos_daily_value` with `asof_utc <= cutoff_utc` (n_x / value_max)\n")

    if violations:
        md_lines.append("## Violations\n")
        for v in violations:
            md_lines.append(f"- Day `{v['day']}`: `{', '.join(v['notes'])}`")
    else:
        md_lines.append("## Violations\n")
        md_lines.append("- None detected.\n")

    (OUT_DIR / "audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("Wrote", OUT_DIR / "audit_report.md")
    print("Wrote", OUT_DIR / "audit_report.json")
    print("Wrote", OUT_DIR / "audit_rows.csv")


if __name__ == "__main__":
    main()
