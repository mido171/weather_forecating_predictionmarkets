from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    allowed_signal_column,
    feature_family,
    load_features,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)

FOLDER_NAME = "0025_longhist_signal_atlas"
LONG_START = pd.Timestamp("1949-01-01")
EPOCHS = {
    "1949_1979": (pd.Timestamp("1949-01-01"), pd.Timestamp("1979-12-31")),
    "1980_1999": (pd.Timestamp("1980-01-01"), pd.Timestamp("1999-12-31")),
    "2000_2019": (pd.Timestamp("2000-01-01"), pd.Timestamp("2019-12-31")),
    "2020_2023": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")),
}
STATION_RE = re.compile(r"^isd_station_(?P<metric>.+)_(?P<station>\d{5,6}_\d{5})$")
PAIR_METRICS = {
    "air_temperature_c",
    "dew_point_c",
    "sea_level_pressure_hpa",
    "wind_speed_mps",
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_station_feature(column: str) -> tuple[str, str] | None:
    match = STATION_RE.match(column)
    if not match:
        return None
    return match.group("station"), match.group("metric")


def safe_corr(a: pd.Series, b: pd.Series, *, min_rows: int = 500, method: str = "pearson") -> float:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() <= 2 or pair.iloc[:, 1].nunique() <= 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def tail_spread(feature: pd.Series, target: pd.Series, *, min_rows: int = 500) -> tuple[int, float, float, float]:
    values = pd.to_numeric(feature, errors="coerce")
    labels = pd.to_numeric(target, errors="coerce")
    pair = pd.DataFrame({"feature": values, "target": labels}).dropna()
    if len(pair) < min_rows or pair["feature"].nunique() <= 2:
        return 0, math.nan, math.nan, math.nan
    low = pair["feature"].quantile(0.10)
    high = pair["feature"].quantile(0.90)
    low_target = pair.loc[pair["feature"] <= low, "target"]
    high_target = pair.loc[pair["feature"] >= high, "target"]
    if low_target.empty or high_target.empty:
        return 0, math.nan, math.nan, math.nan
    return int(len(pair)), float(high_target.mean() - low_target.mean()), float(low_target.mean()), float(high_target.mean())


def add_analysis_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(out["target_date"], context="long-history signal atlas")
    if "clim_constrained_equal_blend_lag7_c" in out.columns:
        out["target_anomaly_vs_clim_c"] = (
            pd.to_numeric(out["target_tmax_c"], errors="coerce")
            - pd.to_numeric(out["clim_constrained_equal_blend_lag7_c"], errors="coerce")
        )
    else:
        out["target_anomaly_vs_clim_c"] = np.nan
    if "target_lag7_tmax_c" in out.columns:
        out["target_change_vs_lag7_c"] = (
            pd.to_numeric(out["target_tmax_c"], errors="coerce")
            - pd.to_numeric(out["target_lag7_tmax_c"], errors="coerce")
        )
    else:
        out["target_change_vs_lag7_c"] = np.nan
    out["hot_tail_flag"] = out["target_anomaly_vs_clim_c"] >= out["target_anomaly_vs_clim_c"].quantile(0.90)
    out["cool_tail_flag"] = out["target_anomaly_vs_clim_c"] <= out["target_anomaly_vs_clim_c"].quantile(0.10)
    return out[(out["target_date"] >= LONG_START) & (out["target_date"] < CONFIRMATION_START)].copy()


def numeric_signal_columns(frame: pd.DataFrame) -> list[str]:
    forbidden = {"target_anomaly_vs_clim_c", "target_change_vs_lag7_c", "hot_tail_flag", "cool_tail_flag"}
    return [
        column
        for column in frame.columns
        if column not in forbidden
        and allowed_signal_column(column)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def epoch_mask(frame: pd.DataFrame, epoch: str) -> pd.Series:
    start, end = EPOCHS[epoch]
    dates = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    return (dates >= start) & (dates <= end)


def sign_consistency(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value) and value != 0.0]
    if not finite:
        return math.nan
    positives = sum(value > 0 for value in finite)
    negatives = sum(value < 0 for value in finite)
    return float(max(positives, negatives) / len(finite))


def build_feature_atlas(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    columns = numeric_signal_columns(frame)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        n_all = int(values.notna().sum())
        if n_all < 500 or values.nunique(dropna=True) <= 2:
            continue
        valid_dates = pd.to_datetime(frame.loc[values.notna(), "target_date"], errors="coerce")
        epoch_anomaly_corrs: list[float] = []
        epoch_change_corrs: list[float] = []
        for epoch in EPOCHS:
            subset = frame.loc[epoch_mask(frame, epoch)]
            anomaly_corr = safe_corr(subset[column], subset["target_anomaly_vs_clim_c"])
            change_corr = safe_corr(subset[column], subset["target_change_vs_lag7_c"])
            target_corr = safe_corr(subset[column], subset["target_tmax_c"])
            epoch_rows.append(
                {
                    "feature": column,
                    "family": feature_family(column),
                    "epoch": epoch,
                    "n": int(pd.to_numeric(subset[column], errors="coerce").notna().sum()),
                    "target_corr": target_corr,
                    "anomaly_corr": anomaly_corr,
                    "change_corr": change_corr,
                }
            )
            if np.isfinite(anomaly_corr):
                epoch_anomaly_corrs.append(anomaly_corr)
            if np.isfinite(change_corr):
                epoch_change_corrs.append(change_corr)

        _, anomaly_tail_spread, anomaly_low_mean, anomaly_high_mean = tail_spread(values, frame["target_anomaly_vs_clim_c"])
        _, change_tail_spread, change_low_mean, change_high_mean = tail_spread(values, frame["target_change_vs_lag7_c"])
        long_target_corr = safe_corr(values, frame["target_tmax_c"], min_rows=2000)
        long_anomaly_corr = safe_corr(values, frame["target_anomaly_vs_clim_c"], min_rows=2000)
        long_change_corr = safe_corr(values, frame["target_change_vs_lag7_c"], min_rows=2000)
        hot_tail_corr = safe_corr(values, frame["hot_tail_flag"].astype(float), min_rows=2000)
        rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "n_all": n_all,
                "coverage": float(n_all / len(frame)),
                "first_date": "" if valid_dates.empty else str(valid_dates.min().date()),
                "last_date": "" if valid_dates.empty else str(valid_dates.max().date()),
                "long_target_corr": long_target_corr,
                "long_anomaly_corr": long_anomaly_corr,
                "long_change_corr": long_change_corr,
                "hot_tail_corr": hot_tail_corr,
                "anomaly_tail_spread_c": anomaly_tail_spread,
                "anomaly_low_decile_mean_c": anomaly_low_mean,
                "anomaly_high_decile_mean_c": anomaly_high_mean,
                "change_tail_spread_c": change_tail_spread,
                "change_low_decile_mean_c": change_low_mean,
                "change_high_decile_mean_c": change_high_mean,
                "epochs_with_anomaly_signal": len(epoch_anomaly_corrs),
                "epoch_anomaly_sign_consistency": sign_consistency(epoch_anomaly_corrs),
                "median_abs_epoch_anomaly_corr": float(np.nanmedian(np.abs(epoch_anomaly_corrs)))
                if epoch_anomaly_corrs
                else math.nan,
                "epochs_with_change_signal": len(epoch_change_corrs),
                "epoch_change_sign_consistency": sign_consistency(epoch_change_corrs),
                "median_abs_epoch_change_corr": float(np.nanmedian(np.abs(epoch_change_corrs)))
                if epoch_change_corrs
                else math.nan,
            }
        )

    atlas = pd.DataFrame(rows)
    if not atlas.empty:
        atlas["information_priority"] = (
            atlas["long_anomaly_corr"].abs().fillna(0.0) * 2.0
            + atlas["long_change_corr"].abs().fillna(0.0) * 1.5
            + atlas["hot_tail_corr"].abs().fillna(0.0)
            + atlas["anomaly_tail_spread_c"].abs().fillna(0.0) / 5.0
            + atlas["change_tail_spread_c"].abs().fillna(0.0) / 5.0
            + atlas["epoch_anomaly_sign_consistency"].fillna(0.0) * atlas["median_abs_epoch_anomaly_corr"].fillna(0.0)
        )
        atlas = atlas.sort_values("information_priority", ascending=False).reset_index(drop=True)
    epochs = pd.DataFrame(epoch_rows)
    return atlas, epochs


def build_family_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return atlas
    return (
        atlas.groupby("family", observed=True)
        .agg(
            feature_count=("feature", "count"),
            best_information_priority=("information_priority", "max"),
            best_abs_anomaly_corr=("long_anomaly_corr", lambda s: float(s.abs().max())),
            best_abs_change_corr=("long_change_corr", lambda s: float(s.abs().max())),
            best_abs_hot_tail_corr=("hot_tail_corr", lambda s: float(s.abs().max())),
            best_abs_anomaly_tail_spread_c=("anomaly_tail_spread_c", lambda s: float(s.abs().max())),
        )
        .reset_index()
        .sort_values("best_information_priority", ascending=False)
    )


def build_station_metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for column in numeric_signal_columns(frame):
        parsed = parse_station_feature(column)
        if parsed is None:
            continue
        station, metric = parsed
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 500:
            continue
        rows.append(
            {
                "station": station,
                "metric": metric,
                "feature": column,
                "n": int(values.notna().sum()),
                "first_date": str(pd.to_datetime(frame.loc[values.notna(), "target_date"]).min().date()),
                "last_date": str(pd.to_datetime(frame.loc[values.notna(), "target_date"]).max().date()),
                "target_corr": safe_corr(values, frame["target_tmax_c"], min_rows=2000),
                "anomaly_corr": safe_corr(values, frame["target_anomaly_vs_clim_c"], min_rows=2000),
                "change_corr": safe_corr(values, frame["target_change_vs_lag7_c"], min_rows=2000),
                "hot_tail_corr": safe_corr(values, frame["hot_tail_flag"].astype(float), min_rows=2000),
                "anomaly_tail_spread_c": tail_spread(values, frame["target_anomaly_vs_clim_c"])[1],
            }
        )
    metric_table = pd.DataFrame(rows)
    if metric_table.empty:
        return metric_table, pd.DataFrame()
    metric_table["station_metric_priority"] = (
        metric_table["anomaly_corr"].abs().fillna(0.0) * 2.0
        + metric_table["change_corr"].abs().fillna(0.0)
        + metric_table["hot_tail_corr"].abs().fillna(0.0)
        + metric_table["anomaly_tail_spread_c"].abs().fillna(0.0) / 5.0
    )
    metric_table = metric_table.sort_values("station_metric_priority", ascending=False).reset_index(drop=True)
    station_summary = (
        metric_table.groupby("station", observed=True)
        .agg(
            metric_count=("metric", "nunique"),
            best_metric_priority=("station_metric_priority", "max"),
            best_abs_anomaly_corr=("anomaly_corr", lambda s: float(s.abs().max())),
            best_abs_change_corr=("change_corr", lambda s: float(s.abs().max())),
            best_abs_hot_tail_corr=("hot_tail_corr", lambda s: float(s.abs().max())),
            best_abs_anomaly_tail_spread_c=("anomaly_tail_spread_c", lambda s: float(s.abs().max())),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
        )
        .reset_index()
        .sort_values("best_metric_priority", ascending=False)
    )
    return metric_table, station_summary


def build_station_pair_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    station_columns: dict[str, list[tuple[str, str]]] = {}
    for column in numeric_signal_columns(frame):
        parsed = parse_station_feature(column)
        if parsed is None:
            continue
        station, metric = parsed
        if metric in PAIR_METRICS:
            station_columns.setdefault(metric, []).append((station, column))

    rows: list[dict[str, object]] = []
    for metric, items in station_columns.items():
        for (left_station, left_col), (right_station, right_col) in combinations(sorted(items), 2):
            spread = pd.to_numeric(frame[left_col], errors="coerce") - pd.to_numeric(frame[right_col], errors="coerce")
            if spread.notna().sum() < 500 or spread.nunique(dropna=True) <= 2:
                continue
            rows.append(
                {
                    "metric": metric,
                    "left_station": left_station,
                    "right_station": right_station,
                    "spread_feature": f"{left_col}__minus__{right_col}",
                    "n": int(spread.notna().sum()),
                    "spread_std": float(spread.std(skipna=True)),
                    "target_corr": safe_corr(spread, frame["target_tmax_c"], min_rows=2000),
                    "anomaly_corr": safe_corr(spread, frame["target_anomaly_vs_clim_c"], min_rows=2000),
                    "change_corr": safe_corr(spread, frame["target_change_vs_lag7_c"], min_rows=2000),
                    "hot_tail_corr": safe_corr(spread, frame["hot_tail_flag"].astype(float), min_rows=2000),
                    "anomaly_tail_spread_c": tail_spread(spread, frame["target_anomaly_vs_clim_c"])[1],
                }
            )
    spreads = pd.DataFrame(rows)
    if not spreads.empty:
        spreads["spread_priority"] = (
            spreads["anomaly_corr"].abs().fillna(0.0) * 2.0
            + spreads["change_corr"].abs().fillna(0.0)
            + spreads["hot_tail_corr"].abs().fillna(0.0)
            + spreads["anomaly_tail_spread_c"].abs().fillna(0.0) / 5.0
        )
        spreads = spreads.sort_values("spread_priority", ascending=False).reset_index(drop=True)
    return spreads


def write_outputs(
    *,
    frame: pd.DataFrame,
    atlas: pd.DataFrame,
    epochs: pd.DataFrame,
    family_summary: pd.DataFrame,
    station_metrics: pd.DataFrame,
    station_summary: pd.DataFrame,
    station_spreads: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    write_csv(artifacts / "feature_atlas.csv", atlas)
    write_csv(artifacts / "epoch_stability.csv", epochs)
    write_csv(artifacts / "family_summary.csv", family_summary)
    write_csv(artifacts / "station_metrics.csv", station_metrics)
    write_csv(artifacts / "station_summary.csv", station_summary)
    write_csv(artifacts / "station_spreads.csv", station_spreads)

    best_feature = atlas.iloc[0] if not atlas.empty else None
    best_station = station_summary.iloc[0] if not station_summary.empty else None
    best_spread = station_spreads.iloc[0] if not station_spreads.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "feature_rows": int(len(atlas)),
        "station_metric_rows": int(len(station_metrics)),
        "station_spread_rows": int(len(station_spreads)),
        "best_feature": "" if best_feature is None else str(best_feature["feature"]),
        "best_feature_family": "" if best_feature is None else str(best_feature["family"]),
        "best_feature_priority": None if best_feature is None else float(best_feature["information_priority"]),
        "best_station": "" if best_station is None else str(best_station["station"]),
        "best_station_priority": None if best_station is None else float(best_station["best_metric_priority"]),
        "best_spread": "" if best_spread is None else str(best_spread["spread_feature"]),
        "best_spread_priority": None if best_spread is None else float(best_spread["spread_priority"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "longhist_signal_atlas_manifest.json", manifest)

    best_feature_text = "No feature was scoreable."
    if best_feature is not None:
        best_feature_text = (
            f"Best feature by calendar-adjusted priority: `{best_feature['feature']}` "
            f"({best_feature['family']}) with anomaly corr `{best_feature['long_anomaly_corr']:.4f}`, "
            f"change corr `{best_feature['long_change_corr']:.4f}`, and anomaly decile spread "
            f"`{best_feature['anomaly_tail_spread_c']:.4f} C`."
        )
    best_station_text = "No station metric was scoreable."
    if best_station is not None:
        best_station_text = (
            f"Best station by metric aggregation: `{best_station['station']}` with best anomaly corr "
            f"`{best_station['best_abs_anomaly_corr']:.4f}` and best anomaly decile spread "
            f"`{best_station['best_abs_anomaly_tail_spread_c']:.4f} C`."
        )
    best_spread_text = "No station-pair spread was scoreable."
    if best_spread is not None:
        best_spread_text = (
            f"Best station-pair spread: `{best_spread['left_station']} - {best_spread['right_station']}` "
            f"for `{best_spread['metric']}`, anomaly corr `{best_spread['anomaly_corr']:.4f}`, "
            f"anomaly decile spread `{best_spread['anomaly_tail_spread_c']:.4f} C`."
        )

    readme = f"""# Long-History Calendar-Adjusted Signal Atlas

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight scans the long-history feature matrix from `{manifest['first_target_date']}` to `{manifest['last_target_date']}`, restricted to pre-2024 labels and focused on the requested deep station/weather signal discovery. It evaluates every numeric leakage-eligible feature against:

- raw target Tmax;
- calendar-adjusted target anomaly: `target_tmax_c - clim_constrained_equal_blend_lag7_c`;
- target change versus the seven-day lag: `target_tmax_c - target_lag7_tmax_c`;
- hot-tail probability based on top-decile calendar-adjusted anomaly.

It separately summarizes all station metrics and all pairwise station spreads for air temperature, dew point, sea-level pressure, and wind speed.

## Leakage Contract

- All rows are before `{CONFIRMATION_START.date()}`.
- The scan uses only columns already present in the T24 feature matrix and excludes current target labels as features.
- Calendar adjustment uses the existing lagged/fold-safe climatology column where available.
- This is signal discovery, not a trained production model.

## Main Results

{best_feature_text}

{best_station_text}

{best_spread_text}

## Family Summary

{markdown_table(family_summary, max_rows=20)}

## Top Features

{markdown_table(atlas.head(30), max_rows=30)}

## Top Stations

{markdown_table(station_summary.head(25), max_rows=25)}

## Top Station-Pair Spreads

{markdown_table(station_spreads.head(25), max_rows=25)}

## Interpretation

This atlas is meant to identify where the long historical record contains stable, calendar-adjusted information beyond simple seasonality. Strong target correlations are expected for thermal fields, but anomaly/change correlations and station-pair spread signals are more useful for residual correction and regime identification. These findings should feed the next official-anchor residual model only after the official forecast archive has enough continuous coverage.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Long-History Calendar-Adjusted Signal Atlas\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_longhist_signal_atlas.py`:

- `{FOLDER_NAME}`: long-history all-feature, station, and station-pair spread atlas against target Tmax anomaly/change.

| Metric | Value |
|---|---:|
| Rows | {manifest['rows']} |
| Feature rows | {manifest['feature_rows']} |
| Station metric rows | {manifest['station_metric_rows']} |
| Station spread rows | {manifest['station_spread_rows']} |
| Best feature | {manifest['best_feature']} |
| Best station | {manifest['best_station']} |
| Best station spread | {manifest['best_spread']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; current target labels are used only as analysis outcomes, not as features.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = add_analysis_targets(load_features())
    atlas, epochs = build_feature_atlas(frame)
    family_summary = build_family_summary(atlas)
    station_metrics, station_summary = build_station_metric_tables(frame)
    station_spreads = build_station_pair_spreads(frame)
    return write_outputs(
        frame=frame,
        atlas=atlas,
        epochs=epochs,
        family_summary=family_summary,
        station_metrics=station_metrics,
        station_summary=station_summary,
        station_spreads=station_spreads,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 long-history calendar-adjusted signal atlas.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
