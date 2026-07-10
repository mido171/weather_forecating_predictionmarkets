from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    HEADLINE_END,
    HEADLINE_START,
    RESEARCH_ROOT,
    allowed_signal_column,
    build_analysis_frame,
    feature_family,
    load_champion_predictions,
    load_features,
    markdown_table,
    require_no_confirmation_dates,
    scan_features,
    write_csv,
    write_json,
    write_text,
)

TOP_FEATURE_LIMIT = 80
STATION_RE = re.compile(r"^isd_station_(?P<metric>.+)_(?P<station>\d{5,6}_\d{5})$")


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_corr(a: pd.Series, b: pd.Series, *, method: str = "pearson", min_rows: int = 120) -> float:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() <= 2 or pair.iloc[:, 1].nunique() <= 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def score_prediction(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int | str]:
    scored = frame[["target_date", "target_tmax_c", prediction_col]].dropna().copy()
    if scored.empty:
        return {"n": 0, "first_date": "", "last_date": "", "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    error = scored[prediction_col] - scored["target_tmax_c"]
    return {
        "n": int(len(scored)),
        "first_date": str(pd.to_datetime(scored["target_date"]).min().date()),
        "last_date": str(pd.to_datetime(scored["target_date"]).max().date()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
    }


def numeric_signal_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if allowed_signal_column(col) and pd.api.types.is_numeric_dtype(frame[col])
    ]


def epoch_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    dates = pd.to_datetime(frame["target_date"]).dt.normalize()
    return {
        "1949_1979": (dates >= pd.Timestamp("1949-01-01")) & (dates <= pd.Timestamp("1979-12-31")),
        "1980_1999": (dates >= pd.Timestamp("1980-01-01")) & (dates <= pd.Timestamp("1999-12-31")),
        "2000_2019": (dates >= pd.Timestamp("2000-01-01")) & (dates <= pd.Timestamp("2019-12-31")),
        "2020_2023": (dates >= HEADLINE_START) & (dates <= HEADLINE_END),
    }


def build_epoch_stability(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_no_confirmation_dates(frame["target_date"], context="deep forensics epoch stability")
    masks = epoch_masks(frame)
    columns = numeric_signal_columns(frame)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    headline = frame[(frame["target_date"] >= HEADLINE_START) & (frame["target_date"] <= HEADLINE_END)].copy()
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        epoch_corrs: list[float] = []
        epoch_signs: list[int] = []
        epoch_count = 0
        for epoch, mask in masks.items():
            subset = frame.loc[mask]
            corr = safe_corr(subset[col], subset["target_tmax_c"], min_rows=500)
            n = int(pd.to_numeric(subset[col], errors="coerce").notna().sum())
            detail_rows.append(
                {
                    "feature": col,
                    "family": feature_family(col),
                    "epoch": epoch,
                    "n": n,
                    "target_corr": corr,
                }
            )
            if np.isfinite(corr):
                epoch_corrs.append(float(corr))
                epoch_signs.append(1 if corr > 0 else -1)
                epoch_count += 1

        valid_dates = pd.to_datetime(frame.loc[series.notna(), "target_date"], errors="coerce")
        residual_corr = safe_corr(headline[col], headline["champion_error_c"], min_rows=500)
        abs_error_corr = safe_corr(headline[col], headline["champion_abs_error_c"], min_rows=500)
        if epoch_signs:
            sign_consistency = max(epoch_signs.count(1), epoch_signs.count(-1)) / len(epoch_signs)
            median_abs_epoch_corr = float(np.nanmedian(np.abs(epoch_corrs)))
            min_abs_epoch_corr = float(np.nanmin(np.abs(epoch_corrs)))
        else:
            sign_consistency = math.nan
            median_abs_epoch_corr = math.nan
            min_abs_epoch_corr = math.nan
        summary_rows.append(
            {
                "feature": col,
                "family": feature_family(col),
                "n_all": int(series.notna().sum()),
                "first_date": "" if valid_dates.empty else str(valid_dates.min().date()),
                "last_date": "" if valid_dates.empty else str(valid_dates.max().date()),
                "epochs_with_signal": epoch_count,
                "sign_consistency": sign_consistency,
                "median_abs_epoch_target_corr": median_abs_epoch_corr,
                "min_abs_epoch_target_corr": min_abs_epoch_corr,
                "headline_residual_corr": residual_corr,
                "headline_abs_error_corr": abs_error_corr,
                "stability_priority": (
                    (0.0 if math.isnan(sign_consistency) else sign_consistency)
                    * (0.0 if math.isnan(median_abs_epoch_corr) else median_abs_epoch_corr)
                    + 2.0 * abs(0.0 if math.isnan(residual_corr) else residual_corr)
                    + abs(0.0 if math.isnan(abs_error_corr) else abs_error_corr)
                ),
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(["feature", "epoch"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values("stability_priority", ascending=False).reset_index(drop=True)
    return detail, summary


def parse_station_feature(column: str) -> tuple[str, str] | None:
    match = STATION_RE.match(column)
    if not match:
        return None
    return match.group("station"), match.group("metric")


def build_station_forensics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_no_confirmation_dates(frame["target_date"], context="deep forensics station attribution")
    headline = frame[(frame["target_date"] >= HEADLINE_START) & (frame["target_date"] <= HEADLINE_END)].copy()
    rows: list[dict[str, object]] = []
    for col in numeric_signal_columns(frame):
        parsed = parse_station_feature(col)
        if parsed is None:
            continue
        station, metric = parsed
        values = pd.to_numeric(frame[col], errors="coerce")
        valid_dates = pd.to_datetime(frame.loc[values.notna(), "target_date"], errors="coerce")
        rows.append(
            {
                "station": station,
                "metric": metric,
                "feature": col,
                "n_all": int(values.notna().sum()),
                "first_date": "" if valid_dates.empty else str(valid_dates.min().date()),
                "last_date": "" if valid_dates.empty else str(valid_dates.max().date()),
                "long_target_corr": safe_corr(frame[col], frame["target_tmax_c"], min_rows=2000),
                "headline_target_corr": safe_corr(headline[col], headline["target_tmax_c"], min_rows=500),
                "headline_residual_corr": safe_corr(headline[col], headline["champion_error_c"], min_rows=500),
                "headline_abs_error_corr": safe_corr(headline[col], headline["champion_abs_error_c"], min_rows=500),
            }
        )
    station_metric = pd.DataFrame(rows)
    if station_metric.empty:
        return station_metric, pd.DataFrame(), pd.DataFrame()

    station_metric["attribution_priority"] = (
        station_metric["long_target_corr"].abs().fillna(0.0) * 0.5
        + station_metric["headline_residual_corr"].abs().fillna(0.0) * 2.0
        + station_metric["headline_abs_error_corr"].abs().fillna(0.0)
    )
    station_metric = station_metric.sort_values("attribution_priority", ascending=False).reset_index(drop=True)
    station_summary = (
        station_metric.groupby("station", observed=True)
        .agg(
            metric_count=("metric", "nunique"),
            best_long_target_corr=("long_target_corr", lambda s: float(s.abs().max())),
            best_headline_residual_corr=("headline_residual_corr", lambda s: float(s.abs().max())),
            best_headline_abs_error_corr=("headline_abs_error_corr", lambda s: float(s.abs().max())),
            best_priority=("attribution_priority", "max"),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
        )
        .sort_values("best_priority", ascending=False)
        .reset_index()
    )

    temp_cols = [
        col
        for col in station_metric.loc[station_metric["metric"].eq("air_temperature_c"), "feature"].to_list()
        if col in headline.columns
    ]
    spread_rows: list[dict[str, object]] = []
    for left_idx, left in enumerate(temp_cols):
        left_station = parse_station_feature(left)[0] if parse_station_feature(left) else left
        for right in temp_cols[left_idx + 1 :]:
            right_station = parse_station_feature(right)[0] if parse_station_feature(right) else right
            diff = pd.to_numeric(headline[left], errors="coerce") - pd.to_numeric(headline[right], errors="coerce")
            spread_rows.append(
                {
                    "left_station": left_station,
                    "right_station": right_station,
                    "spread_feature": f"{left}__minus__{right}",
                    "n": int(diff.notna().sum()),
                    "target_corr": safe_corr(diff, headline["target_tmax_c"], min_rows=500),
                    "residual_corr": safe_corr(diff, headline["champion_error_c"], min_rows=500),
                    "abs_error_corr": safe_corr(diff, headline["champion_abs_error_c"], min_rows=500),
                    "spread_std_c": float(diff.std(skipna=True)) if diff.notna().sum() else math.nan,
                }
            )
    spreads = pd.DataFrame(spread_rows)
    if not spreads.empty:
        spreads["spread_priority"] = (
            spreads["residual_corr"].abs().fillna(0.0) * 2.0
            + spreads["abs_error_corr"].abs().fillna(0.0)
            + spreads["target_corr"].abs().fillna(0.0) * 0.5
        )
        spreads = spreads.sort_values("spread_priority", ascending=False).reset_index(drop=True)
    return station_metric, station_summary, spreads


def qcut_labels(values: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = [f"{prefix}{idx + 1}" for idx in range(bins)]
    if numeric.notna().sum() < bins * 20 or numeric.nunique(dropna=True) < bins:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    return pd.qcut(ranked, bins, labels=labels).astype(str).where(numeric.notna(), "missing")


def select_deep_features(feature_scan: pd.DataFrame, *, limit: int = TOP_FEATURE_LIMIT) -> list[str]:
    if feature_scan.empty:
        return []
    sort_cols = ["discovery_priority_score", "max_abs_residual_corr", "max_abs_abs_error_corr", "max_abs_target_corr"]
    ordered = feature_scan.sort_values(sort_cols, ascending=False)
    chosen: list[str] = []
    for feature in ordered["feature"].to_list():
        if feature not in chosen:
            chosen.append(feature)
        if len(chosen) >= limit:
            break
    return chosen


def build_nonlinear_thresholds(frame: pd.DataFrame, feature_scan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = frame[(frame["target_date"] >= HEADLINE_START) & (frame["target_date"] <= HEADLINE_END)].copy()
    require_no_confirmation_dates(headline["target_date"], context="deep forensics nonlinear thresholds")
    features = select_deep_features(feature_scan, limit=50)
    table_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in headline.columns:
            continue
        labels = qcut_labels(headline[feature], 10, "d")
        if labels.eq("insufficient").all():
            continue
        work = headline[["target_date", "target_tmax_c", "point_forecast", "champion_error_c", "champion_abs_error_c", feature]].copy()
        work["decile"] = labels
        work["feature_value"] = pd.to_numeric(work[feature], errors="coerce")
        grouped = (
            work.groupby("decile", dropna=False, observed=True)
            .agg(
                n=("target_date", "count"),
                feature_min=("feature_value", "min"),
                feature_max=("feature_value", "max"),
                feature_mean=("feature_value", "mean"),
                target_mean_c=("target_tmax_c", "mean"),
                champion_mae_c=("champion_abs_error_c", "mean"),
                champion_bias_c=("champion_error_c", "mean"),
            )
            .reset_index()
        )
        grouped["feature"] = feature
        grouped["family"] = feature_family(feature)
        table_rows.extend(grouped.to_dict("records"))
        valid = grouped[grouped["n"] >= 50].copy()
        if valid.empty:
            continue
        summary_rows.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "decile_count": int(len(valid)),
                "target_mean_spread_c": float(valid["target_mean_c"].max() - valid["target_mean_c"].min()),
                "champion_mae_spread_c": float(valid["champion_mae_c"].max() - valid["champion_mae_c"].min()),
                "champion_bias_spread_c": float(valid["champion_bias_c"].max() - valid["champion_bias_c"].min()),
                "worst_mae_decile": str(valid.sort_values("champion_mae_c", ascending=False).iloc[0]["decile"]),
                "largest_underprediction_decile": str(valid.sort_values("champion_bias_c").iloc[0]["decile"]),
                "largest_overprediction_decile": str(valid.sort_values("champion_bias_c", ascending=False).iloc[0]["decile"]),
            }
        )
    detail = pd.DataFrame(table_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["threshold_priority"] = (
            summary["champion_bias_spread_c"].abs()
            + summary["champion_mae_spread_c"].abs()
            + 0.25 * summary["target_mean_spread_c"].abs()
        )
        summary = summary.sort_values("threshold_priority", ascending=False).reset_index(drop=True)
    return detail, summary


def build_interaction_screens(frame: pd.DataFrame, feature_scan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = frame[(frame["target_date"] >= HEADLINE_START) & (frame["target_date"] <= HEADLINE_END)].copy()
    require_no_confirmation_dates(headline["target_date"], context="deep forensics interaction screens")
    selected = select_deep_features(feature_scan, limit=18)
    grid_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for left_idx, left in enumerate(selected):
        if left not in headline.columns:
            continue
        left_bin = qcut_labels(headline[left], 3, "lq")
        if left_bin.eq("insufficient").all():
            continue
        for right in selected[left_idx + 1 :]:
            if right not in headline.columns or feature_family(left) == feature_family(right):
                continue
            right_bin = qcut_labels(headline[right], 3, "rq")
            if right_bin.eq("insufficient").all():
                continue
            work = headline[["target_date", "target_tmax_c", "champion_error_c", "champion_abs_error_c"]].copy()
            work["left_bin"] = left_bin
            work["right_bin"] = right_bin
            grouped = (
                work.groupby(["left_bin", "right_bin"], dropna=False, observed=True)
                .agg(
                    n=("target_date", "count"),
                    target_mean_c=("target_tmax_c", "mean"),
                    champion_mae_c=("champion_abs_error_c", "mean"),
                    champion_bias_c=("champion_error_c", "mean"),
                )
                .reset_index()
            )
            valid = grouped[grouped["n"] >= 40].copy()
            if len(valid) < 4:
                continue
            for row in valid.to_dict("records"):
                grid_rows.append(
                    {
                        "left_feature": left,
                        "left_family": feature_family(left),
                        "right_feature": right,
                        "right_family": feature_family(right),
                        **row,
                    }
                )
            summary_rows.append(
                {
                    "left_feature": left,
                    "left_family": feature_family(left),
                    "right_feature": right,
                    "right_family": feature_family(right),
                    "valid_cells": int(len(valid)),
                    "min_cell_n": int(valid["n"].min()),
                    "target_mean_spread_c": float(valid["target_mean_c"].max() - valid["target_mean_c"].min()),
                    "champion_mae_spread_c": float(valid["champion_mae_c"].max() - valid["champion_mae_c"].min()),
                    "champion_bias_spread_c": float(valid["champion_bias_c"].max() - valid["champion_bias_c"].min()),
                }
            )
    grid = pd.DataFrame(grid_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["interaction_priority"] = (
            summary["champion_bias_spread_c"].abs()
            + summary["champion_mae_spread_c"].abs()
            + 0.15 * summary["target_mean_spread_c"].abs()
        )
        summary = summary.sort_values("interaction_priority", ascending=False).reset_index(drop=True)
    return grid, summary


def past_only_feature_bucket_predictions(
    frame: pd.DataFrame,
    feature: str,
    *,
    bins: int,
    season_conditioned: bool,
    min_history: int = 180,
    min_group: int = 25,
) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    feature_values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
    corrections_actual_minus_champion = (
        pd.to_numeric(ordered["target_tmax_c"], errors="coerce")
        - pd.to_numeric(ordered["point_forecast"], errors="coerce")
    ).to_numpy(dtype=float)
    point_forecast = pd.to_numeric(ordered["point_forecast"], errors="coerce").to_numpy(dtype=float)
    seasons = ordered["season"].astype(str).to_numpy() if "season" in ordered.columns else np.array([""] * len(ordered))
    predictions: list[float] = []
    correction_rows: list[int] = []
    correction_values: list[float] = []
    for index, current_value in enumerate(feature_values):
        if not np.isfinite(current_value):
            predictions.append(math.nan)
            correction_rows.append(0)
            correction_values.append(math.nan)
            continue

        prior_values = feature_values[:index]
        prior_corrections = corrections_actual_minus_champion[:index]
        prior_seasons = seasons[:index]
        valid_prior = np.isfinite(prior_values) & np.isfinite(prior_corrections)
        prior_values = prior_values[valid_prior]
        prior_corrections = prior_corrections[valid_prior]
        prior_seasons = prior_seasons[valid_prior]
        if len(prior_values) < min_history or len(np.unique(prior_values)) < bins:
            predictions.append(math.nan)
            correction_rows.append(int(len(prior_values)))
            correction_values.append(math.nan)
            continue

        quantiles = np.quantile(prior_values, np.linspace(0, 1, bins + 1))
        quantiles = np.unique(quantiles)
        if len(quantiles) <= 2:
            group_mask = prior_values == current_value
        else:
            bin_index = int(np.searchsorted(quantiles[1:-1], current_value, side="right"))
            low = quantiles[bin_index]
            high = quantiles[bin_index + 1]
            if bin_index == 0:
                group_mask = prior_values <= high
            elif bin_index == len(quantiles) - 2:
                group_mask = prior_values >= low
            else:
                group_mask = (prior_values >= low) & (prior_values <= high)

        base_group_mask = group_mask.copy()
        if season_conditioned and "season" in ordered.columns:
            season_group_mask = group_mask & (prior_seasons == seasons[index])
            if int(season_group_mask.sum()) >= min_group:
                group_mask = season_group_mask
        if int(group_mask.sum()) < min_group:
            group_mask = base_group_mask
        if int(group_mask.sum()) < min_group:
            group_mask = np.ones(len(prior_values), dtype=bool)

        correction = float(np.mean(prior_corrections[group_mask]))
        predictions.append(float(point_forecast[index] + correction))
        correction_rows.append(int(group_mask.sum()))
        correction_values.append(correction)
    out = ordered[["target_date", "target_tmax_c", "point_forecast", "season", feature]].copy()
    out["candidate_prediction_c"] = predictions
    out["past_correction_c"] = correction_values
    out["past_rows_used"] = correction_rows
    return out


def build_residual_correction_screen(frame: pd.DataFrame, feature_scan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = frame[(frame["target_date"] >= HEADLINE_START) & (frame["target_date"] <= HEADLINE_END)].copy()
    require_no_confirmation_dates(headline["target_date"], context="deep forensics residual correction screen")
    features = select_deep_features(feature_scan, limit=TOP_FEATURE_LIMIT)
    score_rows: list[dict[str, object]] = []
    best_predictions = pd.DataFrame()
    for feature in features:
        if feature not in headline.columns:
            continue
        for bins in (3, 5):
            for season_conditioned in (False, True):
                preds = past_only_feature_bucket_predictions(
                    headline,
                    feature,
                    bins=bins,
                    season_conditioned=season_conditioned,
                )
                scored = preds.dropna(subset=["candidate_prediction_c"]).copy()
                if len(scored) < 500:
                    continue
                candidate = score_prediction(scored, "candidate_prediction_c")
                baseline = score_prediction(scored.rename(columns={"point_forecast": "baseline_prediction_c"}), "baseline_prediction_c")
                row = {
                    "feature": feature,
                    "family": feature_family(feature),
                    "bins": bins,
                    "season_conditioned": season_conditioned,
                    "candidate_n": candidate["n"],
                    "first_date": candidate["first_date"],
                    "last_date": candidate["last_date"],
                    "candidate_mae": candidate["mae"],
                    "candidate_rmse": candidate["rmse"],
                    "candidate_bias": candidate["bias"],
                    "champion_same_rows_mae": baseline["mae"],
                    "champion_same_rows_rmse": baseline["rmse"],
                    "mae_delta_vs_champion": float(candidate["mae"] - baseline["mae"]),
                    "median_abs_correction_c": float(preds["past_correction_c"].abs().median(skipna=True)),
                }
                score_rows.append(row)
                if best_predictions.empty or row["mae_delta_vs_champion"] < float(
                    best_predictions.attrs.get("best_delta", math.inf)
                ):
                    best_predictions = preds.copy()
                    best_predictions.attrs["best_delta"] = row["mae_delta_vs_champion"]
                    best_predictions.attrs["best_feature"] = feature
                    best_predictions.attrs["best_bins"] = bins
                    best_predictions.attrs["best_season_conditioned"] = season_conditioned
    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        scores = scores.sort_values(["mae_delta_vs_champion", "candidate_mae"]).reset_index(drop=True)
    return scores, best_predictions


def write_epoch_stability(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0007_long_history_signal_stability"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "feature_target_correlation_by_epoch.csv", detail)
    write_csv(artifacts / "feature_stability_rank.csv", summary)
    text = f"""# Long-History Signal Stability

Generated: `{now_utc()}`

## What Was Tested

This insight asks whether candidate signals are stable across long historical eras, not just useful in the 2020-2023 headline window. Every leakage-eligible numeric feature was scored against HKO Tmax in four periods: 1949-1979, 1980-1999, 2000-2019, and 2020-2023. The same features were also checked against the champion residual and absolute error in 2020-2023.

## Why This Matters

A system targeting very low MAE cannot promote a feature because it works in one modern window. The strongest candidates should either have stable physical relationships over decades or explain a specific documented modern residual regime.

## Top Stable Signals

{markdown_table(summary.head(25), max_rows=25)}

## Interpretation

`stability_priority` is a research ranking, not a model score. High values mean a feature has a stable long-history relationship and/or explains champion residuals. Candidate promotion still requires rolling-origin OOF tests and ablation.
"""
    write_text(folder / "README.md", text)


def write_station_forensics(
    station_metric: pd.DataFrame,
    station_summary: pd.DataFrame,
    spreads: pd.DataFrame,
) -> None:
    folder = RESEARCH_ROOT / "0008_station_network_forensics"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "station_metric_attribution.csv", station_metric)
    write_csv(artifacts / "station_summary_rank.csv", station_summary)
    write_csv(artifacts / "station_temperature_spread_signals.csv", spreads)
    text = f"""# Station Network Forensics

Generated: `{now_utc()}`

## What Was Tested

This insight breaks the ISD station network into station-level attributes: air temperature, dew point, sea-level pressure, and wind speed. It also tests pairwise station-temperature spreads during the 2020-2023 headline OOF period.

## Why This Matters

The target station is not isolated. Surrounding stations can expose air-mass position, coastline/inland contrast, advection, humidity caps, and station disagreement. A station with weak direct residual correlation can still be valuable as a regime flag or uncertainty trigger.

## Best Stations By Attribution

{markdown_table(station_summary.head(20), max_rows=20)}

## Best Individual Station Metrics

{markdown_table(station_metric.head(25), max_rows=25)}

## Best Pairwise Temperature Spreads

{markdown_table(spreads.head(25), max_rows=25)}

## Interpretation

The station-temperature features strongly track target level. Residual correlations are smaller, so the next practical use is not a naive linear add-on. The better path is station-regime experts: coastal/inland spread, airport/HKO proxy, and station-disagreement uncertainty scaling.
"""
    write_text(folder / "README.md", text)


def write_nonlinear_thresholds(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0009_nonlinear_threshold_lifts"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "feature_decile_error_tables.csv", detail)
    write_csv(artifacts / "nonlinear_threshold_summary.csv", summary)
    text = f"""# Nonlinear Threshold Lifts

Generated: `{now_utc()}`

## What Was Tested

Top candidate signals were split into deciles inside the 2020-2023 headline OOF window. For each decile, the report records target Tmax level, champion MAE, and signed champion bias.

## Leakage Note

This is a diagnostic map, not a deployed rule. The deciles are used to discover shape and threshold behavior. Any actual correction derived from these features must be recreated with past-only fold-local thresholds.

## Strongest Threshold Candidates

{markdown_table(summary.head(25), max_rows=25)}

## Interpretation

Large bias spreads are especially important: they mean a feature may identify when the current champion systematically overpredicts or underpredicts. Large MAE spreads mean the feature may be useful for uncertainty scaling or model blending.
"""
    write_text(folder / "README.md", text)


def write_interactions(grid: pd.DataFrame, summary: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0010_cross_feature_interaction_screens"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "interaction_cell_error_grid.csv", grid)
    write_csv(artifacts / "interaction_summary_rank.csv", summary)
    text = f"""# Cross-Feature Interaction Screens

Generated: `{now_utc()}`

## What Was Tested

This insight pairs high-priority features from different families and bins each pair into 3x3 cells. Each cell reports target level, champion MAE, and champion bias.

## Why This Matters

Important weather signals are often conditional. A humidity feature, for example, can mean something different under one upper-air thermal profile than another. Pair screens identify combinations that deserve explicit regime experts.

## Top Interaction Candidates

{markdown_table(summary.head(25), max_rows=25)}

## Interpretation

These interactions are candidate maps, not accepted improvements. Large bias or MAE spreads indicate where the next generation of residual experts should focus, using rolling-origin validation and 2024+ locked confirmation only at the end.
"""
    write_text(folder / "README.md", text)


def write_residual_correction(scores: pd.DataFrame, predictions: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0011_past_only_residual_correction_screen"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "past_only_feature_bucket_scoreboard.csv", scores)
    if not predictions.empty:
        write_csv(artifacts / "best_past_only_candidate_predictions.csv", predictions)
    best_text = "No candidate produced enough past-only rows to score."
    if not scores.empty:
        row = scores.iloc[0]
        best_text = (
            f"Best candidate: `{row['feature']}`, bins `{row['bins']}`, season-conditioned "
            f"`{row['season_conditioned']}`, MAE `{row['candidate_mae']:.4f}` versus same-row "
            f"champion MAE `{row['champion_same_rows_mae']:.4f}` "
            f"(delta `{row['mae_delta_vs_champion']:.4f}`)."
        )
    text = f"""# Past-Only Residual Correction Screen

Generated: `{now_utc()}`

## What Was Tested

This is the first leakage-safe candidate screen for residual correction. For each candidate feature, the script walks through 2020-2023 date order. On each date, it uses only earlier rows to build feature buckets and estimates the champion correction from prior rows in the matching bucket. The current row's target is never used to choose its correction.

## Main Result

{best_text}

## Scoreboard

{markdown_table(scores.head(30), max_rows=30)}

## Interpretation

Negative `mae_delta_vs_champion` means the past-only bucket correction beat the champion on the same scored rows. This is not yet a promotion model: it is a screen for which features deserve proper rolling-origin experiments with ablations, uncertainty intervals, and locked confirmation.
"""
    write_text(folder / "README.md", text)


def update_master_research_index(summary: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Second-Stage Deep Forensics\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{summary['generated_at_utc']}`

Additional insight folders created by `scripts/run_hkg_t24_deep_signal_forensics.py`:

- `0007_long_history_signal_stability`: long-history era stability for every leakage-eligible numeric feature.
- `0008_station_network_forensics`: station-level attribution and pairwise station-temperature spread signals.
- `0009_nonlinear_threshold_lifts`: decile-level target, bias, and MAE threshold maps.
- `0010_cross_feature_interaction_screens`: 3x3 interaction screens across feature families.
- `0011_past_only_residual_correction_screen`: first leakage-safe past-only bucket correction screen.

Key run counts:

| Metric | Value |
|---|---:|
| Feature rows | {summary['feature_rows']} |
| Feature columns | {summary['feature_columns']} |
| Scanned features | {summary['feature_scan_rows']} |
| Epoch-stability rows | {summary['epoch_stability_rows']} |
| Station metric rows | {summary['station_metric_rows']} |
| Station spread rows | {summary['station_spread_rows']} |
| Nonlinear threshold features | {summary['threshold_rows']} |
| Interaction candidates | {summary['interaction_rows']} |
| Past-only residual candidates | {summary['residual_candidate_rows']} |

Leakage contract: confirmation labels from `{CONFIRMATION_START.date()}` onward remain blocked; residual screens use only prior rows for each correction.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    features = load_features()
    champion = load_champion_predictions()
    frame = build_analysis_frame(features, champion)
    require_no_confirmation_dates(frame["target_date"], context="deep forensics run frame")
    feature_scan = scan_features(frame)

    epoch_detail, epoch_summary = build_epoch_stability(frame)
    station_metric, station_summary, station_spreads = build_station_forensics(frame)
    threshold_detail, threshold_summary = build_nonlinear_thresholds(frame, feature_scan)
    interaction_grid, interaction_summary = build_interaction_screens(frame, feature_scan)
    residual_scores, residual_predictions = build_residual_correction_screen(frame, feature_scan)

    write_epoch_stability(epoch_detail, epoch_summary)
    write_station_forensics(station_metric, station_summary, station_spreads)
    write_nonlinear_thresholds(threshold_detail, threshold_summary)
    write_interactions(interaction_grid, interaction_summary)
    write_residual_correction(residual_scores, residual_predictions)

    manifest = {
        "generated_at_utc": now_utc(),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "feature_rows": int(len(frame)),
        "feature_columns": int(frame.shape[1]),
        "feature_scan_rows": int(len(feature_scan)),
        "epoch_detail_rows": int(len(epoch_detail)),
        "epoch_stability_rows": int(len(epoch_summary)),
        "station_metric_rows": int(len(station_metric)),
        "station_summary_rows": int(len(station_summary)),
        "station_spread_rows": int(len(station_spreads)),
        "threshold_detail_rows": int(len(threshold_detail)),
        "threshold_rows": int(len(threshold_summary)),
        "interaction_grid_rows": int(len(interaction_grid)),
        "interaction_rows": int(len(interaction_summary)),
        "residual_candidate_rows": int(len(residual_scores)),
        "folders": [
            "0007_long_history_signal_stability",
            "0008_station_network_forensics",
            "0009_nonlinear_threshold_lifts",
            "0010_cross_feature_interaction_screens",
            "0011_past_only_residual_correction_screen",
        ],
    }
    write_json(RESEARCH_ROOT / "deep_signal_forensics_manifest.json", manifest)
    update_master_research_index(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run second-stage HKG T24 deep signal forensics.").parse_args()


def main() -> None:
    parse_args()
    manifest = run()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
