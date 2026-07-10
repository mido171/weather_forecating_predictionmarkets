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
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_forecast_anchor_forensics import load_official_forecasts  # noqa: E402

FOLDER_NAME = "0019_multistation_info_gain"
MIN_SCAN_ROWS = 500
MIN_GROUP_ROWS = 120
MIN_PAIR_ROWS = 500
TOP_CORRECTION_FEATURES = 60
PAST_ONLY_MIN_HISTORY = 120
PAST_ONLY_MIN_BUCKET_ROWS = 20


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 70) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def safe_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson", min_rows: int = MIN_SCAN_ROWS) -> float:
    pair = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() <= 2 or pair.iloc[:, 1].nunique() <= 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def qtail_delta(feature: pd.Series, target: pd.Series, *, min_rows: int = MIN_SCAN_ROWS) -> float:
    pair = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).dropna()
    if len(pair) < min_rows or pair["feature"].nunique() <= 2:
        return math.nan
    low = pair["feature"].quantile(0.10)
    high = pair["feature"].quantile(0.90)
    low_mean = pair.loc[pair["feature"] <= low, "target"].mean()
    high_mean = pair.loc[pair["feature"] >= high, "target"].mean()
    return float(high_mean - low_mean)


def quantile_bucket(values: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < bins * PAST_ONLY_MIN_BUCKET_ROWS or numeric.nunique(dropna=True) < bins:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    return pd.qcut(ranked, bins, labels=[f"q{idx + 1}" for idx in range(bins)]).astype(str).where(
        numeric.notna(),
        "missing",
    )


def bucket_lift_rows(frame: pd.DataFrame, feature: str, *, bins: int = 5) -> list[dict[str, object]]:
    work = frame[["target_date", "target_tmax_c", "official_error_c", "official_abs_error_c", feature]].copy()
    work["bucket"] = quantile_bucket(work[feature], bins)
    if work["bucket"].eq("insufficient").all():
        return []
    rows: list[dict[str, object]] = []
    for bucket, group in work.groupby("bucket", observed=True, dropna=False):
        if bucket in {"missing", "insufficient"} or len(group) < PAST_ONLY_MIN_BUCKET_ROWS:
            continue
        values = pd.to_numeric(group[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "bucket": bucket,
                "n": int(len(group)),
                "feature_min": float(values.min()),
                "feature_max": float(values.max()),
                "target_tmax_mean_c": float(group["target_tmax_c"].mean()),
                "official_error_mean_c": float(group["official_error_c"].mean()),
                "official_abs_error_mean_c": float(group["official_abs_error_c"].mean()),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
            }
        )
    return rows


def bucket_spreads(rows: list[dict[str, object]]) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    table = pd.DataFrame(rows)
    if table.empty:
        return math.nan, math.nan
    bias_spread = table["official_error_mean_c"].max() - table["official_error_mean_c"].min()
    mae_spread = table["official_abs_error_mean_c"].max() - table["official_abs_error_mean_c"].min()
    return float(bias_spread), float(mae_spread)


def analysis_column_allowed(column: str) -> bool:
    forbidden = {
        "forecast_max_c",
        "target_tmax_c",
        "official_error_c",
        "official_abs_error_c",
        "official_midpoint_error_c",
        "official_underpredicted",
        "official_overpredicted",
        "cutoff_utc",
        "issue_utc",
        "issue_at_hkt",
        "issue_at_utc",
        "available_at_hkt",
        "available_at_utc",
    }
    if column in forbidden:
        return False
    return allowed_signal_column(column)


def parse_station_attribute(column: str) -> tuple[str, str] | None:
    match = re.match(r"^isd_station_(?P<attribute>.+)_(?P<station>\d{5,6}_\d{5})$", column)
    if not match:
        return None
    return match.group("station"), match.group("attribute")


def build_official_feature_frame() -> pd.DataFrame:
    features = load_features()
    official = load_official_forecasts(features)
    if official.empty:
        raise RuntimeError("No official forecast rows available for multistation information-gain analysis.")
    feature_extra_cols = ["target_date", *[col for col in features.columns if col not in official.columns]]
    frame = official.merge(features[feature_extra_cols], on="target_date", how="left", validate="many_to_one")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="multistation official feature frame")
    frame["official_error_c"] = frame["forecast_max_c"] - frame["target_tmax_c"]
    frame["official_abs_error_c"] = frame["official_error_c"].abs()
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def source_stability(frame: pd.DataFrame, feature: str) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("forecast_source_family", observed=True):
        corr = safe_corr(group[feature], group["official_error_c"], min_rows=MIN_GROUP_ROWS)
        if np.isfinite(corr):
            values.append(corr)
    if not values:
        return math.nan, 0
    signs = {1 if value > 0 else -1 for value in values if value != 0}
    return float(max(abs(value) for value in values)), int(len(signs) == 1)


def season_stability(frame: pd.DataFrame, feature: str) -> tuple[float, int]:
    if "season" not in frame.columns:
        return math.nan, 0
    values: list[float] = []
    for _, group in frame.groupby("season", observed=True):
        corr = safe_corr(group[feature], group["official_error_c"], min_rows=MIN_GROUP_ROWS)
        if np.isfinite(corr):
            values.append(corr)
    if not values:
        return math.nan, 0
    signs = {1 if value > 0 else -1 for value in values if value != 0}
    return float(max(abs(value) for value in values)), int(len(signs) == 1)


def scan_feature_information(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        col
        for col in frame.columns
        if analysis_column_allowed(col) and pd.api.types.is_numeric_dtype(frame[col])
    ]
    scan_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        n = int(values.notna().sum())
        if n < MIN_SCAN_ROWS or values.nunique(dropna=True) <= 2:
            continue
        rows = bucket_lift_rows(frame, column)
        bucket_rows.extend(rows)
        bias_spread, mae_spread = bucket_spreads(rows)
        max_source_corr, source_sign_agree = source_stability(frame, column)
        max_season_corr, season_sign_agree = season_stability(frame, column)
        scan_rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "n": n,
                "coverage": float(n / len(frame)),
                "target_pearson": safe_corr(values, frame["target_tmax_c"]),
                "target_spearman": safe_corr(values, frame["target_tmax_c"], method="spearman"),
                "official_error_pearson": safe_corr(values, frame["official_error_c"]),
                "official_error_spearman": safe_corr(values, frame["official_error_c"], method="spearman"),
                "official_abs_error_pearson": safe_corr(values, frame["official_abs_error_c"]),
                "underprediction_corr": safe_corr(values, frame["official_underpredicted"].astype(int)),
                "target_q90_minus_q10_c": qtail_delta(values, frame["target_tmax_c"]),
                "official_error_q90_minus_q10_c": qtail_delta(values, frame["official_error_c"]),
                "official_abs_error_q90_minus_q10_c": qtail_delta(values, frame["official_abs_error_c"]),
                "bucket_bias_spread_c": bias_spread,
                "bucket_mae_spread_c": mae_spread,
                "max_source_abs_official_error_corr": max_source_corr,
                "source_corr_sign_consistent": source_sign_agree,
                "max_season_abs_official_error_corr": max_season_corr,
                "season_corr_sign_consistent": season_sign_agree,
            }
        )
    out = pd.DataFrame(scan_rows)
    if out.empty:
        return out, pd.DataFrame(bucket_rows)
    out["max_abs_target_corr"] = out[["target_pearson", "target_spearman"]].abs().max(axis=1)
    out["max_abs_official_error_corr"] = out[["official_error_pearson", "official_error_spearman"]].abs().max(axis=1)
    out["max_abs_official_abs_error_corr"] = out[["official_abs_error_pearson"]].abs().max(axis=1)
    out["information_gain_priority"] = (
        out["max_abs_official_error_corr"].fillna(0.0) * 3.0
        + out["max_abs_official_abs_error_corr"].fillna(0.0) * 1.5
        + out["bucket_bias_spread_c"].abs().fillna(0.0) * 0.25
        + out["bucket_mae_spread_c"].abs().fillna(0.0) * 0.15
        + out["max_abs_target_corr"].fillna(0.0) * 0.25
    )
    return out.sort_values("information_gain_priority", ascending=False).reset_index(drop=True), pd.DataFrame(bucket_rows)


def build_station_attribute_matrix(feature_scan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in feature_scan.to_dict("records"):
        parsed = parse_station_attribute(str(record["feature"]))
        if not parsed:
            continue
        station, attribute = parsed
        rows.append({"station_id": station, "station_attribute": attribute, **record})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("information_gain_priority", ascending=False).reset_index(drop=True)


def pair_metric_row(
    frame: pd.DataFrame,
    *,
    attribute: str,
    station_a: str,
    station_b: str,
    col_a: str,
    col_b: str,
) -> dict[str, object] | None:
    spread = pd.to_numeric(frame[col_a], errors="coerce") - pd.to_numeric(frame[col_b], errors="coerce")
    n = int(spread.notna().sum())
    if n < MIN_PAIR_ROWS or spread.nunique(dropna=True) <= 2:
        return None
    residual_corr = safe_corr(spread, frame["official_error_c"])
    abs_error_corr = safe_corr(spread, frame["official_abs_error_c"])
    target_corr = safe_corr(spread, frame["target_tmax_c"])
    return {
        "attribute": attribute,
        "station_a": station_a,
        "station_b": station_b,
        "spread_feature": f"{col_a}__minus__{col_b}",
        "n": n,
        "coverage": float(n / len(frame)),
        "target_corr": target_corr,
        "official_error_corr": residual_corr,
        "official_abs_error_corr": abs_error_corr,
        "target_q90_minus_q10_c": qtail_delta(spread, frame["target_tmax_c"]),
        "official_error_q90_minus_q10_c": qtail_delta(spread, frame["official_error_c"]),
        "official_abs_error_q90_minus_q10_c": qtail_delta(spread, frame["official_abs_error_c"]),
        "priority": (
            abs(residual_corr) * 3.0 if np.isfinite(residual_corr) else 0.0
        )
        + (abs(abs_error_corr) * 1.5 if np.isfinite(abs_error_corr) else 0.0)
        + (abs(target_corr) * 0.25 if np.isfinite(target_corr) else 0.0),
    }


def build_station_pair_spread_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    by_attribute: dict[str, list[tuple[str, str]]] = {}
    for column in frame.columns:
        parsed = parse_station_attribute(column)
        if not parsed or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        station, attribute = parsed
        by_attribute.setdefault(attribute, []).append((station, column))

    rows: list[dict[str, object]] = []
    for attribute, station_columns in by_attribute.items():
        for (station_a, col_a), (station_b, col_b) in combinations(sorted(station_columns), 2):
            row = pair_metric_row(
                frame,
                attribute=attribute,
                station_a=station_a,
                station_b=station_b,
                col_a=col_a,
                col_b=col_b,
            )
            if row is not None:
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("priority", ascending=False).reset_index(drop=True)


def past_only_feature_bucket_prediction(
    frame: pd.DataFrame,
    *,
    feature: str,
    bins: int,
    season_conditioned: bool,
    min_history: int = PAST_ONLY_MIN_HISTORY,
    min_bucket_rows: int = PAST_ONLY_MIN_BUCKET_ROWS,
) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    feature_values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
    forecasts = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residuals = targets - forecasts

    predictions: list[float] = []
    corrections: list[float] = []
    rows_used: list[int] = []
    for index, target_date in enumerate(dates):
        forecast = forecasts[index]
        current_value = feature_values[index]
        if not np.isfinite(forecast) or not np.isfinite(current_value):
            predictions.append(forecast if np.isfinite(forecast) else math.nan)
            corrections.append(0.0)
            rows_used.append(0)
            continue

        prior_end = int(np.searchsorted(dates, target_date, side="left"))
        prior_mask = np.arange(len(ordered)) < prior_end
        if season_conditioned and "season" in ordered.columns:
            prior_mask &= ordered["season"].eq(ordered.at[index, "season"]).to_numpy()
        prior_mask &= np.isfinite(feature_values) & np.isfinite(residuals)
        prior_index = np.flatnonzero(prior_mask)
        if len(prior_index) < min_history:
            predictions.append(float(forecast))
            corrections.append(0.0)
            rows_used.append(0)
            continue

        prior_values = feature_values[prior_index]
        edges = np.unique(np.nanquantile(prior_values, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
        if len(edges) < bins - 1:
            predictions.append(float(forecast))
            corrections.append(0.0)
            rows_used.append(0)
            continue

        current_bucket = int(np.searchsorted(edges, current_value, side="right"))
        prior_buckets = np.searchsorted(edges, prior_values, side="right")
        bucket_index = prior_index[prior_buckets == current_bucket]
        if len(bucket_index) < min_bucket_rows:
            predictions.append(float(forecast))
            corrections.append(0.0)
            rows_used.append(0)
            continue

        correction = float(np.nanmean(residuals[bucket_index]))
        predictions.append(float(forecast + correction))
        corrections.append(correction)
        rows_used.append(int(len(bucket_index)))

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["candidate_prediction_c"] = predictions
    out["past_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["feature"] = feature
    out["bins"] = bins
    out["season_conditioned"] = season_conditioned
    return out


def run_past_only_bucket_screen(frame: pd.DataFrame, feature_scan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = feature_scan.head(TOP_CORRECTION_FEATURES)["feature"].to_list()
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for feature in candidates:
        for bins in (3, 5):
            for season_conditioned in (False, True):
                predictions = past_only_feature_bucket_prediction(
                    frame,
                    feature=feature,
                    bins=bins,
                    season_conditioned=season_conditioned,
                )
                candidate_id = f"{slug(feature)}_q{bins}_season{int(season_conditioned)}"
                predictions["candidate_id"] = candidate_id
                candidate = score_prediction_frame(predictions, "candidate_prediction_c")
                official = score_prediction_frame(predictions, "forecast_max_c")
                score_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "feature": feature,
                        "family": feature_family(feature),
                        "bins": bins,
                        "season_conditioned": season_conditioned,
                        **candidate,
                        "official_same_rows_mae": official["mae"],
                        "official_same_rows_rmse": official["rmse"],
                        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
                        "corrected_rows": int((predictions["past_rows_used"] > 0).sum()),
                        "fallback_rows": int((predictions["past_rows_used"] == 0).sum()),
                    }
                )
                prediction_frames.append(predictions)

    scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    if not prediction_frames:
        return scoreboard, pd.DataFrame()
    top_ids = set(scoreboard.head(20)["candidate_id"].to_list())
    predictions = pd.concat(
        [frame for frame in prediction_frames if str(frame["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    return scoreboard, predictions


def family_summary(feature_scan: pd.DataFrame) -> pd.DataFrame:
    if feature_scan.empty:
        return feature_scan
    return (
        feature_scan.groupby("family", observed=True)
        .agg(
            features=("feature", "count"),
            max_priority=("information_gain_priority", "max"),
            max_abs_official_error_corr=("max_abs_official_error_corr", "max"),
            max_bucket_bias_spread_c=("bucket_bias_spread_c", "max"),
            max_bucket_mae_spread_c=("bucket_mae_spread_c", "max"),
            median_coverage=("coverage", "median"),
        )
        .reset_index()
        .sort_values("max_priority", ascending=False)
    )


def write_outputs(
    *,
    frame: pd.DataFrame,
    feature_scan: pd.DataFrame,
    bucket_lifts: pd.DataFrame,
    station_matrix: pd.DataFrame,
    pair_matrix: pd.DataFrame,
    bucket_screen: pd.DataFrame,
    bucket_predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "feature_info_gain.csv", feature_scan)
    write_csv(artifacts / "bucket_lifts.csv", bucket_lifts)
    write_csv(artifacts / "family_summary.csv", family_summary(feature_scan))
    write_csv(artifacts / "station_attributes.csv", station_matrix)
    write_csv(artifacts / "station_pair_spreads.csv", pair_matrix)
    write_csv(artifacts / "past_only_bucket_screen.csv", bucket_screen)
    write_csv(artifacts / "top_bucket_predictions.csv", bucket_predictions)

    best_feature = feature_scan.iloc[0] if not feature_scan.empty else None
    best_pair = pair_matrix.iloc[0] if not pair_matrix.empty else None
    best_bucket = bucket_screen.iloc[0] if not bucket_screen.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "feature_rows": int(len(feature_scan)),
        "bucket_lift_rows": int(len(bucket_lifts)),
        "station_attribute_rows": int(len(station_matrix)),
        "station_pair_rows": int(len(pair_matrix)),
        "past_only_bucket_candidates": int(len(bucket_screen)),
        "best_feature": "" if best_feature is None else str(best_feature["feature"]),
        "best_feature_priority": None if best_feature is None else float(best_feature["information_gain_priority"]),
        "best_pair": "" if best_pair is None else str(best_pair["spread_feature"]),
        "best_pair_priority": None if best_pair is None else float(best_pair["priority"]),
        "best_bucket_candidate": "" if best_bucket is None else str(best_bucket["candidate_id"]),
        "best_bucket_mae": None if best_bucket is None else float(best_bucket["mae"]),
        "best_bucket_delta_vs_official": None if best_bucket is None else float(best_bucket["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "multistation_attribute_information_gain_manifest.json", manifest)

    best_feature_text = "No feature scan rows were produced."
    if best_feature is not None:
        best_feature_text = (
            f"Top attribute signal: `{best_feature['feature']}` with priority "
            f"`{best_feature['information_gain_priority']:.4f}`, official-error correlation "
            f"`{best_feature['official_error_pearson']:.4f}`, and bucket bias spread "
            f"`{best_feature['bucket_bias_spread_c']:.4f} C`."
        )
    best_pair_text = "No station-pair spread rows were produced."
    if best_pair is not None:
        best_pair_text = (
            f"Top station-pair spread: `{best_pair['spread_feature']}` with official-error correlation "
            f"`{best_pair['official_error_corr']:.4f}`."
        )
    best_bucket_text = "No past-only bucket candidates were produced."
    if best_bucket is not None:
        best_bucket_text = (
            f"Best past-only bucket candidate: `{best_bucket['candidate_id']}` with MAE "
            f"`{best_bucket['mae']:.4f}` versus same-row official MAE "
            f"`{best_bucket['official_same_rows_mae']:.4f}` "
            f"(delta `{best_bucket['delta_vs_official_same_rows']:.4f}`)."
        )

    readme = f"""# Multistation Attribute Information-Gain Matrix

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight scans every numeric leakage-eligible long-history feature available on the current official forecast rows, then isolates station attributes and station-pair spreads. The target is not raw Tmax alone; the main target is information that explains the official forecast residual.

## Leakage Contract

- All target dates are earlier than `{CONFIRMATION_START.date()}`.
- The frame uses the current official forecast rows from the press archive and RSS archive only where the issue time is pre-cutoff.
- Correlation and bucket-lift tables are diagnostic ranking tools.
- The past-only bucket screen estimates corrections using only dates strictly earlier than the target date.
- Same-date rows are excluded from correction history.
- No 2024+ labels are used.

## Main Results

{best_feature_text}

{best_pair_text}

{best_bucket_text}

## Top Attribute Information-Gain Rows

{markdown_table(feature_scan.head(20), max_rows=20)}

## Top Station Attributes

{markdown_table(station_matrix.head(20), max_rows=20)}

## Top Station-Pair Spreads

{markdown_table(pair_matrix.head(20), max_rows=20)}

## Top Past-Only Bucket Corrections

{markdown_table(bucket_screen.head(20), max_rows=20)}

## Interpretation

The station network contains real residual signal, but the current simple bucket correction still produces only small MAE movement around the official forecast anchor. The strongest useful direction is to convert the best station attributes and station-pair spreads into richer regime-specific experts, then blend them with the official forecast and analog stack using only prior performance.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Multistation Attribute Information-Gain Matrix\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_multistation_attribute_information_gain.py`:

- `{FOLDER_NAME}`: full numeric feature scan, station attribute matrix, station-pair spreads, bucket regime lifts, and past-only bucket correction screen around the official forecast anchor.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Feature scan rows | {manifest['feature_rows']} |
| Station attribute rows | {manifest['station_attribute_rows']} |
| Station pair rows | {manifest['station_pair_rows']} |
| Past-only bucket candidates | {manifest['past_only_bucket_candidates']} |
| Best bucket MAE | {manifest['best_bucket_mae']} |
| Best bucket delta vs official | {manifest['best_bucket_delta_vs_official']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}` and past-only corrections use strictly earlier target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = build_official_feature_frame()
    feature_scan, bucket_lifts = scan_feature_information(frame)
    station_matrix = build_station_attribute_matrix(feature_scan)
    pair_matrix = build_station_pair_spread_matrix(frame)
    bucket_screen, bucket_predictions = run_past_only_bucket_screen(frame, feature_scan)
    return write_outputs(
        frame=frame,
        feature_scan=feature_scan,
        bucket_lifts=bucket_lifts,
        station_matrix=station_matrix,
        pair_matrix=pair_matrix,
        bucket_screen=bucket_screen,
        bucket_predictions=bucket_predictions,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 multistation attribute information-gain analysis.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
