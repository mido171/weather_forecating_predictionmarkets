from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
RESEARCH_ROOT = REPO_ROOT / "experiments"

FEATURE_MATRIX_PATH = (
    DATASETS_ROOT
    / "12_hkg_t24_robust_experiment_outputs"
    / "hkg_t24_exp0050_0099_feature_matrix.parquet"
)
CHAMPION_PREDICTIONS_PATH = (
    REPO_ROOT / "experiments" / "EXP-0050" / "results" / "headline_oof_2020_2023_predictions.parquet"
)
RSS_FORECAST_PATH = (
    DATASETS_ROOT
    / "05_hko_historical_rss_forecasts"
    / "hko_historical_rss_temperature_forecasts.parquet"
)
PRESS_FORECAST_EXPORT_PATH = (
    DATASETS_ROOT
    / "05_hko_historical_rss_forecasts"
    / "hko_press_archive_temperature_forecast_days.parquet"
)
DEFAULT_PRESS_ARCHIVE_DB = Path(r"C:\hko_press_2000_2026\metadata\archive.sqlite3")

HEADLINE_START = pd.Timestamp("2020-01-01")
HEADLINE_END = pd.Timestamp("2023-12-31")
CONFIRMATION_START = pd.Timestamp("2024-01-01")
CUTOFF_HOUR_HKT = 15
TOP_N = 75


@dataclass(frozen=True)
class SourceCoverage:
    source_id: str
    status: str
    rows: int
    first_date: str
    last_date: str
    note: str


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).copy()
    columns = [str(col) for col in clipped.columns]
    rows = [
        [
            "" if pd.isna(value) else str(value)
            for value in row
        ]
        for row in clipped.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def require_no_confirmation_dates(dates: Iterable[object], *, context: str) -> None:
    series = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    bad = series[series >= CONFIRMATION_START]
    if not bad.empty:
        examples = ", ".join(str(value.date()) for value in bad.head(10))
        raise RuntimeError(f"{context} attempted to use confirmation dates >= 2024-01-01: {examples}")


def hkt_cutoff_utc_for_target_dates(target_dates: pd.Series, *, cutoff_hour_hkt: int = CUTOFF_HOUR_HKT) -> pd.Series:
    normalized = pd.to_datetime(target_dates, errors="coerce").dt.normalize()
    cutoff_local = normalized - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour_hkt)
    return cutoff_local.dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC")


def sanitize_temperature_forecasts(
    frame: pd.DataFrame,
    *,
    target_col: str,
    issue_col: str,
    max_col: str,
    min_col: str | None,
    source_name: str,
) -> pd.DataFrame:
    required = {target_col, issue_col, max_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")

    clean = frame.copy()
    clean["target_date"] = pd.to_datetime(clean[target_col], errors="coerce").dt.normalize()
    clean["issue_utc"] = pd.to_datetime(clean[issue_col], errors="coerce", utc=True)
    clean["forecast_max_c"] = pd.to_numeric(clean[max_col], errors="coerce")
    if min_col and min_col in clean.columns:
        clean["forecast_min_c"] = pd.to_numeric(clean[min_col], errors="coerce")
    else:
        clean["forecast_min_c"] = np.nan

    valid = clean["target_date"].notna() & clean["issue_utc"].notna()
    valid &= clean["forecast_max_c"].between(-5.0, 45.0, inclusive="both")
    min_present = clean["forecast_min_c"].notna()
    valid &= ~min_present | clean["forecast_min_c"].between(-5.0, 45.0, inclusive="both")
    valid &= ~min_present | (clean["forecast_min_c"] <= clean["forecast_max_c"])
    clean = clean.loc[valid].copy()
    clean["forecast_span_c"] = clean["forecast_max_c"] - clean["forecast_min_c"]
    clean["forecast_source_name"] = source_name
    return clean.sort_values(["target_date", "issue_utc"]).reset_index(drop=True)


def select_latest_pre_cutoff_forecast(
    frame: pd.DataFrame,
    *,
    target_col: str,
    issue_col: str,
    max_col: str,
    min_col: str | None,
    source_name: str,
    cutoff_hour_hkt: int = CUTOFF_HOUR_HKT,
) -> pd.DataFrame:
    clean = sanitize_temperature_forecasts(
        frame,
        target_col=target_col,
        issue_col=issue_col,
        max_col=max_col,
        min_col=min_col,
        source_name=source_name,
    )
    clean["cutoff_utc"] = hkt_cutoff_utc_for_target_dates(clean["target_date"], cutoff_hour_hkt=cutoff_hour_hkt)
    eligible = clean[clean["issue_utc"] <= clean["cutoff_utc"]].copy()
    if eligible.empty:
        return eligible
    eligible["lead_hours_at_cutoff"] = (
        (eligible["target_date"].dt.tz_localize("Asia/Hong_Kong") - eligible["issue_utc"]).dt.total_seconds()
        / 3600.0
    )
    return (
        eligible.sort_values(["target_date", "issue_utc"])
        .groupby("target_date", observed=True)
        .tail(1)
        .sort_values("target_date")
        .reset_index(drop=True)
    )


def score_prediction_frame(frame: pd.DataFrame, prediction_col: str, *, label_col: str = "target_tmax_c") -> dict[str, float | int | str]:
    scored = frame[[label_col, prediction_col, "target_date"]].dropna().copy()
    if scored.empty:
        return {"n": 0, "first_date": "", "last_date": "", "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    error = scored[prediction_col] - scored[label_col]
    return {
        "n": int(len(scored)),
        "first_date": str(scored["target_date"].min().date()),
        "last_date": str(scored["target_date"].max().date()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
        "median_abs_error": float(error.abs().median()),
    }


def apply_half_life_bias(
    frame: pd.DataFrame,
    *,
    half_life_days: float,
    forecast_col: str = "forecast_max_c",
    label_col: str = "target_tmax_c",
    min_history: int = 14,
) -> pd.Series:
    ordered = frame.sort_values("target_date").reset_index(drop=True)
    residual_history = ordered[label_col] - ordered[forecast_col]
    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    corrections: list[float] = []
    for index, target_date in enumerate(dates):
        if index == 0:
            corrections.append(0.0)
            continue
        prior_residuals = residual_history.iloc[:index]
        prior_dates = dates.iloc[:index]
        valid = prior_residuals.notna() & prior_dates.notna()
        if valid.sum() < min_history:
            corrections.append(0.0)
            continue
        age_days = (target_date - prior_dates[valid]).dt.days.astype(float)
        weights = np.power(0.5, age_days / float(half_life_days))
        corrections.append(float(np.average(prior_residuals[valid].astype(float), weights=weights)))
    return pd.Series(corrections, index=ordered.index)


def rolling_bias_correction_scores(frame: pd.DataFrame, *, half_lives: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    score_rows = [{"method": "raw_latest_pre_cutoff", **score_prediction_frame(ordered, "forecast_max_c")}]
    for half_life in half_lives:
        correction = apply_half_life_bias(ordered, half_life_days=half_life)
        col = f"forecast_max_bias_corrected_hl{half_life}_c"
        ordered[col] = ordered["forecast_max_c"] + correction
        score_rows.append({"method": f"past_only_half_life_{half_life}d", **score_prediction_frame(ordered, col)})
    scores = pd.DataFrame(score_rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)
    return ordered, scores


def load_features() -> pd.DataFrame:
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing feature matrix: {FEATURE_MATRIX_PATH}")
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"]).dt.normalize()
    features = features[features["target_date"] <= HEADLINE_END].copy()
    require_no_confirmation_dates(features["target_date"], context="feature matrix load")
    return features.sort_values("target_date").reset_index(drop=True)


def load_champion_predictions() -> pd.DataFrame:
    if not CHAMPION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing champion predictions: {CHAMPION_PREDICTIONS_PATH}")
    predictions = pd.read_parquet(CHAMPION_PREDICTIONS_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"]).dt.normalize()
    predictions = predictions[predictions["model_id"].eq("long_history_core_v1")].copy()
    predictions = predictions[(predictions["target_date"] >= HEADLINE_START) & (predictions["target_date"] <= HEADLINE_END)].copy()
    require_no_confirmation_dates(predictions["target_date"], context="champion prediction load")
    keep = ["target_date", "point_forecast", "distribution_sigma_c", "q05", "q95"]
    return predictions[keep].sort_values("target_date").reset_index(drop=True)


def build_analysis_frame(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = features.merge(predictions, on="target_date", how="left", validate="one_to_one")
    frame["champion_error_c"] = frame["point_forecast"] - frame["target_tmax_c"]
    frame["champion_abs_error_c"] = frame["champion_error_c"].abs()
    frame["season"] = frame["month"].map(season_name)
    frame["headline_oof"] = (frame["target_date"] >= HEADLINE_START) & (frame["target_date"] <= HEADLINE_END)
    return frame


def season_name(month: object) -> str:
    value = int(month)
    if value in (12, 1, 2):
        return "DJF"
    if value in (3, 4, 5):
        return "MAM"
    if value in (6, 7, 8):
        return "JJA"
    return "SON"


def allowed_signal_column(column: str) -> bool:
    forbidden_exact = {
        "target_tmax_c",
        "target_date",
        "content_sha256",
        "raw_retrieved_at_utc",
        "valid_at_utc",
        "valid_at_hkt",
        "point_forecast",
        "distribution_sigma_c",
        "q05",
        "q95",
        "champion_error_c",
        "champion_abs_error_c",
        "headline_oof",
    }
    if column in forbidden_exact:
        return False
    return not (column.endswith("_sha256") or column.endswith("_at_utc") or column.endswith("_at_hkt"))


def feature_family(column: str) -> str:
    checks = [
        ("station_network", ("isd_station_", "isd_graph_", "isd_temp_plane", "isd_north_south", "isd_east_west")),
        ("isd_regional_surface", ("isd_",)),
        ("upper_air", ("igra_", "ua_")),
        ("daily_climate", ("daily_",)),
        ("target_memory", ("target_", "clim_", "spell_", "trajectory_", "spectral_", "volatility_")),
        ("calendar_trend", ("doy_", "year_", "month", "day_of_year")),
        ("wind", ("wind", "wspd", "wdir", "shear", "veering")),
        ("humidity_moisture", ("dew", "humidity", "moist", "theta_e", "wet_bulb")),
        ("pressure", ("pressure", "hpa", "mslp", "sea_level")),
        ("rain_cloud_radiation", ("rain", "cloud", "solar", "sunshine", "radiation")),
        ("temperature_slope_change", ("change", "slope", "gradient", "rise", "tendency", "anomaly")),
    ]
    lowered = column.lower()
    for family, needles in checks:
        if any(needle in lowered for needle in needles):
            return family
    return "other"


def safe_corr(a: pd.Series, b: pd.Series, *, method: str, min_rows: int = 500) -> float:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows:
        return math.nan
    if pair.iloc[:, 0].nunique() <= 2 or pair.iloc[:, 1].nunique() <= 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def qtail_delta(feature: pd.Series, target: pd.Series, *, q_low: float = 0.1, q_high: float = 0.9) -> float:
    values = pd.to_numeric(feature, errors="coerce")
    labels = pd.to_numeric(target, errors="coerce")
    pair = pd.DataFrame({"feature": values, "target": labels}).dropna()
    if len(pair) < 500 or pair["feature"].nunique() <= 2:
        return math.nan
    low = pair["feature"].quantile(q_low)
    high = pair["feature"].quantile(q_high)
    low_mean = pair.loc[pair["feature"] <= low, "target"].mean()
    high_mean = pair.loc[pair["feature"] >= high, "target"].mean()
    return float(high_mean - low_mean)


def scan_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    headline = frame[frame["headline_oof"]].copy()
    require_no_confirmation_dates(headline["target_date"], context="feature scan headline frame")
    numeric_columns = [col for col in headline.columns if allowed_signal_column(col) and pd.api.types.is_numeric_dtype(headline[col])]
    for column in numeric_columns:
        values = pd.to_numeric(headline[column], errors="coerce")
        n = int(values.notna().sum())
        if n < 500 or values.nunique(dropna=True) <= 2:
            continue
        rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "n": n,
                "coverage": float(n / len(headline)),
                "target_pearson": safe_corr(values, headline["target_tmax_c"], method="pearson"),
                "target_spearman": safe_corr(values, headline["target_tmax_c"], method="spearman"),
                "champion_residual_pearson": safe_corr(values, headline["champion_error_c"], method="pearson"),
                "champion_residual_spearman": safe_corr(values, headline["champion_error_c"], method="spearman"),
                "champion_abs_error_pearson": safe_corr(values, headline["champion_abs_error_c"], method="pearson"),
                "target_top10_minus_bottom10_c": qtail_delta(values, headline["target_tmax_c"]),
                "residual_top10_minus_bottom10_c": qtail_delta(values, headline["champion_error_c"]),
                "abs_error_top10_minus_bottom10_c": qtail_delta(values, headline["champion_abs_error_c"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["max_abs_target_corr"] = out[["target_pearson", "target_spearman"]].abs().max(axis=1)
    out["max_abs_residual_corr"] = out[["champion_residual_pearson", "champion_residual_spearman"]].abs().max(axis=1)
    out["max_abs_abs_error_corr"] = out[["champion_abs_error_pearson"]].abs().max(axis=1)
    out["discovery_priority_score"] = (
        out["max_abs_residual_corr"].fillna(0.0) * 2.0
        + out["max_abs_abs_error_corr"].fillna(0.0)
        + out["max_abs_target_corr"].fillna(0.0) * 0.5
    )
    return out.sort_values("discovery_priority_score", ascending=False).reset_index(drop=True)


def station_scan(feature_scan: pd.DataFrame) -> pd.DataFrame:
    if feature_scan.empty:
        return feature_scan
    mask = feature_scan["feature"].str.contains("isd_station_|isd_graph_|isd_temp_plane|isd_north_south|isd_east_west", regex=True)
    return feature_scan.loc[mask].copy().sort_values("discovery_priority_score", ascending=False).reset_index(drop=True)


def trend_scan(feature_scan: pd.DataFrame) -> pd.DataFrame:
    if feature_scan.empty:
        return feature_scan
    mask = feature_scan["feature"].str.contains(
        "change|slope|gradient|rise|tendency|anomaly|spell|trajectory|volatility|entropy|roll",
        case=False,
        regex=True,
    )
    return feature_scan.loc[mask].copy().sort_values("discovery_priority_score", ascending=False).reset_index(drop=True)


def regime_error_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    headline = frame[frame["headline_oof"] & frame["champion_error_c"].notna()].copy()
    require_no_confirmation_dates(headline["target_date"], context="regime analysis headline frame")
    tables: dict[str, pd.DataFrame] = {}
    tables["season"] = aggregate_error_by(headline, ["season"])
    tables["month"] = aggregate_error_by(headline, ["month"])

    for source_col, label in [
        ("target_tmax_c", "actual_tmax_tertile"),
        ("target_volatility_forecastability_score_lag7", "forecastability_tertile"),
        ("igra_lower_mean_temp_c", "upper_air_temp_tertile"),
        ("isd_north_south_temp_gradient_c", "north_south_gradient_tertile"),
        ("isd_air_temp_range_c", "station_temp_range_tertile"),
        ("daily_hong_kong_observatory_daily_rainfall_lag7", "rainfall_lag7_tertile"),
        ("daily_hong_kong_observatory_mean_cloud_amount_lag7", "cloud_lag7_tertile"),
    ]:
        if source_col not in headline.columns:
            continue
        binned = headline.copy()
        binned[label] = quantile_bucket(binned[source_col], 3)
        tables[label] = aggregate_error_by(binned, [label])
    return tables


def quantile_bucket(values: pd.Series, buckets: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < buckets * 20 or numeric.nunique(dropna=True) < buckets:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    codes = pd.qcut(ranked, buckets, labels=[f"q{idx + 1}" for idx in range(buckets)])
    return codes.astype(str).where(numeric.notna(), "missing")


def aggregate_error_by(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(group_cols), dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = group["champion_error_c"]
        out = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        out.update(
            {
                "n": int(len(group)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(error.mean()),
                "p90_abs_error": float(error.abs().quantile(0.9)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mae"], ascending=False).reset_index(drop=True)


def rolling_inverse_mae_blend(
    frame: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    window_days: int,
    label_col: str = "target_tmax_c",
    min_history: int = 30,
) -> pd.Series:
    ordered = frame.sort_values("target_date").reset_index(drop=True)
    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    predictions: list[float] = []
    for index, _target_date in enumerate(dates):
        left = float(ordered.loc[index, left_col])
        right = float(ordered.loc[index, right_col])
        if index == 0:
            predictions.append(0.5 * left + 0.5 * right)
            continue
        prior = ordered.iloc[:index].copy()
        prior_dates = dates.iloc[:index]
        recent = prior_dates >= (dates.iloc[index] - pd.Timedelta(days=window_days))
        prior = prior.loc[recent]
        if len(prior) < min_history:
            predictions.append(0.5 * left + 0.5 * right)
            continue
        left_mae = (prior[left_col] - prior[label_col]).abs().mean()
        right_mae = (prior[right_col] - prior[label_col]).abs().mean()
        if not np.isfinite(left_mae) or not np.isfinite(right_mae):
            predictions.append(0.5 * left + 0.5 * right)
            continue
        if left_mae <= 0.0 and right_mae <= 0.0:
            left_weight = 0.5
        elif left_mae <= 0.0:
            left_weight = 1.0
        elif right_mae <= 0.0:
            left_weight = 0.0
        else:
            left_weight = (1.0 / left_mae) / ((1.0 / left_mae) + (1.0 / right_mae))
        predictions.append(float(left_weight * left + (1.0 - left_weight) * right))
    return pd.Series(predictions, index=ordered.index)


def official_champion_blend_scores(corrected: pd.DataFrame) -> pd.DataFrame:
    if corrected.empty or "point_forecast" not in corrected.columns:
        return pd.DataFrame()
    scored = corrected.sort_values("target_date").reset_index(drop=True).copy()
    rows: list[dict[str, object]] = [
        {"method": "champion_same_dates", **score_prediction_frame(scored, "point_forecast")},
        {"method": "official_raw_same_dates", **score_prediction_frame(scored, "forecast_max_c")},
    ]
    forecast_cols = ["forecast_max_c"]
    forecast_cols.extend(
        col
        for col in scored.columns
        if col.startswith("forecast_max_bias_corrected_hl") and col.endswith("_c")
    )
    for col in forecast_cols:
        if col == "forecast_max_c":
            method = "official_raw"
        else:
            method = "official_" + col.removeprefix("forecast_max_").removesuffix("_c")
        for official_weight in (0.25, 0.50, 0.75):
            blend_col = f"blend_{method}_official_weight_{official_weight:.2f}"
            scored[blend_col] = official_weight * scored[col] + (1.0 - official_weight) * scored["point_forecast"]
            rows.append({"method": blend_col, **score_prediction_frame(scored, blend_col)})
        rolling_col = f"rolling_inverse_mae_180d_{method}_vs_champion"
        scored[rolling_col] = rolling_inverse_mae_blend(
            scored,
            left_col=col,
            right_col="point_forecast",
            window_days=180,
        )
        rows.append({"method": rolling_col, **score_prediction_frame(scored, rolling_col)})
    return pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)


def load_rss_forecast_analysis(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not RSS_FORECAST_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rss = pd.read_parquet(RSS_FORECAST_PATH)
    latest = select_latest_pre_cutoff_forecast(
        rss,
        target_col="forecast_date",
        issue_col="available_at_hkt",
        max_col="forecast_max_temperature_c",
        min_col="forecast_min_temperature_c",
        source_name="hko_rss_temperature_forecast",
    )
    latest = latest[(latest["target_date"] >= HEADLINE_START) & (latest["target_date"] <= HEADLINE_END)].copy()
    require_no_confirmation_dates(latest["target_date"], context="RSS pre-cutoff latest forecast")
    joined = latest.merge(
        features[["target_date", "target_tmax_c", "season", "month", "point_forecast"]],
        on="target_date",
        how="inner",
    )
    corrected, scores = rolling_bias_correction_scores(joined, half_lives=[7, 14, 30, 60, 90, 180, 365])
    blend_scores = official_champion_blend_scores(corrected)
    by_season = aggregate_prediction_by(corrected, "forecast_max_c", ["season"])
    return corrected, scores, by_season, blend_scores


def aggregate_prediction_by(frame: pd.DataFrame, prediction_col: str, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(group_cols), dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        row.update(score_prediction_frame(group, prediction_col))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def inspect_press_archive(db_path: Path, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exported_forecast_days = pd.DataFrame()
    if PRESS_FORECAST_EXPORT_PATH.exists():
        exported_forecast_days = pd.read_parquet(PRESS_FORECAST_EXPORT_PATH)

    if not db_path.exists():
        coverage = pd.DataFrame(
            [
                {
                    "source_id": "hko_press_archive_sqlite",
                    "status": "missing",
                    "rows": 0,
                    "first_date": "",
                    "last_date": "",
                    "note": f"Database not found at {db_path}",
                }
            ]
        )
        forecast_days = exported_forecast_days
    else:
        with sqlite3.connect(db_path) as connection:
            tables = pd.read_sql("select name from sqlite_master where type='table' order by name", connection)["name"].to_list()
            coverage_rows: list[dict[str, object]] = []
            for table in tables:
                columns = pd.read_sql(f"pragma table_info({table})", connection)["name"].to_list()
                row_count = int(pd.read_sql(f"select count(*) as n from {table}", connection).iloc[0]["n"])
                date_col = next((col for col in ("target_date", "issue_at_hkt", "index_date", "attempted_at_utc") if col in columns), None)
                first_date = ""
                last_date = ""
                if date_col:
                    dates = pd.read_sql(f"select min({date_col}) as first_date, max({date_col}) as last_date from {table}", connection)
                    first_date = str(dates.iloc[0]["first_date"] or "")
                    last_date = str(dates.iloc[0]["last_date"] or "")
                coverage_rows.append(
                    {
                        "source_id": f"hko_press_archive_sqlite.{table}",
                        "status": "indexed" if table == "candidates" else "parsed_partial" if table in {"bulletins", "forecast_days"} else "empty_or_metadata",
                        "rows": row_count,
                        "first_date": first_date,
                        "last_date": last_date,
                        "note": "Candidates cover index discovery; forecast_days is the parsed table usable for scoring.",
                    }
                )
            if exported_forecast_days.empty and "forecast_days" in tables:
                forecast_days = pd.read_sql("select * from forecast_days", connection)
            else:
                forecast_days = exported_forecast_days
        coverage = pd.DataFrame(coverage_rows)

    if not exported_forecast_days.empty:
        exported_dates = pd.to_datetime(exported_forecast_days["target_date"], errors="coerce")
        export_row = pd.DataFrame(
            [
                {
                    "source_id": "hko_press_archive_offline_export.temperature_forecast_days",
                    "status": "parsed_partial_scoreable_export",
                    "rows": int(len(exported_forecast_days)),
                    "first_date": str(exported_dates.min().date()),
                    "last_date": str(exported_dates.max().date()),
                    "note": f"Repo-local export used for scoring: {PRESS_FORECAST_EXPORT_PATH}",
                }
            ]
        )
        coverage = pd.concat([coverage, export_row], ignore_index=True)

    if forecast_days.empty:
        return coverage, pd.DataFrame(), pd.DataFrame()

    selected = select_latest_pre_cutoff_forecast(
        forecast_days,
        target_col="target_date",
        issue_col="issue_at_hkt",
        max_col="forecast_max_c",
        min_col="forecast_min_c",
        source_name="hko_press_archive_parsed_forecast_days",
    )
    selected = selected[selected["target_date"] <= HEADLINE_END].copy()
    require_no_confirmation_dates(selected["target_date"], context="press archive parsed selected forecast")
    joined = selected.merge(features[["target_date", "target_tmax_c", "season", "month"]], on="target_date", how="inner")
    if joined.empty:
        scores = pd.DataFrame()
    else:
        _corrected, scores = rolling_bias_correction_scores(joined, half_lives=[30, 90, 180, 365])
    return coverage, joined, scores


def summarize_sources(features: pd.DataFrame, rss_selected: pd.DataFrame, press_coverage: pd.DataFrame, press_joined: pd.DataFrame) -> pd.DataFrame:
    rows = [
        SourceCoverage(
            source_id="long_history_feature_matrix",
            status="downloaded_normalized",
            rows=int(len(features)),
            first_date=str(features["target_date"].min().date()),
            last_date=str(features["target_date"].max().date()),
            note=f"{features.shape[1]} columns; confirmation guard excludes 2024+ labels.",
        ).__dict__,
        SourceCoverage(
            source_id="headline_champion_oof",
            status="downloaded_normalized",
            rows=1460,
            first_date="2020-01-01",
            last_date="2023-12-31",
            note="EXP-0050 long_history_core_v1; used only for residual analysis.",
        ).__dict__,
    ]
    if rss_selected.empty:
        rows.append(
            SourceCoverage(
                source_id="hko_rss_temperature_forecast",
                status="missing_or_unusable",
                rows=0,
                first_date="",
                last_date="",
                note="RSS forecast parquet missing or no pre-cutoff rows.",
            ).__dict__
        )
    else:
        rows.append(
            SourceCoverage(
                source_id="hko_rss_temperature_forecast",
                status="downloaded_normalized",
                rows=int(len(rss_selected)),
                first_date=str(rss_selected["target_date"].min().date()),
                last_date=str(rss_selected["target_date"].max().date()),
                note="Latest pre-cutoff T-1 15:00 HKT selection; scored only through 2023-12-31.",
            ).__dict__
        )
    if press_coverage.empty:
        rows.append(
            SourceCoverage(
                source_id="hko_press_archive_sqlite",
                status="missing",
                rows=0,
                first_date="",
                last_date="",
                note="No local SQLite archive found.",
            ).__dict__
        )
    else:
        for row in press_coverage.to_dict("records"):
            rows.append(row)
        if not press_joined.empty:
            rows.append(
                SourceCoverage(
                    source_id="hko_press_archive_selected_scoreable",
                    status="parsed_partial_scoreable",
                    rows=int(len(press_joined)),
                    first_date=str(press_joined["target_date"].min().date()),
                    last_date=str(press_joined["target_date"].max().date()),
                    note="Parsed selected forecasts are currently not a continuous 2000-2026 usable archive.",
                ).__dict__
            )
    return pd.DataFrame(rows)


def write_research_state(
    source_summary: pd.DataFrame,
    feature_scan: pd.DataFrame,
    rss_scores: pd.DataFrame,
    press_scores: pd.DataFrame,
) -> None:
    folder = RESEARCH_ROOT / "0000_research_state_and_data_contract"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "source_summary.csv", source_summary)
    if not feature_scan.empty:
        family_summary = (
            feature_scan.groupby("family", observed=True)
            .agg(
                feature_count=("feature", "count"),
                best_target_corr=("max_abs_target_corr", "max"),
                best_residual_corr=("max_abs_residual_corr", "max"),
                best_abs_error_corr=("max_abs_abs_error_corr", "max"),
            )
            .sort_values("best_residual_corr", ascending=False)
            .reset_index()
        )
    else:
        family_summary = pd.DataFrame()
    write_csv(artifacts / "feature_family_summary.csv", family_summary)

    overview = f"""# Research State And Data Contract

Generated: `{now_utc()}`

This folder is the control sheet for the beastmode signal-discovery phase. It records what was actually usable in this run, which date ranges were scored, and which leakage guard was enforced.

## Leakage Contract

- Target horizon under analysis: HKG Tmax T-24 style cutoff.
- Operational cutoff used for forecast archive selection: `T-1 {CUTOFF_HOUR_HKT}:00 HKT`.
- Confirmation-period labels are blocked by code: any target date on or after `2024-01-01` raises an error.
- The 2024-2026 rows are inventory/prospective only until the user explicitly releases a final confirmation pass.

## Source Summary

{markdown_table(source_summary, max_rows=30)}

## Feature Family Summary

{markdown_table(family_summary, max_rows=30)}

## Forecast Scoreboard Snapshot

RSS latest-pre-cutoff temperature forecast:

{markdown_table(rss_scores, max_rows=12)}

Parsed press archive selected forecast:

{markdown_table(press_scores, max_rows=12)}

## Important Interpretation

The long-history feature matrix is usable back to the 1949 upper-air era and earlier target/climate history, but the current champion OOF residuals exist only for 2020-2023. Feature-vs-target scans are useful over the headline OOF period; later modelling must still run long rolling-origin validation before promotion.

The local press archive database indexes weather forecast candidates from 2000-2026, but this run treats only parsed `forecast_days` as scoreable forecast data. If `forecast_days` is partial, the report says so explicitly instead of pretending the indexed archive is already fully parsed.
"""
    write_text(folder / "README.md", overview)


def write_forecast_insight(
    rss_selected: pd.DataFrame,
    rss_scores: pd.DataFrame,
    rss_by_season: pd.DataFrame,
    rss_blend_scores: pd.DataFrame,
    press_joined: pd.DataFrame,
    press_scores: pd.DataFrame,
) -> None:
    folder = RESEARCH_ROOT / "0001_official_forecast_pre_cutoff_bias"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "rss_latest_pre_cutoff_selected.csv", rss_selected)
    write_csv(artifacts / "rss_bias_correction_scores.csv", rss_scores)
    write_csv(artifacts / "rss_champion_official_blend_scores.csv", rss_blend_scores)
    write_csv(artifacts / "rss_raw_by_season.csv", rss_by_season)
    write_csv(artifacts / "press_parsed_selected_scoreable.csv", press_joined)
    write_csv(artifacts / "press_parsed_bias_correction_scores.csv", press_scores)

    best = rss_scores.head(1).to_dict("records")
    best_text = "No RSS scoreable rows were available." if not best else f"Best RSS method in this run: `{best[0]['method']}` with MAE `{best[0]['mae']:.4f}` and RMSE `{best[0]['rmse']:.4f}` over `{best[0]['n']}` rows."
    press_text = "No parsed press forecast rows were scoreable." if press_scores.empty else f"Parsed press archive best scoreable method: `{press_scores.iloc[0]['method']}` with MAE `{press_scores.iloc[0]['mae']:.4f}` over `{int(press_scores.iloc[0]['n'])}` rows."
    blend_text = "No champion/official overlap blend was scoreable." if rss_blend_scores.empty else f"Best same-date champion/official blend: `{rss_blend_scores.iloc[0]['method']}` with MAE `{rss_blend_scores.iloc[0]['mae']:.4f}` and RMSE `{rss_blend_scores.iloc[0]['rmse']:.4f}`."

    text = f"""# Official Forecast Pre-Cutoff Bias

Generated: `{now_utc()}`

## What Was Tested

This insight tests the official HKO temperature forecast archive as a direct Tmax signal. The selector uses the latest forecast available no later than `T-1 {CUTOFF_HOUR_HKT}:00 HKT`. Rows with null/invalid maximum temperature, impossible values, or min greater than max are removed. Null minimum temperature does not automatically invalidate a row because the immediate target is highest temperature, but null maximum temperature does.

## Main Finding

{best_text}

{press_text}

{blend_text}

## RSS Bias Correction Scoreboard

{markdown_table(rss_scores, max_rows=12)}

## RSS Official Forecast Versus Champion Same-Date Blends

{markdown_table(rss_blend_scores, max_rows=16)}

## RSS Raw Forecast By Season

{markdown_table(rss_by_season, max_rows=8)}

## Why This Matters

If the competitor truly reached roughly 0.45 MAE using the same broad datasets, the official forecast archive is the most plausible anchor signal. The long-history weather-only champion is currently around 1.24 MAE. A corrected official forecast can compress much of that gap if enough exact vintages are available and selected leakage-free.

## Current Limitation

The RSS exact-vintage archive currently contributes only 2020-06 through 2023-12 for sealed research scoring, which is useful but shorter than the requested four full years. The local press archive has a 2000-2026 candidate index, but the parsed scoreable forecast table is currently partial. Completing that parser/backfill is a top-priority data task before any official-forecast-driven model can be promoted.
"""
    write_text(folder / "README.md", text)
    write_text(folder / "METHOD.md", "Latest pre-cutoff forecast selection, sanity filtering, and past-only half-life bias correction. Corrections for each date use only earlier target dates.\n")


def write_feature_correlation_insight(feature_scan: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0002_feature_correlation_atlas"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "all_feature_signal_scan.csv", feature_scan)
    write_csv(artifacts / "top_target_correlations.csv", feature_scan.sort_values("max_abs_target_corr", ascending=False).head(TOP_N))
    write_csv(artifacts / "top_champion_residual_correlations.csv", feature_scan.sort_values("max_abs_residual_corr", ascending=False).head(TOP_N))
    write_csv(artifacts / "top_champion_abs_error_correlations.csv", feature_scan.sort_values("max_abs_abs_error_corr", ascending=False).head(TOP_N))

    family = (
        feature_scan.groupby("family", observed=True)
        .agg(
            feature_count=("feature", "count"),
            best_target_corr=("max_abs_target_corr", "max"),
            best_residual_corr=("max_abs_residual_corr", "max"),
            best_abs_error_corr=("max_abs_abs_error_corr", "max"),
            median_priority=("discovery_priority_score", "median"),
        )
        .sort_values("best_residual_corr", ascending=False)
        .reset_index()
        if not feature_scan.empty
        else pd.DataFrame()
    )
    write_csv(artifacts / "family_signal_summary.csv", family)
    text = f"""# Feature Correlation Atlas

Generated: `{now_utc()}`

## What Was Tested

Every numeric leakage-eligible feature in the long-history feature matrix was scanned against:

- actual HKO Tmax;
- signed champion residual (`forecast - actual`);
- champion absolute error.

This is not a promotion model. It is a signal atlas. Features that correlate with residuals or absolute error are especially valuable because they point to where the current best model is systematically wrong.

## Strongest Families By Residual Signal

{markdown_table(family, max_rows=20)}

## Top Residual Features

{markdown_table(feature_scan.sort_values("max_abs_residual_corr", ascending=False).head(20), max_rows=20)}

## Interpretation

Target correlation tells us which variables track next-day Tmax level. Residual correlation is more strategically important because it tells us which variables could correct the current champion. Absolute-error correlation identifies regimes where the champion becomes less trustworthy and where dynamic blending or uncertainty scaling is needed.
"""
    write_text(folder / "README.md", text)


def write_station_insight(stations: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0003_station_network_information_gain"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "station_network_signal_scan.csv", stations)
    text = f"""# Station Network Information Gain

Generated: `{now_utc()}`

## What Was Tested

This insight isolates station-panel, station-graph, and spatial-gradient features from the larger correlation atlas. The goal is to identify surrounding stations or spatial field shapes that add information beyond the target station's historical climatology.

## Top Station-Network Residual Signals

{markdown_table(stations.sort_values("max_abs_residual_corr", ascending=False).head(25), max_rows=25)}

## Top Station-Network Target Signals

{markdown_table(stations.sort_values("max_abs_target_corr", ascending=False).head(25), max_rows=25)}

## What To Do Next

The highest-priority station-network features should be converted into explicit candidate experts: coastal-vs-inland gradient expert, airport-vs-HKO proxy expert, graph-mode regime expert, and station disagreement uncertainty expert. Each must be validated with rolling-origin folds and no 2024+ tuning.
"""
    write_text(folder / "README.md", text)


def write_trend_regime_insight(trends: pd.DataFrame, regime_tables: dict[str, pd.DataFrame]) -> None:
    folder = RESEARCH_ROOT / "0004_slope_trend_regime_signals"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "slope_trend_signal_scan.csv", trends)
    for name, table in regime_tables.items():
        write_csv(artifacts / f"champion_error_by_{name}.csv", table)
    sections = []
    for name, table in regime_tables.items():
        sections.append(f"### {name}\n\n{markdown_table(table, max_rows=12)}")
    text = f"""# Slope, Trend, And Regime Signals

Generated: `{now_utc()}`

## What Was Tested

This insight focuses on dynamic features: slopes, changes, gradients, tendency terms, rolling means, anomalies, spell lengths, volatility, entropy, and broad weather-regime bins. These are exactly the signals that can explain whether tomorrow's Tmax follows persistence, reverts, jumps, or is suppressed.

## Top Dynamic Residual Signals

{markdown_table(trends.sort_values("max_abs_residual_corr", ascending=False).head(25), max_rows=25)}

## Champion Error By Regime

{chr(10).join(sections)}

## Interpretation

The most useful dynamic features are not necessarily the highest direct Tmax correlations. A slope or regime feature can be valuable if it tells us when the champion's residual changes sign, when uncertainty widens, or when the model should hand weight to an official-forecast, station-gradient, or upper-air expert.
"""
    write_text(folder / "README.md", text)


def write_error_autopsy(frame: pd.DataFrame, feature_scan: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0005_champion_error_autopsy"
    artifacts = folder / "artifacts"
    headline = frame[frame["headline_oof"] & frame["champion_error_c"].notna()].copy()
    top_features = feature_scan.sort_values("discovery_priority_score", ascending=False)["feature"].head(20).to_list()
    keep = ["target_date", "target_tmax_c", "point_forecast", "champion_error_c", "champion_abs_error_c", "season", "month"]
    keep.extend([col for col in top_features if col in headline.columns])
    worst = headline.sort_values("champion_abs_error_c", ascending=False)[keep].head(100)
    write_csv(artifacts / "top_100_champion_error_cases_with_signal_features.csv", worst)
    write_csv(artifacts / "top_20_candidate_error_explainers.csv", feature_scan.head(20))
    text = f"""# Champion Error Autopsy

Generated: `{now_utc()}`

## What Was Tested

This folder collects the worst EXP-0050 champion misses and attaches the strongest candidate explanatory features from the signal scan. The purpose is to turn bad rows into specific hypotheses rather than just saying the model missed.

## Worst Error Rows

{markdown_table(worst.head(20), max_rows=20)}

## Candidate Error Explainers

{markdown_table(feature_scan.head(20), max_rows=20)}

## Next Hypotheses

- Build a residual-correction expert trained only on past folds and the top residual-correlated features.
- Build an uncertainty-scaling expert using top absolute-error correlated features.
- Split errors by season and station-gradient regime before adding complexity.
- Use official forecast disagreement as soon as full 2000-2026 parsed forecasts are available.
"""
    write_text(folder / "README.md", text)


def write_master_index(
    source_summary: pd.DataFrame,
    feature_scan: pd.DataFrame,
    rss_scores: pd.DataFrame,
    rss_blend_scores: pd.DataFrame,
) -> None:
    best_forecast = "No RSS forecast score available."
    if not rss_scores.empty:
        row = rss_scores.iloc[0]
        best_forecast = f"Best RSS forecast correction: `{row['method']}` MAE `{row['mae']:.4f}`, RMSE `{row['rmse']:.4f}`, n `{int(row['n'])}`."
    best_blend = "No RSS/champion blend score available."
    if not rss_blend_scores.empty:
        row = rss_blend_scores.iloc[0]
        best_blend = f"Best RSS/champion same-date blend: `{row['method']}` MAE `{row['mae']:.4f}`, RMSE `{row['rmse']:.4f}`, n `{int(row['n'])}`."
    top_residual = feature_scan.sort_values("max_abs_residual_corr", ascending=False).head(10) if not feature_scan.empty else pd.DataFrame()
    text = f"""# HKG Tmax Beastmode Data Analysis Index

Generated: `{now_utc()}`

This directory is the durable record for the deep signal-discovery phase. Each numbered folder is one insight family with its own methodology and artifacts.

## Current Best Research Facts

- EXP-0050 `long_history_core_v1` remains the current sealed 2020-2023 champion from prior experiments: MAE `1.2366`, RMSE `1.5714`.
- {best_forecast}
- {best_blend}
- Confirmation-period labels beginning `2024-01-01` are not used by this pipeline.
- The local press forecast archive is indexed for 2000-2026, but scoreable parsed forecast rows must be verified separately before promotion.

## Insight Folders

- `0000_research_state_and_data_contract`: source coverage, leakage contract, and feature-family summary.
- `0001_official_forecast_pre_cutoff_bias`: official RSS/press forecast selection and past-only bias correction.
- `0002_feature_correlation_atlas`: all-feature correlation and residual signal atlas.
- `0003_station_network_information_gain`: multi-station and spatial-gradient signal scan.
- `0004_slope_trend_regime_signals`: slopes, tendencies, anomalies, and champion error regimes.
- `0005_champion_error_autopsy`: worst champion misses with candidate explanatory features.

## Source Snapshot

{markdown_table(source_summary, max_rows=30)}

## Top Residual Signals

{markdown_table(top_residual, max_rows=10)}
"""
    write_text(RESEARCH_ROOT / "README.md", text)


def run(press_archive_db: Path) -> dict[str, object]:
    features = load_features()
    champion = load_champion_predictions()
    frame = build_analysis_frame(features, champion)
    feature_scan = scan_features(frame)
    stations = station_scan(feature_scan)
    trends = trend_scan(feature_scan)
    regime_tables = regime_error_tables(frame)
    rss_selected, rss_scores, rss_by_season, rss_blend_scores = load_rss_forecast_analysis(frame)
    press_coverage, press_joined, press_scores = inspect_press_archive(press_archive_db, frame)
    source_summary = summarize_sources(features, rss_selected, press_coverage, press_joined)

    write_research_state(source_summary, feature_scan, rss_scores, press_scores)
    write_forecast_insight(rss_selected, rss_scores, rss_by_season, rss_blend_scores, press_joined, press_scores)
    write_feature_correlation_insight(feature_scan)
    write_station_insight(stations)
    write_trend_regime_insight(trends, regime_tables)
    write_error_autopsy(frame, feature_scan)
    write_master_index(source_summary, feature_scan, rss_scores, rss_blend_scores)

    manifest = {
        "generated_at_utc": now_utc(),
        "feature_matrix_path": str(FEATURE_MATRIX_PATH),
        "champion_predictions_path": str(CHAMPION_PREDICTIONS_PATH),
        "rss_forecast_path": str(RSS_FORECAST_PATH),
        "press_forecast_export_path": str(PRESS_FORECAST_EXPORT_PATH),
        "press_archive_db": str(press_archive_db),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "folders": [
            "0000_research_state_and_data_contract",
            "0001_official_forecast_pre_cutoff_bias",
            "0002_feature_correlation_atlas",
            "0003_station_network_information_gain",
            "0004_slope_trend_regime_signals",
            "0005_champion_error_autopsy",
        ],
        "feature_rows": int(len(frame)),
        "feature_columns": int(frame.shape[1]),
        "feature_scan_rows": int(len(feature_scan)),
        "station_scan_rows": int(len(stations)),
        "trend_scan_rows": int(len(trends)),
        "rss_selected_rows": int(len(rss_selected)),
    }
    write_json(RESEARCH_ROOT / "signal_discovery_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe HKG T24 beastmode signal discovery.")
    parser.add_argument(
        "--press-archive-db",
        type=Path,
        default=DEFAULT_PRESS_ARCHIVE_DB,
        help="Local SQLite database for the HKO press forecast archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(args.press_archive_db)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
