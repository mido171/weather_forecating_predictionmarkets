from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
EXPERIMENT_ID = "0184"
SLUG = "hf_teacher_proxy_causal_memory_router"
TITLE = "HF Teacher Proxy Causal Memory Router"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"

DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
OFFICIAL_PATH = (
    DATASETS_ROOT
    / "05_hko_historical_rss_forecasts"
    / "hko_official_t15_scored_pre2024.parquet"
)
FEATURE_MATRIX_PATH = (
    DATASETS_ROOT
    / "12_hkg_t24_robust_experiment_outputs"
    / "hkg_t24_exp0050_0099_feature_matrix.parquet"
)

CONFIRMATION_START = pd.Timestamp("2024-01-01")
MODEL_FOLDS = ((2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2023))
CONFIDENCE_GRID = [
    {"alpha": 3.0, "cap_c": 0.15},
    {"alpha": 10.0, "cap_c": 0.15},
    {"alpha": 30.0, "cap_c": 0.15},
    {"alpha": 100.0, "cap_c": 0.15},
    {"alpha": 3.0, "cap_c": 0.25},
    {"alpha": 10.0, "cap_c": 0.25},
    {"alpha": 30.0, "cap_c": 0.25},
    {"alpha": 100.0, "cap_c": 0.25},
    {"alpha": 3.0, "cap_c": 0.35},
    {"alpha": 10.0, "cap_c": 0.35},
    {"alpha": 30.0, "cap_c": 0.35},
    {"alpha": 100.0, "cap_c": 0.35},
]
PRIMARY_CANDIDATE_ID = "0184_hf_teacher_proxy_ridge_nested_cap"

TARGET_MEMORY_COLUMNS = [
    "target_lag7_tmax_c",
    "target_lag8_tmax_c",
    "target_lag14_tmax_c",
    "target_lag21_tmax_c",
    "target_lag30_tmax_c",
    "target_lag60_tmax_c",
    "target_lag90_tmax_c",
    "target_lag365_tmax_c",
    "target_roll7_mean_lag7_c",
    "target_roll14_mean_lag7_c",
    "target_roll30_mean_lag7_c",
    "target_roll60_mean_lag7_c",
    "target_roll90_mean_lag7_c",
    "target_roll365_mean_lag7_c",
    "target_roll7_std_lag7_c",
    "target_roll14_std_lag7_c",
    "target_roll30_std_lag7_c",
    "target_roll60_std_lag7_c",
    "target_lag7_minus_lag14_c",
    "target_lag7_minus_roll30_c",
    "target_lag7_minus_roll365_c",
    "target_spell_hot_lag7",
    "target_spell_cold_lag7",
    "target_reversal_pressure_lag7",
    "target_entropy_30_lag7",
    "target_abs_change_7_14_c",
    "trajectory_range_lag7_60_c",
    "target_volatility_forecastability_score_lag7",
]

SAFE_FEATURE_MATRIX_COLUMNS = ["target_date", *TARGET_MEMORY_COLUMNS]
TEXT_PATTERNS = {
    "text_sunny": r"\bsunny\b|\bfine\b|\bbright\b",
    "text_cloudy": r"\bcloudy\b|\bcloud\b|\bovercast\b",
    "text_rain": r"\brain\b|\bshowers?\b|\brain patches\b",
    "text_thunder": r"\bthunder\b|\bsqually\b",
    "text_haze": r"\bhaze\b|\bhazy\b",
    "text_mist": r"\bmist\b|\bfog\b",
    "text_dry": r"\bdry\b",
    "text_humid": r"\bhumid\b",
    "text_cool": r"\bcool\b|\bcold\b",
    "text_hot": r"\bhot\b|\bwarm\b",
    "wind_easterly": r"\beast\b|\beasterly\b|\bnortheast\b|\bsoutheast\b",
    "wind_northerly": r"\bnorth\b|\bnortherly\b|\bnortheast\b|\bnorthwest\b",
    "wind_light": r"\blight winds?\b|\bforce 2\b",
    "wind_strong": r"\bforce [5-9]\b|\bstrong\b|\bwindy\b",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def assert_pre2024(frame: pd.DataFrame, context: str) -> None:
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    bad = dates[dates >= CONFIRMATION_START]
    if not bad.empty:
        examples = ", ".join(date_text(value) for value in bad.head(5))
        raise RuntimeError(f"{context} attempted to use sealed 2024+ dates: {examples}")


def metric_row(frame: pd.DataFrame, pred_col: str, *, label: str, group_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    clean = frame[["target_tmax_c", pred_col]].dropna()
    target = clean["target_tmax_c"].astype(float)
    pred = clean[pred_col].astype(float)
    err = pred - target
    ae = err.abs()
    row: dict[str, Any] = {
        "candidate_id": label,
        "n": int(len(clean)),
        "mae_c": float(ae.mean()) if len(clean) else math.nan,
        "rmse_c": float(np.sqrt(np.mean(np.square(err)))) if len(clean) else math.nan,
        "bias_c": float(err.mean()) if len(clean) else math.nan,
        "median_abs_error_c": float(ae.median()) if len(clean) else math.nan,
        "p90_abs_error_c": float(ae.quantile(0.90)) if len(clean) else math.nan,
        "p95_abs_error_c": float(ae.quantile(0.95)) if len(clean) else math.nan,
        "max_abs_error_c": float(ae.max()) if len(clean) else math.nan,
        "gt2c_count": int(ae.gt(2.0).sum()) if len(clean) else 0,
        "gt2c_rate": float(ae.gt(2.0).mean()) if len(clean) else math.nan,
        "gt3c_count": int(ae.gt(3.0).sum()) if len(clean) else 0,
        "gt3c_rate": float(ae.gt(3.0).mean()) if len(clean) else math.nan,
        "hot_underforecast_count": int(((target >= 30.0) & (err < 0.0)).sum()) if len(clean) else 0,
        "cold_overforecast_count": int(((target <= 20.0) & (err > 0.0)).sum()) if len(clean) else 0,
    }
    if group_fields:
        row.update(group_fields)
    return row


def compare_metrics(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = metric_row(frame, "official_prediction_c", label="official_raw")
    candidate = metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "official_rmse_c": official["rmse_c"],
        "candidate_rmse_c": candidate["rmse_c"],
        "candidate_bias_c": candidate["bias_c"],
        "official_p95_abs_error_c": official["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
        "official_gt2c_rate": official["gt2c_rate"],
        "candidate_gt2c_rate": candidate["gt2c_rate"],
        "official_gt3c_rate": official["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "candidate_hot_underforecast_count": candidate["hot_underforecast_count"],
        "candidate_cold_overforecast_count": candidate["cold_overforecast_count"],
    }


def load_inputs() -> pd.DataFrame:
    official = pd.read_parquet(OFFICIAL_PATH)
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[official["target_date"].notna() & (official["target_date"] < CONFIRMATION_START)].copy()
    official = official.drop_duplicates("target_date", keep="last").sort_values("target_date").reset_index(drop=True)
    for column in ("target_tmax_c", "forecast_max_c", "forecast_min_c", "rh_min_pct", "rh_max_pct"):
        if column in official.columns:
            official[column] = pd.to_numeric(official[column], errors="coerce")
    official["official_prediction_c"] = official["forecast_max_c"]
    official["official_residual_c"] = official["target_tmax_c"] - official["official_prediction_c"]
    official["forecast_range_c"] = official["forecast_max_c"] - official["forecast_min_c"]
    official["forecast_midpoint_c"] = (official["forecast_max_c"] + official["forecast_min_c"]) / 2.0
    official["rh_range_pct"] = official["rh_max_pct"] - official["rh_min_pct"]
    official["month"] = pd.to_numeric(official["month"], errors="coerce").fillna(official["target_date"].dt.month).astype(int)
    official["season"] = official["month"].map(month_to_season)
    official["day_of_year"] = official["target_date"].dt.dayofyear.astype(int)
    official["doy_sin"] = np.sin(2.0 * np.pi * official["day_of_year"] / 366.0)
    official["doy_cos"] = np.cos(2.0 * np.pi * official["day_of_year"] / 366.0)
    assert_pre2024(official, "official scored frame")

    available = pd.read_parquet(FEATURE_MATRIX_PATH).columns.tolist()
    feature_cols = [column for column in SAFE_FEATURE_MATRIX_COLUMNS if column == "target_date" or column in available]
    features = pd.read_parquet(FEATURE_MATRIX_PATH, columns=feature_cols)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()
    features = features.drop_duplicates("target_date", keep="last").sort_values("target_date").reset_index(drop=True)
    assert_pre2024(features, "safe target-memory feature matrix")

    merged = official.merge(features, on="target_date", how="left", validate="one_to_one")
    return merged.sort_values("target_date").reset_index(drop=True)


def add_predeclared_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = frame.copy()
    text = (
        out.get("weather_text", pd.Series("", index=out.index)).fillna("").astype(str)
        + " "
        + out.get("wind_text", pd.Series("", index=out.index)).fillna("").astype(str)
        + " "
        + out.get("description_text", pd.Series("", index=out.index)).fillna("").astype(str)
    ).str.lower()
    for name, pattern in TEXT_PATTERNS.items():
        out[name] = text.str.contains(pattern, regex=True).astype(float)
    out["wind_force_max"] = text.map(
        lambda value: max([int(match) for match in re.findall(r"force\s+([0-9])", value)] or [np.nan])
    )

    out["source_press_archive"] = out["forecast_source_family"].eq("press_archive").astype(float)
    out["source_rss_archive"] = out["forecast_source_family"].eq("rss_archive").astype(float)
    out["product_5day"] = out.get("product_type", pd.Series("", index=out.index)).fillna("").astype(str).str.contains("5").astype(float)
    out["product_9day"] = out.get("feed_type", pd.Series("", index=out.index)).fillna("").astype(str).str.contains("9").astype(float)

    if "target_lag7_tmax_c" in out.columns:
        out["official_minus_target_lag7_c"] = out["forecast_max_c"] - out["target_lag7_tmax_c"]
    if "target_roll30_mean_lag7_c" in out.columns:
        out["official_minus_target_roll30_c"] = out["forecast_max_c"] - out["target_roll30_mean_lag7_c"]
    if "target_roll365_mean_lag7_c" in out.columns:
        out["official_minus_target_roll365_c"] = out["forecast_max_c"] - out["target_roll365_mean_lag7_c"]
    out["range_x_rh_range"] = out["forecast_range_c"] * out["rh_range_pct"]
    out["solar_proxy_score"] = out[["text_sunny", "text_dry"]].sum(axis=1) - out[["text_cloudy", "text_rain", "text_thunder", "text_mist"]].sum(axis=1)
    out["humidity_suppression_proxy"] = out[["rh_max_pct", "rh_range_pct"]].mean(axis=1) + 10.0 * out["text_humid"]
    out["cool_surge_text_proxy"] = out["text_cool"] + out["wind_northerly"] + out["wind_strong"]
    out["warm_cloudbreak_text_proxy"] = out["text_hot"] + out["text_sunny"] - out["text_rain"] - out["text_cloudy"]
    if "target_lag7_minus_roll30_c" in out.columns:
        out["lagged_rebound_x_solar_proxy"] = out["target_lag7_minus_roll30_c"] * out["solar_proxy_score"]
    if "target_roll30_std_lag7_c" in out.columns:
        out["lagged_volatility_x_forecast_range"] = out["target_roll30_std_lag7_c"] * out["forecast_range_c"]

    base_feature_cols = [
        "forecast_max_c",
        "forecast_min_c",
        "forecast_range_c",
        "forecast_midpoint_c",
        "rh_min_pct",
        "rh_max_pct",
        "rh_range_pct",
        "issue_to_cutoff_hours",
        "lead_hours_at_cutoff",
        "forecast_span_c",
        "month",
        "day_of_year",
        "doy_sin",
        "doy_cos",
        "source_press_archive",
        "source_rss_archive",
        "product_5day",
        "product_9day",
        "wind_force_max",
        *TEXT_PATTERNS.keys(),
        *[column for column in TARGET_MEMORY_COLUMNS if column in out.columns],
        "official_minus_target_lag7_c",
        "official_minus_target_roll30_c",
        "official_minus_target_roll365_c",
        "range_x_rh_range",
        "solar_proxy_score",
        "humidity_suppression_proxy",
        "cool_surge_text_proxy",
        "warm_cloudbreak_text_proxy",
        "lagged_rebound_x_solar_proxy",
        "lagged_volatility_x_forecast_range",
    ]
    feature_cols = []
    for column in base_feature_cols:
        if column in out.columns and column not in feature_cols:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            feature_cols.append(column)

    feature_defs = []
    for column in feature_cols:
        family = "official_forecast_text_proxy"
        availability = "Exact-vintage official forecast field issued before the T-24 cutoff."
        if column.startswith("target_") or column.startswith("trajectory_") or column.startswith("official_minus_target") or column.startswith("lagged_"):
            family = "causal_target_memory_lag7"
            availability = "Uses only target outcomes shifted by at least seven local days before T."
        elif column in {"month", "day_of_year", "doy_sin", "doy_cos"}:
            family = "calendar"
            availability = "Deterministic calendar known before cutoff."
        feature_defs.append(
            {
                "feature_name": column,
                "role": "candidate_predictor",
                "family": family,
                "formula": "Predeclared proxy formula in src/run_0184.py; no target T value, residual, or future row is used.",
                "input_columns": "official forecast exact-vintage fields; target-memory lag7 columns; deterministic calendar",
                "units": "degC/pct/binary/derived",
                "lag": ">=7 days for target-memory columns; current issued forecast for official text/range columns",
                "window": "fixed feature-specific lag/rolling window from robust feature matrix",
                "fit_scope": "Fold-local imputation, scaling, ridge fit, and nested grid selection",
                "availability_rule": availability,
                "missingness_policy": "Median imputation fitted inside each training fold only",
            }
        )
    return out, feature_cols, pd.DataFrame(feature_defs)


def pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def inner_select(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    train = train.sort_values("target_date").copy()
    max_year = int(train["target_date"].dt.year.max())
    min_year = int(train["target_date"].dt.year.min())
    validation_start = max(min_year + 2, max_year - 3)
    inner_fit = train[train["target_date"].dt.year < validation_start].copy()
    inner_val = train[train["target_date"].dt.year >= validation_start].copy()
    if len(inner_fit) < 365 or len(inner_val) < 90:
        return {"alpha": 30.0, "cap_c": 0.25, "inner_mae_c": math.nan}

    y_fit = inner_fit["target_tmax_c"] - inner_fit["official_prediction_c"]
    best: dict[str, float] | None = None
    for config in CONFIDENCE_GRID:
        model = pipeline(config["alpha"])
        model.fit(inner_fit[feature_cols], y_fit)
        correction = np.clip(model.predict(inner_val[feature_cols]), -config["cap_c"], config["cap_c"])
        candidate = inner_val["official_prediction_c"].to_numpy(dtype=float) + correction
        mae = float(np.mean(np.abs(candidate - inner_val["target_tmax_c"].to_numpy(dtype=float))))
        row = {"alpha": config["alpha"], "cap_c": config["cap_c"], "inner_mae_c": mae}
        if best is None or (mae, config["cap_c"], config["alpha"]) < (best["inner_mae_c"], best["cap_c"], best["alpha"]):
            best = row
    if best is None:
        return {"alpha": 30.0, "cap_c": 0.25, "inner_mae_c": math.nan}
    return best


def run_walk_forward(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for start_year, end_year in MODEL_FOLDS:
        test_mask = frame["target_date"].dt.year.between(start_year, end_year)
        test = frame[test_mask].copy()
        train = frame[frame["target_date"].dt.year < start_year].copy()
        if test.empty:
            continue
        if len(train) < 365:
            test["candidate_correction_c"] = 0.0
            test["candidate_prediction_c"] = test["official_prediction_c"]
            selected = {"alpha": math.nan, "cap_c": 0.0, "inner_mae_c": math.nan}
            fold_id = f"fold_{start_year}_{end_year}_no_prior_train"
            train_end = ""
        else:
            selected = inner_select(train, feature_cols)
            model = pipeline(selected["alpha"])
            y_train = train["target_tmax_c"] - train["official_prediction_c"]
            model.fit(train[feature_cols], y_train)
            correction = np.clip(model.predict(test[feature_cols]), -selected["cap_c"], selected["cap_c"])
            test["candidate_correction_c"] = correction
            test["candidate_prediction_c"] = test["official_prediction_c"] + test["candidate_correction_c"]
            fold_id = f"fold_{start_year}_{end_year}"
            train_end = date_text(train["target_date"].max())
        test["fold_id"] = fold_id
        test["training_end"] = train_end
        test["selected_alpha"] = selected["alpha"]
        test["selected_cap_c"] = selected["cap_c"]
        test["selected_inner_mae_c"] = selected["inner_mae_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["official_error_c_signed"] = test["official_prediction_c"] - test["target_tmax_c"]
        prediction_parts.append(test)

        metrics = compare_metrics(test, slice_type="fold", slice_value=fold_id)
        metrics.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "training_rows": int(len(train)),
                "training_end": train_end,
                "selected_alpha": selected["alpha"],
                "selected_cap_c": selected["cap_c"],
                "selected_inner_mae_c": selected["inner_mae_c"],
            }
        )
        fold_rows.append(metrics)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    return predictions, pd.DataFrame(fold_rows)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [compare_metrics(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(compare_metrics(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(compare_metrics(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(compare_metrics(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    if not late.empty:
        rows.append(compare_metrics(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["official_abs_error_c"] >= 2.0] if "official_abs_error_c" in predictions.columns else pd.DataFrame()
    if not tail.empty:
        rows.append(compare_metrics(tail, slice_type="official_tail", slice_value="official_abs_error_ge_2c"))
    slice_metrics = pd.DataFrame(rows)

    yearly = pd.DataFrame(
        [
            compare_metrics(group, slice_type="year", slice_value=year)
            for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)
        ]
    )
    seasonal = pd.DataFrame(
        [
            compare_metrics(group, slice_type="season", slice_value=season)
            for season, group in predictions.groupby("season", dropna=False)
        ]
    )
    return slice_metrics, yearly, seasonal


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy().replace({np.nan: ""})
    lines = [
        "| " + " | ".join(str(column) for column in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("|", r"\|") for column in display.columns) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": (
            "A fold-local residual correction using exact-vintage official forecast text/range fields and "
            "lag-seven target-memory proxies for high-frequency heating dispersion can reduce HKO T-24 Tmax MAE "
            "without reading target-day observations or 2024+ confirmation outcomes."
        ),
        "rationale": (
            "Experiments 0179-0183 found strong diagnostic signal in same-day high-frequency temperature dispersion, "
            "solar, humidity, and cloud-break features, but those are not production-promoted. This experiment tests "
            "whether a deployable proxy for that teacher signal is already present in the official forecast wording, "
            "forecast range, humidity range, source family, and conservative lag-seven target-memory trajectory."
        ),
        "expected_sign_and_falsification": (
            "Expected sign is lower MAE versus raw official forecast on identical rows. The hypothesis is falsified if "
            "nested walk-forward correction fails to beat the official baseline or improves mean MAE only by increasing "
            "severe-tail or seasonal damage."
        ),
        "novelty": {
            "prior_experiments": ["0074", "0075", "0179", "0180", "0181", "0182", "0183"],
            "difference": (
                "Prior HF experiments used same-day high-frequency observations and prior online residual memory runs "
                "did not explicitly distill HF teacher mechanisms into exact-vintage forecast text plus lag-seven target memory."
            ),
            "similarity_audit_path": "RESULTS.md#comparison-limitations",
        },
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": "data/datasets/05_hko_historical_rss_forecasts/hko_official_t15_scored_pre2024.parquet",
            "cutoff_function": "Use official forecast entries issued/available no later than the stored cutoff_utc for target date T.",
            "daily_boundary_contract": "HKO local daily maximum temperature for target local date T.",
        },
        "frame": {
            "frame_id": "official_t15_pre2024_5265_rows",
            "development_start": "2000-01-02",
            "development_end_exclusive": "2024-01-01",
            "confirmation_locked": True,
            "row_universe_artifact": rel(OFFICIAL_PATH),
        },
        "data_sources": [
            {
                "source_id": "official_t15_scored_pre2024",
                "paths": [rel(OFFICIAL_PATH)],
                "attributes": ["forecast_max_c", "forecast_min_c", "rh_min_pct", "rh_max_pct", "weather_text", "wind_text", "issue_at_utc", "available_at_utc", "cutoff_utc"],
                "eligibility": "DEPLOYABLE_PROVEN",
                "availability_proof": "Rows are exact-vintage official HKO press/RSS forecasts with issue/available timestamps no later than the stored T-24 cutoff.",
            },
            {
                "source_id": "lag7_target_memory_from_robust_matrix",
                "paths": [rel(FEATURE_MATRIX_PATH)],
                "attributes": TARGET_MEMORY_COLUMNS,
                "eligibility": "DEPLOYABLE_LAGGED_ONLY",
                "availability_proof": "Only lag-seven-or-older target-memory columns are used; the current target date and target-day observations are not predictors.",
            },
        ],
        "stations": [
            {
                "station_id": "HKO",
                "role": "target",
                "attributes": ["daily maximum temperature label; lag-seven-or-older target memory only"],
            }
        ],
        "features": {
            "generation_rule": "Official text/range/RH/source proxy features plus conservative lag-seven target-memory trajectory; fold-local imputation/scaling only.",
            "explicit_exclusions": ["target_tmax_c as predictor", "official_residual_c as predictor", "official_abs_error_c as predictor", "all 2024+ rows", "same-day high-frequency observations"],
            "grid": CONFIDENCE_GRID,
        },
        "response": {
            "variable": "official residual, target_tmax_c - forecast_max_c",
            "prediction": "forecast_max_c + clipped residual correction",
        },
        "baseline": {
            "id": "official_forecast_max_c",
            "definition": "Raw official maximum-temperature forecast on the exact same target dates.",
        },
        "validation": {
            "outer_folds": [list(item) for item in MODEL_FOLDS],
            "inner_selection": "For each outer fold, choose ridge alpha and correction cap using only the last four years inside the prior training history.",
            "minimum_train_rows": 365,
        },
        "metrics": ["MAE", "RMSE", "bias", "median_abs_error", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source slices"],
        "sample_rules": {
            "row_policy": "Use every pre-2024 official row; first fold receives zero correction if no prior training history exists.",
            "missing_policy": "Fold-local median imputation only.",
        },
        "acceptance_gates": {
            "minimum_mae_lift_c": 0.01,
            "no_tail_harm": "Candidate >3C rate must not exceed official by more than 0.005.",
            "confirmation": "No 2024+ rows may be used.",
        },
        "rejection_conditions": [
            "Any target_date >= 2024-01-01 is read into scoring.",
            "Any feature contains target T, residual, absolute error, or same-day high-frequency observation.",
            "Candidate and baseline rows differ.",
        ],
        "required_outputs": [
            "README.md",
            "RESULTS.md",
            "CONCLUSION.md",
            "scoreboard.csv",
            "slice_metrics.csv",
            "yearly_metrics.csv",
            "fold_metrics.csv",
            "predictions.parquet",
        ],
        "owner_authorized_confirmation": False,
    }


def write_documents(
    predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    summary: dict[str, Any],
    feature_cols: list[str],
) -> None:
    top_slices = slice_metrics.sort_values("mae_delta_c").head(10)
    damaged_slices = slice_metrics.sort_values("mae_delta_c", ascending=False).head(10)
    readme = f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a single-folder HKG T-24 development experiment created to test a deployable proxy for the high-frequency teacher signal found in 0179-0183.

## One-Sentence Hypothesis

Exact-vintage official forecast text/range/RH information plus conservative lag-seven target-memory features can proxy the same cloud-break, heating-dispersion, and humidity suppression mechanisms that the high-frequency diagnostics exposed.

## Why It Is Worth Doing

The recent high-frequency experiments were information-rich but not production-promoted. They showed that same-day temperature dispersion, solar, and humidity structure can move MAE, but the live decision is at T-24. This experiment asks whether the official bulletin itself and old target trajectory already encode enough of that mechanism to improve the raw official maximum forecast without target-day observations.

## Prior Evidence And Novelty

Prior residual-memory work established that small causal corrections can help. The 0179-0183 batch showed a stronger teacher signal but used short-history same-day data. This experiment differs because it uses no same-day high-frequency values and no current target residual features.

## Target, Horizon, And Exact Cutoff

Target is HKO daily Tmax for local day T. Horizon is `T-24` in `Asia/Hong_Kong`. The cutoff is the stored official T15/cutoff contract in `{rel(OFFICIAL_PATH)}`; forecast rows must be issued or available no later than the stored cutoff for T.

## Datasets, Stations, And Attributes

Inputs are the official pre-2024 forecast row set and lag-seven-or-older HKO target-memory columns from the robust feature matrix. The only station-specific outcome used as a predictor is historical HKO target memory shifted by at least seven days.

## Feature Definitions

The candidate uses {len(feature_cols)} predictors. They include official forecast max/min/range, RH range, weather and wind text flags, source-family flags, calendar fields, and lag-seven target-memory trajectory/proxy interactions. Full definitions are in `feature_definitions.csv`.

## Response And Baseline

The response is official residual `target_tmax_c - forecast_max_c`. The baseline is the raw official `forecast_max_c` on the exact same rows.

## Walk-Forward Design

Outer folds are `{MODEL_FOLDS}`. For each fold, all fitting, imputation, scaling, and ridge hyperparameter/cap selection use only earlier target dates. Prediction happens before the target outcome is used for any later fold.

## Acceptance And Rejection Criteria

Acceptance requires lower MAE than official on identical rows, no 2024+ access, no target T predictors, and no material severe-tail harm. Rejection would occur for timestamp failure, target leakage, or row-set mismatch.

## Expected Failure Modes

The lane can fail if official forecasters already fully encode the teacher signal, if lag-seven memory is too stale, or if text proxies are too coarse to represent same-day heating dispersion.

## Reproduction Command

Run `python scripts/run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router.py` from the repository root.
"""
    write_text(EXP_DIR / "README.md", readme)

    results = f"""# Results

## Headline Result Table

{markdown_table(scoreboard)}

## Coverage And Row Identity

The scored frame has {summary['n_common']} common rows from {summary['date_start']} through {summary['date_end']}. Candidate and official baseline share identical target dates and target values. The common row hash is `{summary['common_row_hash']}`.

## Global Metrics

Candidate MAE is `{summary['candidate_mae_c']:.6f}` C versus official baseline `{summary['baseline_mae_c']:.6f}` C, for a delta of `{summary['mae_delta_c']:.6f}` C. Candidate RMSE is `{summary['candidate_rmse_c']:.6f}` C and candidate bias is `{summary['candidate_bias_c']:.6f}` C.

## Fold Stability

{markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'selected_alpha', 'selected_cap_c']], max_rows=20)}

## Yearly And Seasonal Results

{markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season summary:

{markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Candidate hot-underforecast and cold-overforecast counts are saved in `slice_metrics.csv`. The root predictions file stores signed official and candidate errors for row-level inspection.

## Ablations

No post-hoc ablation was used to select the candidate. The finite alpha/correction-cap grid is saved in `artifacts/candidate_grid.csv`, and fold-local selected configurations are in `fold_metrics.csv`.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. No 2024+ confirmation rows were used. No predictor contains target-day Tmax, official residual, official absolute error, or same-day high-frequency observations.

## Comparison Limitations

This result compares only to raw official forecast on the same official T15 frame. It should not be compared directly to older scores on 2670-row frames or to 0179-0183 same-day high-frequency diagnostic scores. Better MAE on this development frame does not authorize confirmation-period access.

## Best And Worst Slices

Best slices:

{markdown_table(top_slices[['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

Damaged slices:

{markdown_table(damaged_slices[['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}
"""
    write_text(EXP_DIR / "RESULTS.md", results)

    verdict = "promotable for deeper replay" if summary["status"] == "COMPLETED_PROMOTION_CANDIDATE" else "information-gain only"
    conclusion = f"""# Conclusion

## Verdict

The experiment verdict is `{verdict}` with status `{summary['status']}`.

## What Was Learned

The run directly tested whether same-day high-frequency teacher mechanisms can be represented by T-24-safe official forecast text/range/RH fields and conservative lag-seven target memory. The saved slice files show where that proxy helps and where it is redundant or harmful.

## Realized Point-MAE Change

Candidate MAE changed by `{summary['mae_delta_c']:.6f}` C against raw official forecast on the identical {summary['n_common']}-row frame. Negative delta means improvement.

## Information Gain Outside Point MAE

The correction distribution and fold-local selected caps quantify how much residual movement the safe proxy can justify. Even if the candidate is not a final champion, it tells us whether the HF teacher lane should be distilled into official-text and old-target-memory features rather than relying on same-day observations.

## Robustness And Uncertainty

All model fitting was chronological. Imputation, scaling, alpha selection, and cap selection were fold-local. The first fold used zero correction because no prior training history existed, which keeps row coverage identical without inventing future knowledge.

## Failure Diagnosis

If the lane failed, the likely reason is that official forecasts already internalize most cloud-break and humidity information, or that lag-seven target trajectory is too stale to proxy same-day heating dispersion. If it succeeded, the lift is a deployable clue that exact-vintage bulletin language and conservative target memory deserve a stronger specialist.

## Promotion Status

Promotion decision: `{summary['promotion_decision']}`. Confirmation remains locked and was not opened.

## Implication For Future Research

The next Director step should use this evidence to decide whether to build a narrower official-text specialist, a source-aware online residual memory variant, or a lagged target-memory/official-range router with more explicit no-harm gates.
"""
    write_text(EXP_DIR / "CONCLUSION.md", conclusion)


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(exist_ok=True)

    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    write_json(
        EXP_DIR / "run_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "slug": SLUG,
            "created_at_utc": created_at,
            "repo_root": str(REPO_ROOT),
            "script": rel(Path(__file__).resolve()),
            "spec_sha256": spec_sha,
            "state": "SPEC_WRITTEN_BEFORE_SCORING",
        },
    )

    shutil.copy2(Path(__file__).resolve(), EXP_DIR / "src" / Path(__file__).name)

    frame = load_inputs()
    model_frame, feature_cols, feature_defs = add_predeclared_features(frame)
    forbidden = [
        column
        for column in feature_cols
        if column in {"target_tmax_c", "official_residual_c", "official_abs_error_c", "official_error_c"}
        or "residual" in column.lower()
        or "abs_error" in column.lower()
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden target/error features selected: {forbidden}")
    assert_pre2024(model_frame, "model frame")

    predictions, fold_metrics = run_walk_forward(model_frame, feature_cols)
    predictions["official_abs_error_c"] = (predictions["official_prediction_c"] - predictions["target_tmax_c"]).abs()
    predictions["candidate_abs_error_c"] = (predictions["candidate_prediction_c"] - predictions["target_tmax_c"]).abs()
    predictions["candidate_id"] = PRIMARY_CANDIDATE_ID
    predictions["baseline_id"] = "official_forecast_max_c"
    predictions["model_family"] = "nested_ridge_hf_teacher_proxy"
    predictions["feature_count"] = len(feature_cols)

    slice_metrics, yearly_metrics, season_metrics = build_slice_metrics(predictions)
    overall = compare_metrics(predictions, slice_type="overall", slice_value="all")
    official_global = metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)

    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max()) if not fold_metrics.empty else math.nan
    if overall["mae_delta_c"] <= -0.01 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_TO_DEEPER_REPLAY_NO_CONFIRMATION"
    elif overall["mae_delta_c"] < 0:
        status = "COMPLETED_INFORMATION_GAIN_ONLY"
        promotion_decision = "DO_NOT_PROMOTE_YET_INFORMATION_GAIN"
    else:
        status = "COMPLETED_NULL_OR_NEGATIVE"
        promotion_decision = "DO_NOT_PROMOTE"

    common_row_hash = sha256_text("\n".join(date_text(value) for value in predictions["target_date"]))
    scoreboard = pd.DataFrame(
        [
            {
                "candidate_id": "official_forecast_max_c",
                "model_family": "baseline",
                "n": official_global["n"],
                "mae_c": official_global["mae_c"],
                "rmse_c": official_global["rmse_c"],
                "bias_c": official_global["bias_c"],
                "median_abs_error_c": official_global["median_abs_error_c"],
                "p95_abs_error_c": official_global["p95_abs_error_c"],
                "gt2c_rate": official_global["gt2c_rate"],
                "gt3c_rate": official_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": 0.0,
            },
            {
                "candidate_id": PRIMARY_CANDIDATE_ID,
                "model_family": "nested_ridge_hf_teacher_proxy",
                "n": candidate_global["n"],
                "mae_c": candidate_global["mae_c"],
                "rmse_c": candidate_global["rmse_c"],
                "bias_c": candidate_global["bias_c"],
                "median_abs_error_c": candidate_global["median_abs_error_c"],
                "p95_abs_error_c": candidate_global["p95_abs_error_c"],
                "gt2c_rate": candidate_global["gt2c_rate"],
                "gt3c_rate": candidate_global["gt3c_rate"],
                "baseline_mae_c": official_global["mae_c"],
                "mae_delta_c": candidate_global["mae_c"] - official_global["mae_c"],
            },
        ]
    )

    row_coverage = pd.DataFrame(
        [
            {
                "frame_id": "official_t15_pre2024_5265_rows",
                "parent_rows": int(len(frame)),
                "candidate_rows": int(len(predictions)),
                "baseline_rows": int(len(predictions)),
                "common_rows": int(len(predictions)),
                "date_start": date_text(predictions["target_date"].min()),
                "date_end": date_text(predictions["target_date"].max()),
                "row_policy": "identical official T15 pre-2024 rows; no row dropped for candidate",
                "common_row_hash": common_row_hash,
            }
        ]
    )

    correction_distribution = predictions["candidate_correction_c"].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).reset_index()
    correction_distribution.columns = ["statistic", "candidate_correction_c"]

    data_manifest = pd.DataFrame(
        [
            {
                "source_id": "official_t15_scored_pre2024",
                "path": rel(OFFICIAL_PATH),
                "sha256": sha256_file(OFFICIAL_PATH),
                "size_bytes": OFFICIAL_PATH.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "issue_at_utc;available_at_utc;cutoff_utc;published_at_utc",
                "availability_class": "DEPLOYABLE_PROVEN",
                "notes": "Exact-vintage official forecast archive, pre-2024 only.",
            },
            {
                "source_id": "lag7_target_memory_from_robust_matrix",
                "path": rel(FEATURE_MATRIX_PATH),
                "sha256": sha256_file(FEATURE_MATRIX_PATH),
                "size_bytes": FEATURE_MATRIX_PATH.stat().st_size,
                "row_count": int(len(frame)),
                "date_start": date_text(frame["target_date"].min()),
                "date_end": date_text(frame["target_date"].max()),
                "timestamp_fields": "target_date;lag7_or_older_target_memory_only",
                "availability_class": "DEPLOYABLE_LAGGED_ONLY",
                "notes": "Only conservative lag-seven-or-older target-memory columns were admitted.",
            },
        ]
    )

    prediction_cols = [
        "target_date",
        "target_tmax_c",
        "forecast_source_family",
        "season",
        "month",
        "official_prediction_c",
        "candidate_prediction_c",
        "candidate_correction_c",
        "official_error_c_signed",
        "candidate_error_c",
        "official_abs_error_c",
        "candidate_abs_error_c",
        "fold_id",
        "training_end",
        "selected_alpha",
        "selected_cap_c",
        "selected_inner_mae_c",
        "candidate_id",
        "baseline_id",
        "model_family",
        "feature_count",
    ]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[prediction_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", correction_distribution)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "candidate_grid.csv", pd.DataFrame(CONFIDENCE_GRID))
    write_csv(EXP_DIR / "artifacts" / "season_metrics.csv", season_metrics)
    write_text(EXP_DIR / "diagnostics" / "feature_columns.txt", "\n".join(feature_cols) + "\n")

    audit = f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff Function

The experiment uses the official T15 pre-2024 row set. Each official forecast row carries issue/availability/cutoff fields, and only rows with target dates before 2024-01-01 are scored. The operational cutoff is the stored T-24 cutoff for target date T in `{rel(OFFICIAL_PATH)}`.

## Feature Eligibility

Official forecast max/min/range, humidity range, wind text, weather text, source family, and publication metadata are exact-vintage forecast fields available before the cutoff. Target-memory predictors are restricted to lag-seven-or-older robust matrix columns. Same-day high-frequency features from 0179-0183 are deliberately excluded.

## Available-At, Target, And Rolling Checks

No feature is named `target_tmax_c`, `official_residual_c`, `official_abs_error_c`, or any residual/error derivative. The code rejects such features before fitting. Lagged target-memory columns are already shifted by at least seven days, so rolling windows do not include target T. Fold-local imputation, scaling, alpha selection, cap selection, and model fitting use only dates earlier than the predicted outer fold.

## Confirmation Exclusion Proof

The maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Online And Fold Ordering

The first fold receives zero correction because no prior training history exists. Later folds train only on target dates before the fold start year, then predict the whole fold. Target outcomes from a fold are not used until subsequent folds.
"""
    write_text(EXP_DIR / "leakage_audit.md", audit)

    write_text(
        EXP_DIR / "REPRODUCE.md",
        f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router.py
```

Expected headline:

- Experiment folder: `{EXP_DIR}`
- Candidate: `{PRIMARY_CANDIDATE_ID}`
- Baseline MAE: `{official_global['mae_c']:.12f}`
- Candidate MAE: `{candidate_global['mae_c']:.12f}`
- MAE delta: `{candidate_global['mae_c'] - official_global['mae_c']:.12f}`

The script rewrites only this experiment folder and does not open 2024+ confirmation rows.
""",
    )

    code_sha = sha256_file(EXP_DIR / "src" / Path(__file__).name)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "slug": SLUG,
        "status": status,
        "created_at_utc": created_at,
        "target": "HKO daily Tmax T-24",
        "frame_id": "official_t15_pre2024_5265_rows",
        "date_start": date_text(predictions["target_date"].min()),
        "date_end": date_text(predictions["target_date"].max()),
        "n_candidate": int(len(predictions)),
        "n_common": int(len(predictions)),
        "baseline_id": "official_forecast_max_c",
        "baseline_mae_c": official_global["mae_c"],
        "candidate_id": PRIMARY_CANDIDATE_ID,
        "candidate_mae_c": candidate_global["mae_c"],
        "mae_delta_c": candidate_global["mae_c"] - official_global["mae_c"],
        "candidate_rmse_c": candidate_global["rmse_c"],
        "candidate_bias_c": candidate_global["bias_c"],
        "leakage_status": "PASS",
        "confirmation_rows_used": 0,
        "owner_authorized_confirmation": False,
        "promotion_decision": promotion_decision,
        "spec_sha256": spec_sha,
        "code_sha256": code_sha,
        "data_manifest_sha256": data_manifest_sha,
        "common_row_hash": common_row_hash,
        "baseline_n": int(len(predictions)),
        "candidate_n": int(len(predictions)),
        "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45),
        "fold_worst_mae_delta_c": fold_worst_delta,
        "severe_gt3_rate_delta_c": severe_harm,
        "feature_count": len(feature_cols),
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_documents(predictions, scoreboard, slice_metrics, yearly_metrics, fold_metrics, summary, feature_cols)
    write_json(
        EXP_DIR / "run_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "slug": SLUG,
            "created_at_utc": created_at,
            "completed_at_utc": utc_now(),
            "repo_root": str(REPO_ROOT),
            "script": rel(Path(__file__).resolve()),
            "spec_sha256": spec_sha,
            "code_sha256": code_sha,
            "state": "COMPLETED",
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
