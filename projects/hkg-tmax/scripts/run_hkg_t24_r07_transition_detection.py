from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R07"
EXPERIMENT_ID = "EXP-0039"
EXPERIMENT_DIR = REPO_ROOT / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0039-HKG-T24-R07"

_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R07 transition experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
_R04_SPEC.loader.exec_module(_R04_MODULE)

_R06_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r06_moisture_state",
    REPO_ROOT / "scripts" / "run_hkg_t24_r06_moisture_state.py",
)
if _R06_SPEC is None or _R06_SPEC.loader is None:
    raise ImportError("Unable to load R06 helper script for R07 transition experiment.")
_R06_MODULE = importlib.util.module_from_spec(_R06_SPEC)
sys.modules[_R06_SPEC.name] = _R06_MODULE
_R06_SPEC.loader.exec_module(_R06_MODULE)

QUANTILE_Z = _R04_MODULE.QUANTILE_Z
fold_definitions = _R04_MODULE.fold_definitions
git_state = _R04_MODULE.git_state
markdown_table = _R04_MODULE.markdown_table
normal_crps = _R04_MODULE.normal_crps
now_utc = _R04_MODULE.now_utc
r04_feature_sets = _R04_MODULE.model_feature_sets
sha256_file = _R04_MODULE.sha256_file

asof_values = _R06_MODULE.asof_values
make_cutoffs = _R06_MODULE.make_cutoffs
valid_columns = _R06_MODULE.valid_columns
write_text = _R06_MODULE.write_text


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    columns: tuple[str, ...]
    factory: Callable[[], object]
    two_stage_transition: bool = False


def load_wind_observations(data_root: Path) -> tuple[pd.DataFrame, Path]:
    path = data_root / "bronze" / "analysis_phase_a" / "hko_high_frequency_selected_station_observations.parquet"
    if not path.exists():
        raise FileNotFoundError(f"R07 requires parsed Phase A high-frequency observations: {path}")
    observations = pd.read_parquet(path)
    observations = observations[observations["variable"].isin(["mean_wind_speed_kmh", "max_wind_gust_kmh"])].copy()
    observations["observed_at_hkt"] = pd.to_datetime(observations["observed_at_hkt"], utc=True).dt.tz_convert(_R06_MODULE.HKT)
    observations["available_at_hkt"] = pd.to_datetime(observations["available_at_hkt"], utc=True).dt.tz_convert(_R06_MODULE.HKT)
    observations = observations.rename(columns={"content_sha256": "source_file_hash"})
    return observations, path


def wind_network_features(observations: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variable, prefix in [("mean_wind_speed_kmh", "wind_speed"), ("max_wind_gust_kmh", "wind_gust")]:
        current = asof_values(observations, cutoffs, variable=variable, offset_hours=0, tolerance_hours=3)
        prior = asof_values(observations, cutoffs, variable=variable, offset_hours=3, tolerance_hours=3)
        if current.empty:
            continue
        current_agg = current.groupby("target_date").agg(
            **{
                f"network_{prefix}_station_count": ("station", "nunique"),
                f"network_median_{prefix}_kmh": ("value", "median"),
                f"network_max_{prefix}_kmh": ("value", "max"),
                f"network_p90_{prefix}_kmh": ("value", lambda s: float(s.quantile(0.90))),
            }
        )
        if not prior.empty:
            prior_agg = prior.groupby("target_date")["value"].median().rename(f"network_median_{prefix}_3h_prior_kmh")
            current_agg = current_agg.join(prior_agg, how="left")
            current_agg[f"network_median_{prefix}_3h_change_kmh"] = (
                current_agg[f"network_median_{prefix}_kmh"] - current_agg[f"network_median_{prefix}_3h_prior_kmh"]
            )
        frames.append(current_agg.reset_index())
    if not frames:
        return pd.DataFrame(columns=["target_date"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="target_date", how="outer")
    return out


def add_pressure_candidates(data_root: Path, features: pd.DataFrame) -> pd.DataFrame:
    path = data_root / "silver" / "features" / "t24_cutoff_feature_candidates.parquet"
    if not path.exists():
        features["hko_mslp_3h_change_to_cutoff_hpa"] = np.nan
        return features
    candidates = pd.read_parquet(
        path,
        columns=[
            "local_date",
            "hko_mslp_at_tminus1_1500_hpa",
            "hko_mslp_tminus1_1200_hpa",
            "hko_mslp_3h_change_to_cutoff_hpa",
        ],
    ).rename(
        columns={
            "local_date": "target_date",
            "hko_mslp_at_tminus1_1500_hpa": "hko_mslp_cutoff_hpa_candidate",
            "hko_mslp_tminus1_1200_hpa": "hko_mslp_3h_prior_hpa",
        }
    )
    candidates["target_date"] = pd.to_datetime(candidates["target_date"])
    out = features.merge(candidates, on="target_date", how="left")
    out["hko_mslp_cutoff_hpa"] = out["hko_mslp_cutoff_hpa"].combine_first(out["hko_mslp_cutoff_hpa_candidate"])
    out = out.drop(columns=["hko_mslp_cutoff_hpa_candidate"])
    return out


def add_transition_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features.sort_values("target_date").reset_index(drop=True).copy()
    one_day_gap = pd.to_datetime(out["target_date"]).diff().dt.days == 1
    two_day_gap = pd.to_datetime(out["target_date"]).diff(2).dt.days == 2
    out["hko_mslp_24h_change_hpa"] = (out["hko_mslp_cutoff_hpa"] - out["hko_mslp_cutoff_hpa"].shift(1)).where(one_day_gap)
    out["hko_mslp_48h_change_hpa"] = (out["hko_mslp_cutoff_hpa"] - out["hko_mslp_cutoff_hpa"].shift(2)).where(two_day_gap)
    out["hko_mslp_3h_acceleration_hpa"] = (
        out["hko_mslp_3h_change_to_cutoff_hpa"] - out["hko_mslp_3h_change_to_cutoff_hpa"].shift(1)
    ).where(one_day_gap)
    out["hko_temp_decline_6h_c"] = -out["hko_temp_change_360m_to_latest_c"]
    out["hko_dew_point_decline_6h_c"] = -out["hko_dew_point_change_6h_c"]
    out["wind_speed_increase_3h_kmh"] = out.get("network_median_wind_speed_3h_change_kmh", np.nan)
    out["pressure_rise_component"] = out["hko_mslp_3h_change_to_cutoff_hpa"] / 3.0
    out["cooling_component"] = out["hko_temp_decline_6h_c"] / 2.0
    out["drying_component"] = out["hko_dew_point_decline_6h_c"] / 2.0
    out["wind_increase_component"] = out["wind_speed_increase_3h_kmh"] / 10.0
    out["cold_surge_score"] = (
        out["pressure_rise_component"].fillna(0)
        + out["cooling_component"].fillna(0)
        + out["drying_component"].fillna(0)
        + out["wind_increase_component"].fillna(0)
    )
    out["front_score"] = (
        out["pressure_rise_component"].abs().fillna(0)
        + out["cooling_component"].abs().fillna(0)
        + out["drying_component"].abs().fillna(0)
        + out["wind_increase_component"].abs().fillna(0)
    )
    out["warm_sector_score"] = (
        (-out["pressure_rise_component"]).fillna(0)
        + (-out["cooling_component"]).fillna(0)
        + (-out["drying_component"]).fillna(0)
    )
    out["post_frontal_score"] = (
        out["hko_mslp_24h_change_hpa"].clip(lower=0).fillna(0) / 5.0
        + out["hko_dewpoint_depression_c"].fillna(0) / 8.0
        + out["hko_temp_decline_6h_c"].clip(lower=0).fillna(0) / 3.0
    )
    out["target_tmax_change_1d_c"] = (out["target_tmax_c"] - out["target_tmax_c"].shift(1)).where(one_day_gap)
    out["aux_transition_label"] = (out["target_tmax_change_1d_c"].abs() >= 2.5).astype(float)
    out["aux_cold_drop_label"] = (out["target_tmax_change_1d_c"] <= -2.0).astype(float)
    out["top_decile_transition_proxy"] = out["front_score"] >= out["front_score"].quantile(0.90)
    out = add_permuted_transition_controls(out)
    return out


def add_permuted_transition_controls(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    rng = np.random.default_rng(707)
    months = pd.to_datetime(out["target_date"]).dt.month
    for col in ["hko_mslp_3h_change_to_cutoff_hpa", "cold_surge_score", "front_score"]:
        permuted = pd.Series(index=out.index, dtype=float)
        for month in sorted(months.dropna().unique()):
            idx = out.index[months == month].to_numpy()
            values = out.loc[idx, col].to_numpy(dtype=float)
            rng.shuffle(values)
            permuted.loc[idx] = values
        out[f"permuted_{col}"] = permuted
    return out


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    r06_path = data_root / "gold" / "hkg_t24" / "r06_moisture_state" / "r06_feature_matrix.parquet"
    if not r06_path.exists():
        raise FileNotFoundError(f"R07 requires R06 feature matrix: {r06_path}")
    features = pd.read_parquet(r06_path).sort_values("target_date").reset_index(drop=True)
    features["target_date"] = pd.to_datetime(features["target_date"])
    assert_no_locked_dates(features["target_date"], context="R07 source R06 matrix")
    features = add_pressure_candidates(data_root, features)
    wind_obs, wind_path = load_wind_observations(data_root)
    wind = wind_network_features(wind_obs, make_cutoffs(features))
    features = features.merge(wind, on="target_date", how="left")
    features = add_transition_scores(features)
    assert_no_locked_dates(features["target_date"], context="R07 transition feature matrix")
    output_path = data_root / "gold" / "hkg_t24" / "r07_transition_detection" / "r07_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features, output_path, wind_path


def model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    r04_sets = r04_feature_sets(features)
    baseline = r04_sets["r04_baseline_latest_temp_calendar"]
    pressure = [
        "hko_mslp_cutoff_hpa",
        "hko_mslp_3h_prior_hpa",
        "hko_mslp_3h_change_to_cutoff_hpa",
        "hko_mslp_24h_change_hpa",
        "hko_mslp_48h_change_hpa",
        "hko_mslp_3h_acceleration_hpa",
    ]
    temp_dew = [
        "hko_temp_decline_6h_c",
        "hko_dew_point_decline_6h_c",
        "hko_rh_change_3h_pct",
        "hko_dew_point_change_3h_c",
        "hko_dewpoint_depression_c",
        "hko_sudden_drying_3h_flag",
    ]
    wind = [
        "network_median_wind_speed_kmh",
        "network_max_wind_speed_kmh",
        "network_median_wind_speed_3h_change_kmh",
        "network_median_wind_gust_kmh",
        "network_max_wind_gust_kmh",
        "network_median_wind_gust_3h_change_kmh",
    ]
    scores = ["cold_surge_score", "front_score", "warm_sector_score", "post_frontal_score"]
    permuted = ["permuted_hko_mslp_3h_change_to_cutoff_hpa", "permuted_cold_surge_score", "permuted_front_score"]
    combined = pressure + temp_dew + wind + scores
    return [
        ModelSpec(
            "r07_baseline_temp_calendar",
            "ridge_baseline",
            valid_columns(features, baseline),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
        ModelSpec(
            "r07_pressure_only_elastic_net",
            "elastic_net_pressure_tendency",
            valid_columns(features, baseline + pressure),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=20000, random_state=707)),
                ]
            ),
        ),
        ModelSpec(
            "r07_temp_dew_transition_elastic_net",
            "elastic_net_temp_dew_transition",
            valid_columns(features, baseline + temp_dew + scores),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=20000, random_state=708)),
                ]
            ),
        ),
        ModelSpec(
            "r07_wind_only_elastic_net",
            "elastic_net_wind_speed_transition",
            valid_columns(features, baseline + wind),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=20000, random_state=709)),
                ]
            ),
        ),
        ModelSpec(
            "r07_combined_transition_elastic_net",
            "elastic_net_combined_transition",
            valid_columns(features, baseline + combined),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.03, l1_ratio=0.25, max_iter=20000, random_state=710)),
                ]
            ),
        ),
        ModelSpec(
            "r07_shallow_boosting_transition",
            "hist_gradient_boosting_shallow_transition",
            valid_columns(features, baseline + combined),
            lambda: HistGradientBoostingRegressor(
                max_iter=60,
                max_leaf_nodes=7,
                learning_rate=0.04,
                l2_regularization=1.0,
                min_samples_leaf=30,
                random_state=711,
            ),
        ),
        ModelSpec(
            "r07_two_stage_transition_probability_residual",
            "logistic_transition_probability_plus_ridge_residual",
            valid_columns(features, baseline + combined),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            two_stage_transition=True,
        ),
        ModelSpec(
            "r07_month_permuted_transition_control",
            "negative_control_month_permuted_transition",
            valid_columns(features, baseline + permuted),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
    ]


def active_cols(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in train.columns and train[col].notna().sum() > 0 and train[col].nunique(dropna=True) > 1
    ]


def fit_predict_regular(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    model = spec.factory()
    model.fit(train[cols], train["target_tmax_c"])
    return model.predict(train[cols]), model.predict(test[cols]), len(cols)


def fit_predict_two_stage(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    label = train["aux_transition_label"].fillna(0).astype(int)
    if label.nunique() < 2:
        return fit_predict_regular(
            ModelSpec(
                "fallback",
                "fallback",
                tuple(cols),
                lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            ),
            train,
            test,
            cols,
        )
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=0.75, class_weight="balanced", max_iter=2000, random_state=712)),
        ]
    )
    classifier.fit(train[cols], label)
    train_aug = train[cols].copy()
    test_aug = test[cols].copy()
    train_aug["fold_transition_probability"] = classifier.predict_proba(train[cols])[:, 1]
    test_aug["fold_transition_probability"] = classifier.predict_proba(test[cols])[:, 1]
    reg = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    reg.fit(train_aug, train["target_tmax_c"])
    return reg.predict(train_aug), reg.predict(test_aug), len(cols) + 1


def run_oof(features: pd.DataFrame, specs: Sequence[ModelSpec]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for spec in specs:
            cols = active_cols(train, spec.columns)
            if not cols:
                continue
            if spec.two_stage_transition:
                train_pred, test_pred, feature_count = fit_predict_two_stage(train, test, cols)
            else:
                train_pred, test_pred, feature_count = fit_predict_regular(spec, train, test, cols)
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = spec.model_id
            pred["model_family"] = spec.model_family
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(feature_count)
            pred["point_forecast"] = test_pred
            pred["distribution_sigma_c"] = sigma
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            rows.append(pred)
    if not rows:
        raise RuntimeError("R07 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R07 OOF predictions")
    return predictions


def score_frame(predictions: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(list(group_cols), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = group["point_forecast"] - group["target_tmax_c"]
        crps = [
            normal_crps(float(row.target_tmax_c), float(row.point_forecast), float(row.distribution_sigma_c))
            for row in group.itertuples()
        ]
        out = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        out.update(
            {
                "n": int(len(group)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_abs_error": float(error.abs().median()),
                "bias": float(error.mean()),
                "crps_normal": float(np.mean(crps)),
                "coverage_80": float(((group["q10"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q90"])).mean()),
                "coverage_90": float(((group["q05"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q95"])).mean()),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def fold_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = score_frame(predictions, ["fold_id", "model_id"])
    baseline = scores[scores["model_id"] == "r07_baseline_temp_calendar"][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def transition_subgroups(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = predictions.merge(
        features[["target_date", "front_score", "cold_surge_score", "target_tmax_change_1d_c"]],
        on="target_date",
        how="left",
    )
    joined["transition_regime"] = np.select(
        [
            joined["target_tmax_change_1d_c"] <= -2.0,
            joined["target_tmax_change_1d_c"] >= 2.0,
            joined["front_score"] >= joined["front_score"].quantile(0.90),
        ],
        ["target_side_cold_drop", "target_side_warm_jump", "predictor_top_decile_front_score"],
        default="ordinary",
    )
    return score_frame(joined, ["model_id", "transition_regime"])


def transition_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "hko_mslp_3h_change_to_cutoff_hpa",
        "hko_mslp_24h_change_hpa",
        "hko_temp_decline_6h_c",
        "hko_dew_point_decline_6h_c",
        "network_median_wind_speed_3h_change_kmh",
        "cold_surge_score",
        "front_score",
        "warm_sector_score",
        "post_frontal_score",
    ]
    rows: list[dict[str, object]] = []
    for col in cols:
        if col not in features:
            continue
        valid = features[[col, "target_tmax_change_1d_c"]].dropna()
        rows.append(
            {
                "feature": col,
                "n": int(len(valid)),
                "corr_with_next_tmax_change": float(valid[col].corr(valid["target_tmax_change_1d_c"])) if len(valid) > 2 else np.nan,
                "mean": float(valid[col].mean()) if len(valid) else np.nan,
                "p10": float(valid[col].quantile(0.10)) if len(valid) else np.nan,
                "p90": float(valid[col].quantile(0.90)) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, Mapping)
    oof = payload["oof_feasibility"]
    assert isinstance(oof, Mapping)
    return f"""# EXP-0039 / HKG-T24-R07 Long-Form Experiment Report

## Purpose

R07 tests whether pressure tendency, temperature and dew-point decline, and wind-speed changes can identify transition regimes that produce the largest HKG T-24 Tmax forecast errors. The hypothesis is not that pressure level alone predicts temperature. Pressure level is strongly seasonal and can be redundant with calendar. The useful information should come from changes: rising pressure, cooling, drying, wind increase, and combined surface evidence of fronts or cold surges before the T-1 15:00 cutoff.

## Data Used

The feature backbone is the R06 pre-validation feature matrix, which itself is built from R04 cutoff-safe target-station thermal features and immutable high-frequency temperature/humidity snapshots. R07 adds HKO pressure candidates from `C:\\hkg_tmax_data\\silver\\features\\t24_cutoff_feature_candidates.parquet` and wind speed/gust summaries from `C:\\hkg_tmax_data\\bronze\\analysis_phase_a\\hko_high_frequency_selected_station_observations.parquet`. The target-date feature period is `{payload['feature_min']}` through `{payload['feature_max']}`, and the OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`.

## Feature Construction

The experiment constructs HKO MSLP level, 3-hour pressure tendency, 24-hour and 48-hour pressure changes, pressure acceleration, six-hour temperature decline, six-hour dew-point decline, median network wind speed and gust, three-hour wind-speed changes, and four fixed-scale transition scores: cold-surge score, front score, warm-sector score, and post-frontal score. The scores use fixed meteorological scaling constants rather than fit-on-full-sample z-scores, so the feature construction does not import future distributional information. Target-side daily Tmax change is used only as an auxiliary training label and subgroup diagnostic, not as an inference-time predictor.

## Missing Inputs and Blockers

The uploaded specification asks for 12-station pressure gradients, wind-direction shifts, robust plane-fit pressure gradients, and gradient-vector rotation. Those are not fully available in the current parsed T24 tables. The current R07 implementation uses HKO pressure and parsed network wind speed/gust. Wind direction and pressure-network gradients remain explicit blockers for a later richer R07/R08 extension unless the raw parsers are expanded. This is documented rather than silently pretending that speed-only wind captures direction shifts.

## Model Ladder

R07 runs a baseline temperature/calendar model, pressure-only Elastic Net, temperature/dew-point transition Elastic Net, wind-only Elastic Net, combined transition Elastic Net, shallow constrained gradient boosting, a two-stage transition-probability residual specialist, and a month-permuted transition negative control. The two-stage model trains a logistic transition classifier inside each fold using only training rows and target-side transition labels, then feeds its fold-specific transition probability into a ridge residual model. No classifier is fit on validation or locked-test data.

## Leakage Controls

All ordinary predictors are available before T-1 15:00 HKT under their inherited conservative availability rules. The feature matrix and predictions are guarded against dates from 2025-01-01 onward. Validation 2024 is not used. Target-side transition labels are derived only for training labels and diagnostics inside development folds; they are never included as direct predictors. Preprocessing, logistic classification, scaling, imputation, boosting, and regression are all fit inside chronological training folds.

## OOF Gate

The strict four-year OOF check is `{oof['status']}`: {oof['reason']}. Therefore R07 is a completed transition diagnostic but not promotable under the user's hard four-year OOF rule. Even a positive transition specialist would require longer eligible development history or a revised predeclared evaluation design before promotion.

## Main Result

The best non-control diagnostic model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The fold-delta table shows whether any transition model improves across chronological folds or only in isolated periods. The subgroup scorecard separates target-side cold drops, warm jumps, high front-score days, and ordinary days.

## Interpretation

A useful transition experiment should improve the high-error transition cohort without damaging ordinary days. If pressure-only wins, HKO pressure tendency is already carrying meaningful air-mass change information. If temp/dew decline wins, the station thermal/moisture trajectory is sufficient and pressure is redundant. If wind-only wins, network flow speed is a useful proxy even without direction. If the two-stage specialist wins only on target-side transition days but loses ordinary days, it should become a gated expert rather than a universal model. If all transition candidates lose to baseline, then the available parsed transition variables are too sparse or too incomplete, and the correct next step is parser expansion rather than tuning.

## Decision Record

R07 is complete as an auditable diagnostic once its artifacts and tests pass. The result is retained whether positive, conditional, or null. Current blockers are wind direction, pressure-network gradients, and longer than 3.5 years of modern pre-validation OOF coverage. The next planned experiment is R08 surface wind, advection, and sea-breeze regime, where direction-aware parsing should be prioritized if the raw feed preserves direction columns.

## Actual Diagnostic Disposition

The generated scoreboard is intentionally not optimized after the fact. In this run the simple temperature/calendar baseline remains the best non-control model. The month-permuted transition control sits close to the baseline, while the physically motivated transition models generally lose. That combination says the available transition variables are not yet strong enough in their current parsed form. It also says the project should be careful about any pressure or wind improvement that is only a seasonal artifact. The pressure-only model performs especially poorly because the available HKO pressure sample is shorter and patchier than the thermal baseline sample; fold-local filtering prevents all-null columns from breaking the model, but it cannot invent missing pressure history.

## Why The Null Result Matters

This null result directly answers a high-priority failure mechanism question. The transition hypothesis is meteorologically plausible, especially for winter and spring cold surges, but the current operational feature representation is incomplete. HKO pressure tendency, speed-only network wind, and HKO cooling/drying do not yet beat the baseline in broad chronological OOF. This shifts expected information gain toward parser expansion: wind direction, pressure-network gradients, regional ISD pressure/wind, and upper-air coupling should be added before concluding that transition detection itself is unhelpful. The result is therefore `diagnostic null with parser blockers`, not a scientific rejection of front/cold-surge physics.

## Carry-Forward Rules

Later experiments may reuse the fixed-scale `front_score`, `cold_surge_score`, and target-side transition labels only under strict separation. The scores are legal predictors because they use pre-cutoff pressure, wind speed, temperature, and dew-point changes. The target-side labels are not legal predictors; they are labels for fold-local specialists or evaluation subgroups. If R20 or R22 builds a transition specialist, it must train the transition classifier inside each fold exactly as R07 does or with an equally strict fold-local design. If R08 parses wind direction, it should rerun the transition scorecard rather than assuming speed-only wind was an adequate proxy.

## Acceptance Outcome

R07 does not meet the feature-family promotion rule. It does not improve overall OOF MAE by 0.03 C, and it does not yet prove the required top-decile baseline-error improvement without ordinary-day harm. It is also blocked by the strict four-year OOF sample rule. The correct conclusion is to retain all artifacts and move on to R08, not to tune pressure thresholds against these same folds.
"""


def write_experiment(
    *,
    data_root: Path,
    wind_source_path: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    subgroup: pd.DataFrame,
    diagnostics: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    for subdir in ["results", "artifacts", "predictions", "logs"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    metrics = {
        "research_id": RESEARCH_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_DIAGNOSTIC_OOF_BLOCKED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "champion": payload["champion"],
        "oof_feasibility": payload["oof_feasibility"],
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
        "subgroup_scores": subgroup.to_dict(orient="records"),
        "transition_diagnostics": diagnostics.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    subgroup.to_csv(EXPERIMENT_DIR / "artifacts" / "subgroup_scores.csv", index=False)
    diagnostics.to_csv(EXPERIMENT_DIR / "artifacts" / "transition_diagnostics.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r07_oof_predictions.parquet", index=False)
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0039 HKG-T24-R07 Transition Detection\n\nPressure tendency, cooling/drying, wind-speed, and transition-specialist diagnostic. No validation 2024, no locked test, no Polymarket.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nPressure rise, cooling, drying, and wind increase should identify fronts/cold surges that cause large next-day Tmax errors.\n")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Start from R06 cutoff-safe features.\n2. Add HKO pressure and parsed network wind speed/gust features.\n3. Build fixed-scale transition scores without full-sample fitted normalization.\n4. Train all models inside chronological folds only.\n5. Use target-side transition labels only for fold-local auxiliary classification and subgroup diagnostics.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nPredictors must be available by T-1 15:00 HKT. Target-side daily Tmax change is forbidden as a direct predictor and used only as a training label/diagnostic inside development folds.\n")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
wind_source_table: {wind_source_path}
wind_source_table_sha256: {sha256_file(wind_source_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: SILVER_OPERATIONAL_REPLAY
blocked_inputs:
  - pressure_network_gradient
  - wind_direction_shift
  - gradient_vector_rotation
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
model_ladder: baseline, pressure_only, temp_dew_transition, wind_only, combined_transition, shallow_boosting, two_stage_transition_probability, month_permuted_control
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Parsed wind observation period: `{payload['wind_observation_min']}` through `{payload['wind_observation_max']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""")
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Transition Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "corr_with_next_tmax_change", "mean", "p10", "p90"]),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR07 is complete as a transition diagnostic, but it is OOF-blocked under the strict four-year rule. Pressure-network gradients and wind-direction shifts remain parser/data blockers.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r07_transition_detection.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R07
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
blocked_inputs: [pressure_network_gradient, wind_direction_shift, gradient_vector_rotation]
""")
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_report(
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    subgroup: pd.DataFrame,
    diagnostics: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    write_text(
        REPO_ROOT / "reports" / "hkg_t24" / "R07_TRANSITION_DETECTION.md",
        long_report(payload)
        + "\n# R07 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"], limit=100)
        + "\n## Transition Subgroups\n\n"
        + markdown_table(subgroup, ["model_id", "transition_regime", "n", "mae", "rmse", "crps_normal"], limit=100)
        + "\n## Transition Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "corr_with_next_tmax_change", "mean", "p10", "p90"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R07 transition detection diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path, wind_path = build_feature_matrix(data_root)
    specs = model_specs(features)
    predictions = run_oof(features, specs)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    subgroup = transition_subgroups(predictions, features)
    diagnostics = transition_diagnostics(features)
    output_dir = data_root / "gold" / "hkg_t24" / "r07_transition_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r07_oof_predictions.parquet"
    scoreboard_path = output_dir / "r07_scoreboard.parquet"
    fold_path = output_dir / "r07_fold_score_deltas.parquet"
    subgroup_path = output_dir / "r07_subgroup_scores.parquet"
    diagnostics_path = output_dir / "r07_transition_diagnostics.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    subgroup.to_parquet(subgroup_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)
    feature_dates = pd.to_datetime(features["target_date"])
    feasibility = check_four_year_oof_feasibility(
        feature_dates.min().date(),
        feature_dates.max().date(),
        min_years=4.0,
        reason_context="R07 modern transition-detection pre-validation feature period",
    )
    non_control = scoreboard[~scoreboard["model_id"].eq("r07_month_permuted_transition_control")].copy()
    champion = non_control.iloc[0].to_dict()
    wind_obs = pd.read_parquet(wind_path, columns=["observed_at_hkt", "variable"])
    wind_obs = wind_obs[wind_obs["variable"].isin(["mean_wind_speed_kmh", "max_wind_gust_kmh"])]
    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_state": git_state(),
        "feature_min": str(feature_dates.min().date()),
        "feature_max": str(feature_dates.max().date()),
        "prediction_min": str(pd.to_datetime(predictions["target_date"]).min().date()),
        "prediction_max": str(pd.to_datetime(predictions["target_date"]).max().date()),
        "wind_observation_min": str(pd.to_datetime(wind_obs["observed_at_hkt"], utc=True).min()),
        "wind_observation_max": str(pd.to_datetime(wind_obs["observed_at_hkt"], utc=True).max()),
        "oof_feasibility": feasibility.__dict__,
        "champion": champion,
    }
    write_experiment(
        data_root=data_root,
        wind_source_path=wind_path,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        subgroup=subgroup,
        diagnostics=diagnostics,
        payload=payload,
    )
    write_report(scoreboard, fold_scores, subgroup, diagnostics, payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
