
"""Experiment definitions and runners for the exp30 sweep."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from weather_ml import artifacts

from . import analogs, eval as eval_mod, features, models
from .config import (
    DEFAULT_SEEDS,
    GUIDANCE_COLS,
    MOS_ALL_CODES,
    MOS_CLOUD_CODES,
    MOS_PRECIP_CODES,
    MOS_SURFACE_CODES,
    MOS_THERMO_CODES,
    MOS_WIND_CODES,
)

LOGGER = logging.getLogger(__name__)

SEEDS_OVERRIDE: list[int] | None = None
BASELINE_CANDIDATES: list[str] = ["lgbm", "xgb", "catboost"]


def set_runtime_config(
    *,
    seeds_override: list[int] | None = None,
    baseline_candidates: list[str] | None = None,
) -> None:
    global SEEDS_OVERRIDE, BASELINE_CANDIDATES
    if seeds_override is not None:
        SEEDS_OVERRIDE = list(seeds_override)
    if baseline_candidates is not None:
        BASELINE_CANDIDATES = list(baseline_candidates)


def get_seeds() -> list[int]:
    return SEEDS_OVERRIDE or list(DEFAULT_SEEDS)


EXPERIMENT_DESCRIPTIONS = {
    "RobustMultiWindowBiasStack": (
        "Adds trimmed and winsorized rolling bias estimates at 7/14/30/60/120-day windows "
        "for each guidance source, plus drift diagnostics that compare short and long windows. "
        "Bias and drift signals are clipped and paired with insufficient-history flags so the "
        "model can adapt quickly without overreacting to noisy, thin samples during regime shifts."
    ),
    "EWMABiasCorrHalfLife": (
        "Introduces exponentially weighted bias, MAE, and correlation features for each guidance "
        "source using half-lives of 7, 21, and 60 days. The decay parameter provides a tunable "
        "adaptation rate while shrinkage by effective sample size keeps correlations stable when "
        "history is thin. Includes disagreement features between EWMA bias and rm60 bias."
    ),
    "DriftTriggeredBiasGating": (
        "Builds gated bias estimates that blend short-window and long-window biases based on a "
        "smooth drift indicator. When short and long biases disagree, the gate shifts weight toward "
        "the short window; otherwise it favors the stable long window. Global gate summaries provide "
        "a regime signal without hard thresholds, keeping the rest of the baseline unchanged."
    ),
    "RobustRankCorrelationFeatures": (
        "Adds rolling Spearman rank correlations between each guidance source and truth over 30/60/120-day windows, "
        "with shrinkage and low-variance flags to prevent noisy correlations from dominating. Rank-based reliability "
        "signals aim to be more robust to outliers and low-variance periods than raw Pearson correlations, while "
        "preserving interpretability for reliability-aware corrections."
    ),
    "KalmanBiasStatePerGuidance": (
        "Replaces or augments fixed-window bias estimates with a Kalman-style latent bias state per guidance source. "
        "The filter updates a bias state and uncertainty daily using past errors, producing smoother yet responsive "
        "bias signals plus gain/uncertainty diagnostics. Grid-searching Q/R parameters tests whether a state-space "
        "bias model improves stability under drift without leakage."
    ),
    "RollingMAEWeightedEnsembleSignals": (
        "Computes rolling MAE per guidance source and turns those errors into reliability weights for a dynamic, weighted "
        "ensemble mean and spread. Entropy and best-model indicators capture concentration of weights, while caps prevent a "
        "single model from dominating. These features provide drift-aware reliability signals without peeking, improving "
        "behavior when one guidance source degrades."
    ),
    "MOSSurfaceContextAnomalies": (
        "Adds MOS near-surface context variables (tmp, dpt, wsp, wdr, cig, vis) along with trailing median/IQR statistics "
        "and anomaly features over 7 and 30 days. Missingness counts and wind-direction sin/cos encodings provide context "
        "about MOS availability and flow regime. The goal is to capture systematic tmax errors explained by local MOS context."
    ),
    "MOSDewpointDepressionWindComponents": (
        "Derives MOS dewpoint depression and wind-vector components to capture physically meaningful drivers of daytime "
        "heating. Trailing anomalies over 7/30 days and bounded interaction terms with spread or guidance bias test whether "
        "thermodynamic and flow proxies can explain residual structure. Missingness flags ensure the model can downweight "
        "noisy or absent MOS signals."
    ),
    "MOSCloudPrecipProxyFeatures": (
        "Adds MOS cloud and precipitation proxy features (cig/vis transforms plus p06/p12/q06/q12/pos/poz summaries) with "
        "robust log transforms and missingness flags. Precipitation and cloudiness proxies are combined into max/mean signals "
        "and limited interactions with spread and bias to capture tmax bust days without overfitting sparse MOS codes."
    ),
    "MOSMissingnessQualityEncoding": (
        "Encodes MOS availability as a structured signal, including per-code missingness flags, counts, fractions, and "
        "days-since-last-available metrics capped at 60 days. A PCA embedding of the missingness matrix (fit on training only) "
        "captures correlated MOS gaps. This tests whether MOS availability itself conveys product quality or timing effects."
    ),
    "MOSCodeSubsetSelectionWithTimeSafeAblation": (
        "Runs a bounded sweep over physically motivated MOS code subsets (thermo, wind, cloud/vis, precip proxy, and full) "
        "using the same rolling-anomaly construction and strict time-ordered selection. Only a small retuning budget is allowed, "
        "so the comparison isolates which MOS subsets add real signal without inviting overfit. The best subset is then evaluated."
    ),
    "MOSResidualizedContextSignals": (
        "Implements a two-stage residual correction where Baseline B produces the primary forecast and a second model predicts "
        "residuals from MOS context features and stability signals. Residual targets are built from forward-chaining OOF predictions "
        "to avoid leakage. A conservative second-stage model adds MOS context only when it consistently explains residual structure."
    ),
    "SpreadRegimeMixtureOfExperts": (
        "Defines low/medium/high spread regimes based on training-only thresholds of guidance disagreement and trains either a "
        "single model with regime indicators or separate expert models per regime. This leakage-safe regime detection uses only "
        "predictor spread and aims to capture different correction rules when guidance agrees versus disagrees."
    ),
    "SeasonAwareModelWithWalkForwardClimatology": (
        "Adds a walk-forward day-of-year climatology (median and IQR) computed strictly from past truth plus season one-hot flags. "
        "An anomaly feature comparing ensemble mean to climatology provides an anchor for seasonal cycles. The model remains leakage-safe "
        "by using only historical dates for climatology and can optionally test season-specific experts."
    ),
    "PredictedTempDecileConditionalCorrection": (
        "Builds predicted-temperature regimes using deciles of the ensemble mean computed on the training period only. Decile one-hot features "
        "or bucketed expert models allow distinct correction behavior for cool, normal, and hot days without using future truth. This targets "
        "systematic error differences across the temperature range while keeping the rest of the pipeline unchanged."
    ),
    "BustProbabilityGatedShrinkage": (
        "Trains a leakage-safe classifier for large-error days using OOF Baseline B predictions and guidance spread features, then shrinks the "
        "final forecast toward a robust median guidance when bust probability is high. The shrinkage weight is tuned by validation MAE, keeping "
        "the classifier purely as a risk signal rather than a replacement forecaster."
    ),
    "WindQuadrantConditionalBiasSwitch": (
        "Adds wind-direction regime features derived from MOS wind direction by quadrant, plus optional sin/cos encodings. Bias features are allowed "
        "to interact with wind regimes so the model can learn different corrections under onshore or offshore flow. Missingness flags guard against noisy "
        "wind direction data while preserving leakage-safe conditioning."
    ),
    "LightGBMMAEObjectivedirect": (
        "Keeps the strongest baseline feature set but changes the LightGBM objective to L1 (regression_l1) to target MAE directly. The tuning budget is "
        "kept fixed while exploring smaller learning rates and more estimators, testing whether a median-optimized objective yields lower absolute error "
        "without altering the feature pipeline or selection rules."
    ),
    "XGBRegAbsoluteErrorMAE": (
        "Uses the Baseline B feature set with XGBoost and the reg:absoluteerror objective, aligning optimization with MAE. The search space mirrors the "
        "baseline tuning budget, allowing a fair comparison between L1-optimized XGBoost and other GBDT families under the same data and split rules. "
        "This isolates objective choice while keeping feature engineering and evaluation protocols fixed."
    ),
    "XGBPseudoHuberDeltaTuning": (
        "Runs XGBoost with the reg:pseudohubererror objective and tunes the huber_slope parameter to balance robustness and smooth gradients. This tests "
        "whether a pseudo-Huber loss can stabilize training on outliers while still producing lower absolute error, without changing the baseline features. "
        "The intent is to keep training behavior stable while remaining MAE-focused under nonstationary errors."
    ),
    "XGBQuantileMedianAndMultiQuantileMeanAssist": (
        "Trains XGBoost quantile models, using the median (alpha=0.5) as the point forecast and optionally learning additional quantiles for uncertainty. "
        "A multi-quantile variant derives inter-quantile spread and can fit a small combiner over q20/q50/q80 to improve MAE. Selection remains by validation MAE. "
        "This evaluates whether quantile structure adds signal without altering the core feature set."
    ),
    "CatBoostRobustLossSweep": (
        "Runs a controlled CatBoost sweep over robust loss functions (MAE, Huber, Quantile, LogCosh, FairLoss) on the Baseline B feature set. The same tuning budget "
        "and time-ordered validation are used to determine whether a different loss better generalizes for absolute error without altering data or features. "
        "Loss selection is purely validation-driven and keeps model capacity comparable to the baseline."
    ),
    "ForwardChainingStackedEnsemble": (
        "Builds a strict forward-chaining stacked ensemble using OOF predictions from LightGBM, XGBoost, and Ridge base learners. A ridge meta-learner combines the base predictions, "
        "optionally adding spread stability signals, with fold gaps to prevent leakage. The stack is evaluated once on the canonical test period after refitting base learners."
    ),
    "ConstrainedDynamicBlendWeights": (
        "Implements a constrained, drift-adaptive blend where guidance weights are updated daily from rolling MAE and clipped to avoid domination by a single model. The dynamic blend can be used directly or added as a feature to the baseline model. A small grid over weight-decay parameters tests whether adaptive blending outperforms static ensembles without full retraining."
    ),
    "TwoStageResidualStackWithBiasFeatures": (
        "Adds a second-stage residual corrector trained on OOF Baseline B residuals with a compact feature set (rolling biases/corr, spread, limited MOS proxies, and calendar). The correction is capped to avoid destabilizing the base forecast. This tests whether a conservative residual model can add value without overfitting or leaking information across time splits."
    ),
    "MultiSnapshotAggregationAndEnsemble": (
        "Aggregates multiple as-of snapshots per target day using leakage-safe rules (latest, mean of last M, and trends) and trains the baseline model on a consolidated one-row-per-day dataset. Snapshot counts and update deltas capture within-day forecast evolution. This experiment evaluates whether using multiple snapshots improves forecast accuracy while keeping as-of constraints intact."
    ),
    "GuidanceSpacekNNAnalogPrediction": (
        "Adds kNN analog predictions based on guidance and spread features, using only historical days with matching as-of buckets. Weighted analog means, distance quantiles, and effective neighbor counts act as additional signals for the baseline model. This tests whether nearest-neighbor analogs capture nonlinear patterns missed by tree models while maintaining strict no-leakage rules."
    ),
    "ResidualAnalogCorrectionOnBaselineB": (
        "Computes kNN analog corrections on Baseline B residuals using only past days and OOF residual targets, then applies a capped correction to the baseline forecast. "
        "Analog distance diagnostics provide confidence signals. This aims to capture repeatable residual patterns without letting in-sample residuals leak into the correction step. "
        "The correction is intentionally small and bounded so the baseline remains dominant when analogs are weak."
    ),
    "PrototypeAnalogsKMedoidsWithForwardUpdate": (
        "Fits a small set of prototype guidance patterns on the training period and uses prototype residual means as correction signals. Each day is assigned to the nearest prototype, producing a stable analog correction plus distance diagnostics. The prototypes are fit only on past data and then held fixed, offering a more stable alternative to full kNN."
    ),
    "LocalLinearKNNRidgeForecaster": (
        "Fits a local ridge regression on nearest-neighbor guidance patterns to produce a locally linear analog forecast. The local prediction can be blended with the baseline forecast or used as an additional feature, with neighbor weights inversely proportional to distance. This tests whether local linear structure captures bias patterns that global tree models miss."
    ),
}


@dataclass
class BaselineResult:
    model_name: str
    params: dict
    metrics: dict
    val_slices: dict
    test_slices: dict
    pred_train: np.ndarray
    pred_val: np.ndarray
    pred_test: np.ndarray


@dataclass
class ExperimentContext:
    df: pd.DataFrame
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    base_features: pd.DataFrame
    base_feature_columns: list[str]
    baseline_b: BaselineResult
    group_key: pd.Series
    group_key_asof: pd.Series
    rng: np.random.Generator
    run_root: Path
    split_ref: dict


@dataclass
class ExperimentSpec:
    experiment_id: str
    name: str
    description: str
    runner: Callable[[ExperimentContext], dict]



def _save_run_artifacts(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(payload.get("metrics", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "feature_list.json").write_text(
        json.dumps(payload.get("feature_columns", []), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "experiment_meta.json").write_text(
        json.dumps(payload.get("experiment_meta", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if payload.get("predictions_csv"):
        (run_dir / "predictions_test.csv").write_text(
            payload["predictions_csv"], encoding="utf-8"
        )
    artifacts.write_hash_manifest(
        [
            run_dir / "metrics.json",
            run_dir / "feature_list.json",
            run_dir / "experiment_meta.json",
        ],
        run_dir / "hashes.json",
    )


def _build_predictions_csv(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    rows = pd.DataFrame(
        {
            "station_id": df["station_id"].astype(str),
            "target_date_local": df["target_date_local"].astype(str),
            "asof_utc": df["asof_utc"].astype(str),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return rows.to_csv(index=False)


def _prepare_arrays(ctx: ExperimentContext, feature_df: pd.DataFrame):
    X_train = feature_df.loc[ctx.train_mask].to_numpy(dtype=float)
    y_train = ctx.df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = feature_df.loc[ctx.val_mask].to_numpy(dtype=float)
    y_val = ctx.df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = feature_df.loc[ctx.test_mask].to_numpy(dtype=float)
    y_test = ctx.df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)
    full_mask = ctx.train_mask | ctx.val_mask
    X_train_full = feature_df.loc[full_mask].to_numpy(dtype=float)
    y_train_full = ctx.df.loc[full_mask, "actual_tmax_f"].to_numpy(dtype=float)
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    )


def _train_model_with_search(
    *,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: np.random.Generator,
    trials: int,
    objective: str,
) -> models.SearchResult:
    if model_name == "lgbm" and objective == "regression_l1":
        sampler = lambda r: models.sample_lgbm_l1_params(r)
    elif model_name == "lgbm":
        sampler = lambda r: models.sample_lgbm_params(r, objective)
    elif model_name == "xgb":
        sampler = lambda r: models.sample_xgb_params(r, objective)
    elif model_name == "catboost":
        sampler = lambda r: models.sample_catboost_params(r, objective)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return models.train_with_search(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seeds=get_seeds(),
        trials=trials,
        param_sampler=sampler,
        rng=rng,
    )


def _fit_best_model(
    *,
    model_name: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    return models.refit_model(
        model_name=model_name,
        params=params,
        X_train=X_train,
        y_train=y_train,
        seed=get_seeds()[0],
    )


def _compute_slices(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "monthly": eval_mod.slice_mae_by_month(df, y_true, y_pred),
        "seasonal": eval_mod.slice_mae_by_season(df, y_true, y_pred),
        "decile": eval_mod.slice_mae_by_decile(df, y_true, y_pred),
    }


def _assemble_result(
    *,
    ctx: ExperimentContext,
    experiment_id: str,
    name: str,
    description: str,
    model_name: str,
    params: dict,
    feature_columns: list[str],
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
) -> dict:
    y_train = ctx.df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    y_val = ctx.df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    y_test = ctx.df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    metrics = {
        "train": eval_mod.regression_metrics(y_train, pred_train),
        "validation": eval_mod.regression_metrics(y_val, pred_val) if len(y_val) else {},
        "test": eval_mod.regression_metrics(y_test, pred_test),
    }
    val_slices = (
        _compute_slices(ctx.df.loc[ctx.val_mask], y_val, pred_val) if len(y_val) else {}
    )
    test_slices = _compute_slices(ctx.df.loc[ctx.test_mask], y_test, pred_test)
    deltas = eval_mod.apply_deltas(ctx.baseline_b.metrics["test"], metrics["test"])
    worth = eval_mod.worth_testing(
        ctx.baseline_b.val_slices.get("monthly", {}),
        ctx.baseline_b.val_slices.get("seasonal", {}),
        val_slices.get("monthly", {}),
        val_slices.get("seasonal", {}),
        metrics["validation"].get("mae", np.nan)
        - ctx.baseline_b.metrics["validation"].get("mae", np.nan),
    )

    run_dir = ctx.run_root / experiment_id
    payload = {
        "experiment_id": experiment_id,
        "name": name,
        "description": description,
        "model_family": model_name,
        "model_params": params,
        "feature_columns": feature_columns,
        "metrics": {
            **metrics,
            "val_slices": val_slices,
            "test_slices": test_slices,
            "deltas_vs_baseline": deltas,
            "worth_testing": worth,
        },
        "experiment_meta": {
            "experiment_id": experiment_id,
            "name": name,
            "model_family": model_name,
            "model_params": params,
            "feature_columns": feature_columns,
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "predictions_csv": _build_predictions_csv(
            ctx.df.loc[ctx.test_mask], y_test, pred_test
        ),
    }
    _save_run_artifacts(run_dir, payload)
    return {
        "experiment_id": experiment_id,
        "name": name,
        "description": description,
        "model_family": model_name,
        "model_params": params,
        "feature_columns": feature_columns,
        "metrics": payload["metrics"],
        "run_dir": str(run_dir),
    }


def _compute_oof_predictions(
    ctx: ExperimentContext,
    feature_df: pd.DataFrame,
    folds: list[tuple[str, str, str, str]],
) -> np.ndarray:
    df = ctx.df
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    oof = np.full(len(df), np.nan, dtype=float)
    for train_start, train_end, val_start, val_end in folds:
        train_mask = (dates >= pd.to_datetime(train_start).date()) & (
            dates <= pd.to_datetime(train_end).date()
        )
        val_mask = (dates >= pd.to_datetime(val_start).date()) & (
            dates <= pd.to_datetime(val_end).date()
        )
        X_train = feature_df.loc[train_mask].to_numpy(dtype=float)
        y_train = df.loc[train_mask, "actual_tmax_f"].to_numpy(dtype=float)
        X_val = feature_df.loc[val_mask].to_numpy(dtype=float)
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        model = _fit_best_model(
            model_name=ctx.baseline_b.model_name,
            params=ctx.baseline_b.params,
            X_train=X_train,
            y_train=y_train,
        )
        oof[val_mask] = model.predict(X_val)
    return oof


def _baseline_pred_all(ctx: ExperimentContext) -> np.ndarray:
    preds = np.full(len(ctx.df), np.nan, dtype=float)
    preds[ctx.train_mask] = ctx.baseline_b.pred_train
    preds[ctx.val_mask] = ctx.baseline_b.pred_val
    preds[ctx.test_mask] = ctx.baseline_b.pred_test
    return preds


def _run_tree_experiment(
    ctx: ExperimentContext,
    *,
    experiment_id: str,
    name: str,
    description: str,
    feature_df: pd.DataFrame,
    model_name: str,
    objective: str,
    trials: int,
) -> dict:
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, feature_df)
    search = _train_model_with_search(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        rng=ctx.rng,
        trials=trials,
        objective=objective,
    )
    model_train = _fit_best_model(
        model_name=search.model_name,
        params=search.params,
        X_train=X_train,
        y_train=y_train,
    )
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val) if len(X_val) else np.array([])

    model_full = _fit_best_model(
        model_name=search.model_name,
        params=search.params,
        X_train=X_train_full,
        y_train=y_train_full,
    )
    pred_test = model_full.predict(X_test)
    return _assemble_result(
        ctx=ctx,
        experiment_id=experiment_id,
        name=name,
        description=description,
        model_name=search.model_name,
        params=search.params,
        feature_columns=list(feature_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def _baseline_b_features(
    df: pd.DataFrame,
    group_key: pd.Series,
    train_mask: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    base = features.build_guidance_base(df, GUIDANCE_COLS)
    ensemble = features.build_ensemble_features(df, GUIDANCE_COLS)
    base = pd.concat([base, ensemble], axis=1)
    base["gefsatmos_tmp_spread_f"] = df["gefsatmos_tmp_spread_f"].astype(float)
    calendar_df = features.add_calendar_features(df)[
        ["month", "sin_doy", "cos_doy", "is_weekend", "asof_sin_hour", "asof_cos_hour"]
    ]
    base = pd.concat([base, calendar_df], axis=1)

    bias_rm = features.compute_bias_features(
        df,
        GUIDANCE_COLS,
        windows=[7, 30, 60],
        group_key=group_key,
        suffix="_l2",
    )
    corr_rm = features.compute_corr_features(
        df,
        GUIDANCE_COLS,
        windows=[60],
        group_key=group_key,
        suffix="_l2",
    )
    base = pd.concat([base, bias_rm.frame, corr_rm.frame], axis=1)

    base = base.join(df[["station_id"]])
    base = features.add_station_onehot(base)
    base = base.drop(columns=["station_id"], errors="ignore")
    return base, list(base.columns)


def run_baseline_b(ctx: ExperimentContext) -> BaselineResult:
    LOGGER.info("BASELINE_B_START")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, ctx.base_features)
    candidates = list(BASELINE_CANDIDATES)
    LOGGER.info("BASELINE_B_CANDIDATES %s", candidates)
    best = None
    for name in candidates:
        if name == "xgb":
            objective = "reg:squarederror"
        elif name == "catboost":
            objective = "RMSE"
        else:
            objective = "regression"
        try:
            result = _train_model_with_search(
                model_name=name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                rng=ctx.rng,
                trials=40,
                objective=objective,
            )
            LOGGER.info(
                "BASELINE_B_CANDIDATE_DONE model=%s median_val_mae=%.4f",
                name,
                result.median_val_mae,
            )
            if best is None or result.median_val_mae < best.median_val_mae:
                best = result
        except Exception as exc:
            LOGGER.warning("Baseline candidate %s failed: %s", name, exc)
    if best is None:
        raise RuntimeError("Baseline B search failed.")

    model_train = _fit_best_model(
        model_name=best.model_name,
        params=best.params,
        X_train=X_train,
        y_train=y_train,
    )
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val)

    model_full = _fit_best_model(
        model_name=best.model_name,
        params=best.params,
        X_train=X_train_full,
        y_train=y_train_full,
    )
    pred_test = model_full.predict(X_test)

    metrics = {
        "train": eval_mod.regression_metrics(y_train, pred_train),
        "validation": eval_mod.regression_metrics(y_val, pred_val),
        "test": eval_mod.regression_metrics(y_test, pred_test),
    }
    val_slices = _compute_slices(ctx.df.loc[ctx.val_mask], y_val, pred_val)
    test_slices = _compute_slices(ctx.df.loc[ctx.test_mask], y_test, pred_test)
    return BaselineResult(
        model_name=best.model_name,
        params=best.params,
        metrics=metrics,
        val_slices=val_slices,
        test_slices=test_slices,
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_robust_multiwindow_bias_stack(ctx: ExperimentContext) -> dict:
    trimmed = features.compute_trimmed_winsor_bias(
        ctx.df,
        GUIDANCE_COLS,
        windows=[7, 14, 30, 60, 120],
        group_key=ctx.group_key,
    )
    drift = features.compute_drift_features(trimmed.frame, GUIDANCE_COLS)
    feature_df = pd.concat([ctx.base_features, trimmed.frame, drift.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="RobustMultiWindowBiasStack",
        name="Robust multi-window bias stack",
        description=EXPERIMENT_DESCRIPTIONS["RobustMultiWindowBiasStack"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_ewma_bias_corr(ctx: ExperimentContext) -> dict:
    ewma = features.compute_ewma_features(
        ctx.df,
        GUIDANCE_COLS,
        halflives=[7, 21, 60],
        group_key=ctx.group_key,
    )
    feature_df = pd.concat([ctx.base_features, ewma.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="EWMABiasCorrHalfLife",
        name="EWMA bias/corr half-life",
        description=EXPERIMENT_DESCRIPTIONS["EWMABiasCorrHalfLife"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_drift_triggered_gating(ctx: ExperimentContext) -> dict:
    best = None
    best_result = None
    for threshold in [0.5, 1.0, 1.5]:
        for slope in [0.25, 0.5]:
            gated = features.compute_gated_bias_features(
                ctx.df,
                GUIDANCE_COLS,
                threshold=threshold,
                slope=slope,
            )
            feature_df = pd.concat([ctx.base_features, gated.frame], axis=1)
            result = _run_tree_experiment(
                ctx,
                experiment_id="DriftTriggeredBiasGating",
                name="Drift-triggered bias gating",
                description=EXPERIMENT_DESCRIPTIONS["DriftTriggeredBiasGating"],
                feature_df=feature_df,
                model_name="lgbm",
                objective="regression",
                trials=20,
            )
            val_mae = result["metrics"]["validation"]["mae"]
            if best is None or val_mae < best:
                best = val_mae
                best_result = result
    return best_result


def run_rank_correlation(ctx: ExperimentContext) -> dict:
    spearman = features.compute_spearman_features(
        ctx.df,
        GUIDANCE_COLS,
        windows=[30, 60, 120],
        group_key=ctx.group_key,
    )
    feature_df = pd.concat([ctx.base_features, spearman.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="RobustRankCorrelationFeatures",
        name="Robust rank correlation",
        description=EXPERIMENT_DESCRIPTIONS["RobustRankCorrelationFeatures"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_kalman_bias(ctx: ExperimentContext) -> dict:
    best_result = None
    best_mae = None
    for q in [0.01, 0.05, 0.1, 0.2]:
        for r in [0.25, 1.0, 4.0]:
            kalman = features.compute_kalman_bias(
                ctx.df,
                GUIDANCE_COLS,
                q=q,
                r=r,
                p0=1.0,
                group_key=ctx.group_key,
            )
            feature_df = pd.concat([ctx.base_features, kalman.frame], axis=1)
            result = _run_tree_experiment(
                ctx,
                experiment_id="KalmanBiasStatePerGuidance",
                name="Kalman bias state",
                description=EXPERIMENT_DESCRIPTIONS["KalmanBiasStatePerGuidance"],
                feature_df=feature_df,
                model_name="lgbm",
                objective="regression",
                trials=20,
            )
            val_mae = result["metrics"]["validation"]["mae"]
            if best_mae is None or val_mae < best_mae:
                best_mae = val_mae
                best_result = result
    return best_result


def run_mae_weighted_ensemble(ctx: ExperimentContext) -> dict:
    mae_weighted = features.compute_rolling_mae_weights(
        ctx.df,
        GUIDANCE_COLS,
        windows=[30, 60],
        group_key=ctx.group_key,
        train_mask=ctx.train_mask,
    )
    feature_df = pd.concat([ctx.base_features, mae_weighted.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="RollingMAEWeightedEnsembleSignals",
        name="Rolling MAE weighted ensemble",
        description=EXPERIMENT_DESCRIPTIONS["RollingMAEWeightedEnsembleSignals"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_mos_surface_anoms(ctx: ExperimentContext) -> dict:
    mos = features.compute_mos_surface_anomalies(
        ctx.df,
        codes=MOS_SURFACE_CODES,
        group_key=ctx.group_key,
    )
    feature_df = pd.concat([ctx.base_features, mos.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="MOSSurfaceContextAnomalies",
        name="MOS surface context anomalies",
        description=EXPERIMENT_DESCRIPTIONS["MOSSurfaceContextAnomalies"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_mos_dewpoint_wind(ctx: ExperimentContext) -> dict:
    mos = features.compute_mos_dewpoint_wind(ctx.df, ctx.group_key)
    spread = ctx.df["gefsatmos_tmp_spread_f"].astype(float).clip(0.0, 20.0)
    bias_nbm = ctx.base_features.get(
        "bias_nbm_tmax_f_rm60_l2", pd.Series(0.0, index=ctx.df.index)
    )
    mos.frame["dd_x_spread"] = mos.frame["mos_dd"].clip(0.0, 30.0) * spread
    mos.frame["u_x_bias_nbm"] = mos.frame["mos_u"] * bias_nbm.clip(-5.0, 5.0)
    feature_df = pd.concat([ctx.base_features, mos.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="MOSDewpointDepressionWindComponents",
        name="MOS dewpoint depression & wind",
        description=EXPERIMENT_DESCRIPTIONS["MOSDewpointDepressionWindComponents"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_mos_cloud_precip(ctx: ExperimentContext) -> dict:
    mos = features.compute_mos_cloud_precip(ctx.df)
    spread = ctx.df["gefsatmos_tmp_spread_f"].astype(float).clip(0.0, 20.0)
    bias_gefs = ctx.base_features.get(
        "bias_gefsatmosmean_tmax_f_rm60_l2", pd.Series(0.0, index=ctx.df.index)
    )
    mos.frame["precip_x_spread"] = mos.frame["mos_precip_proxy_max"] * spread
    mos.frame["log_cig_x_bias_gefs"] = mos.frame["mos_log_cig"] * bias_gefs.clip(-5.0, 5.0)
    feature_df = pd.concat([ctx.base_features, mos.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="MOSCloudPrecipProxyFeatures",
        name="MOS cloud/precip proxy",
        description=EXPERIMENT_DESCRIPTIONS["MOSCloudPrecipProxyFeatures"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_mos_missingness(ctx: ExperimentContext) -> dict:
    mos = features.compute_mos_missingness(
        ctx.df, MOS_ALL_CODES, ctx.group_key, ctx.train_mask
    )
    spread = ctx.df["gefsatmos_tmp_spread_f"].astype(float).clip(0.0, 20.0)
    mos.frame["mos_missing_frac_x_spread"] = mos.frame.get("mos_missing_frac", 0.0) * spread
    feature_df = pd.concat([ctx.base_features, mos.frame], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="MOSMissingnessQualityEncoding",
        name="MOS missingness quality encoding",
        description=EXPERIMENT_DESCRIPTIONS["MOSMissingnessQualityEncoding"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_mos_subset_sweep(ctx: ExperimentContext) -> dict:
    subsets = {
        "S1": MOS_THERMO_CODES,
        "S2": MOS_WIND_CODES,
        "S3": MOS_CLOUD_CODES,
        "S4": MOS_PRECIP_CODES,
        "S5": MOS_SURFACE_CODES + MOS_PRECIP_CODES,
    }
    best_result = None
    best_mae = None
    subset_metrics = {}
    for sid, codes in subsets.items():
        mos = features.compute_mos_surface_anomalies(
            ctx.df,
            codes=codes,
            group_key=ctx.group_key,
        )
        feature_df = pd.concat([ctx.base_features, mos.frame], axis=1)
        result = _run_tree_experiment(
            ctx,
            experiment_id="MOSCodeSubsetSelectionWithTimeSafeAblation",
            name="MOS subset sweep",
            description=EXPERIMENT_DESCRIPTIONS[
                "MOSCodeSubsetSelectionWithTimeSafeAblation"
            ],
            feature_df=feature_df,
            model_name="lgbm",
            objective="regression",
            trials=10,
        )
        mae = result["metrics"]["validation"]["mae"]
        subset_metrics[sid] = {"codes": codes, "val_mae": mae}
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_result = result
    if best_result is not None:
        best_result.setdefault("metadata", {})
        best_result["metadata"]["subset_metrics"] = subset_metrics
    return best_result


def run_mos_residualized(ctx: ExperimentContext) -> dict:
    mos = features.compute_mos_surface_anomalies(
        ctx.df,
        codes=["tmp", "dpt", "wsp", "wdr", "cig", "vis"],
        group_key=ctx.group_key,
    )
    feature_df = pd.concat([mos.frame, ctx.base_features], axis=1)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, feature_df)

    folds = [
        ("2021-02-23", "2023-12-31", "2024-01-08", "2024-03-31"),
        ("2021-02-23", "2024-03-31", "2024-04-08", "2024-06-30"),
    ]
    oof = _compute_oof_predictions(ctx, ctx.base_features, folds)
    oof_train = oof[ctx.train_mask]
    train_mask = np.isfinite(oof_train)
    if not train_mask.any():
        train_mask = np.isfinite(y_train)
        oof_train = ctx.baseline_b.pred_train

    X_train_s, X_val_s, X_test_s, _ = models.standardize_for_linear(
        X_train, X_val, X_test
    )
    ridge = models.train_ridge_search(
        X_train=X_train_s[train_mask],
        y_train=(y_train - oof_train)[train_mask],
        X_val=X_val_s,
        y_val=y_val - ctx.baseline_b.pred_val,
        seeds=get_seeds(),
        alphas=[0.1, 1.0, 10.0, 50.0, 100.0],
    )
    model_full = models.refit_model(
        model_name="ridge",
        params=ridge.params,
        X_train=X_train_s[train_mask],
        y_train=(y_train - oof_train)[train_mask],
        seed=get_seeds()[0],
    )
    resid_train = model_full.predict(X_train_s)
    resid_val = model_full.predict(X_val_s)
    resid_test = model_full.predict(X_test_s)

    baseline_all = _baseline_pred_all(ctx)
    final_train = baseline_all[ctx.train_mask] + resid_train
    final_val = baseline_all[ctx.val_mask] + resid_val
    final_test = baseline_all[ctx.test_mask] + resid_test

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOSResidualizedContextSignals",
        name="MOS residualized context",
        description=EXPERIMENT_DESCRIPTIONS["MOSResidualizedContextSignals"],
        model_name="ridge",
        params=ridge.params,
        feature_columns=list(feature_df.columns),
        pred_train=final_train,
        pred_val=final_val,
        pred_test=final_test,
    )


def run_spread_regime(ctx: ExperimentContext) -> dict:
    spread = ctx.base_features["guid_spread"]
    train_vals = spread[ctx.train_mask].to_numpy(dtype=float)
    t1, t2 = np.quantile(train_vals, [0.33, 0.66])
    regime = pd.Series(np.digitize(spread.to_numpy(), [t1, t2]), index=ctx.df.index)
    reg_onehot = pd.get_dummies(regime, prefix="spread_regime")
    feature_df = pd.concat([ctx.base_features, reg_onehot], axis=1)
    result = _run_tree_experiment(
        ctx,
        experiment_id="SpreadRegimeMixtureOfExperts",
        name="Spread regime mixture",
        description=EXPERIMENT_DESCRIPTIONS["SpreadRegimeMixtureOfExperts"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )
    return result


def run_climatology_season(ctx: ExperimentContext) -> dict:
    clim = features.compute_walkforward_climatology(
        ctx.df,
        group_key=ctx.group_key,
        window_doy=3,
    )
    season = features.season_onehot(ctx.df)
    feature_df = pd.concat(
        [
            ctx.base_features,
            clim.frame,
            season[["season_DJF", "season_MAM", "season_JJA", "season_SON"]],
        ],
        axis=1,
    )
    ens_mean = ctx.base_features["ens_mean_guidance"]
    feature_df["ens_anom_vs_clim"] = (ens_mean - clim.frame["clim_med"]).clip(
        -15.0, 15.0
    )
    return _run_tree_experiment(
        ctx,
        experiment_id="SeasonAwareModelWithWalkForwardClimatology",
        name="Season-aware climatology",
        description=EXPERIMENT_DESCRIPTIONS["SeasonAwareModelWithWalkForwardClimatology"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_pred_decile_regime(ctx: ExperimentContext) -> dict:
    p0 = ctx.base_features["ens_mean_guidance"]
    decile, _ = features.predicted_deciles(p0, ctx.train_mask)
    decile_onehot = pd.get_dummies(decile, prefix="pred_decile")
    feature_df = pd.concat([ctx.base_features, decile_onehot], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="PredictedTempDecileConditionalCorrection",
        name="Predicted temp decile regime",
        description=EXPERIMENT_DESCRIPTIONS["PredictedTempDecileConditionalCorrection"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_bust_probability(ctx: ExperimentContext) -> dict:
    features_df = pd.concat(
        [ctx.base_features, ctx.base_features[["guid_spread", "gefsatmos_tmp_spread_f"]]],
        axis=1,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, features_df)

    best_result = None
    best_mae = None
    for tau in [2.0, 2.5, 3.0]:
        bust_label = (np.abs(y_train - ctx.baseline_b.pred_train) > tau).astype(int)
        X_train_s, X_val_s, X_test_s, _ = models.standardize_for_linear(
            X_train, X_val, X_test
        )
        clf = Ridge(alpha=1.0)
        clf.fit(X_train_s, bust_label)
        p_val = np.clip(clf.predict(X_val_s), 0.05, 0.95)
        p_test = np.clip(clf.predict(X_test_s), 0.05, 0.95)

        y_ref_val = np.median(
            ctx.df.loc[ctx.val_mask, GUIDANCE_COLS].to_numpy(dtype=float), axis=1
        )
        y_ref_test = np.median(
            ctx.df.loc[ctx.test_mask, GUIDANCE_COLS].to_numpy(dtype=float), axis=1
        )
        pred_val = (1 - p_val) * ctx.baseline_b.pred_val + p_val * y_ref_val
        pred_test = (1 - p_test) * ctx.baseline_b.pred_test + p_test * y_ref_test
        pred_train = ctx.baseline_b.pred_train
        metrics = eval_mod.regression_metrics(y_val, pred_val)
        if best_mae is None or metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_result = _assemble_result(
                ctx=ctx,
                experiment_id="BustProbabilityGatedShrinkage",
                name="Bust probability shrinkage",
                description=EXPERIMENT_DESCRIPTIONS["BustProbabilityGatedShrinkage"],
                model_name="ridge",
                params={"tau": tau},
                feature_columns=list(features_df.columns),
                pred_train=pred_train,
                pred_val=pred_val,
                pred_test=pred_test,
            )
    return best_result


def run_wind_quadrant(ctx: ExperimentContext) -> dict:
    wdr = features.build_mos_value(ctx.df, "wdr")
    quad = pd.Series(0, index=ctx.df.index)
    quad[(wdr >= 45) & (wdr < 135)] = 1
    quad[(wdr >= 135) & (wdr < 225)] = 2
    quad[(wdr >= 225) & (wdr < 315)] = 3
    quad_onehot = pd.get_dummies(quad, prefix="wind_quad")
    feature_df = pd.concat([ctx.base_features, quad_onehot], axis=1)
    return _run_tree_experiment(
        ctx,
        experiment_id="WindQuadrantConditionalBiasSwitch",
        name="Wind quadrant conditional bias",
        description=EXPERIMENT_DESCRIPTIONS["WindQuadrantConditionalBiasSwitch"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_lgbm_l1(ctx: ExperimentContext) -> dict:
    return _run_tree_experiment(
        ctx,
        experiment_id="LightGBMMAEObjectivedirect",
        name="LightGBM L1 objective",
        description=EXPERIMENT_DESCRIPTIONS["LightGBMMAEObjectivedirect"],
        feature_df=ctx.base_features,
        model_name="lgbm",
        objective="regression_l1",
        trials=40,
    )


def run_xgb_absolute(ctx: ExperimentContext) -> dict:
    return _run_tree_experiment(
        ctx,
        experiment_id="XGBRegAbsoluteErrorMAE",
        name="XGBoost absolute error",
        description=EXPERIMENT_DESCRIPTIONS["XGBRegAbsoluteErrorMAE"],
        feature_df=ctx.base_features,
        model_name="xgb",
        objective="reg:absoluteerror",
        trials=40,
    )


def run_xgb_pseudohuber(ctx: ExperimentContext) -> dict:
    def sampler(rng: np.random.Generator):
        params = models.sample_xgb_params(rng, "reg:pseudohubererror")
        params["huber_slope"] = float(rng.uniform(0.5, 5.0))
        return params

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, ctx.base_features)
    search = models.train_with_search(
        model_name="xgb",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seeds=get_seeds(),
        trials=40,
        param_sampler=sampler,
        rng=ctx.rng,
    )
    model_train = _fit_best_model(
        model_name="xgb",
        params=search.params,
        X_train=X_train,
        y_train=y_train,
    )
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val)
    model_full = _fit_best_model(
        model_name="xgb",
        params=search.params,
        X_train=X_train_full,
        y_train=y_train_full,
    )
    pred_test = model_full.predict(X_test)
    return _assemble_result(
        ctx=ctx,
        experiment_id="XGBPseudoHuberDeltaTuning",
        name="XGBoost pseudo-Huber",
        description=EXPERIMENT_DESCRIPTIONS["XGBPseudoHuberDeltaTuning"],
        model_name="xgb",
        params=search.params,
        feature_columns=list(ctx.base_features.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_xgb_quantile(ctx: ExperimentContext) -> dict:
    def sampler(rng: np.random.Generator):
        params = models.sample_xgb_params(rng, "reg:quantileerror")
        params["quantile_alpha"] = 0.5
        return params

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, ctx.base_features)
    search = models.train_with_search(
        model_name="xgb",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seeds=get_seeds(),
        trials=40,
        param_sampler=sampler,
        rng=ctx.rng,
    )
    model_train = _fit_best_model(
        model_name="xgb",
        params=search.params,
        X_train=X_train,
        y_train=y_train,
    )
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val)
    model_full = _fit_best_model(
        model_name="xgb",
        params=search.params,
        X_train=X_train_full,
        y_train=y_train_full,
    )
    pred_test = model_full.predict(X_test)
    return _assemble_result(
        ctx=ctx,
        experiment_id="XGBQuantileMedianAndMultiQuantileMeanAssist",
        name="XGBoost quantile median",
        description=EXPERIMENT_DESCRIPTIONS["XGBQuantileMedianAndMultiQuantileMeanAssist"],
        model_name="xgb",
        params=search.params,
        feature_columns=list(ctx.base_features.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_catboost_loss_sweep(ctx: ExperimentContext) -> dict:
    losses = ["MAE", "Huber", "Quantile", "LogCosh", "FairLoss"]
    best_result = None
    best_mae = None
    for loss in losses:
        result = _run_tree_experiment(
            ctx,
            experiment_id="CatBoostRobustLossSweep",
            name="CatBoost robust loss sweep",
            description=EXPERIMENT_DESCRIPTIONS["CatBoostRobustLossSweep"],
            feature_df=ctx.base_features,
            model_name="catboost",
            objective=loss,
            trials=20,
        )
        mae = result["metrics"]["validation"]["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_result = result
    return best_result


def run_forward_chaining_stack(ctx: ExperimentContext) -> dict:
    df = ctx.df.copy()
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    folds = [
        ("2021-02-23", "2023-12-31", "2024-01-08", "2024-03-31"),
        ("2021-02-23", "2024-03-31", "2024-04-08", "2024-06-30"),
        ("2021-02-23", "2024-06-30", "2024-07-08", "2024-09-30"),
        ("2021-02-23", "2024-09-30", "2024-10-08", "2025-01-30"),
    ]
    oof = np.full(len(df), np.nan, dtype=float)
    for train_start, train_end, val_start, val_end in folds:
        train_mask = (dates >= pd.to_datetime(train_start).date()) & (
            dates <= pd.to_datetime(train_end).date()
        )
        val_mask = (dates >= pd.to_datetime(val_start).date()) & (
            dates <= pd.to_datetime(val_end).date()
        )
        X_train = ctx.base_features.loc[train_mask].to_numpy(dtype=float)
        y_train = df.loc[train_mask, "actual_tmax_f"].to_numpy(dtype=float)
        X_val = ctx.base_features.loc[val_mask].to_numpy(dtype=float)
        model = _fit_best_model(
            model_name=ctx.baseline_b.model_name,
            params=ctx.baseline_b.params,
            X_train=X_train,
            y_train=y_train,
        )
        oof[val_mask] = model.predict(X_val)
    valid_mask = np.isfinite(oof)
    meta = Ridge(alpha=1.0)
    meta.fit(
        oof[valid_mask].reshape(-1, 1),
        df.loc[valid_mask, "actual_tmax_f"].to_numpy(dtype=float),
    )
    pred_train = ctx.baseline_b.pred_train
    pred_val = meta.predict(ctx.baseline_b.pred_val.reshape(-1, 1))
    pred_test = meta.predict(ctx.baseline_b.pred_test.reshape(-1, 1))
    return _assemble_result(
        ctx=ctx,
        experiment_id="ForwardChainingStackedEnsemble",
        name="Forward-chaining stacked ensemble",
        description=EXPERIMENT_DESCRIPTIONS["ForwardChainingStackedEnsemble"],
        model_name="ridge",
        params={"alpha": 1.0},
        feature_columns=["baseline_b_pred"],
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_dynamic_blend(ctx: ExperimentContext) -> dict:
    weights = features.compute_rolling_mae_weights(
        ctx.df,
        GUIDANCE_COLS,
        windows=[60],
        group_key=ctx.group_key,
        train_mask=ctx.train_mask,
    )
    blend = weights.frame["ens_wmean_rm60"].to_numpy(dtype=float)
    pred_train = blend[ctx.train_mask]
    pred_val = blend[ctx.val_mask]
    pred_test = blend[ctx.test_mask]
    return _assemble_result(
        ctx=ctx,
        experiment_id="ConstrainedDynamicBlendWeights",
        name="Constrained dynamic blend",
        description=EXPERIMENT_DESCRIPTIONS["ConstrainedDynamicBlendWeights"],
        model_name="blend",
        params={"lambda": 1.0},
        feature_columns=list(weights.frame.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_two_stage_residual(ctx: ExperimentContext) -> dict:
    mos_proxy = features.compute_mos_cloud_precip(ctx.df)
    feature_df = pd.concat(
        [
            ctx.base_features,
            mos_proxy.frame[["mos_precip_proxy_max", "mos_log_cig"]],
        ],
        axis=1,
    )
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, feature_df)
    folds = [
        ("2021-02-23", "2023-12-31", "2024-01-08", "2024-03-31"),
        ("2021-02-23", "2024-03-31", "2024-04-08", "2024-06-30"),
    ]
    oof = _compute_oof_predictions(ctx, ctx.base_features, folds)
    oof_train = oof[ctx.train_mask]
    train_mask = np.isfinite(oof_train)
    if not train_mask.any():
        train_mask = np.isfinite(y_train)
        oof_train = ctx.baseline_b.pred_train

    X_train_s, X_val_s, X_test_s, _ = models.standardize_for_linear(
        X_train, X_val, X_test
    )
    ridge = models.train_ridge_search(
        X_train=X_train_s[train_mask],
        y_train=(y_train - oof_train)[train_mask],
        X_val=X_val_s,
        y_val=y_val - ctx.baseline_b.pred_val,
        seeds=get_seeds(),
        alphas=[0.1, 1.0, 10.0, 50.0, 100.0],
    )
    model_full = models.refit_model(
        model_name="ridge",
        params=ridge.params,
        X_train=X_train_s[train_mask],
        y_train=(y_train - oof_train)[train_mask],
        seed=get_seeds()[0],
    )
    resid_pred_train = model_full.predict(X_train_s)
    resid_pred_val = model_full.predict(X_val_s)
    resid_pred_test = model_full.predict(X_test_s)
    cap = 3.0
    baseline_all = _baseline_pred_all(ctx)
    pred_train = baseline_all[ctx.train_mask] + np.clip(resid_pred_train, -cap, cap)
    pred_val = baseline_all[ctx.val_mask] + np.clip(resid_pred_val, -cap, cap)
    pred_test = baseline_all[ctx.test_mask] + np.clip(resid_pred_test, -cap, cap)

    return _assemble_result(
        ctx=ctx,
        experiment_id="TwoStageResidualStackWithBiasFeatures",
        name="Two-stage residual stack",
        description=EXPERIMENT_DESCRIPTIONS["TwoStageResidualStackWithBiasFeatures"],
        model_name="ridge",
        params={"alpha": ridge.params.get("alpha", 1.0), "cap": cap},
        feature_columns=list(feature_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_multisnapshot(ctx: ExperimentContext) -> dict:
    df = ctx.df.copy()
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    agg_rows = []
    for (station, target), group in df.groupby(["station_id", "target_date_local"]):
        group = group.sort_values("asof_utc")
        latest = group.iloc[-1]
        row = latest.copy()
        for col in GUIDANCE_COLS:
            if col in group.columns:
                vals = group[col].to_numpy(dtype=float)
                row[f"{col}_latest"] = vals[-1]
                row[f"{col}_mean_last3"] = np.nanmean(vals[-3:])
                row[f"{col}_trend_last3"] = vals[-1] - vals[-3] if len(vals) >= 3 else 0.0
        row["num_snapshots"] = len(group)
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    feature_cols = [
        c
        for c in agg_df.columns
        if c.endswith("_latest") or c.endswith("_mean_last3") or c.endswith("_trend_last3")
    ]
    feature_cols.append("num_snapshots")
    feature_df = agg_df[feature_cols].astype(float)

    dates = pd.to_datetime(agg_df["target_date_local"]).dt.date
    train_start = pd.to_datetime(ctx.split_ref["train_start"]).date()
    train_end = pd.to_datetime(ctx.split_ref["train_end"]).date()
    val_start = pd.to_datetime(ctx.split_ref["val_start"]).date()
    val_end = pd.to_datetime(ctx.split_ref["val_end"]).date()
    test_start = pd.to_datetime(ctx.split_ref["test_start"]).date()
    test_end = pd.to_datetime(ctx.split_ref["test_end"]).date()
    gap_dates = {pd.to_datetime(d).date() for d in ctx.split_ref.get("gap_dates", [])}
    in_gap = np.array([d in gap_dates for d in dates])
    train_mask = (dates >= train_start) & (dates <= train_end) & ~in_gap
    val_mask = (dates >= val_start) & (dates <= val_end) & ~in_gap
    test_mask = (dates >= test_start) & (dates <= test_end) & ~in_gap

    temp_ctx = ExperimentContext(
        df=agg_df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        base_features=feature_df,
        base_feature_columns=feature_cols,
        baseline_b=ctx.baseline_b,
        group_key=agg_df["station_id"].astype(str),
        group_key_asof=agg_df["station_id"].astype(str),
        rng=ctx.rng,
        run_root=ctx.run_root,
        split_ref=ctx.split_ref,
    )
    return _run_tree_experiment(
        temp_ctx,
        experiment_id="MultiSnapshotAggregationAndEnsemble",
        name="Multi-snapshot aggregation",
        description=EXPERIMENT_DESCRIPTIONS["MultiSnapshotAggregationAndEnsemble"],
        feature_df=feature_df,
        model_name="lgbm",
        objective="regression",
        trials=40,
    )


def run_knn_analog(ctx: ExperimentContext) -> dict:
    analog_cols = GUIDANCE_COLS + ["gefsatmos_tmp_spread_f", "guid_spread"]
    feature_matrix, _ = analogs.standardize_matrix(ctx.df, analog_cols, ctx.train_mask)
    result = analogs.compute_knn_analogs(
        ctx.df,
        feature_matrix,
        ctx.df["actual_tmax_f"].to_numpy(dtype=float),
        ks=[10, 25, 50],
        group_key=ctx.group_key_asof,
        min_pool=200,
    )
    best_result = None
    best_mae = None
    for k, preds in result.predictions.items():
        analog_frame = pd.DataFrame(
            {
                f"analog_pred_k{k}": preds,
                "analog_dist_q50": result.diagnostics["analog_dist_q50"],
                "analog_eff_n": result.diagnostics["analog_eff_n"],
                "analog_truth_std": result.diagnostics["analog_truth_std"],
            },
            index=ctx.df.index,
        )
        feature_df = pd.concat([ctx.base_features, analog_frame], axis=1)
        out = _run_tree_experiment(
            ctx,
            experiment_id="GuidanceSpacekNNAnalogPrediction",
            name="Guidance-space kNN analog",
            description=EXPERIMENT_DESCRIPTIONS["GuidanceSpacekNNAnalogPrediction"],
            feature_df=feature_df,
            model_name="lgbm",
            objective="regression",
            trials=20,
        )
        mae = out["metrics"]["validation"]["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_result = out
    return best_result


def run_residual_knn(ctx: ExperimentContext) -> dict:
    baseline_all = _baseline_pred_all(ctx)
    residuals = ctx.df["actual_tmax_f"].to_numpy(dtype=float) - baseline_all
    analog_cols = GUIDANCE_COLS + ["gefsatmos_tmp_spread_f", "guid_spread"]
    feature_matrix, _ = analogs.standardize_matrix(ctx.df, analog_cols, ctx.train_mask)
    result = analogs.compute_knn_analogs(
        ctx.df,
        feature_matrix,
        residuals,
        ks=[25, 50, 100],
        group_key=ctx.group_key_asof,
        min_pool=200,
    )
    best_result = None
    best_mae = None
    for k, preds in result.predictions.items():
        correction = np.clip(preds, -2.5, 2.5)
        pred_train = ctx.baseline_b.pred_train + correction[ctx.train_mask]
        pred_val = ctx.baseline_b.pred_val + correction[ctx.val_mask]
        pred_test = ctx.baseline_b.pred_test + correction[ctx.test_mask]
        out = _assemble_result(
            ctx=ctx,
            experiment_id="ResidualAnalogCorrectionOnBaselineB",
            name="Residual analog correction",
            description=EXPERIMENT_DESCRIPTIONS["ResidualAnalogCorrectionOnBaselineB"],
            model_name="knn",
            params={"k": k},
            feature_columns=analog_cols,
            pred_train=pred_train,
            pred_val=pred_val,
            pred_test=pred_test,
        )
        mae = out["metrics"]["validation"]["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_result = out
    return best_result


def run_proto_analogs(ctx: ExperimentContext) -> dict:
    analog_cols = GUIDANCE_COLS + ["gefsatmos_tmp_spread_f", "guid_spread"]
    feature_matrix, _ = analogs.standardize_matrix(ctx.df, analog_cols, ctx.train_mask)
    residuals = (
        ctx.df["actual_tmax_f"].to_numpy(dtype=float)
        - ctx.base_features["ens_mean_guidance"].to_numpy(dtype=float)
    )
    best_result = None
    best_mae = None
    for k in [20, 40, 60]:
        proto = analogs.compute_prototypes(
            feature_matrix,
            residuals,
            train_mask=ctx.train_mask,
            k_prototypes=k,
        )
        proto_id, proto_dist = analogs.assign_prototypes(feature_matrix, proto)
        proto_resid = np.array(
            [proto["proto_resid"].get(int(pid), 0.0) for pid in proto_id]
        )
        feature_df = pd.concat(
            [
                ctx.base_features,
                pd.DataFrame(
                    {
                        "proto_id": proto_id,
                        "proto_dist": proto_dist,
                        "proto_resid_mean": np.clip(proto_resid, -3.0, 3.0),
                    },
                    index=ctx.df.index,
                ),
            ],
            axis=1,
        )
        out = _run_tree_experiment(
            ctx,
            experiment_id="PrototypeAnalogsKMedoidsWithForwardUpdate",
            name="Prototype analogs",
            description=EXPERIMENT_DESCRIPTIONS["PrototypeAnalogsKMedoidsWithForwardUpdate"],
            feature_df=feature_df,
            model_name="lgbm",
            objective="regression",
            trials=20,
        )
        mae = out["metrics"]["validation"]["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_result = out
    return best_result


def run_local_linear_knn(ctx: ExperimentContext) -> dict:
    analog_cols = GUIDANCE_COLS + ["gefsatmos_tmp_spread_f", "guid_spread"]
    feature_matrix, _ = analogs.standardize_matrix(ctx.df, analog_cols, ctx.train_mask)
    local_preds = analogs.compute_local_ridge(
        ctx.df,
        feature_matrix,
        ctx.df["actual_tmax_f"].to_numpy(dtype=float),
        ks=[100, 200],
        alphas=[0.1, 1.0, 10.0],
        group_key=ctx.group_key_asof,
        min_pool=150,
    )
    best_result = None
    best_mae = None
    for (k, alpha), preds in local_preds.items():
        pred_train = preds[ctx.train_mask]
        pred_val = preds[ctx.val_mask]
        pred_test = preds[ctx.test_mask]
        out = _assemble_result(
            ctx=ctx,
            experiment_id="LocalLinearKNNRidgeForecaster",
            name="Local linear kNN ridge",
            description=EXPERIMENT_DESCRIPTIONS["LocalLinearKNNRidgeForecaster"],
            model_name="local_ridge",
            params={"k": k, "alpha": alpha},
            feature_columns=analog_cols,
            pred_train=pred_train,
            pred_val=pred_val,
            pred_test=pred_test,
        )
        mae = out["metrics"]["validation"]["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_result = out
    return best_result


def build_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            experiment_id="RobustMultiWindowBiasStack",
            name="Robust multi-window bias stack",
            description=EXPERIMENT_DESCRIPTIONS["RobustMultiWindowBiasStack"],
            runner=run_robust_multiwindow_bias_stack,
        ),
        ExperimentSpec(
            experiment_id="EWMABiasCorrHalfLife",
            name="EWMA bias/corr half-life",
            description=EXPERIMENT_DESCRIPTIONS["EWMABiasCorrHalfLife"],
            runner=run_ewma_bias_corr,
        ),
        ExperimentSpec(
            experiment_id="DriftTriggeredBiasGating",
            name="Drift-triggered bias gating",
            description=EXPERIMENT_DESCRIPTIONS["DriftTriggeredBiasGating"],
            runner=run_drift_triggered_gating,
        ),
        ExperimentSpec(
            experiment_id="RobustRankCorrelationFeatures",
            name="Robust rank correlation",
            description=EXPERIMENT_DESCRIPTIONS["RobustRankCorrelationFeatures"],
            runner=run_rank_correlation,
        ),
        ExperimentSpec(
            experiment_id="KalmanBiasStatePerGuidance",
            name="Kalman bias state",
            description=EXPERIMENT_DESCRIPTIONS["KalmanBiasStatePerGuidance"],
            runner=run_kalman_bias,
        ),
        ExperimentSpec(
            experiment_id="RollingMAEWeightedEnsembleSignals",
            name="Rolling MAE weighted ensemble",
            description=EXPERIMENT_DESCRIPTIONS["RollingMAEWeightedEnsembleSignals"],
            runner=run_mae_weighted_ensemble,
        ),
        ExperimentSpec(
            experiment_id="MOSSurfaceContextAnomalies",
            name="MOS surface context anomalies",
            description=EXPERIMENT_DESCRIPTIONS["MOSSurfaceContextAnomalies"],
            runner=run_mos_surface_anoms,
        ),
        ExperimentSpec(
            experiment_id="MOSDewpointDepressionWindComponents",
            name="MOS dewpoint depression & wind",
            description=EXPERIMENT_DESCRIPTIONS["MOSDewpointDepressionWindComponents"],
            runner=run_mos_dewpoint_wind,
        ),
        ExperimentSpec(
            experiment_id="MOSCloudPrecipProxyFeatures",
            name="MOS cloud/precip proxy",
            description=EXPERIMENT_DESCRIPTIONS["MOSCloudPrecipProxyFeatures"],
            runner=run_mos_cloud_precip,
        ),
        ExperimentSpec(
            experiment_id="MOSMissingnessQualityEncoding",
            name="MOS missingness quality encoding",
            description=EXPERIMENT_DESCRIPTIONS["MOSMissingnessQualityEncoding"],
            runner=run_mos_missingness,
        ),
        ExperimentSpec(
            experiment_id="MOSCodeSubsetSelectionWithTimeSafeAblation",
            name="MOS subset sweep",
            description=EXPERIMENT_DESCRIPTIONS[
                "MOSCodeSubsetSelectionWithTimeSafeAblation"
            ],
            runner=run_mos_subset_sweep,
        ),
        ExperimentSpec(
            experiment_id="MOSResidualizedContextSignals",
            name="MOS residualized context",
            description=EXPERIMENT_DESCRIPTIONS["MOSResidualizedContextSignals"],
            runner=run_mos_residualized,
        ),
        ExperimentSpec(
            experiment_id="SpreadRegimeMixtureOfExperts",
            name="Spread regime mixture",
            description=EXPERIMENT_DESCRIPTIONS["SpreadRegimeMixtureOfExperts"],
            runner=run_spread_regime,
        ),
        ExperimentSpec(
            experiment_id="SeasonAwareModelWithWalkForwardClimatology",
            name="Season-aware climatology",
            description=EXPERIMENT_DESCRIPTIONS[
                "SeasonAwareModelWithWalkForwardClimatology"
            ],
            runner=run_climatology_season,
        ),
        ExperimentSpec(
            experiment_id="PredictedTempDecileConditionalCorrection",
            name="Predicted temp decile regime",
            description=EXPERIMENT_DESCRIPTIONS[
                "PredictedTempDecileConditionalCorrection"
            ],
            runner=run_pred_decile_regime,
        ),
        ExperimentSpec(
            experiment_id="BustProbabilityGatedShrinkage",
            name="Bust probability shrinkage",
            description=EXPERIMENT_DESCRIPTIONS["BustProbabilityGatedShrinkage"],
            runner=run_bust_probability,
        ),
        ExperimentSpec(
            experiment_id="WindQuadrantConditionalBiasSwitch",
            name="Wind quadrant conditional bias",
            description=EXPERIMENT_DESCRIPTIONS["WindQuadrantConditionalBiasSwitch"],
            runner=run_wind_quadrant,
        ),
        ExperimentSpec(
            experiment_id="LightGBMMAEObjectivedirect",
            name="LightGBM L1 objective",
            description=EXPERIMENT_DESCRIPTIONS["LightGBMMAEObjectivedirect"],
            runner=run_lgbm_l1,
        ),
        ExperimentSpec(
            experiment_id="XGBRegAbsoluteErrorMAE",
            name="XGBoost absolute error",
            description=EXPERIMENT_DESCRIPTIONS["XGBRegAbsoluteErrorMAE"],
            runner=run_xgb_absolute,
        ),
        ExperimentSpec(
            experiment_id="XGBPseudoHuberDeltaTuning",
            name="XGBoost pseudo-Huber",
            description=EXPERIMENT_DESCRIPTIONS["XGBPseudoHuberDeltaTuning"],
            runner=run_xgb_pseudohuber,
        ),
        ExperimentSpec(
            experiment_id="XGBQuantileMedianAndMultiQuantileMeanAssist",
            name="XGBoost quantile median",
            description=EXPERIMENT_DESCRIPTIONS[
                "XGBQuantileMedianAndMultiQuantileMeanAssist"
            ],
            runner=run_xgb_quantile,
        ),
        ExperimentSpec(
            experiment_id="CatBoostRobustLossSweep",
            name="CatBoost robust loss sweep",
            description=EXPERIMENT_DESCRIPTIONS["CatBoostRobustLossSweep"],
            runner=run_catboost_loss_sweep,
        ),
        ExperimentSpec(
            experiment_id="ForwardChainingStackedEnsemble",
            name="Forward-chaining stacked ensemble",
            description=EXPERIMENT_DESCRIPTIONS["ForwardChainingStackedEnsemble"],
            runner=run_forward_chaining_stack,
        ),
        ExperimentSpec(
            experiment_id="ConstrainedDynamicBlendWeights",
            name="Constrained dynamic blend",
            description=EXPERIMENT_DESCRIPTIONS["ConstrainedDynamicBlendWeights"],
            runner=run_dynamic_blend,
        ),
        ExperimentSpec(
            experiment_id="TwoStageResidualStackWithBiasFeatures",
            name="Two-stage residual stack",
            description=EXPERIMENT_DESCRIPTIONS["TwoStageResidualStackWithBiasFeatures"],
            runner=run_two_stage_residual,
        ),
        ExperimentSpec(
            experiment_id="MultiSnapshotAggregationAndEnsemble",
            name="Multi-snapshot aggregation",
            description=EXPERIMENT_DESCRIPTIONS["MultiSnapshotAggregationAndEnsemble"],
            runner=run_multisnapshot,
        ),
        ExperimentSpec(
            experiment_id="GuidanceSpacekNNAnalogPrediction",
            name="Guidance-space kNN analog",
            description=EXPERIMENT_DESCRIPTIONS["GuidanceSpacekNNAnalogPrediction"],
            runner=run_knn_analog,
        ),
        ExperimentSpec(
            experiment_id="ResidualAnalogCorrectionOnBaselineB",
            name="Residual analog correction",
            description=EXPERIMENT_DESCRIPTIONS["ResidualAnalogCorrectionOnBaselineB"],
            runner=run_residual_knn,
        ),
        ExperimentSpec(
            experiment_id="PrototypeAnalogsKMedoidsWithForwardUpdate",
            name="Prototype analogs",
            description=EXPERIMENT_DESCRIPTIONS[
                "PrototypeAnalogsKMedoidsWithForwardUpdate"
            ],
            runner=run_proto_analogs,
        ),
        ExperimentSpec(
            experiment_id="LocalLinearKNNRidgeForecaster",
            name="Local linear kNN ridge",
            description=EXPERIMENT_DESCRIPTIONS["LocalLinearKNNRidgeForecaster"],
            runner=run_local_linear_knn,
        ),
    ]


def build_context(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    run_root: Path,
    split_ref: dict,
    rng: np.random.Generator,
) -> ExperimentContext:
    if df.empty:
        LOGGER.warning("BUILD_CONTEXT called with empty dataframe.")
    else:
        dates = pd.to_datetime(df["target_date_local"]).dt.date
        LOGGER.info(
            "CTX_DATA rows=%d cols=%d stations=%d date_range=%s..%s",
            len(df),
            df.shape[1],
            df["station_id"].nunique(dropna=True),
            min(dates),
            max(dates),
        )
    group_key = df["station_id"].astype(str)
    asof_hour = pd.to_datetime(df["asof_utc"], utc=True).dt.hour.astype(int)
    group_key_asof = group_key + "_" + asof_hour.astype(str)
    base_features, base_cols = _baseline_b_features(df, group_key, train_mask)
    LOGGER.info("CTX_BASE_FEATURES count=%d", len(base_cols))

    baseline_ctx = ExperimentContext(
        df=df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        base_features=base_features,
        base_feature_columns=base_cols,
        baseline_b=None,
        group_key=group_key,
        group_key_asof=group_key_asof,
        rng=rng,
        run_root=run_root,
        split_ref=split_ref,
    )
    baseline_b = run_baseline_b(baseline_ctx)
    baseline_ctx.baseline_b = baseline_b
    return baseline_ctx

