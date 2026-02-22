"""Experiment definitions and runners for TFS2 sweep."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture

from weather_ml import artifacts
from weather_ml import rs_moe
from weather_ml import splits
from weather_ml import time_feature_library as tfl

from . import eval as eval_mod
from . import features, models
from .config import BASELINE_PARAMS, DEFAULT_SPLIT, GUIDANCE_COLS, SPREAD_COL, TRUTH_LAG_DAYS, SplitConfig

LOGGER = logging.getLogger(__name__)


EXPERIMENT_DESCRIPTIONS = {
    "PerModelAdaptiveEWMA-BanditCorr": "This experiment replaces the single-window bias idea with per-guidance adaptive EWMA biases at multiple half-lives. It computes robust, clipped error histories using only data available by D-2, selects half-lives via bandit-style weights, and builds a reliability-weighted corrected ensemble. The intent is to track fast drift without overreacting, while exposing entropy and stability diagnostics to keep corrections bounded.",
    "FourierK3+AnomalyEWMA-PerModel": "This experiment models seasonal bias explicitly with three Fourier harmonics per guidance source, then removes remaining drift using lag-safe EWMA anomalies. It yields corrected per-model forecasts and corrected ensemble summaries, plus higher-harmonic phase features. The goal is to capture stable annual structure while still adapting to shorter-term shifts, improving MAE in shoulder seasons without introducing leakage from future truth.",
    "DualTailDepthBiasMap": "This experiment extends the cool-tail depth mapping to include warm-tail depth and tail asymmetry, using quantile bins built on training residuals only. Each bin receives a shrunk median bias correction, and the corrected ensemble mean is added as a feature. By separating cool vs warm tail structure, it targets extreme-day errors and improves performance on the hottest and coolest regimes without changing the base model family.",
    "ErrorCorrelationDiversityWeights": "This experiment measures rolling error correlation among guidance sources using lag-safe residual windows and builds diversity-aware weights. It penalizes models whose errors move together and favors sources that are both skillful and complementary, producing a corrected ensemble mean plus diversity diagnostics. The intent is to reduce MAE in low-diversity regimes where a simple mean behaves like a single model, while keeping corrections stable and leakage-safe.",
    "CUSUMReset+BOCPDGatedBias": "This experiment combines a BOCPD change probability gate with a simple CUSUM drift reset on the ensemble residual. A fast EWMA bias is blended with a slower bias state that is reset when CUSUM alarms, using only D-2 residuals. The corrected ensemble mean and drift diagnostics are added as features, aiming to adapt quickly during regime shifts without the overfitting seen when change-point probabilities are used directly.",
    "SeasonalAdaptiveHalfLifeBias": "This experiment lets the bias half-life vary smoothly with day-of-year, producing a seasonal EWMA bias that adapts at different speeds across the annual cycle. The half-life curve is tuned on validation and applied causally using D-2 residuals. A corrected ensemble mean and half-life features are added so the model can exploit seasonal drift patterns without adding any new data sources or leakage risk.",
    "MOS-DewpointDepressionHumiditySignals": "This experiment adds MOS-derived dewpoint depression, its lag-safe anomalies, and quantile bin indicators to represent moisture and boundary-layer stability effects on daily maxima. Missing MOS values are handled explicitly, and interactions with bust risk and spread are included. The goal is to capture humidity-driven error structure that is not fully explained by raw guidance, improving summer performance while preserving strict as-of integrity.",
    "MOS-WindVectorSeaBreezeGating": "This experiment introduces MOS wind direction and speed as vector components, plus rolling wind-shift features, to better represent sea-breeze regimes. A sea-breeze index from guidance is combined with wind vectors and bust risk interactions. All rolling signals are lag-safe. The intent is to let the model learn flow-dependent bias patterns and reduce MAE on onshore-flow days without expanding the model family or data sources.",
    "MOS-CloudVisibilityRadiationProxy": "This experiment builds a cloudiness index from MOS ceiling and visibility using log transforms and lag-safe anomalies. It includes interactions with ensemble mean, spread, and bust risk to capture cloudy-day cooling effects that often cause overprediction. Missingness is encoded explicitly. The hypothesis is that cloud proxies provide high-leverage context for Miami tmax, improving MAE in wet and hazy regimes while remaining leakage-safe.",
    "MOS-PrecipLikelihoodCoolingHedge": "This experiment aggregates MOS precipitation probability and amount proxies into a precipitation index and uses it to gate a conservative hedge toward a robust forecast. The gate is a smooth sigmoid tuned on validation, and the resulting hedged forecast is added as a feature alongside precip interactions. The objective is to reduce large errors on convective days by hedging toward safer guidance when precip likelihood is high.",
    "MOS-T06HeatingPotentialFeatures": "This experiment uses MOS early-hour temperature proxies (t06 variants) to estimate overnight setup and heating potential. It computes a diurnal rise proxy, lag-safe anomalies, and interactions with bust risk. When t06 data are missing, the feature falls back to MOS temperature or ensemble mean with flags. The aim is to capture boundary-layer preconditioning effects that drive daily maxima without altering the baseline model.",
    "MOS-MissingnessQualityEncoding": "This experiment encodes MOS availability as a first-class signal: per-code missing flags, days-since-last-available, pooled missingness fractions, and a reliability weight. These features help the model learn when MOS-driven corrections should be trusted versus ignored. The design emphasizes leakage-safe recency tracking and avoids spurious imputation bias, so MOS-rich regimes can help while sparse regimes do not degrade MAE.",
    "BustClassifierV2-Ordinal5Class": "This experiment upgrades the bust classifier to five ordered residual classes, yielding probabilities for large/small cool and warm regimes plus a neutral class. Class thresholds are derived from training residuals only, with forward-chaining cross-fit to avoid leakage. The resulting probabilities and expected signed error are added to the regression model, aiming to refine regime separation beyond the 3-class baseline while maintaining strict as-of integrity.",
    "SpreadxSeaBreezeMixtureOfExpertsBias": "This experiment builds a soft-gated mixture of two corrected experts: a Fourier-plus-anomaly correction and a tail-depth correction. A logistic gate trained on spread and sea-breeze indicators selects which expert is more reliable, producing a blended corrected forecast and gate features. The goal is to apply different correction logic in stable vs convective regimes without sacrificing the fixed baseline model capacity or introducing test leakage.",
    "ResidualHMM-StateProbBias": "This experiment fits a small Gaussian HMM to standardized residuals using training data only, then filters state probabilities forward with a D-2 lag. State means are converted into a bias correction scaled by spread, producing a corrected ensemble mean and regime probabilities. The intent is to capture persistent warm or cool bias states with a stable probabilistic signal, improving MAE without relying on direct future information.",
    "SeasonTransitionSpreadGate": "This experiment adds smooth transition-season bump features centered on spring and fall, plus interactions with spread and bust risk. The bumps are deterministic functions of day-of-year and do not use truth. The purpose is to give the model a dedicated handle for shoulder-season behavior where errors often change sign, while keeping the baseline feature set intact and leakage-safe.",
    "PCA2+MOSRegimeGMMProbFeatures": "This experiment fits a Gaussian mixture model on a regime vector combining disagreement PC1/PC2, spread, sea-breeze index, and MOS humidity/cloud proxies when available. The GMM is fit on training data only and yields regime probabilities, entropy, and component assignments. These features are intended to capture latent physical regimes that alter forecast reliability, improving MAE without adding new data sources or violating as-of constraints.",
    "LightGBM-RegressionL1MAEObjective": "This experiment keeps the baseline feature set but changes the LightGBM objective to L1 to align training with MAE. A bounded random search tunes tree capacity and learning rate under the same fixed budget, with early stopping on validation. The hypothesis is that the conditional median is a better point forecast under heavy-tailed residuals, yielding small but consistent MAE gains while preserving the baseline pipeline.",
    "CatBoost-MAEWithBustMOS": "This experiment trains CatBoost with MAE loss on the baseline plus high-signal MOS features (dewpoint depression, cloud index, precip proxy, wind vectors). CatBoost’s ordered boosting and robust handling of missing values are leveraged to improve stability in small time-series splits. A fixed-budget search tunes depth, learning rate, and regularization, testing whether CatBoost can outperform the LightGBM baseline without changing data sources.",
    "XGB-QuantileTrioMeanReconstruction": "This experiment trains XGBoost quantile regressors for the 10th, 50th, and 90th percentiles using the baseline plus select MOS features. The median is MAE-optimal, and a reconstructed mean from the three quantiles provides a robust point forecast and uncertainty proxy. The intent is to capture asymmetry and heteroskedasticity, improving MAE on tail days while remaining strictly time-ordered.",
    "TwoStageResidualShrinkageByUncertainty": "This experiment uses a two-stage model: a baseline predictor, followed by a residual corrector and an uncertainty model for |residual|. Corrections are shrunk when predicted uncertainty is high, using a tuned shrinkage constant. The approach targets large-error days without destabilizing normal conditions. Training is time-ordered and leakage-safe, and the final prediction balances correction strength with uncertainty to reduce MAE.",
    "HuberObjectiveTuning+MAESelection": "This experiment replaces the squared-loss objective with Huber loss and tunes the delta parameter alongside model capacity. Huber offers a smooth compromise between L2 and L1, aiming to improve MAE while retaining stability. A fixed-budget random search with early stopping selects hyperparameters by validation MAE. The goal is to reduce the influence of outliers without sacrificing central performance on typical days.",
    "CorrectedForecastLibrary-StackRidge": "This experiment builds a library of corrected forecasts (bandit bias, Fourier correction, tail correction, online weights, precip hedge, and raw mean) and trains a ridge meta-learner to combine them. The stacker uses only lag-safe candidates and adds stability features like candidate range and spread. It seeks to capture complementary corrections while limiting overfit, producing a robust ensemble that can outperform any single correction.",
    "ConvexBlend-MAEOptimized-ShrunkToEqual": "This experiment optimizes a convex blend of the same corrected-forecast library by directly minimizing validation MAE with a shrinkage penalty toward equal weights. The optimization enforces nonnegative weights that sum to one, preventing unstable extrapolation. The resulting blend is a transparent, low-variance alternative to stacking, designed to deliver small but reliable MAE gains without introducing additional model complexity or leakage risk.",
    "OnlineExpertWeights+BustHedge": "This experiment implements online exponential weighting for guidance experts using only D-2 absolute errors, with shrinkage toward uniform weights. The online forecast is then hedged toward a conservative median when bust probability is high. The resulting features capture drift-adaptive reliability while preventing weight collapse. The goal is to reduce MAE during model degradations without sacrificing performance in stable periods.",
    "SeasonSpecificSubmodels-SmoothBlend": "This experiment trains four seasonal submodels plus a global model on the baseline feature set, then blends them smoothly using day-of-year bump weights. A tuned mixing parameter controls how strongly the seasonal experts influence the final prediction. The design respects time ordering and avoids leakage while allowing season-dependent relationships to differ. It aims to improve MAE in regimes where a single global model underfits seasonal dynamics.",
    "KNNResidualCorrection-ForecastSpace": "This experiment adds a kNN analog residual correction in forecast space using a lag-safe historical pool. The neighbor search uses ensemble mean, spread, range, sea-breeze index, and calendar signals; the median neighbor residual is added to the ensemble mean. Distance diagnostics are included as features. The intent is to capture repeatable nonlinear patterns from similar forecast states while enforcing strict D-2 neighbor constraints.",
    "KNNQuantileResiduals+UncertaintyShrink": "This experiment extends the kNN residual correction by using neighbor residual quantiles (q10/q50/q90) to estimate uncertainty. The correction is shrunk when the neighbor IQR is large, reducing overcorrection in ambiguous regimes. The features remain lag-safe and use only past neighbors. This approach targets improved MAE on tail days while avoiding the instability that can occur with naive analog corrections.",
    "RegimeAwareAnalogs-Season+SBSign": "This experiment restricts analog pools by season and sea-breeze sign to enforce regime similarity before computing kNN residual corrections. When the constrained pool is too small, it falls back to the unconstrained pool and records the fallback rate. Pool size and distance metrics become features. The goal is to improve analog quality in physically consistent regimes while remaining robust when history is sparse.",
    "LocalLinearAnalogCalibration-LLR": "This experiment fits a local ridge regression on the K nearest historical analogs for each day, using guidance, spread, and calendar features. The local model yields a calibrated analog forecast and stability diagnostics such as selected alpha, coefficient norm, and condition proxy. Corrections are applied only with lag-safe neighbors and fall back to the ensemble mean when history is insufficient. It targets MAE gains via local linear structure.",
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
    baseline_a: BaselineResult
    baseline_b: BaselineResult
    bust_probs: pd.DataFrame
    group_key: pd.Series
    rng: np.random.Generator
    run_root: Path
    split_ref: dict
    cache: dict


@dataclass
class ExperimentSpec:
    experiment_id: str
    name: str
    description: str
    runner: Callable[[ExperimentContext], dict]


# ---------- Helpers ----------


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
        [run_dir / "metrics.json", run_dir / "feature_list.json", run_dir / "experiment_meta.json"],
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
    X_train_full = feature_df.loc[ctx.train_mask | ctx.val_mask].to_numpy(dtype=float)
    y_train_full = ctx.df.loc[ctx.train_mask | ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_full, y_train_full


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
        ctx.baseline_b.val_slices.get("seasonal", {}),
        ctx.baseline_b.val_slices.get("decile", {}),
        val_slices.get("seasonal", {}),
        val_slices.get("decile", {}),
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
        "metrics": metrics,
        "val_slices": val_slices,
        "test_slices": test_slices,
        "deltas_vs_baseline": deltas,
        "worth_testing": worth,
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
        "val_slices": val_slices,
        "test_slices": test_slices,
        "deltas_vs_baseline": deltas,
        "worth_testing": worth,
        "run_dir": str(run_dir),
    }


def _train_fixed_lgbm(ctx: ExperimentContext, feature_df: pd.DataFrame, objective: str = "regression") -> dict:
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
    model_train = models.train_lgbm(X_train, y_train, BASELINE_PARAMS, X_val=X_val, y_val=y_val, objective=objective)
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val) if len(y_val) else np.array([])
    model_full = models.train_lgbm(X_train_full, y_train_full, BASELINE_PARAMS, objective=objective)
    pred_test = model_full.predict(X_test)
    return {
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def _run_lgbm_experiment(
    ctx: ExperimentContext,
    *,
    experiment_id: str,
    name: str,
    description: str,
    feature_df: pd.DataFrame,
    objective: str = "regression",
) -> dict:
    preds = _train_fixed_lgbm(ctx, feature_df, objective=objective)
    return _assemble_result(
        ctx=ctx,
        experiment_id=experiment_id,
        name=name,
        description=description,
        model_name="lgbm",
        params={"objective": objective, **BASELINE_PARAMS},
        feature_columns=list(feature_df.columns),
        pred_train=preds["pred_train"],
        pred_val=preds["pred_val"],
        pred_test=preds["pred_test"],
    )


def _compute_bust_probs(ctx: ExperimentContext) -> pd.DataFrame:
    df = ctx.df
    model_cols = [c for c in GUIDANCE_COLS if c in df.columns]
    stats = features.ensemble_stats(df, model_cols)
    ens_mean = stats["mean"]
    spread = stats["range"]
    ens_range = stats["range"]
    sb_idx = features.sb_index(df).to_numpy(dtype=float)
    pc = features.disagreement_pca(df, model_cols, ctx.train_mask, n_components=1)
    pc1 = pc["pc1"].to_numpy(dtype=float)
    tail = features.compute_tail_depths(df, model_cols)
    cool_tail = tail["cool_tail"].to_numpy(dtype=float)
    cal = features.add_calendar_features(df)
    sin_doy = cal["sin_doy"].to_numpy(dtype=float)
    cos_doy = cal["cos_doy"].to_numpy(dtype=float)

    X = np.column_stack([ens_mean, spread, ens_range, sb_idx, pc1, cool_tail, sin_doy, cos_doy])
    train_df = df.loc[ctx.train_mask]
    labeler = rs_moe.BustRegimeLabeler(
        rs_moe.BustRegimeLabelerConfig(type="ex108_compat"),
        model_cols=model_cols,
        target_col="actual_tmax_f",
    ).fit(train_df)
    y_train = labeler.transform(train_df)
    if len(np.unique(y_train)) < 2:
        LOGGER.warning("Bust classifier labels collapsed to a single class; using neutral probabilities.")
        return pd.DataFrame(
            {
                "p_cool": np.zeros(len(df)),
                "p_norm": np.ones(len(df)),
                "p_warm": np.zeros(len(df)),
                "p_bust": np.zeros(len(df)),
            },
            index=df.index,
        )

    splits_idx = splits.make_time_cv_splits(train_df, n_splits=5, gap_days=2)
    oof = np.full((len(train_df), 3), np.nan, dtype=float)
    train_index = train_df.index
    for train_idx, val_idx in splits_idx:
        train_pos = train_index.get_indexer(train_idx)
        val_pos = train_index.get_indexer(val_idx)
        if len(np.unique(y_train[train_pos])) < 2:
            oof[val_pos] = np.array([0.0, 1.0, 0.0])
            continue
        model = LogisticRegression(max_iter=500, random_state=0, multi_class="multinomial")
        model.fit(X[train_idx], y_train[train_pos])
        probs = model.predict_proba(X[val_idx])
        oof_chunk = np.zeros((len(val_idx), 3), dtype=float)
        for cls_idx, cls in enumerate(model.classes_):
            oof_chunk[:, int(cls)] = probs[:, cls_idx]
        oof[val_pos] = oof_chunk

    if len(np.unique(y_train)) < 2:
        probs_full = np.column_stack([np.zeros(len(df)), np.ones(len(df)), np.zeros(len(df))])
    else:
        model_full = LogisticRegression(max_iter=500, random_state=0, multi_class="multinomial")
        model_full.fit(X[ctx.train_mask], y_train)
        probs = model_full.predict_proba(X)
        probs_full = np.zeros((len(df), 3), dtype=float)
        for cls_idx, cls in enumerate(model_full.classes_):
            probs_full[:, int(cls)] = probs[:, cls_idx]
    # fill OOF for train
    train_pos = np.where(ctx.train_mask)[0]
    valid_oof = np.isfinite(oof).all(axis=1)
    probs_full[train_pos[valid_oof]] = oof[valid_oof]

    out = pd.DataFrame(
        {
            "p_cool": probs_full[:, 0],
            "p_norm": probs_full[:, 1],
            "p_warm": probs_full[:, 2],
            "p_bust": 1.0 - probs_full[:, 1],
        },
        index=df.index,
    )
    return out


# ---------- Baselines ----------


def run_baseline_a(ctx: ExperimentContext) -> BaselineResult:
    preds = _train_fixed_lgbm(ctx, ctx.base_features)
    y_train = ctx.df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    y_val = ctx.df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    y_test = ctx.df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)
    pred_train = preds["pred_train"]
    pred_val = preds["pred_val"]
    pred_test = preds["pred_test"]
    metrics = {
        "train": eval_mod.regression_metrics(y_train, pred_train),
        "validation": eval_mod.regression_metrics(y_val, pred_val) if len(y_val) else {},
        "test": eval_mod.regression_metrics(y_test, pred_test),
    }
    val_slices = _compute_slices(ctx.df.loc[ctx.val_mask], y_val, pred_val) if len(y_val) else {}
    test_slices = _compute_slices(ctx.df.loc[ctx.test_mask], y_test, pred_test)
    return BaselineResult(
        model_name="lgbm",
        params=BASELINE_PARAMS,
        metrics=metrics,
        val_slices=val_slices,
        test_slices=test_slices,
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_baseline_b(ctx: ExperimentContext) -> BaselineResult:
    feature_df = pd.concat([ctx.base_features, ctx.bust_probs], axis=1)
    preds = _train_fixed_lgbm(ctx, feature_df)
    y_train = ctx.df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    y_val = ctx.df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    y_test = ctx.df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)
    pred_train = preds["pred_train"]
    pred_val = preds["pred_val"]
    pred_test = preds["pred_test"]
    metrics = {
        "train": eval_mod.regression_metrics(y_train, pred_train),
        "validation": eval_mod.regression_metrics(y_val, pred_val) if len(y_val) else {},
        "test": eval_mod.regression_metrics(y_test, pred_test),
    }
    val_slices = _compute_slices(ctx.df.loc[ctx.val_mask], y_val, pred_val) if len(y_val) else {}
    test_slices = _compute_slices(ctx.df.loc[ctx.test_mask], y_test, pred_test)
    return BaselineResult(
        model_name="lgbm",
        params=BASELINE_PARAMS,
        metrics=metrics,
        val_slices=val_slices,
        test_slices=test_slices,
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


# ---------- Experiment runners ----------

# NOTE: Individual experiment implementations are appended below.

def _base_with_bust(ctx: ExperimentContext) -> pd.DataFrame:
    return pd.concat([ctx.base_features, ctx.bust_probs], axis=1)


def _ens_mean(df: pd.DataFrame) -> pd.Series:
    return df[GUIDANCE_COLS].mean(axis=1)


def _compute_errors(df: pd.DataFrame, cols: list[str], group_key: pd.Series) -> pd.DataFrame:
    errors = {}
    for col in cols:
        err = df[col] - df["actual_tmax_f"]
        err = err.groupby(group_key).shift(TRUTH_LAG_DAYS)
        errors[col] = err
    return pd.DataFrame(errors, index=df.index)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _adaptive_ewma(series: pd.Series, halflife_series: pd.Series, group_key: pd.Series) -> pd.Series:
    out = np.full(len(series), np.nan, dtype=float)
    for g in group_key.unique():
        idx = np.where(group_key == g)[0]
        idx = idx[np.argsort(pd.to_datetime(series.index[idx], errors="coerce"))] if hasattr(series.index, "__len__") else idx
        last = np.nan
        for pos, row_idx in enumerate(idx):
            val = series.iloc[row_idx]
            hl = halflife_series.iloc[row_idx]
            if not np.isfinite(hl) or hl <= 0:
                hl = 10.0
            alpha = 1.0 - np.exp(np.log(0.5) / hl)
            if np.isnan(last):
                last = val if np.isfinite(val) else 0.0
            else:
                if np.isfinite(val):
                    last = alpha * val + (1.0 - alpha) * last
            out[row_idx] = last
    return pd.Series(out, index=series.index)


def _eval_fixed_candidate(ctx: ExperimentContext, feature_df: pd.DataFrame, objective: str = "regression") -> tuple[dict, float]:
    preds = _train_fixed_lgbm(ctx, feature_df, objective=objective)
    y_val = ctx.df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    val_mae = np.nan
    if len(y_val):
        val_mae = eval_mod.regression_metrics(y_val, preds["pred_val"]).get("mae", np.nan)
    return preds, float(val_mae)


def _select_best_fixed(
    ctx: ExperimentContext,
    candidates: list[tuple[str, pd.DataFrame]],
    *,
    objective: str = "regression",
) -> tuple[str, pd.DataFrame, dict]:
    best = None
    best_name = None
    best_df = None
    best_preds = None
    for name, feat in candidates:
        preds, mae = _eval_fixed_candidate(ctx, feat, objective=objective)
        if best is None or mae < best:
            best = mae
            best_name = name
            best_df = feat
            best_preds = preds
    if best_preds is None:
        raise RuntimeError("No candidates evaluated.")
    return best_name or "candidate", best_df, best_preds


def run_per_model_adaptive_ewma_bandit(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    group_key = ctx.group_key
    errors = _compute_errors(df, GUIDANCE_COLS, group_key)

    halflives = [7, 21, 60]
    windows = [7, 21, 60]
    eta_grid = [0.5, 1.0, 2.0, 3.0]

    # precompute rolling MAE for each window
    mae_w = {}
    for w in windows:
        min_obs = max(5, int(np.ceil(0.5 * w)))
        mae = errors.abs().groupby(group_key).rolling(w, min_periods=min_obs).mean()
        mae = mae.reset_index(level=0, drop=True)
        mae_w[w] = mae

    best = None
    best_preds = None
    best_feat = None
    best_eta = None
    for eta in eta_grid:
        feat = pd.DataFrame(index=df.index)
        weight_entropy = np.zeros(len(df))
        s_min = np.full(len(df), np.nan)
        corrected = np.zeros((len(df), len(GUIDANCE_COLS)))
        skill60 = mae_w[60].to_numpy(dtype=float)
        # reliability weights per model (w_i)
        w_raw = 1.0 / (skill60 + 0.25)
        w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
        w_sum = w_raw.sum(axis=1, keepdims=True)
        w_norm = np.where(w_sum == 0, 1.0 / len(GUIDANCE_COLS), w_raw / w_sum)
        weight_entropy = -np.sum(np.clip(w_norm, 1e-12, 1.0) * np.log(np.clip(w_norm, 1e-12, 1.0)), axis=1)
        s_min = np.nanmin(skill60, axis=1)

        for i, col in enumerate(GUIDANCE_COLS):
            err = errors[col].clip(-8.0, 8.0)
            # EWMA bias per halflife
            b_vals = []
            u_vals = []
            for hl in halflives:
                bias = err.groupby(group_key).apply(
                    lambda s: s.ewm(halflife=hl, min_periods=1, adjust=False).mean()
                ).reset_index(level=0, drop=True)
                b_vals.append(bias)
                # skill proxy
                s_h = mae_w[hl][col]
                u_vals.append(np.exp(-eta * s_h))
            u_stack = np.vstack([u.to_numpy(dtype=float) for u in u_vals]).T
            u_sum = np.sum(u_stack, axis=1, keepdims=True)
            u_norm = np.where(u_sum == 0, 1.0 / len(halflives), u_stack / u_sum)
            b_stack = np.vstack([b.to_numpy(dtype=float) for b in b_vals]).T
            b_star = np.sum(u_norm * b_stack, axis=1)
            corrected[:, i] = df[col].to_numpy(dtype=float) - b_star

        m_corr = np.sum(w_norm * corrected, axis=1)
        feat["m_corr"] = m_corr
        feat["weight_entropy"] = weight_entropy
        feat["s_min"] = s_min
        feat["m_corr_x_pbust"] = m_corr * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
        feat["entropy_x_spread"] = weight_entropy * df[SPREAD_COL].to_numpy(dtype=float)
        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_preds = preds
            best_feat = feature_df
            best_eta = eta

    return _assemble_result(
        ctx=ctx,
        experiment_id="PerModelAdaptiveEWMA-BanditCorr",
        name="Per-model adaptive EWMA bandit",
        description=EXPERIMENT_DESCRIPTIONS["PerModelAdaptiveEWMA-BanditCorr"],
        model_name="lgbm",
        params={"eta": best_eta, **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )

def _fit_fourier_coeffs(doy: np.ndarray, resid: np.ndarray, k: int = 3) -> np.ndarray:
    X = []
    for kk in range(1, k + 1):
        X.append(np.sin(2 * np.pi * kk * doy / 365.0))
        X.append(np.cos(2 * np.pi * kk * doy / 365.0))
    X = np.vstack(X).T
    X = np.where(np.isfinite(X), X, 0.0)
    y = np.where(np.isfinite(resid), resid, 0.0)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _apply_fourier(doy: np.ndarray, coef: np.ndarray, k: int = 3) -> np.ndarray:
    X = []
    for kk in range(1, k + 1):
        X.append(np.sin(2 * np.pi * kk * doy / 365.0))
        X.append(np.cos(2 * np.pi * kk * doy / 365.0))
    X = np.vstack(X).T
    return X @ coef


def run_fourier_k3_anom(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    dates = pd.to_datetime(df["target_date_local"])
    doy = dates.dt.dayofyear.to_numpy(dtype=float)
    model_cols = GUIDANCE_COLS
    train_mask = ctx.train_mask
    halflife_pairs = [(7, 30), (10, 40), (14, 60)]

    # fit Fourier bias coefficients on train only
    coeffs = {}
    for col in model_cols:
        resid = (df[col] - df["actual_tmax_f"]).to_numpy(dtype=float)
        coeffs[col] = _fit_fourier_coeffs(doy[train_mask], resid[train_mask], k=3)

    best = None
    best_preds = None
    best_feat = None
    best_pair = None
    for fast_hl, slow_hl in halflife_pairs:
        feat = pd.DataFrame(index=df.index)
        corr_vals = []
        for col in model_cols:
            B = _apply_fourier(doy, coeffs[col], k=3)
            resid = (df[col] - df["actual_tmax_f"]).to_numpy(dtype=float)
            u = resid - B
            u_series = pd.Series(u, index=df.index)
            u_series = u_series.groupby(ctx.group_key).shift(TRUTH_LAG_DAYS)
            u_series = u_series.clip(-8.0, 8.0)
            A_fast = u_series.groupby(ctx.group_key).apply(
                lambda s: s.ewm(halflife=fast_hl, min_periods=1, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            A_slow = u_series.groupby(ctx.group_key).apply(
                lambda s: s.ewm(halflife=slow_hl, min_periods=1, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            f_corr = df[col].to_numpy(dtype=float) - B - A_fast.to_numpy(dtype=float)
            corr_vals.append(f_corr)
            feat[f"fourier_bias_{col}"] = B
            feat[f"anom_fast_{col}"] = A_fast
            feat[f"anom_slow_{col}"] = A_slow
        corr_stack = np.vstack(corr_vals).T
        feat["corr_ens_mean"] = np.nanmean(corr_stack, axis=1)
        feat["corr_ens_median"] = np.nanmedian(corr_stack, axis=1)
        # add harmonic features k=2,3
        feat["sin2_doy"] = np.sin(4 * np.pi * doy / 365.0)
        feat["cos2_doy"] = np.cos(4 * np.pi * doy / 365.0)
        feat["sin3_doy"] = np.sin(6 * np.pi * doy / 365.0)
        feat["cos3_doy"] = np.cos(6 * np.pi * doy / 365.0)
        feat["corr_mean_x_pbust"] = feat["corr_ens_mean"] * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
        feat["corr_mean_x_spread"] = feat["corr_ens_mean"] * df[SPREAD_COL].to_numpy(dtype=float)
        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_preds = preds
            best_feat = feature_df
            best_pair = (fast_hl, slow_hl)

    return _assemble_result(
        ctx=ctx,
        experiment_id="FourierK3+AnomalyEWMA-PerModel",
        name="Fourier k3 + EWMA anomaly",
        description=EXPERIMENT_DESCRIPTIONS["FourierK3+AnomalyEWMA-PerModel"],
        model_name="lgbm",
        params={"halflife_pair": best_pair, **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


# ---------- Shared feature helpers ----------


def _get_ens_stats(ctx: ExperimentContext) -> dict[str, np.ndarray]:
    stats = ctx.cache.get("ens_stats")
    if stats is None:
        stats = features.ensemble_stats(ctx.df, GUIDANCE_COLS)
        ctx.cache["ens_stats"] = stats
    return stats


def _get_calendar(ctx: ExperimentContext) -> pd.DataFrame:
    cal = ctx.cache.get("calendar")
    if cal is None:
        cal = features.add_calendar_features(ctx.df)
        ctx.cache["calendar"] = cal
    return cal


def _get_sb_idx(ctx: ExperimentContext) -> np.ndarray:
    sb = ctx.cache.get("sb_idx")
    if sb is None:
        sb = features.sb_index(ctx.df).to_numpy(dtype=float)
        ctx.cache["sb_idx"] = sb
    return sb


def _get_tail_depths(ctx: ExperimentContext) -> pd.DataFrame:
    tail = ctx.cache.get("tail_depths")
    if tail is None:
        tail = features.compute_tail_depths(ctx.df, GUIDANCE_COLS)
        ctx.cache["tail_depths"] = tail
    return tail


def _get_disagreement_pca(ctx: ExperimentContext, n_components: int = 2) -> pd.DataFrame:
    key = f"pca_{n_components}"
    pc = ctx.cache.get(key)
    if pc is None:
        pc = features.disagreement_pca(ctx.df, GUIDANCE_COLS, ctx.train_mask, n_components=n_components)
        ctx.cache[key] = pc
    return pc


def _mos_series(ctx: ExperimentContext, code: str, stat: str = "mean") -> pd.Series:
    key = f"mos_{code}_{stat}"
    cached = ctx.cache.get(key)
    if cached is None:
        cached = features.mos_value(ctx.df, code, stat=stat)
        ctx.cache[key] = cached
    return cached


def _clip_series(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.clip(lower=low, upper=high)


def _bocpd_student_t(
    series: np.ndarray,
    *,
    hazard: float,
    max_run: int = 400,
    mu0: float = 0.0,
    kappa0: float = 1e-2,
    alpha0: float = 2.0,
    beta0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.special import logsumexp
        from scipy.stats import t as student_t
    except Exception:
        LOGGER.warning("SciPy unavailable; BOCPD probabilities set to zero.")
        n = len(series)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    n = len(series)
    log_r = np.full(max_run + 1, -np.inf, dtype=float)
    log_r[0] = 0.0
    mu = np.full(max_run + 1, mu0, dtype=float)
    kappa = np.full(max_run + 1, kappa0, dtype=float)
    alpha = np.full(max_run + 1, alpha0, dtype=float)
    beta = np.full(max_run + 1, beta0, dtype=float)
    cp_prob = np.full(n, np.nan, dtype=float)
    exp_run = np.full(n, np.nan, dtype=float)

    log_h = np.log(hazard)
    log_1mh = np.log(1.0 - hazard)
    run_idx = np.arange(max_run + 1)

    for t, x in enumerate(series):
        if not np.isfinite(x):
            probs = np.exp(log_r - logsumexp(log_r))
            cp_prob[t] = probs[0]
            exp_run[t] = float(np.sum(run_idx * probs))
            continue

        nu = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        log_pred = student_t.logpdf(x, df=nu, loc=mu, scale=scale)

        log_growth = log_r + log_1mh + log_pred
        log_cp = logsumexp(log_r + log_h + log_pred)

        log_r_new = np.full_like(log_r, -np.inf)
        log_r_new[0] = log_cp
        log_r_new[1:] = log_growth[:-1]
        log_r_new = log_r_new - logsumexp(log_r_new)

        probs = np.exp(log_r_new)
        cp_prob[t] = probs[0]
        exp_run[t] = float(np.sum(run_idx * probs))

        mu_new = np.full_like(mu, mu0)
        kappa_new = np.full_like(kappa, kappa0)
        alpha_new = np.full_like(alpha, alpha0)
        beta_new = np.full_like(beta, beta0)
        mu_prev = mu[:-1]
        kappa_prev = kappa[:-1]
        alpha_prev = alpha[:-1]
        beta_prev = beta[:-1]
        kappa_up = kappa_prev + 1.0
        mu_up = (kappa_prev * mu_prev + x) / kappa_up
        alpha_up = alpha_prev + 0.5
        beta_up = beta_prev + 0.5 * kappa_prev * (x - mu_prev) ** 2 / kappa_up
        mu_new[1:] = mu_up
        kappa_new[1:] = kappa_up
        alpha_new[1:] = alpha_up
        beta_new[1:] = beta_up

        log_r = log_r_new
        mu = mu_new
        kappa = kappa_new
        alpha = alpha_new
        beta = beta_new

    return cp_prob, exp_run


def _rolling_error_corr(
    errors: pd.DataFrame,
    *,
    window: int,
    min_obs: int,
    group_key: pd.Series,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(errors)
    mean_corr = np.full(n, np.nan, dtype=float)
    max_corr = np.full(n, np.nan, dtype=float)
    cbar = np.full((n, errors.shape[1]), np.nan, dtype=float)
    date_vals = pd.to_datetime(dates)
    groups = group_key.to_numpy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(date_vals.iloc[idx])]
        for pos, row_idx in enumerate(idx):
            start = max(0, pos - window + 1)
            window_idx = idx[start:pos + 1]
            if len(window_idx) < min_obs:
                continue
            mat = errors.iloc[window_idx].to_numpy(dtype=float)
            if mat.shape[0] < 2:
                continue
            # replace nan with column mean for correlation
            col_mean = np.nanmean(mat, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            mat = np.where(np.isfinite(mat), mat, col_mean)
            corr = np.corrcoef(mat, rowvar=False)
            if not np.isfinite(corr).any():
                continue
            iu = np.triu_indices_from(corr, k=1)
            vals = corr[iu]
            mean_corr[row_idx] = float(np.nanmean(vals))
            max_corr[row_idx] = float(np.nanmax(vals))
            for i in range(corr.shape[0]):
                cbar[row_idx, i] = float(np.nanmean(np.delete(corr[i], i)))
    return mean_corr, max_corr, cbar


def _adaptive_ewma_varying_halflife(
    series: pd.Series,
    halflife: np.ndarray,
    *,
    group_key: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    out = np.full(len(series), np.nan, dtype=float)
    groups = group_key.to_numpy()
    date_vals = pd.to_datetime(dates)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(date_vals.iloc[idx])]
        last = np.nan
        for row_idx in idx:
            val = series.iloc[row_idx]
            hl = halflife[row_idx]
            if not np.isfinite(hl) or hl <= 0:
                hl = 10.0
            alpha = 1.0 - np.exp(np.log(0.5) / hl)
            if np.isnan(last):
                last = val if np.isfinite(val) else 0.0
            else:
                if np.isfinite(val):
                    last = alpha * val + (1.0 - alpha) * last
            out[row_idx] = last
    return pd.Series(out, index=series.index)


def _random_search_lgbm(
    ctx: ExperimentContext,
    feature_df: pd.DataFrame,
    *,
    objective: str,
    n_trials: int,
    param_sampler: Callable[[], dict],
    early_stopping_rounds: int = 200,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
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
    best_params = None
    best_mae = None
    for _ in range(n_trials):
        params = param_sampler()
        model = models.train_lgbm(
            X_train,
            y_train,
            params,
            X_val=X_val,
            y_val=y_val,
            objective=objective,
            early_stopping_rounds=early_stopping_rounds,
        )
        pred_val = model.predict(X_val)
        mae = float(eval_mod.regression_metrics(y_val, pred_val).get("mae", np.inf))
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_params = params
    if best_params is None:
        best_params = param_sampler()
    model_train = models.train_lgbm(
        X_train,
        y_train,
        best_params,
        X_val=X_val,
        y_val=y_val,
        objective=objective,
        early_stopping_rounds=early_stopping_rounds,
    )
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val)
    model_full = models.train_lgbm(
        X_train_full,
        y_train_full,
        best_params,
        objective=objective,
    )
    pred_test = model_full.predict(X_test)
    return best_params, pred_train, pred_val, pred_test


def _standardize_matrix(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_vals = values[train_mask]
    mean = np.nanmean(train_vals, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(train_vals, axis=0)
    std = np.where(std == 0, 1.0, std)
    values = np.where(np.isfinite(values), values, mean)
    return (values - mean) / std, mean, std


def _compute_online_weights(
    values: np.ndarray,
    actual: np.ndarray,
    *,
    group_key: pd.Series,
    dates: pd.Series,
    eta: float,
    rho: float,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = values.shape
    weights = np.full((n, k), np.nan, dtype=float)
    y_online = np.full(n, np.nan, dtype=float)
    w_entropy = np.full(n, np.nan, dtype=float)
    eff_n = np.full(n, np.nan, dtype=float)
    groups = group_key.to_numpy()
    date_vals = pd.to_datetime(dates)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(date_vals.iloc[idx])]
        w = np.full(k, 1.0 / k, dtype=float)
        for pos, row_idx in enumerate(idx):
            if pos >= lag:
                past_idx = idx[pos - lag]
                err = np.abs(values[past_idx] - actual[past_idx])
                err = np.where(np.isfinite(err), err, 0.0)
                w = w * np.exp(-eta * err)
                w = (1.0 - rho) * w + rho / k
                w_sum = np.sum(w)
                if w_sum == 0:
                    w = np.full(k, 1.0 / k, dtype=float)
                else:
                    w = w / w_sum
            weights[row_idx] = w
            y_online[row_idx] = float(np.nansum(w * values[row_idx]))
            w_entropy[row_idx] = float(
                -np.sum(np.clip(w, 1e-12, 1.0) * np.log(np.clip(w, 1e-12, 1.0)))
            )
            eff_n[row_idx] = float(1.0 / np.sum(w ** 2))
    return y_online, w_entropy, eff_n


def _build_candidate_library(ctx: ExperimentContext) -> pd.DataFrame:
    cached = ctx.cache.get("candidate_library")
    if cached is not None:
        return cached
    df = ctx.df
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    ens_median = ens_stats["median"]
    candidates = {}

    # Raw ensemble mean
    candidates["y_raw"] = ens_mean

    # Bandit-style adaptive EWMA (fixed eta)
    errors = _compute_errors(df, GUIDANCE_COLS, ctx.group_key)
    halflives = [7, 21, 60]
    windows = [7, 21, 60]
    mae_w = {}
    for w in windows:
        min_obs = max(5, int(np.ceil(0.5 * w)))
        mae = errors.abs().groupby(ctx.group_key).rolling(w, min_periods=min_obs).mean()
        mae = mae.reset_index(level=0, drop=True)
        mae_w[w] = mae
    eta = 1.0
    skill60 = mae_w[60].to_numpy(dtype=float)
    w_raw = 1.0 / (skill60 + 0.25)
    w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
    w_sum = w_raw.sum(axis=1, keepdims=True)
    w_norm = np.where(w_sum == 0, 1.0 / len(GUIDANCE_COLS), w_raw / w_sum)
    corrected = np.zeros((len(df), len(GUIDANCE_COLS)))
    for i, col in enumerate(GUIDANCE_COLS):
        err = errors[col].clip(-8.0, 8.0)
        b_vals = []
        u_vals = []
        for hl in halflives:
            bias = err.groupby(ctx.group_key).apply(
                lambda s: s.ewm(halflife=hl, min_periods=1, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            b_vals.append(bias)
            s_h = mae_w[hl][col]
            u_vals.append(np.exp(-eta * s_h))
        u_stack = np.vstack([u.to_numpy(dtype=float) for u in u_vals]).T
        u_sum = np.sum(u_stack, axis=1, keepdims=True)
        u_norm = np.where(u_sum == 0, 1.0 / len(halflives), u_stack / u_sum)
        b_stack = np.vstack([b.to_numpy(dtype=float) for b in b_vals]).T
        b_star = np.sum(u_norm * b_stack, axis=1)
        corrected[:, i] = df[col].to_numpy(dtype=float) - b_star
    candidates["y_bandit"] = np.sum(w_norm * corrected, axis=1)

    # Fourier seasonal bias (k=1) + EWMA anomaly
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear.to_numpy(dtype=float)
    resid = ens_mean - df["actual_tmax_f"].to_numpy(dtype=float)
    coef = _fit_fourier_coeffs(doy[ctx.train_mask], resid[ctx.train_mask], k=1)
    bias_season = _apply_fourier(doy, coef, k=1)
    u = pd.Series(resid - bias_season, index=df.index).groupby(ctx.group_key).shift(TRUTH_LAG_DAYS).fillna(0.0)
    a_fast = u.groupby(ctx.group_key).apply(
        lambda s: s.ewm(halflife=10, min_periods=1, adjust=False).mean()
    ).reset_index(level=0, drop=True)
    candidates["y_fourier"] = ens_mean - bias_season - a_fast.to_numpy(dtype=float)

    # Tail-depth corrected ensemble (fixed bins)
    tail = _get_tail_depths(ctx)
    cool = tail["cool_tail"].to_numpy(dtype=float)
    warm = tail["warm_tail"].to_numpy(dtype=float)
    bins = 5
    shrink_min = 20
    cool_edges = np.quantile(cool[ctx.train_mask], np.linspace(0, 1, bins + 1))
    warm_edges = np.quantile(warm[ctx.train_mask], np.linspace(0, 1, bins + 1))
    cool_bin = np.digitize(cool, cool_edges[1:-1], right=True)
    warm_bin = np.digitize(warm, warm_edges[1:-1], right=True)
    bias_cool = np.zeros(len(df), dtype=float)
    bias_warm = np.zeros(len(df), dtype=float)
    for b in range(bins):
        mask_c = (cool_bin == b) & ctx.train_mask
        vals = resid[mask_c]
        med = float(np.nanmedian(vals)) if vals.size else 0.0
        shrink = min(1.0, vals.size / float(shrink_min)) if vals.size else 0.0
        bias_cool[cool_bin == b] = med * shrink
        mask_w = (warm_bin == b) & ctx.train_mask
        vals_w = resid[mask_w]
        med_w = float(np.nanmedian(vals_w)) if vals_w.size else 0.0
        shrink_w = min(1.0, vals_w.size / float(shrink_min)) if vals_w.size else 0.0
        bias_warm[warm_bin == b] = med_w * shrink_w
    candidates["y_tailcorr"] = ens_mean - bias_cool - bias_warm

    # Online expert weights (guidance only)
    guidance_vals = df[GUIDANCE_COLS].to_numpy(dtype=float)
    y_online, _, _ = _compute_online_weights(
        guidance_vals,
        df["actual_tmax_f"].to_numpy(dtype=float),
        group_key=ctx.group_key,
        dates=df["target_date_local"],
        eta=0.2,
        rho=0.05,
        lag=TRUTH_LAG_DAYS,
    )
    candidates["y_online"] = y_online

    # Precip hedge candidate
    p_precip = np.nanmax(np.vstack([_mos_series(ctx, "p06"), _mos_series(ctx, "p12")]), axis=0)
    q_precip = np.nanmax(np.vstack([_mos_series(ctx, "q06"), _mos_series(ctx, "q12")]), axis=0)
    p_precip = np.clip(p_precip, 0.0, 100.0)
    q_precip = np.clip(q_precip, 0.0, 5.0)
    precip_idx = (p_precip / 100.0) * np.log1p(q_precip)
    precip_idx = np.where(np.isfinite(precip_idx), precip_idx, 0.0)
    gate = _sigmoid((precip_idx - 0.1) / 0.25)
    candidates["y_hedge"] = (1.0 - gate) * ens_mean + gate * ens_median

    cand_df = pd.DataFrame(candidates, index=df.index)
    ctx.cache["candidate_library"] = cand_df
    return cand_df


def _group_sorted_indices(df: pd.DataFrame, group_key: pd.Series) -> list[np.ndarray]:
    groups = group_key.to_numpy()
    dates = pd.to_datetime(df["target_date_local"])
    out = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(dates.iloc[idx])]
        out.append(idx)
    return out


# ---------- Experiments 3-6 ----------


def run_dual_tail_depth_bias_map(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    tail = _get_tail_depths(ctx)
    ens_mean = _get_ens_stats(ctx)["mean"]
    resid = ens_mean - df["actual_tmax_f"].to_numpy(dtype=float)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for bins in (4, 5, 6):
        for shrink_min in (15, 20, 30):
            cool = tail["cool_tail"].to_numpy(dtype=float)
            warm = tail["warm_tail"].to_numpy(dtype=float)
            cool_edges = np.quantile(cool[ctx.train_mask], np.linspace(0, 1, bins + 1))
            warm_edges = np.quantile(warm[ctx.train_mask], np.linspace(0, 1, bins + 1))
            cool_bin = np.digitize(cool, cool_edges[1:-1], right=True)
            warm_bin = np.digitize(warm, warm_edges[1:-1], right=True)

            bias_cool = np.zeros(len(df), dtype=float)
            bias_warm = np.zeros(len(df), dtype=float)
            for b in range(bins):
                mask_c = (cool_bin == b) & ctx.train_mask
                vals_c = resid[mask_c]
                if vals_c.size:
                    med_c = float(np.nanmedian(vals_c))
                    shrink_c = min(1.0, vals_c.size / float(shrink_min))
                else:
                    med_c = 0.0
                    shrink_c = 0.0
                bias_cool[cool_bin == b] = med_c * shrink_c

                mask_w = (warm_bin == b) & ctx.train_mask
                vals_w = resid[mask_w]
                if vals_w.size:
                    med_w = float(np.nanmedian(vals_w))
                    shrink_w = min(1.0, vals_w.size / float(shrink_min))
                else:
                    med_w = 0.0
                    shrink_w = 0.0
                bias_warm[warm_bin == b] = med_w * shrink_w

            feat = pd.DataFrame(index=df.index)
            feat["cool_tail"] = tail["cool_tail"]
            feat["warm_tail"] = tail["warm_tail"]
            feat["tail_asym"] = tail["tail_asym"]
            feat["cool_tail_bin"] = cool_bin
            feat["warm_tail_bin"] = warm_bin
            feat["bias_cooltail_bin"] = bias_cool
            feat["bias_warmtail_bin"] = bias_warm
            feat["ens_mean_tailcorr"] = ens_mean - bias_cool - bias_warm
            feat["tail_asym_x_pwarm"] = feat["tail_asym"] * ctx.bust_probs["p_warm"].to_numpy(dtype=float)
            feat["tail_asym_x_pcool"] = feat["tail_asym"] * ctx.bust_probs["p_cool"].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"bins": bins, "shrink_min": shrink_min}

    return _assemble_result(
        ctx=ctx,
        experiment_id="DualTailDepthBiasMap",
        name="Dual tail-depth bias map",
        description=EXPERIMENT_DESCRIPTIONS["DualTailDepthBiasMap"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_error_correlation_diversity_weights(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    errors = _compute_errors(df, GUIDANCE_COLS, ctx.group_key)
    ens_mean = _get_ens_stats(ctx)["mean"]
    best = None
    best_feat = None
    best_preds = None
    best_params = None
    dates = df["target_date_local"]

    # rolling MAE per model (W=60)
    mae_cols = {}
    for col in GUIDANCE_COLS:
        mae_cols[col] = features.rolling_mae(
            errors[col],
            window=60,
            min_obs=30,
            group_key=ctx.group_key,
            lag=0,
        ).fillna(1.0)

    for window in (60, 90, 120):
        min_obs = max(30, int(window * 0.5))
        mean_corr, max_corr, cbar = _rolling_error_corr(
            errors,
            window=window,
            min_obs=min_obs,
            group_key=ctx.group_key,
            dates=dates,
        )
        mean_corr = np.where(np.isfinite(mean_corr), mean_corr, 0.0)
        max_corr = np.where(np.isfinite(max_corr), max_corr, 0.0)
        cbar = np.where(np.isfinite(cbar), cbar, 0.0)
        for penalty in (0.5, 1.0, 2.0):
            s_vals = np.vstack([mae_cols[c].to_numpy(dtype=float) for c in GUIDANCE_COLS]).T
            w_raw = 1.0 / (s_vals + 0.25)
            w_raw = w_raw / (1.0 + penalty * cbar)
            w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
            w_sum = w_raw.sum(axis=1, keepdims=True)
            w_norm = np.where(w_sum == 0, 1.0 / len(GUIDANCE_COLS), w_raw / w_sum)
            guidance_vals = df[GUIDANCE_COLS].to_numpy(dtype=float)
            ens_mean_div = np.nansum(w_norm * guidance_vals, axis=1)
            eff_n = 1.0 / np.sum(w_norm ** 2, axis=1)

            feat = pd.DataFrame(index=df.index)
            feat["ens_mean_div"] = ens_mean_div
            feat["ens_mean_div_minus_mean"] = ens_mean_div - ens_mean
            feat["err_corr_mean"] = mean_corr
            feat["err_corr_max"] = max_corr
            feat["err_eff_n"] = eff_n
            feat["err_corr_mean_x_spread"] = mean_corr * df[SPREAD_COL].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"window": window, "penalty": penalty}

    return _assemble_result(
        ctx=ctx,
        experiment_id="ErrorCorrelationDiversityWeights",
        name="Error correlation diversity weights",
        description=EXPERIMENT_DESCRIPTIONS["ErrorCorrelationDiversityWeights"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_cusum_bocpd_gated_bias(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_mean = _get_ens_stats(ctx)["mean"]
    resid = pd.Series(ens_mean - df["actual_tmax_f"].to_numpy(dtype=float), index=df.index)
    resid_clip = resid.clip(-8.0, 8.0)
    bias_fast = features.ewma(resid_clip, halflife=7, group_key=ctx.group_key, lag=TRUTH_LAG_DAYS)
    bias_slow = features.ewma(resid_clip, halflife=45, group_key=ctx.group_key, lag=TRUTH_LAG_DAYS)
    resid_shift = resid.groupby(ctx.group_key).shift(TRUTH_LAG_DAYS).to_numpy(dtype=float)
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    groups = ctx.group_key.to_numpy()

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for hazard in (1 / 50, 1 / 100, 1 / 200):
        p_cp = np.full(len(df), np.nan, dtype=float)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            idx = idx[np.argsort(dates[idx])]
            series = resid_shift[idx]
            cp_prob, _ = _bocpd_student_t(series, hazard=hazard, max_run=400)
            p_cp[idx] = cp_prob
        p_cp = np.where(np.isfinite(p_cp), p_cp, 0.0)

        for h in (3, 4, 5, 6):
            g_plus = np.full(len(df), 0.0, dtype=float)
            g_minus = np.full(len(df), 0.0, dtype=float)
            drift_flag = np.zeros(len(df), dtype=int)
            time_since = np.full(len(df), np.nan, dtype=float)
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                idx = idx[np.argsort(dates[idx])]
                last_reset = None
                gp = 0.0
                gm = 0.0
                for pos, row_idx in enumerate(idx):
                    if pos < TRUTH_LAG_DAYS:
                        continue
                    x = resid_shift[row_idx]
                    if not np.isfinite(x):
                        x = 0.0
                    gp = max(0.0, gp + x - 0.25)
                    gm = max(0.0, gm - x - 0.25)
                    g_plus[row_idx] = gp
                    g_minus[row_idx] = gm
                    if max(gp, gm) > h:
                        drift_flag[row_idx] = 1
                        last_reset = pos
                    if last_reset is None:
                        time_since[row_idx] = 60.0
                    else:
                        time_since[row_idx] = float(min(60, pos - last_reset))

            gate = _sigmoid((p_cp - 0.3) / 0.1)
            bias_slow_reset = bias_slow.to_numpy(dtype=float) * (1.0 - drift_flag)
            bias = gate * bias_fast.to_numpy(dtype=float) + (1.0 - gate) * bias_slow_reset
            ens_corr = ens_mean - bias

            feat = pd.DataFrame(index=df.index)
            feat["p_cp"] = p_cp
            feat["bias_fast"] = bias_fast
            feat["bias_slow"] = bias_slow
            feat["bias_final"] = bias
            feat["drift_flag"] = drift_flag
            feat["time_since_reset"] = time_since
            feat["ens_mean_corr"] = ens_corr
            feat["bias_x_pbust"] = bias * ctx.bust_probs["p_bust"].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"hazard": hazard, "h": h}

    return _assemble_result(
        ctx=ctx,
        experiment_id="CUSUMReset+BOCPDGatedBias",
        name="CUSUM reset + BOCPD gated bias",
        description=EXPERIMENT_DESCRIPTIONS["CUSUMReset+BOCPDGatedBias"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_seasonal_adaptive_half_life_bias(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    cal = _get_calendar(ctx)
    doy = cal["day_of_year"].to_numpy(dtype=float)
    ens_mean = _get_ens_stats(ctx)["mean"]
    resid = pd.Series(ens_mean - df["actual_tmax_f"].to_numpy(dtype=float), index=df.index)
    resid_shift = resid.groupby(ctx.group_key).shift(TRUTH_LAG_DAYS).fillna(0.0)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for H0 in (20, 30, 40):
        for H1 in (-15, -10, -5, 0, 5, 10, 15):
            for H2 in (-10, -5, 0, 5, 10):
                H = H0 + H1 * np.cos(2 * np.pi * doy / 365.0) + H2 * np.cos(4 * np.pi * doy / 365.0)
                H = np.clip(H, 5.0, 80.0)
                bias = _adaptive_ewma_varying_halflife(
                    resid_shift,
                    H,
                    group_key=ctx.group_key,
                    dates=df["target_date_local"],
                )
                ens_corr = ens_mean - bias.to_numpy(dtype=float)
                feat = pd.DataFrame(index=df.index)
                feat["bias_seas"] = bias
                feat["bias_seas_x_pbust"] = bias.to_numpy(dtype=float) * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
                feat["bias_seas_x_spread"] = bias.to_numpy(dtype=float) * df[SPREAD_COL].to_numpy(dtype=float)
                feat["half_life_doy"] = H
                feat["ens_mean_seascorr"] = ens_corr
                feature_df = pd.concat([base, feat], axis=1)
                preds, mae = _eval_fixed_candidate(ctx, feature_df)
                if best is None or mae < best:
                    best = mae
                    best_feat = feature_df
                    best_preds = preds
                    best_params = {"H0": H0, "H1": H1, "H2": H2}

    return _assemble_result(
        ctx=ctx,
        experiment_id="SeasonalAdaptiveHalfLifeBias",
        name="Seasonal adaptive half-life bias",
        description=EXPERIMENT_DESCRIPTIONS["SeasonalAdaptiveHalfLifeBias"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_mos_dpd_humidity(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    tmp = _mos_series(ctx, "tmp")
    dpt = _mos_series(ctx, "dpt")
    dpd = (tmp - dpt).clip(lower=0.0, upper=40.0)
    missing = tmp.isna() | dpt.isna()
    train_median = float(np.nanmedian(dpd[ctx.train_mask])) if ctx.train_mask.any() else 0.0
    dpd = dpd.fillna(train_median)
    dpd = dpd.where(~missing, train_median)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for window in (21, 30, 45):
        min_obs = max(5, int(np.ceil(0.5 * window)))
        dpd_rm = features.rolling_mean(dpd, window=window, min_obs=min_obs, group_key=ctx.group_key, lag=TRUTH_LAG_DAYS)
        dpd_anom = dpd - dpd_rm
        for bins in (4, 5, 6):
            edges = np.quantile(dpd[ctx.train_mask], np.linspace(0, 1, bins + 1))
            dpd_bin = np.digitize(dpd.to_numpy(dtype=float), edges[1:-1], right=True)

            feat = pd.DataFrame(index=df.index)
            feat["mos_dpd"] = dpd
            feat["mos_dpd_anom"] = dpd_anom
            feat["mos_dpd_bin"] = dpd_bin
            feat["mos_dpd_missing"] = missing.astype(int)
            feat["mos_dpd_x_pbust"] = dpd.to_numpy(dtype=float) * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
            feat["mos_dpd_anom_x_spread"] = dpd_anom.to_numpy(dtype=float) * df[SPREAD_COL].to_numpy(dtype=float)
            feat["mos_dpd_x_sb"] = dpd.to_numpy(dtype=float) * _get_sb_idx(ctx)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"window": window, "bins": bins}

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOS-DewpointDepressionHumiditySignals",
        name="MOS dewpoint depression humidity signals",
        description=EXPERIMENT_DESCRIPTIONS["MOS-DewpointDepressionHumiditySignals"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_mos_wind_sea_breeze(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    wdr = _mos_series(ctx, "wdr")
    wsp = _mos_series(ctx, "wsp").clip(lower=0.0, upper=40.0)
    missing = wdr.isna() | wsp.isna()
    radians = np.deg2rad(wdr.fillna(0.0))
    u = -wsp.fillna(0.0) * np.sin(radians)
    v = -wsp.fillna(0.0) * np.cos(radians)
    sb_idx = _get_sb_idx(ctx)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for window in (7, 14, 21):
        min_obs = max(3, int(np.ceil(0.5 * window)))
        u_rm = features.rolling_mean(u, window=window, min_obs=min_obs, group_key=ctx.group_key, lag=TRUTH_LAG_DAYS)
        v_rm = features.rolling_mean(v, window=window, min_obs=min_obs, group_key=ctx.group_key, lag=TRUTH_LAG_DAYS)
        du = u - u_rm
        dv = v - v_rm

        feat = pd.DataFrame(index=df.index)
        feat["mos_u"] = u
        feat["mos_v"] = v
        feat["mos_u_rm"] = u_rm
        feat["mos_v_rm"] = v_rm
        feat["mos_du"] = du
        feat["mos_dv"] = dv
        feat["mos_wind_missing"] = missing.astype(int)
        feat["sb_idx"] = sb_idx
        feat["sb_x_u"] = sb_idx * u.to_numpy(dtype=float)
        feat["sb_x_v"] = sb_idx * v.to_numpy(dtype=float)
        feat["du_x_pbust"] = du.to_numpy(dtype=float) * ctx.bust_probs["p_bust"].to_numpy(dtype=float)

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"window": window}

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOS-WindVectorSeaBreezeGating",
        name="MOS wind vector sea-breeze gating",
        description=EXPERIMENT_DESCRIPTIONS["MOS-WindVectorSeaBreezeGating"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_mos_cloud_visibility(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    cig = _mos_series(ctx, "cig")
    vis = _mos_series(ctx, "vis")
    cig_clip = cig.clip(lower=0.0, upper=20000.0)
    vis_clip = vis.clip(lower=0.0, upper=10.0)
    log_cig = np.log1p(cig_clip.fillna(0.0))
    log_vis = np.log1p(vis_clip.fillna(0.0))
    missing = cig.isna() | vis.isna()
    ens_mean = _get_ens_stats(ctx)["mean"]

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for window in (14, 21, 30):
        min_obs = max(5, int(np.ceil(0.5 * window)))
        for coef in (0.0, 0.5, 1.0):
            cloud_idx = -log_cig + coef * log_vis
            cloud_rm = features.rolling_mean(
                pd.Series(cloud_idx, index=df.index),
                window=window,
                min_obs=min_obs,
                group_key=ctx.group_key,
                lag=TRUTH_LAG_DAYS,
            )
            cloud_anom = cloud_idx - cloud_rm

            feat = pd.DataFrame(index=df.index)
            feat["cloud_idx"] = cloud_idx
            feat["cloud_anom"] = cloud_anom
            feat["log_cig"] = log_cig
            feat["log_vis"] = log_vis
            feat["cloud_missing"] = missing.astype(int)
            feat["cloud_x_ens_mean"] = cloud_idx * ens_mean
            feat["cloud_anom_x_pbust"] = cloud_anom.to_numpy(dtype=float) * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
            feat["cloud_x_spread"] = cloud_idx * df[SPREAD_COL].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"window": window, "coef": coef}

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOS-CloudVisibilityRadiationProxy",
        name="MOS cloud/visibility radiation proxy",
        description=EXPERIMENT_DESCRIPTIONS["MOS-CloudVisibilityRadiationProxy"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_mos_precip_hedge(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    codes = ["p06", "p12", "q06", "q12", "pos", "poz"]
    values = {c: _mos_series(ctx, c) for c in codes}
    p_precip = np.nanmax(np.vstack([values["p06"], values["p12"]]), axis=0)
    q_precip = np.nanmax(np.vstack([values["q06"], values["q12"]]), axis=0)
    p_precip = np.clip(p_precip, 0.0, 100.0)
    q_precip = np.clip(q_precip, 0.0, 5.0)
    precip_idx = (p_precip / 100.0) * np.log1p(q_precip)
    missing = np.isnan(p_precip) & np.isnan(q_precip)
    precip_idx = np.where(np.isfinite(precip_idx), precip_idx, 0.0)

    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    ens_median = ens_stats["median"]
    ens_min = ens_stats["min"]

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for t0 in (0.05, 0.1, 0.15, 0.2):
        for conservative in ("median", "min"):
            y_cons = ens_median if conservative == "median" else ens_min
            gate = _sigmoid((precip_idx - t0) / 0.25)
            y_hedge = (1.0 - gate) * ens_mean + gate * y_cons

            feat = pd.DataFrame(index=df.index)
            feat["precip_idx"] = precip_idx
            feat["precip_gate"] = gate
            feat["y_hedge"] = y_hedge
            feat["precip_missing"] = missing.astype(int)
            feat["precip_idx_x_pbust"] = precip_idx * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
            feat["precip_idx_x_spread"] = precip_idx * df[SPREAD_COL].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"t0": t0, "conservative": conservative}

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOS-PrecipLikelihoodCoolingHedge",
        name="MOS precip likelihood cooling hedge",
        description=EXPERIMENT_DESCRIPTIONS["MOS-PrecipLikelihoodCoolingHedge"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_mos_t06_heating(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    t06 = _mos_series(ctx, "t06")
    t06_1 = _mos_series(ctx, "t06_1")
    t06_2 = _mos_series(ctx, "t06_2")
    t06_stack = np.vstack([t06, t06_1, t06_2])
    t06_mean = np.nanmean(t06_stack, axis=0)
    tmp = _mos_series(ctx, "tmp")
    ens_mean = _get_ens_stats(ctx)["mean"]
    fallback = np.where(np.isfinite(tmp), tmp.to_numpy(dtype=float), ens_mean)
    t06_mean = np.where(np.isfinite(t06_mean), t06_mean, fallback)
    heat_potential = np.clip(ens_mean - t06_mean, 0.0, 30.0)
    missing = np.isnan(t06_stack).all(axis=0)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for window in (14, 21, 30):
        min_obs = max(5, int(np.ceil(0.5 * window)))
        hp_rm = features.rolling_mean(
            pd.Series(heat_potential, index=df.index),
            window=window,
            min_obs=min_obs,
            group_key=ctx.group_key,
            lag=TRUTH_LAG_DAYS,
        )
        hp_anom = heat_potential - hp_rm

        feat = pd.DataFrame(index=df.index)
        feat["heat_potential"] = heat_potential
        feat["heat_potential_anom"] = hp_anom
        feat["heat_missing"] = missing.astype(int)
        feat["heat_x_pbust"] = heat_potential * ctx.bust_probs["p_bust"].to_numpy(dtype=float)

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"window": window}

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOS-T06HeatingPotentialFeatures",
        name="MOS t06 heating potential",
        description=EXPERIMENT_DESCRIPTIONS["MOS-T06HeatingPotentialFeatures"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_mos_missingness_quality(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    code_sets = {
        "A": ["tmp", "dpt", "wdr", "wsp", "cig", "vis"],
        "B": ["tmp", "dpt", "wdr", "wsp", "cig", "vis", "p06", "p12", "q06", "q12"],
    }
    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for label, codes in code_sets.items():
        feat = pd.DataFrame(index=df.index)
        missing_matrix = []
        dsl_matrix = []
        for code in codes:
            vals = _mos_series(ctx, code)
            is_missing = vals.isna().astype(int)
            feat[f"mos_{code}_missing"] = is_missing
            missing_matrix.append(is_missing.to_numpy(dtype=float))
            dsl = tfl.days_since_event(~is_missing.astype(bool), lag=TRUTH_LAG_DAYS, cap=60, group_key=ctx.group_key)
            feat[f"mos_{code}_dsl"] = dsl
            feat[f"mos_{code}_dsl_capped"] = (dsl >= 60).astype(int)
            dsl_matrix.append(dsl.to_numpy(dtype=float))
        miss_stack = np.vstack(missing_matrix).T if missing_matrix else np.zeros((len(df), 0))
        dsl_stack = np.vstack(dsl_matrix).T if dsl_matrix else np.zeros((len(df), 0))
        miss_count = miss_stack.sum(axis=1) if miss_stack.size else np.zeros(len(df))
        feat["mos_missing_frac"] = miss_count / float(len(codes))
        feat["mos_available_cnt"] = len(codes) - miss_count
        days_since_any = np.nanmin(dsl_stack, axis=1) if dsl_stack.size else np.full(len(df), 60.0)
        days_since_any = np.where(np.isfinite(days_since_any), days_since_any, 60.0)
        feat["mos_days_since_any"] = days_since_any
        feat["mos_reliability"] = np.exp(-days_since_any / 14.0)
        feat["mos_missing_frac_x_pbust"] = feat["mos_missing_frac"].to_numpy(dtype=float) * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
        feat["mos_reliability_x_spread"] = feat["mos_reliability"].to_numpy(dtype=float) * df[SPREAD_COL].to_numpy(dtype=float)

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"code_set": label}

    return _assemble_result(
        ctx=ctx,
        experiment_id="MOS-MissingnessQualityEncoding",
        name="MOS missingness quality encoding",
        description=EXPERIMENT_DESCRIPTIONS["MOS-MissingnessQualityEncoding"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_bust_classifier_v2(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = ctx.base_features
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    model_range = ens_stats["max"] - ens_stats["min"]
    sb_idx = _get_sb_idx(ctx)
    pc1 = _get_disagreement_pca(ctx, n_components=1)["pc1"].to_numpy(dtype=float)
    cal = _get_calendar(ctx)
    sin_doy = cal["sin_doy"].to_numpy(dtype=float)
    cos_doy = cal["cos_doy"].to_numpy(dtype=float)
    dpd = (_mos_series(ctx, "tmp") - _mos_series(ctx, "dpt")).to_numpy(dtype=float)
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    ).to_numpy(dtype=float)

    X = np.column_stack(
        [
            ens_mean,
            df[SPREAD_COL].to_numpy(dtype=float),
            model_range,
            sb_idx,
            pc1,
            sin_doy,
            cos_doy,
            dpd,
            cloud_idx,
        ]
    )
    # impute missing with train means
    train_vals = X[ctx.train_mask]
    col_mean = np.nanmean(train_vals, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X = np.where(np.isfinite(X), X, col_mean)

    resid = ens_mean - df["actual_tmax_f"].to_numpy(dtype=float)
    q10, q30, q70, q90 = np.quantile(resid[ctx.train_mask], [0.1, 0.3, 0.7, 0.9])
    labels = np.zeros(len(df), dtype=int)
    labels[resid <= q10] = 0
    labels[(resid > q10) & (resid <= q30)] = 1
    labels[(resid > q30) & (resid < q70)] = 2
    labels[(resid >= q70) & (resid < q90)] = 3
    labels[resid >= q90] = 4

    train_df = df.loc[ctx.train_mask].copy()
    X_train = X[ctx.train_mask]
    y_train = labels[ctx.train_mask]
    splits_idx_raw = splits.make_time_cv_splits(train_df, n_splits=4, gap_days=7)
    index_map = {idx: pos for pos, idx in enumerate(train_df.index)}
    splits_idx = []
    for tr_idx, val_idx in splits_idx_raw:
        tr_pos = np.array([index_map[i] for i in tr_idx], dtype=int)
        val_pos = np.array([index_map[i] for i in val_idx], dtype=int)
        splits_idx.append((tr_pos, val_pos))
    oof = np.full((len(train_df), 5), np.nan, dtype=float)

    param_grid = []
    rng = ctx.rng
    for _ in range(20):
        param_grid.append(
            {
                "max_depth": int(rng.integers(3, 6)),
                "num_leaves": int(rng.choice([15, 31, 63])),
                "learning_rate": float(rng.uniform(0.03, 0.12)),
                "min_data_in_leaf": int(rng.choice([10, 20, 40])),
            }
        )
    from sklearn.metrics import log_loss

    def _train_classifier(X_tr: np.ndarray, y_tr: np.ndarray, params: dict) -> Any:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=500,
            random_state=0,
            objective="multiclass",
            num_class=5,
            n_jobs=1,
        )
        model.set_params(**params)
        model.fit(X_tr, y_tr)
        return model

    best_params = None
    best_oof = None
    best_loss = None
    for params in param_grid:
        if splits_idx:
            oof[:] = np.nan
            for tr_idx, val_idx in splits_idx:
                model_fold = _train_classifier(X_train[tr_idx], y_train[tr_idx], params)
                oof[val_idx] = model_fold.predict_proba(X_train[val_idx])
            if not np.isfinite(oof).all():
                continue
            loss = float(log_loss(y_train, oof, labels=[0, 1, 2, 3, 4]))
        else:
            model = _train_classifier(X_train, y_train, params)
            preds = model.predict_proba(X_train)
            loss = float(log_loss(y_train, preds, labels=[0, 1, 2, 3, 4]))
            oof = preds
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_params = params
            best_oof = oof.copy()

    if best_params is None:
        best_params = param_grid[0]
    clf = _train_classifier(X_train, y_train, best_params)
    probs_all = clf.predict_proba(X)
    probs = np.zeros((len(df), 5), dtype=float)
    for cls_idx, cls in enumerate(clf.classes_):
        probs[:, int(cls)] = probs_all[:, cls_idx]
    # OOF replacement for train if available
    if best_oof is not None and splits_idx:
        train_pos = np.where(ctx.train_mask)[0]
        probs[train_pos] = best_oof

    # expected signed error per class (train only)
    mu = []
    for k in range(5):
        mask = (labels == k) & ctx.train_mask
        mu.append(float(np.nanmean(resid[mask])) if np.any(mask) else 0.0)
    mu = np.array(mu, dtype=float)
    e_hat = probs @ mu

    feat_probs = pd.DataFrame(
        {
            "p_cool_big": probs[:, 0],
            "p_cool_small": probs[:, 1],
            "p_norm5": probs[:, 2],
            "p_warm_small": probs[:, 3],
            "p_warm_big": probs[:, 4],
            "e_hat": e_hat,
            "bust_mag": probs[:, 0] + probs[:, 4],
        },
        index=df.index,
    )

    candidates = [
        ("new_only", pd.concat([base, feat_probs], axis=1)),
        ("new_plus_old", pd.concat([_base_with_bust(ctx), feat_probs], axis=1)),
    ]
    name, best_feat, preds = _select_best_fixed(ctx, candidates)
    return _assemble_result(
        ctx=ctx,
        experiment_id="BustClassifierV2-Ordinal5Class",
        name="Bust classifier v2 ordinal 5-class",
        description=EXPERIMENT_DESCRIPTIONS["BustClassifierV2-Ordinal5Class"],
        model_name="lgbm",
        params={"classifier": "lgbm", "variant": name, **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=preds["pred_train"],
        pred_val=preds["pred_val"],
        pred_test=preds["pred_test"],
    )


def run_spread_seabreeze_moe(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    tail = _get_tail_depths(ctx)
    sb_idx = _get_sb_idx(ctx)
    spread = df[SPREAD_COL].to_numpy(dtype=float)

    # Expert A: Fourier k=1 bias + EWMA anomaly
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear.to_numpy(dtype=float)
    coeff = _fit_fourier_coeffs(doy[ctx.train_mask], (ens_mean - df["actual_tmax_f"].to_numpy(dtype=float))[ctx.train_mask], k=1)
    bias_season = _apply_fourier(doy, coeff, k=1)
    resid = ens_mean - df["actual_tmax_f"].to_numpy(dtype=float)
    u = pd.Series(resid - bias_season, index=df.index).groupby(ctx.group_key).shift(TRUTH_LAG_DAYS).fillna(0.0).clip(-8.0, 8.0)
    a_fast = u.groupby(ctx.group_key).apply(lambda s: s.ewm(halflife=10, min_periods=1, adjust=False).mean()).reset_index(level=0, drop=True)
    y_a = ens_mean - bias_season - a_fast.to_numpy(dtype=float)

    # Expert B: tail-depth correction (fixed bins)
    cool = tail["cool_tail"].to_numpy(dtype=float)
    warm = tail["warm_tail"].to_numpy(dtype=float)
    bins = 5
    shrink_min = 20
    cool_edges = np.quantile(cool[ctx.train_mask], np.linspace(0, 1, bins + 1))
    warm_edges = np.quantile(warm[ctx.train_mask], np.linspace(0, 1, bins + 1))
    cool_bin = np.digitize(cool, cool_edges[1:-1], right=True)
    warm_bin = np.digitize(warm, warm_edges[1:-1], right=True)
    resid_train = resid
    bias_cool = np.zeros(len(df), dtype=float)
    bias_warm = np.zeros(len(df), dtype=float)
    for b in range(bins):
        mask_c = (cool_bin == b) & ctx.train_mask
        vals = resid_train[mask_c]
        med = float(np.nanmedian(vals)) if vals.size else 0.0
        shrink = min(1.0, vals.size / float(shrink_min)) if vals.size else 0.0
        bias_cool[cool_bin == b] = med * shrink
        mask_w = (warm_bin == b) & ctx.train_mask
        vals_w = resid_train[mask_w]
        med_w = float(np.nanmedian(vals_w)) if vals_w.size else 0.0
        shrink_w = min(1.0, vals_w.size / float(shrink_min)) if vals_w.size else 0.0
        bias_warm[warm_bin == b] = med_w * shrink_w
    y_b = ens_mean - bias_cool - bias_warm

    # Gate inputs
    g1 = (spread - np.nanmedian(spread[ctx.train_mask])) / (np.nanpercentile(spread[ctx.train_mask], 75) - np.nanpercentile(spread[ctx.train_mask], 25) + 1e-6)
    g2 = (sb_idx - np.nanmedian(sb_idx[ctx.train_mask])) / (np.nanpercentile(sb_idx[ctx.train_mask], 75) - np.nanpercentile(sb_idx[ctx.train_mask], 25) + 1e-6)
    gate_X = np.column_stack([g1, g2])
    err_a = resid + (y_a - ens_mean)
    err_b = resid + (y_b - ens_mean)
    label = (np.abs(err_b) < np.abs(err_a)).astype(int)
    train_idx = ctx.train_mask

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for C in (0.1, 1.0, 10.0):
        gate = LogisticRegression(max_iter=500, C=C)
        gate.fit(gate_X[train_idx], label[train_idx])
        p_gate = gate.predict_proba(gate_X)[:, 1]
        y_moe = (1.0 - p_gate) * y_a + p_gate * y_b

        feat = pd.DataFrame(index=df.index)
        feat["p_gate"] = p_gate
        feat["y_expert_a"] = y_a
        feat["y_expert_b"] = y_b
        feat["y_moe"] = y_moe
        feat["p_gate_x_pbust"] = p_gate * ctx.bust_probs["p_bust"].to_numpy(dtype=float)

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"C": C}

    return _assemble_result(
        ctx=ctx,
        experiment_id="SpreadxSeaBreezeMixtureOfExpertsBias",
        name="Spread × sea-breeze mixture-of-experts bias",
        description=EXPERIMENT_DESCRIPTIONS["SpreadxSeaBreezeMixtureOfExpertsBias"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_residual_hmm_stateprob(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    spread = df[SPREAD_COL].to_numpy(dtype=float)
    resid = ens_mean - df["actual_tmax_f"].to_numpy(dtype=float)
    z = resid / (spread + 0.5)
    z_shift = pd.Series(z, index=df.index).groupby(ctx.group_key).shift(TRUTH_LAG_DAYS).fillna(0.0).to_numpy(dtype=float)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for n_states in (2, 3, 4):
        probs = features.fit_hmm_probs(z_shift, ctx.train_mask, n_states=n_states, seed=0)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        # state means from train
        mu = []
        for k in range(n_states):
            w = probs[:, k]
            mask = ctx.train_mask
            denom = np.sum(w[mask]) if np.any(mask) else 0.0
            mu_k = float(np.sum(w[mask] * z[mask]) / denom) if denom > 0 else 0.0
            mu.append(mu_k)
        mu = np.array(mu, dtype=float)
        hmm_bias = (probs @ mu) * (spread + 0.5)
        ens_corr = ens_mean - hmm_bias

        feat = pd.DataFrame(index=df.index)
        for k in range(n_states):
            feat[f"hmm_p{k+1}"] = probs[:, k]
        feat["hmm_bias"] = hmm_bias
        feat["ens_mean_hmmcorr"] = ens_corr
        feat["hmm_bias_x_pbust"] = hmm_bias * ctx.bust_probs["p_bust"].to_numpy(dtype=float)

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"n_states": n_states}

    return _assemble_result(
        ctx=ctx,
        experiment_id="ResidualHMM-StateProbBias",
        name="Residual HMM state-prob bias",
        description=EXPERIMENT_DESCRIPTIONS["ResidualHMM-StateProbBias"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_season_transition_spread_gate(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for centers in ((110, 290), (120, 300), (130, 310)):
        for width in (20, 25, 30):
            spring = features.transition_bumps(df["target_date_local"], centers[0], width)
            fall = features.transition_bumps(df["target_date_local"], centers[1], width)
            transition = spring + fall
            feat = pd.DataFrame(index=df.index)
            feat["transition"] = transition
            feat["transition_spread"] = transition * df[SPREAD_COL].to_numpy(dtype=float)
            feat["transition_bust"] = transition * ctx.bust_probs["p_bust"].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"centers": centers, "width": width}

    return _assemble_result(
        ctx=ctx,
        experiment_id="SeasonTransitionSpreadGate",
        name="Season transition spread gate",
        description=EXPERIMENT_DESCRIPTIONS["SeasonTransitionSpreadGate"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_pca2_mos_regime_gmm(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    pc = _get_disagreement_pca(ctx, n_components=2)
    sb_idx = _get_sb_idx(ctx)
    spread = df[SPREAD_COL].to_numpy(dtype=float)
    dpd = (_mos_series(ctx, "tmp") - _mos_series(ctx, "dpt")).to_numpy(dtype=float)
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    ).to_numpy(dtype=float)

    base_vec = np.column_stack([pc["pc1"].to_numpy(dtype=float), pc["pc2"].to_numpy(dtype=float), spread, sb_idx])
    if np.isfinite(dpd).any() and np.isfinite(cloud_idx).any():
        R = np.column_stack([base_vec, dpd, cloud_idx])
        version = 2
    else:
        R = base_vec
        version = 1

    # impute train mean
    train_R = R[ctx.train_mask]
    col_mean = np.nanmean(train_R, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    R = np.where(np.isfinite(R), R, col_mean)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for k in (3, 4, 5):
        for cov in ("diag", "full"):
            gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=0)
            gmm.fit(R[ctx.train_mask])
            probs = gmm.predict_proba(R)
            entropy = -np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
            kmax = np.argmax(probs, axis=1)

            feat = pd.DataFrame(index=df.index)
            for idx in range(k):
                feat[f"gmm_p{idx+1}"] = probs[:, idx]
            feat["gmm_entropy"] = entropy
            feat["gmm_kmax"] = kmax
            feat["gmm_entropy_x_pbust"] = entropy * ctx.bust_probs["p_bust"].to_numpy(dtype=float)
            feat["regime_vec_version"] = version

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"k": k, "cov": cov, "version": version}

    return _assemble_result(
        ctx=ctx,
        experiment_id="PCA2+MOSRegimeGMMProbFeatures",
        name="PCA2 + MOS regime GMM probs",
        description=EXPERIMENT_DESCRIPTIONS["PCA2+MOSRegimeGMMProbFeatures"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_lgbm_l1_objective(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    )
    feature_df = pd.concat([base, cloud_idx.rename("cloud_idx")], axis=1)

    def sampler() -> dict:
        return {
            "num_leaves": int(ctx.rng.integers(15, 128)),
            "learning_rate": float(ctx.rng.uniform(0.005, 0.08)),
            "n_estimators": int(ctx.rng.integers(800, 6001)),
            "min_data_in_leaf": int(ctx.rng.integers(10, 121)),
            "feature_fraction": float(ctx.rng.uniform(0.6, 1.0)),
            "bagging_fraction": float(ctx.rng.uniform(0.6, 1.0)),
            "bagging_freq": int(ctx.rng.choice([0, 1, 5])),
            "reg_lambda": float(ctx.rng.uniform(0.0, 10.0)),
        }

    best_params, pred_train, pred_val, pred_test = _random_search_lgbm(
        ctx, feature_df, objective="regression_l1", n_trials=30, param_sampler=sampler
    )
    params = {**best_params, "objective": "regression_l1"}
    return _assemble_result(
        ctx=ctx,
        experiment_id="LightGBM-RegressionL1MAEObjective",
        name="LightGBM regression L1 objective",
        description=EXPERIMENT_DESCRIPTIONS["LightGBM-RegressionL1MAEObjective"],
        model_name="lgbm",
        params=params,
        feature_columns=list(feature_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_catboost_mae_bust_mos(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    dpd = (_mos_series(ctx, "tmp") - _mos_series(ctx, "dpt")).rename("mos_dpd")
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    ).rename("cloud_idx")
    precip_idx = (
        np.clip(np.nanmax(np.vstack([_mos_series(ctx, "p06"), _mos_series(ctx, "p12")]), axis=0), 0.0, 100.0)
        / 100.0
    ) * np.log1p(
        np.clip(np.nanmax(np.vstack([_mos_series(ctx, "q06"), _mos_series(ctx, "q12")]), axis=0), 0.0, 5.0)
    )
    precip_idx = pd.Series(precip_idx, index=df.index, name="precip_idx").fillna(0.0)
    wdr = _mos_series(ctx, "wdr")
    wsp = _mos_series(ctx, "wsp")
    radians = np.deg2rad(wdr.fillna(0.0))
    u = (-wsp.fillna(0.0) * np.sin(radians)).rename("mos_u")
    v = (-wsp.fillna(0.0) * np.cos(radians)).rename("mos_v")

    feature_df = pd.concat([base, dpd, cloud_idx, precip_idx, u, v], axis=1)
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

    best_params = None
    best_mae = None
    best_model = None
    for _ in range(30):
        params = {
            "loss_function": "MAE",
            "depth": int(ctx.rng.integers(4, 9)),
            "learning_rate": float(ctx.rng.uniform(0.01, 0.15)),
            "l2_leaf_reg": float(ctx.rng.uniform(1.0, 20.0)),
            "iterations": int(ctx.rng.integers(500, 6001)),
            "subsample": float(ctx.rng.uniform(0.6, 1.0)),
        }
        model = models.train_catboost(X_train, y_train, params, X_val=X_val, y_val=y_val)
        pred_val = model.predict(X_val)
        mae = float(eval_mod.regression_metrics(y_val, pred_val).get("mae", np.inf))
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_params = params
            best_model = model

    if best_params is None:
        best_params = {
            "loss_function": "MAE",
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "iterations": 2000,
        }
    model_train = best_model or models.train_catboost(X_train, y_train, best_params, X_val=X_val, y_val=y_val)
    pred_train = model_train.predict(X_train)
    pred_val = model_train.predict(X_val)
    model_full = models.train_catboost(X_train_full, y_train_full, best_params)
    pred_test = model_full.predict(X_test)

    return _assemble_result(
        ctx=ctx,
        experiment_id="CatBoost-MAEWithBustMOS",
        name="CatBoost MAE with bust+MOS",
        description=EXPERIMENT_DESCRIPTIONS["CatBoost-MAEWithBustMOS"],
        model_name="catboost",
        params=best_params,
        feature_columns=list(feature_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_xgb_quantile_trio(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    )
    dpd = (_mos_series(ctx, "tmp") - _mos_series(ctx, "dpt"))
    precip_idx = (
        np.clip(np.nanmax(np.vstack([_mos_series(ctx, "p06"), _mos_series(ctx, "p12")]), axis=0), 0.0, 100.0)
        / 100.0
    ) * np.log1p(
        np.clip(np.nanmax(np.vstack([_mos_series(ctx, "q06"), _mos_series(ctx, "q12")]), axis=0), 0.0, 5.0)
    )
    precip_idx = pd.Series(precip_idx, index=df.index).fillna(0.0)

    feature_df = pd.concat([base, cloud_idx.rename("cloud_idx"), dpd.rename("dpd"), precip_idx.rename("precip_idx")], axis=1)
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

    best_params = None
    best_mae = None
    for _ in range(20):
        params = {
            "max_depth": int(ctx.rng.integers(2, 7)),
            "learning_rate": float(ctx.rng.uniform(0.02, 0.15)),
            "n_estimators": int(ctx.rng.integers(200, 3001)),
            "subsample": float(ctx.rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(ctx.rng.uniform(0.6, 1.0)),
            "reg_lambda": float(ctx.rng.uniform(0.0, 10.0)),
            "objective": "reg:quantileerror",
        }
        preds_val = []
        for alpha in (0.1, 0.5, 0.9):
            params["quantile_alpha"] = alpha
            model = models.train_xgb(X_train, y_train, params, X_val=X_val, y_val=y_val)
            preds_val.append(model.predict(X_val))
        q10, q50, q90 = preds_val
        mu_hat = 0.5 * q50 + 0.25 * (q10 + q90)
        mae = float(eval_mod.regression_metrics(y_val, mu_hat).get("mae", np.inf))
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_params = params.copy()

    if best_params is None:
        best_params = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 800,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "reg:quantileerror",
        }

    def _train_quantile(alpha: float, X_tr: np.ndarray, y_tr: np.ndarray, X_vali: np.ndarray | None = None, y_vali: np.ndarray | None = None):
        params = dict(best_params)
        params["quantile_alpha"] = alpha
        return models.train_xgb(X_tr, y_tr, params, X_val=X_vali, y_val=y_vali)

    model_q10 = _train_quantile(0.1, X_train, y_train, X_val, y_val)
    model_q50 = _train_quantile(0.5, X_train, y_train, X_val, y_val)
    model_q90 = _train_quantile(0.9, X_train, y_train, X_val, y_val)
    q10_train = model_q10.predict(X_train)
    q50_train = model_q50.predict(X_train)
    q90_train = model_q90.predict(X_train)
    q10_val = model_q10.predict(X_val)
    q50_val = model_q50.predict(X_val)
    q90_val = model_q90.predict(X_val)
    mu_train = 0.5 * q50_train + 0.25 * (q10_train + q90_train)
    mu_val = 0.5 * q50_val + 0.25 * (q10_val + q90_val)

    model_q10_full = _train_quantile(0.1, X_train_full, y_train_full)
    model_q50_full = _train_quantile(0.5, X_train_full, y_train_full)
    model_q90_full = _train_quantile(0.9, X_train_full, y_train_full)
    q10_test = model_q10_full.predict(X_test)
    q50_test = model_q50_full.predict(X_test)
    q90_test = model_q90_full.predict(X_test)
    mu_test = 0.5 * q50_test + 0.25 * (q10_test + q90_test)

    return _assemble_result(
        ctx=ctx,
        experiment_id="XGB-QuantileTrioMeanReconstruction",
        name="XGB quantile trio mean reconstruction",
        description=EXPERIMENT_DESCRIPTIONS["XGB-QuantileTrioMeanReconstruction"],
        model_name="xgb_quantile",
        params=best_params,
        feature_columns=list(feature_df.columns),
        pred_train=mu_train,
        pred_val=mu_val,
        pred_test=mu_test,
    )


def run_two_stage_residual_shrink(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    )
    dpd = (_mos_series(ctx, "tmp") - _mos_series(ctx, "dpt"))
    feature_df = pd.concat([base, cloud_idx.rename("cloud_idx"), dpd.rename("dpd")], axis=1)

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

    # Stage 1 model
    model1 = models.train_lgbm(X_train, y_train, BASELINE_PARAMS, X_val=X_val, y_val=y_val)
    pred1_train = model1.predict(X_train)
    pred1_val = model1.predict(X_val)
    model1_full = models.train_lgbm(X_train_full, y_train_full, BASELINE_PARAMS)
    pred1_full_train = model1_full.predict(X_train_full)
    pred1_test = model1_full.predict(X_test)

    # Stage 2 residual model
    resid_train = y_train - pred1_train
    model2 = models.train_lgbm(X_train, resid_train, BASELINE_PARAMS, X_val=X_val, y_val=y_val)
    rhat_train = model2.predict(X_train)
    rhat_val = model2.predict(X_val)
    resid_full = y_train_full - pred1_full_train
    model2_full = models.train_lgbm(X_train_full, resid_full, BASELINE_PARAMS)
    rhat_test = model2_full.predict(X_test)

    # Stage 2 uncertainty model (abs residual)
    abs_resid_train = np.abs(resid_train)
    model3 = models.train_lgbm(
        X_train, abs_resid_train, BASELINE_PARAMS, X_val=X_val, y_val=y_val, objective="regression_l1"
    )
    ahat_train = model3.predict(X_train)
    ahat_val = model3.predict(X_val)
    abs_resid_full = np.abs(resid_full)
    model3_full = models.train_lgbm(X_train_full, abs_resid_full, BASELINE_PARAMS, objective="regression_l1")
    ahat_test = model3_full.predict(X_test)

    best = None
    best_c = None
    best_preds = None
    for c in (0.5, 1.0, 1.5, 2.0):
        shrink_train = rhat_train * (c / (c + ahat_train))
        shrink_val = rhat_val * (c / (c + ahat_val))
        pred_val = pred1_val + shrink_val
        mae = float(eval_mod.regression_metrics(y_val, pred_val).get("mae", np.inf))
        if best is None or mae < best:
            best = mae
            best_c = c
            best_preds = (pred1_train + shrink_train, pred_val, None)

    shrink_test = rhat_test * (best_c / (best_c + ahat_test))
    pred_test = pred1_test + shrink_test

    return _assemble_result(
        ctx=ctx,
        experiment_id="TwoStageResidualShrinkageByUncertainty",
        name="Two-stage residual shrinkage by uncertainty",
        description=EXPERIMENT_DESCRIPTIONS["TwoStageResidualShrinkageByUncertainty"],
        model_name="lgbm_two_stage",
        params={"c": best_c, **BASELINE_PARAMS},
        feature_columns=list(feature_df.columns),
        pred_train=best_preds[0],
        pred_val=best_preds[1],
        pred_test=pred_test,
    )


def run_huber_objective_tuning(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    cloud_idx = (
        -np.log1p(_mos_series(ctx, "cig").clip(lower=0.0, upper=20000.0).fillna(0.0))
        + 0.5 * np.log1p(_mos_series(ctx, "vis").clip(lower=0.0, upper=10.0).fillna(0.0))
    )
    feature_df = pd.concat([base, cloud_idx.rename("cloud_idx")], axis=1)

    def sampler() -> dict:
        return {
            "num_leaves": int(ctx.rng.integers(15, 128)),
            "learning_rate": float(ctx.rng.uniform(0.01, 0.10)),
            "n_estimators": int(ctx.rng.integers(400, 4001)),
            "min_data_in_leaf": int(ctx.rng.integers(10, 121)),
            "reg_lambda": float(ctx.rng.uniform(0.0, 10.0)),
            "huber_delta": float(ctx.rng.uniform(0.5, 3.0)),
        }

    best_params, pred_train, pred_val, pred_test = _random_search_lgbm(
        ctx, feature_df, objective="huber", n_trials=30, param_sampler=sampler
    )
    params = {**best_params, "objective": "huber"}
    return _assemble_result(
        ctx=ctx,
        experiment_id="HuberObjectiveTuning+MAESelection",
        name="Huber objective tuning",
        description=EXPERIMENT_DESCRIPTIONS["HuberObjectiveTuning+MAESelection"],
        model_name="lgbm",
        params=params,
        feature_columns=list(feature_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_corrected_forecast_stack(ctx: ExperimentContext) -> dict:
    df = ctx.df
    cand_df = _build_candidate_library(ctx)
    ens_mean = _get_ens_stats(ctx)["mean"]
    ens_series = pd.Series(ens_mean, index=df.index)
    cand_df = cand_df.copy()
    cand_cols = list(cand_df.columns)
    for col in cand_cols:
        missing_flag = cand_df[col].isna().astype(int)
        cand_df[f"{col}_missing"] = missing_flag
        cand_df[col] = cand_df[col].fillna(ens_series)
    cand_df["cand_range"] = cand_df[cand_cols].max(axis=1) - cand_df[cand_cols].min(axis=1)
    cand_df["cand_std"] = cand_df[cand_cols].std(axis=1)
    cand_df["spread"] = df[SPREAD_COL].to_numpy(dtype=float)
    cand_df["p_bust"] = ctx.bust_probs["p_bust"].to_numpy(dtype=float)

    X_train = cand_df.loc[ctx.train_mask].to_numpy(dtype=float)
    y_train = df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = cand_df.loc[ctx.val_mask].to_numpy(dtype=float)
    y_val = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = cand_df.loc[ctx.test_mask].to_numpy(dtype=float)
    y_test = df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    best = None
    best_alpha = None
    best_model = None
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        mae = float(eval_mod.regression_metrics(y_val, pred_val).get("mae", np.inf))
        if best is None or mae < best:
            best = mae
            best_alpha = alpha
            best_model = model

    pred_train = best_model.predict(X_train)
    pred_val = best_model.predict(X_val)
    model_full = Ridge(alpha=best_alpha)
    model_full.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    pred_test = model_full.predict(X_test)

    return _assemble_result(
        ctx=ctx,
        experiment_id="CorrectedForecastLibrary-StackRidge",
        name="Corrected forecast library stack (ridge)",
        description=EXPERIMENT_DESCRIPTIONS["CorrectedForecastLibrary-StackRidge"],
        model_name="ridge",
        params={"alpha": best_alpha},
        feature_columns=list(cand_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_convex_blend_mae(ctx: ExperimentContext) -> dict:
    df = ctx.df
    cand_df = _build_candidate_library(ctx).copy()
    ens_mean = _get_ens_stats(ctx)["mean"]
    ens_series = pd.Series(ens_mean, index=df.index)
    for col in cand_df.columns:
        cand_df[col] = cand_df[col].fillna(ens_series)
    X_val = cand_df.loc[ctx.val_mask].to_numpy(dtype=float)
    y_val = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_train = cand_df.loc[ctx.train_mask].to_numpy(dtype=float)
    y_train = df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = cand_df.loc[ctx.test_mask].to_numpy(dtype=float)
    y_test = df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)
    k = X_val.shape[1]

    def _project_simplex(v: np.ndarray) -> np.ndarray:
        v = np.maximum(v, 0.0)
        if v.sum() == 0:
            return np.full_like(v, 1.0 / len(v))
        return v / v.sum()

    best = None
    best_w = None
    best_lambda = None
    for lam in (0.0, 0.1, 0.3, 1.0, 3.0):
        try:
            from scipy.optimize import minimize

            def obj(w):
                w = _project_simplex(w)
                pred = X_val @ w
                mae = np.mean(np.abs(pred - y_val))
                penalty = lam * np.sum((w - 1.0 / k) ** 2)
                return mae + penalty

            cons = {"type": "eq", "fun": lambda w: np.sum(_project_simplex(w)) - 1.0}
            bounds = [(0.0, 1.0) for _ in range(k)]
            res = minimize(obj, np.full(k, 1.0 / k), bounds=bounds, constraints=cons, method="SLSQP")
            w = _project_simplex(res.x if res.success else np.full(k, 1.0 / k))
        except Exception:
            # fallback random search
            best_val = None
            w = np.full(k, 1.0 / k)
            for _ in range(200):
                rnd = ctx.rng.random(k)
                rnd = _project_simplex(rnd)
                pred = X_val @ rnd
                mae = np.mean(np.abs(pred - y_val)) + lam * np.sum((rnd - 1.0 / k) ** 2)
                if best_val is None or mae < best_val:
                    best_val = mae
                    w = rnd
        pred_val = X_val @ w
        mae = float(eval_mod.regression_metrics(y_val, pred_val).get("mae", np.inf))
        if best is None or mae < best:
            best = mae
            best_w = w
            best_lambda = lam

    pred_train = X_train @ best_w
    pred_val = X_val @ best_w
    pred_test = X_test @ best_w

    return _assemble_result(
        ctx=ctx,
        experiment_id="ConvexBlend-MAEOptimized-ShrunkToEqual",
        name="Convex MAE-optimized blend",
        description=EXPERIMENT_DESCRIPTIONS["ConvexBlend-MAEOptimized-ShrunkToEqual"],
        model_name="blend",
        params={"lambda": best_lambda, "weights": best_w.tolist()},
        feature_columns=list(cand_df.columns),
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
    )


def run_online_expert_weights(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    ens_median = ens_stats["median"]
    expert_vals = np.column_stack([df[GUIDANCE_COLS].to_numpy(dtype=float), ens_mean, ens_median])
    actual = df["actual_tmax_f"].to_numpy(dtype=float)
    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for eta in (0.1, 0.2, 0.3):
        for rho in (0.02, 0.05, 0.1):
            y_online, w_entropy, eff_n = _compute_online_weights(
                expert_vals,
                actual,
                group_key=ctx.group_key,
                dates=df["target_date_local"],
                eta=eta,
                rho=rho,
                lag=TRUTH_LAG_DAYS,
            )
            gate = _sigmoid((ctx.bust_probs["p_bust"].to_numpy(dtype=float) - 0.4) / 0.1)
            y_final = (1.0 - gate) * y_online + gate * ens_median

            feat = pd.DataFrame(index=df.index)
            feat["y_online"] = y_online
            feat["y_final"] = y_final
            feat["w_entropy"] = w_entropy
            feat["eff_n"] = eff_n

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"eta": eta, "rho": rho}

    return _assemble_result(
        ctx=ctx,
        experiment_id="OnlineExpertWeights+BustHedge",
        name="Online expert weights + bust hedge",
        description=EXPERIMENT_DESCRIPTIONS["OnlineExpertWeights+BustHedge"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_season_specific_submodels(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    dates = pd.to_datetime(df["target_date_local"])
    month = dates.dt.month
    season = pd.Series(index=df.index, dtype="string")
    season.loc[month.isin([12, 1, 2])] = "DJF"
    season.loc[month.isin([3, 4, 5])] = "MAM"
    season.loc[month.isin([6, 7, 8])] = "JJA"
    season.loc[month.isin([9, 10, 11])] = "SON"

    # regularized params for seasonal models
    season_params = dict(BASELINE_PARAMS)
    season_params["num_leaves"] = min(31, BASELINE_PARAMS.get("num_leaves", 31))
    season_params["min_data_in_leaf"] = max(30, BASELINE_PARAMS.get("min_data_in_leaf", 20))

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_full,
        y_train_full,
    ) = _prepare_arrays(ctx, base)

    def train_mask_for(season_label: str, mask: np.ndarray) -> np.ndarray:
        return mask & (season == season_label).to_numpy()

    def train_models(mask: np.ndarray) -> dict[str, Any]:
        models_out = {}
        for s in ["DJF", "MAM", "JJA", "SON"]:
            s_mask = train_mask_for(s, mask)
            if not np.any(s_mask):
                models_out[s] = None
                continue
            model = models.train_lgbm(
                base.loc[s_mask].to_numpy(dtype=float),
                df.loc[s_mask, "actual_tmax_f"].to_numpy(dtype=float),
                season_params,
            )
            models_out[s] = model
        models_out["GLOBAL"] = models.train_lgbm(
            base.loc[mask].to_numpy(dtype=float),
            df.loc[mask, "actual_tmax_f"].to_numpy(dtype=float),
            season_params,
        )
        return models_out

    def predict_models(models_out: dict[str, Any]) -> dict[str, np.ndarray]:
        preds = {}
        for s in ["DJF", "MAM", "JJA", "SON"]:
            model = models_out.get(s)
            if model is None:
                preds[s] = np.full(len(df), np.nan, dtype=float)
            else:
                preds[s] = model.predict(base.to_numpy(dtype=float))
        preds["GLOBAL"] = models_out["GLOBAL"].predict(base.to_numpy(dtype=float))
        return preds

    best = None
    best_params = None
    best_preds = None
    for width in (10, 15, 20):
        # season weights via gaussian bumps
        doy = dates.dt.dayofyear.to_numpy(dtype=float)
        centers = {"DJF": 15.0, "MAM": 105.0, "JJA": 200.0, "SON": 290.0}
        weights = {}
        for s, c in centers.items():
            weights[s] = np.exp(-((doy - c) / width) ** 2)
        w_stack = np.vstack([weights[s] for s in ["DJF", "MAM", "JJA", "SON"]]).T
        w_sum = w_stack.sum(axis=1, keepdims=True)
        w_stack = np.where(w_sum == 0, 0.25, w_stack / w_sum)

        models_train = train_models(ctx.train_mask)
        preds_all = predict_models(models_train)

        for alpha in (0.2, 0.4, 0.6, 0.8):
            blended = alpha * preds_all["GLOBAL"]
            for i, s in enumerate(["DJF", "MAM", "JJA", "SON"]):
                blended += (1.0 - alpha) * w_stack[:, i] * preds_all[s]
            pred_val = blended[ctx.val_mask]
            mae = float(eval_mod.regression_metrics(y_val, pred_val).get("mae", np.inf))
            if best is None or mae < best:
                best = mae
                best_params = {"alpha": alpha, "width": width}
                pred_train = blended[ctx.train_mask]
                best_preds = (pred_train, pred_val, None)

    # retrain on train+val for test prediction
    models_full = train_models(ctx.train_mask | ctx.val_mask)
    preds_all_full = predict_models(models_full)
    doy = dates.dt.dayofyear.to_numpy(dtype=float)
    width = best_params["width"]
    centers = {"DJF": 15.0, "MAM": 105.0, "JJA": 200.0, "SON": 290.0}
    weights = {s: np.exp(-((doy - c) / width) ** 2) for s, c in centers.items()}
    w_stack = np.vstack([weights[s] for s in ["DJF", "MAM", "JJA", "SON"]]).T
    w_sum = w_stack.sum(axis=1, keepdims=True)
    w_stack = np.where(w_sum == 0, 0.25, w_stack / w_sum)
    alpha = best_params["alpha"]
    blended_full = alpha * preds_all_full["GLOBAL"]
    for i, s in enumerate(["DJF", "MAM", "JJA", "SON"]):
        blended_full += (1.0 - alpha) * w_stack[:, i] * preds_all_full[s]
    pred_test = blended_full[ctx.test_mask]

    return _assemble_result(
        ctx=ctx,
        experiment_id="SeasonSpecificSubmodels-SmoothBlend",
        name="Season-specific submodels smooth blend",
        description=EXPERIMENT_DESCRIPTIONS["SeasonSpecificSubmodels-SmoothBlend"],
        model_name="lgbm_seasonal_blend",
        params=best_params,
        feature_columns=list(base.columns),
        pred_train=best_preds[0],
        pred_val=best_preds[1],
        pred_test=pred_test,
    )


def run_knn_residual_correction(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    model_range = ens_stats["max"] - ens_stats["min"]
    sb_idx = _get_sb_idx(ctx)
    cal = _get_calendar(ctx)
    p_bust = ctx.bust_probs["p_bust"].to_numpy(dtype=float)
    v = np.column_stack(
        [
            ens_mean,
            model_range,
            df[SPREAD_COL].to_numpy(dtype=float),
            sb_idx,
            cal["sin_doy"].to_numpy(dtype=float),
            cal["cos_doy"].to_numpy(dtype=float),
            p_bust,
        ]
    )
    v_std, _, _ = _standardize_matrix(v, ctx.train_mask)
    resid = df["actual_tmax_f"].to_numpy(dtype=float) - ens_mean
    group_indices = _group_sorted_indices(df, ctx.group_key)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for K in (15, 25, 35):
        knn_r50 = np.full(len(df), np.nan, dtype=float)
        dist_mean = np.full(len(df), np.nan, dtype=float)
        dist_p10 = np.full(len(df), np.nan, dtype=float)
        dist_p90 = np.full(len(df), np.nan, dtype=float)
        for idx in group_indices:
            for pos, row_idx in enumerate(idx):
                if pos % 200 == 0:
                    LOGGER.info("KNN_PROGRESS exp=knn_r50 K=%d pos=%d/%d", K, pos, len(idx))
                past_end = pos - TRUTH_LAG_DAYS + 1
                if past_end <= 0:
                    continue
                past_idx = idx[:past_end]
                if len(past_idx) < max(10, K):
                    continue
                current = v_std[row_idx]
                past_mat = v_std[past_idx]
                dists = np.linalg.norm(past_mat - current, axis=1)
                order = np.argsort(dists)
                sel = order[:K]
                sel_resid = resid[past_idx][sel]
                knn_r50[row_idx] = float(np.nanmedian(sel_resid))
                sel_dist = dists[sel]
                dist_mean[row_idx] = float(np.nanmean(sel_dist))
                dist_p10[row_idx] = float(np.nanquantile(sel_dist, 0.1))
                dist_p90[row_idx] = float(np.nanquantile(sel_dist, 0.9))

        knn_r50 = np.where(np.isfinite(knn_r50), knn_r50, 0.0)
        m_knn = ens_mean + knn_r50
        feat = pd.DataFrame(index=df.index)
        feat["knn_r50"] = knn_r50
        feat["knn_dist_mean"] = dist_mean
        feat["knn_dist_p10"] = dist_p10
        feat["knn_dist_p90"] = dist_p90
        feat["m_knn"] = m_knn
        feat["knn_r50_x_pbust"] = knn_r50 * p_bust
        feat["knn_dist_mean_x_spread"] = dist_mean * df[SPREAD_COL].to_numpy(dtype=float)

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"K": K}

    return _assemble_result(
        ctx=ctx,
        experiment_id="KNNResidualCorrection-ForecastSpace",
        name="kNN residual correction (forecast space)",
        description=EXPERIMENT_DESCRIPTIONS["KNNResidualCorrection-ForecastSpace"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_knn_quantile_residuals(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    model_range = ens_stats["max"] - ens_stats["min"]
    sb_idx = _get_sb_idx(ctx)
    cal = _get_calendar(ctx)
    p_bust = ctx.bust_probs["p_bust"].to_numpy(dtype=float)
    v = np.column_stack(
        [
            ens_mean,
            model_range,
            df[SPREAD_COL].to_numpy(dtype=float),
            sb_idx,
            cal["sin_doy"].to_numpy(dtype=float),
            cal["cos_doy"].to_numpy(dtype=float),
            p_bust,
        ]
    )
    v_std, _, _ = _standardize_matrix(v, ctx.train_mask)
    resid = df["actual_tmax_f"].to_numpy(dtype=float) - ens_mean
    group_indices = _group_sorted_indices(df, ctx.group_key)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for K in (15, 25, 35):
        knn_q10 = np.full(len(df), np.nan, dtype=float)
        knn_q50 = np.full(len(df), np.nan, dtype=float)
        knn_q90 = np.full(len(df), np.nan, dtype=float)
        for idx in group_indices:
            for pos, row_idx in enumerate(idx):
                if pos % 200 == 0:
                    LOGGER.info("KNN_PROGRESS exp=knn_quantile K=%d pos=%d/%d", K, pos, len(idx))
                past_end = pos - TRUTH_LAG_DAYS + 1
                if past_end <= 0:
                    continue
                past_idx = idx[:past_end]
                if len(past_idx) < max(10, K):
                    continue
                current = v_std[row_idx]
                past_mat = v_std[past_idx]
                dists = np.linalg.norm(past_mat - current, axis=1)
                order = np.argsort(dists)
                sel = order[:K]
                sel_resid = resid[past_idx][sel]
                knn_q10[row_idx] = float(np.nanquantile(sel_resid, 0.1))
                knn_q50[row_idx] = float(np.nanquantile(sel_resid, 0.5))
                knn_q90[row_idx] = float(np.nanquantile(sel_resid, 0.9))

        knn_q10 = np.where(np.isfinite(knn_q10), knn_q10, 0.0)
        knn_q50 = np.where(np.isfinite(knn_q50), knn_q50, 0.0)
        knn_q90 = np.where(np.isfinite(knn_q90), knn_q90, 0.0)
        knn_iqr = knn_q90 - knn_q10
        for c in (0.5, 1.0, 1.5):
            knn_corr = knn_q50 * (c / (c + knn_iqr))
            m_knn = ens_mean + knn_corr
            feat = pd.DataFrame(index=df.index)
            feat["knn_q10"] = knn_q10
            feat["knn_q50"] = knn_q50
            feat["knn_q90"] = knn_q90
            feat["knn_iqr"] = knn_iqr
            feat["knn_corr"] = knn_corr
            feat["m_knn_shrunk"] = m_knn
            feat["knn_iqr_x_pbust"] = knn_iqr * p_bust
            feat["knn_corr_x_spread"] = knn_corr * df[SPREAD_COL].to_numpy(dtype=float)

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"K": K, "c": c}

    return _assemble_result(
        ctx=ctx,
        experiment_id="KNNQuantileResiduals+UncertaintyShrink",
        name="kNN quantile residuals + shrinkage",
        description=EXPERIMENT_DESCRIPTIONS["KNNQuantileResiduals+UncertaintyShrink"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_regime_aware_analogs(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    model_range = ens_stats["max"] - ens_stats["min"]
    sb_idx = _get_sb_idx(ctx)
    cal = _get_calendar(ctx)
    p_bust = ctx.bust_probs["p_bust"].to_numpy(dtype=float)
    v = np.column_stack(
        [
            ens_mean,
            model_range,
            df[SPREAD_COL].to_numpy(dtype=float),
            sb_idx,
            cal["sin_doy"].to_numpy(dtype=float),
            cal["cos_doy"].to_numpy(dtype=float),
            p_bust,
        ]
    )
    v_std, _, _ = _standardize_matrix(v, ctx.train_mask)
    resid = df["actual_tmax_f"].to_numpy(dtype=float) - ens_mean
    group_indices = _group_sorted_indices(df, ctx.group_key)
    dates = pd.to_datetime(df["target_date_local"])
    month = dates.dt.month
    season = pd.Series(index=df.index, dtype="string")
    season.loc[month.isin([12, 1, 2])] = "DJF"
    season.loc[month.isin([3, 4, 5])] = "MAM"
    season.loc[month.isin([6, 7, 8])] = "JJA"
    season.loc[month.isin([9, 10, 11])] = "SON"

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for K in (10, 20, 30):
        for sb_thresh in (0.3, 0.5, 0.8):
            sb_sign = np.where(sb_idx > sb_thresh, 1, np.where(sb_idx < -sb_thresh, -1, 0))
            knn_r50 = np.full(len(df), np.nan, dtype=float)
            dist_mean = np.full(len(df), np.nan, dtype=float)
            pool_size = np.zeros(len(df), dtype=float)
            fallback = np.zeros(len(df), dtype=int)
            for idx in group_indices:
                for pos, row_idx in enumerate(idx):
                    if pos % 200 == 0:
                        LOGGER.info("KNN_PROGRESS exp=knn_regime K=%d pos=%d/%d", K, pos, len(idx))
                    past_end = pos - TRUTH_LAG_DAYS + 1
                    if past_end <= 0:
                        continue
                    past_idx = idx[:past_end]
                    if len(past_idx) < 10:
                        continue
                    # regime filter
                    reg_mask = (season.iloc[past_idx] == season.iloc[row_idx]).to_numpy() & (
                        sb_sign[past_idx] == sb_sign[row_idx]
                    )
                    reg_idx = past_idx[reg_mask]
                    pool_size[row_idx] = len(reg_idx)
                    use_idx = reg_idx
                    if len(use_idx) < K:
                        use_idx = past_idx
                        fallback[row_idx] = 1
                    if len(use_idx) < max(10, K):
                        continue
                    current = v_std[row_idx]
                    past_mat = v_std[use_idx]
                    dists = np.linalg.norm(past_mat - current, axis=1)
                    order = np.argsort(dists)
                    sel = order[:K]
                    sel_resid = resid[use_idx][sel]
                    knn_r50[row_idx] = float(np.nanmedian(sel_resid))
                    dist_mean[row_idx] = float(np.nanmean(dists[sel]))

            knn_r50 = np.where(np.isfinite(knn_r50), knn_r50, 0.0)
            m_knn = ens_mean + knn_r50
            feat = pd.DataFrame(index=df.index)
            feat["knn_r50_regime"] = knn_r50
            feat["knn_dist_mean_regime"] = dist_mean
            feat["knn_pool_size"] = pool_size
            feat["knn_fallback"] = fallback
            feat["m_knn_regime"] = m_knn
            feat["pool_x_pbust"] = pool_size * p_bust

            feature_df = pd.concat([base, feat], axis=1)
            preds, mae = _eval_fixed_candidate(ctx, feature_df)
            if best is None or mae < best:
                best = mae
                best_feat = feature_df
                best_preds = preds
                best_params = {"K": K, "sb_thresh": sb_thresh}

    return _assemble_result(
        ctx=ctx,
        experiment_id="RegimeAwareAnalogs-Season+SBSign",
        name="Regime-aware analogs (season + sb sign)",
        description=EXPERIMENT_DESCRIPTIONS["RegimeAwareAnalogs-Season+SBSign"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


def run_local_linear_analog(ctx: ExperimentContext) -> dict:
    df = ctx.df
    base = _base_with_bust(ctx)
    ens_stats = _get_ens_stats(ctx)
    ens_mean = ens_stats["mean"]
    cal = _get_calendar(ctx)
    X_feat = np.column_stack(
        [
            df[GUIDANCE_COLS].to_numpy(dtype=float),
            df[SPREAD_COL].to_numpy(dtype=float),
            cal["sin_doy"].to_numpy(dtype=float),
            cal["cos_doy"].to_numpy(dtype=float),
        ]
    )
    X_std, _, _ = _standardize_matrix(X_feat, ctx.train_mask)
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    group_indices = _group_sorted_indices(df, ctx.group_key)

    best = None
    best_feat = None
    best_preds = None
    best_params = None
    for K in (20, 30, 40):
        y_llr = np.full(len(df), np.nan, dtype=float)
        alpha_sel = np.full(len(df), np.nan, dtype=float)
        coef_norm = np.full(len(df), np.nan, dtype=float)
        cond_proxy = np.full(len(df), np.nan, dtype=float)
        fallback = np.zeros(len(df), dtype=int)
        for idx in group_indices:
            for pos, row_idx in enumerate(idx):
                if pos % 200 == 0:
                    LOGGER.info("KNN_PROGRESS exp=llr K=%d pos=%d/%d", K, pos, len(idx))
                past_end = pos - TRUTH_LAG_DAYS + 1
                if past_end <= 0:
                    continue
                past_idx = idx[:past_end]
                if len(past_idx):
                    finite_mask = np.isfinite(y[past_idx])
                    past_idx = past_idx[finite_mask]
                if len(past_idx) < max(10, K):
                    fallback[row_idx] = 1
                    continue
                current = X_std[row_idx]
                past_mat = X_std[past_idx]
                current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
                past_mat = np.nan_to_num(past_mat, nan=0.0, posinf=0.0, neginf=0.0)
                dists = np.linalg.norm(past_mat - current, axis=1)
                order = np.argsort(dists)
                sel = order[:K]
                weights = 1.0 / (dists[sel] + 1e-3)
                weight_sum = float(np.sum(weights))
                if not np.isfinite(weight_sum) or weight_sum <= 0:
                    fallback[row_idx] = 1
                    continue
                weights = weights / weight_sum
                Xn = past_mat[sel]
                yn = y[past_idx][sel]
                best_local = None
                best_alpha = None
                best_coef = None
                for alpha in (0.1, 1.0, 10.0, 100.0):
                    model = Ridge(alpha=alpha)
                    model.fit(Xn, yn, sample_weight=weights)
                    pred = model.predict(current.reshape(1, -1))[0]
                    # weighted MAE on neighbors
                    pred_neighbors = model.predict(Xn)
                    mae = np.sum(weights * np.abs(pred_neighbors - yn))
                    if best_local is None or mae < best_local:
                        best_local = mae
                        best_alpha = alpha
                        best_coef = model.coef_
                        y_llr[row_idx] = pred
                alpha_sel[row_idx] = best_alpha
                if best_coef is not None:
                    coef_norm[row_idx] = float(np.linalg.norm(best_coef))
                    try:
                        cond_proxy[row_idx] = float(np.linalg.cond(Xn))
                    except Exception:
                        cond_proxy[row_idx] = np.nan

        y_llr = np.where(np.isfinite(y_llr), y_llr, ens_mean)
        feat = pd.DataFrame(index=df.index)
        feat["y_llr"] = y_llr
        feat["y_llr_minus_mean"] = y_llr - ens_mean
        feat["alpha_sel"] = alpha_sel
        feat["coef_norm"] = coef_norm
        feat["neighbor_cond"] = cond_proxy
        feat["llr_fallback"] = fallback

        feature_df = pd.concat([base, feat], axis=1)
        preds, mae = _eval_fixed_candidate(ctx, feature_df)
        if best is None or mae < best:
            best = mae
            best_feat = feature_df
            best_preds = preds
            best_params = {"K": K}

    return _assemble_result(
        ctx=ctx,
        experiment_id="LocalLinearAnalogCalibration-LLR",
        name="Local linear analog calibration (LLR)",
        description=EXPERIMENT_DESCRIPTIONS["LocalLinearAnalogCalibration-LLR"],
        model_name="lgbm",
        params={**(best_params or {}), **BASELINE_PARAMS},
        feature_columns=list(best_feat.columns),
        pred_train=best_preds["pred_train"],
        pred_val=best_preds["pred_val"],
        pred_test=best_preds["pred_test"],
    )


# ---------- Context/builders ----------


def build_context(
    df: pd.DataFrame,
    *,
    run_root: Path,
    seed: int = 42,
    split_override: SplitConfig | None = None,
) -> ExperimentContext:
    df = df.copy()
    split = split_override or DEFAULT_SPLIT
    LOGGER.info("CTX_BUILD_START rows=%d cols=%d", len(df), df.shape[1])
    LOGGER.info("CTX_TRUTH_LAG_DAYS %d", TRUTH_LAG_DAYS)

    # Ensure MOS-derived guidance columns exist
    def _pick_mos_col(model: str, stat: str) -> pd.Series:
        return df.get(f"mos_{model}_n_x_{stat}", pd.Series(np.nan, index=df.index)).astype(float)

    gfs_nx = _pick_mos_col("gfs", "max")
    if gfs_nx.isna().all():
        gfs_nx = _pick_mos_col("gfs", "mean")
    if gfs_nx.isna().all() and "gfs_tmax_f" in df.columns:
        gfs_nx = df["gfs_tmax_f"].astype(float)
    df["gfs_n_x_max"] = gfs_nx

    nam_nx = _pick_mos_col("nam", "max")
    if nam_nx.isna().all():
        nam_nx = _pick_mos_col("nam", "mean")
    if nam_nx.isna().all() and "nam_tmax_f" in df.columns:
        nam_nx = df["nam_tmax_f"].astype(float)
    df["nam_n_x_max"] = nam_nx

    # Ensure guidance columns exist
    for col in GUIDANCE_COLS:
        if col not in df.columns:
            df[col] = np.nan
            LOGGER.warning("Missing guidance column added as NaN: %s", col)
    if SPREAD_COL not in df.columns:
        df[SPREAD_COL] = np.nan
        LOGGER.warning("Missing spread column added as NaN: %s", SPREAD_COL)

    cal = features.add_calendar_features(df)
    base_features = pd.concat(
        [
            df[GUIDANCE_COLS].astype(float),
            df[[SPREAD_COL]].astype(float),
            cal[["month", "day_of_year", "sin_doy", "cos_doy", "is_weekend"]],
        ],
        axis=1,
    )

    dates = pd.to_datetime(df["target_date_local"]).dt.date
    gap_set = set(split.gap_dates)
    in_gap = dates.isin(gap_set)
    train_mask = (dates >= split.train_start) & (dates <= split.train_end) & ~in_gap
    val_mask = (dates >= split.val_start) & (dates <= split.val_end) & ~in_gap
    test_mask = (dates >= split.test_start) & (dates <= split.test_end) & ~in_gap
    valid_target = df["actual_tmax_f"].notna()
    removed = int((~valid_target & (train_mask | val_mask | test_mask)).sum())
    if removed:
        LOGGER.warning("Dropping %d rows with missing actual_tmax_f from splits.", removed)
    train_mask = train_mask & valid_target
    val_mask = val_mask & valid_target
    test_mask = test_mask & valid_target
    LOGGER.info(
        "CTX_SPLIT_COUNTS train=%d val=%d test=%d gap=%d",
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
        int(in_gap.sum()),
    )
    LOGGER.info("CTX_BASE_FEATURES cols=%d", base_features.shape[1])

    ctx = ExperimentContext(
        df=df,
        train_mask=train_mask.to_numpy(),
        val_mask=val_mask.to_numpy(),
        test_mask=test_mask.to_numpy(),
        base_features=base_features,
        base_feature_columns=list(base_features.columns),
        baseline_a=None,  # type: ignore[arg-type]
        baseline_b=None,  # type: ignore[arg-type]
        bust_probs=pd.DataFrame(index=df.index),
        group_key=df["station_id"].fillna("UNKNOWN"),
        rng=np.random.default_rng(seed),
        run_root=run_root,
        split_ref={
            "train_start": str(split.train_start),
            "train_end": str(split.train_end),
            "val_start": str(split.val_start),
            "val_end": str(split.val_end),
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
            "gap_dates": [str(d) for d in split.gap_dates],
        },
        cache={},
    )

    LOGGER.info("CTX_BUST_PROBS_START")
    ctx.bust_probs = _compute_bust_probs(ctx)
    LOGGER.info("CTX_BUST_PROBS_DONE cols=%d", ctx.bust_probs.shape[1])
    LOGGER.info("CTX_BASELINE_A_START")
    ctx.baseline_a = run_baseline_a(ctx)
    LOGGER.info("CTX_BASELINE_A_DONE")
    LOGGER.info("CTX_BASELINE_B_START")
    ctx.baseline_b = run_baseline_b(ctx)
    LOGGER.info("CTX_BASELINE_B_DONE")
    return ctx


def build_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            experiment_id="PerModelAdaptiveEWMA-BanditCorr",
            name="Per-model adaptive EWMA bandit",
            description=EXPERIMENT_DESCRIPTIONS["PerModelAdaptiveEWMA-BanditCorr"],
            runner=run_per_model_adaptive_ewma_bandit,
        ),
        ExperimentSpec(
            experiment_id="FourierK3+AnomalyEWMA-PerModel",
            name="Fourier k3 + EWMA anomaly",
            description=EXPERIMENT_DESCRIPTIONS["FourierK3+AnomalyEWMA-PerModel"],
            runner=run_fourier_k3_anom,
        ),
        ExperimentSpec(
            experiment_id="DualTailDepthBiasMap",
            name="Dual tail-depth bias map",
            description=EXPERIMENT_DESCRIPTIONS["DualTailDepthBiasMap"],
            runner=run_dual_tail_depth_bias_map,
        ),
        ExperimentSpec(
            experiment_id="ErrorCorrelationDiversityWeights",
            name="Error correlation diversity weights",
            description=EXPERIMENT_DESCRIPTIONS["ErrorCorrelationDiversityWeights"],
            runner=run_error_correlation_diversity_weights,
        ),
        ExperimentSpec(
            experiment_id="CUSUMReset+BOCPDGatedBias",
            name="CUSUM reset + BOCPD gated bias",
            description=EXPERIMENT_DESCRIPTIONS["CUSUMReset+BOCPDGatedBias"],
            runner=run_cusum_bocpd_gated_bias,
        ),
        ExperimentSpec(
            experiment_id="SeasonalAdaptiveHalfLifeBias",
            name="Seasonal adaptive half-life bias",
            description=EXPERIMENT_DESCRIPTIONS["SeasonalAdaptiveHalfLifeBias"],
            runner=run_seasonal_adaptive_half_life_bias,
        ),
        ExperimentSpec(
            experiment_id="MOS-DewpointDepressionHumiditySignals",
            name="MOS dewpoint depression humidity signals",
            description=EXPERIMENT_DESCRIPTIONS["MOS-DewpointDepressionHumiditySignals"],
            runner=run_mos_dpd_humidity,
        ),
        ExperimentSpec(
            experiment_id="MOS-WindVectorSeaBreezeGating",
            name="MOS wind vector sea-breeze gating",
            description=EXPERIMENT_DESCRIPTIONS["MOS-WindVectorSeaBreezeGating"],
            runner=run_mos_wind_sea_breeze,
        ),
        ExperimentSpec(
            experiment_id="MOS-CloudVisibilityRadiationProxy",
            name="MOS cloud/visibility radiation proxy",
            description=EXPERIMENT_DESCRIPTIONS["MOS-CloudVisibilityRadiationProxy"],
            runner=run_mos_cloud_visibility,
        ),
        ExperimentSpec(
            experiment_id="MOS-PrecipLikelihoodCoolingHedge",
            name="MOS precip likelihood cooling hedge",
            description=EXPERIMENT_DESCRIPTIONS["MOS-PrecipLikelihoodCoolingHedge"],
            runner=run_mos_precip_hedge,
        ),
        ExperimentSpec(
            experiment_id="MOS-T06HeatingPotentialFeatures",
            name="MOS t06 heating potential",
            description=EXPERIMENT_DESCRIPTIONS["MOS-T06HeatingPotentialFeatures"],
            runner=run_mos_t06_heating,
        ),
        ExperimentSpec(
            experiment_id="MOS-MissingnessQualityEncoding",
            name="MOS missingness quality encoding",
            description=EXPERIMENT_DESCRIPTIONS["MOS-MissingnessQualityEncoding"],
            runner=run_mos_missingness_quality,
        ),
        ExperimentSpec(
            experiment_id="BustClassifierV2-Ordinal5Class",
            name="Bust classifier v2 ordinal 5-class",
            description=EXPERIMENT_DESCRIPTIONS["BustClassifierV2-Ordinal5Class"],
            runner=run_bust_classifier_v2,
        ),
        ExperimentSpec(
            experiment_id="SpreadxSeaBreezeMixtureOfExpertsBias",
            name="Spread × sea-breeze mixture-of-experts bias",
            description=EXPERIMENT_DESCRIPTIONS["SpreadxSeaBreezeMixtureOfExpertsBias"],
            runner=run_spread_seabreeze_moe,
        ),
        ExperimentSpec(
            experiment_id="ResidualHMM-StateProbBias",
            name="Residual HMM state-prob bias",
            description=EXPERIMENT_DESCRIPTIONS["ResidualHMM-StateProbBias"],
            runner=run_residual_hmm_stateprob,
        ),
        ExperimentSpec(
            experiment_id="SeasonTransitionSpreadGate",
            name="Season transition spread gate",
            description=EXPERIMENT_DESCRIPTIONS["SeasonTransitionSpreadGate"],
            runner=run_season_transition_spread_gate,
        ),
        ExperimentSpec(
            experiment_id="PCA2+MOSRegimeGMMProbFeatures",
            name="PCA2 + MOS regime GMM probs",
            description=EXPERIMENT_DESCRIPTIONS["PCA2+MOSRegimeGMMProbFeatures"],
            runner=run_pca2_mos_regime_gmm,
        ),
        ExperimentSpec(
            experiment_id="LightGBM-RegressionL1MAEObjective",
            name="LightGBM regression L1 objective",
            description=EXPERIMENT_DESCRIPTIONS["LightGBM-RegressionL1MAEObjective"],
            runner=run_lgbm_l1_objective,
        ),
        ExperimentSpec(
            experiment_id="CatBoost-MAEWithBustMOS",
            name="CatBoost MAE with bust+MOS",
            description=EXPERIMENT_DESCRIPTIONS["CatBoost-MAEWithBustMOS"],
            runner=run_catboost_mae_bust_mos,
        ),
        ExperimentSpec(
            experiment_id="XGB-QuantileTrioMeanReconstruction",
            name="XGB quantile trio mean reconstruction",
            description=EXPERIMENT_DESCRIPTIONS["XGB-QuantileTrioMeanReconstruction"],
            runner=run_xgb_quantile_trio,
        ),
        ExperimentSpec(
            experiment_id="TwoStageResidualShrinkageByUncertainty",
            name="Two-stage residual shrinkage by uncertainty",
            description=EXPERIMENT_DESCRIPTIONS["TwoStageResidualShrinkageByUncertainty"],
            runner=run_two_stage_residual_shrink,
        ),
        ExperimentSpec(
            experiment_id="HuberObjectiveTuning+MAESelection",
            name="Huber objective tuning",
            description=EXPERIMENT_DESCRIPTIONS["HuberObjectiveTuning+MAESelection"],
            runner=run_huber_objective_tuning,
        ),
        ExperimentSpec(
            experiment_id="CorrectedForecastLibrary-StackRidge",
            name="Corrected forecast library stack (ridge)",
            description=EXPERIMENT_DESCRIPTIONS["CorrectedForecastLibrary-StackRidge"],
            runner=run_corrected_forecast_stack,
        ),
        ExperimentSpec(
            experiment_id="ConvexBlend-MAEOptimized-ShrunkToEqual",
            name="Convex MAE-optimized blend",
            description=EXPERIMENT_DESCRIPTIONS["ConvexBlend-MAEOptimized-ShrunkToEqual"],
            runner=run_convex_blend_mae,
        ),
        ExperimentSpec(
            experiment_id="OnlineExpertWeights+BustHedge",
            name="Online expert weights + bust hedge",
            description=EXPERIMENT_DESCRIPTIONS["OnlineExpertWeights+BustHedge"],
            runner=run_online_expert_weights,
        ),
        ExperimentSpec(
            experiment_id="SeasonSpecificSubmodels-SmoothBlend",
            name="Season-specific submodels smooth blend",
            description=EXPERIMENT_DESCRIPTIONS["SeasonSpecificSubmodels-SmoothBlend"],
            runner=run_season_specific_submodels,
        ),
        ExperimentSpec(
            experiment_id="KNNResidualCorrection-ForecastSpace",
            name="kNN residual correction (forecast space)",
            description=EXPERIMENT_DESCRIPTIONS["KNNResidualCorrection-ForecastSpace"],
            runner=run_knn_residual_correction,
        ),
        ExperimentSpec(
            experiment_id="KNNQuantileResiduals+UncertaintyShrink",
            name="kNN quantile residuals + shrinkage",
            description=EXPERIMENT_DESCRIPTIONS["KNNQuantileResiduals+UncertaintyShrink"],
            runner=run_knn_quantile_residuals,
        ),
        ExperimentSpec(
            experiment_id="RegimeAwareAnalogs-Season+SBSign",
            name="Regime-aware analogs (season + sb sign)",
            description=EXPERIMENT_DESCRIPTIONS["RegimeAwareAnalogs-Season+SBSign"],
            runner=run_regime_aware_analogs,
        ),
        ExperimentSpec(
            experiment_id="LocalLinearAnalogCalibration-LLR",
            name="Local linear analog calibration (LLR)",
            description=EXPERIMENT_DESCRIPTIONS["LocalLinearAnalogCalibration-LLR"],
            runner=run_local_linear_analog,
        ),
    ]
