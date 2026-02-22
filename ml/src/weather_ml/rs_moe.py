"""Regime-switching mixture-of-experts (RS-MoE) mean model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


REGIMES_DEFAULT = ("cool", "normal", "warm")


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = logits / float(temperature)
    scaled = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(scaled)


def average_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-15, 1.0)
    return float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))


def average_max_prob(probs: np.ndarray) -> float:
    return float(np.mean(np.max(probs, axis=1)))


def multiclass_nll(probs: np.ndarray, y_int: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-15, 1.0)
    return float(np.mean(-np.log(clipped[np.arange(len(y_int)), y_int])))


def confusion_matrix_3(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_from_cm(cm: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for cls in range(3):
        tp = float(cm[cls, cls])
        fp = float(np.sum(cm[:, cls]) - tp)
        fn = float(np.sum(cm[cls, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        stats[str(cls)] = {"precision": precision, "recall": recall}
    return stats


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * np.abs(y_true - y_pred)) / denom)


def assert_mixture_identity(
    *,
    p: np.ndarray,
    mu_cool: np.ndarray,
    mu_normal: np.ndarray,
    mu_warm: np.ndarray,
    mu_hat: np.ndarray,
    atol: float = 1e-10,
) -> None:
    recomputed = p[:, 0] * mu_cool + p[:, 1] * mu_normal + p[:, 2] * mu_warm
    diff = np.max(np.abs(recomputed - mu_hat)) if len(mu_hat) else 0.0
    if not np.isfinite(diff) or diff > atol:
        raise AssertionError(f"RS-MoE mixture identity failed: max_abs_diff={diff} atol={atol}")


def _ensure_2d_logits(logits: Any, *, n_rows: int) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 1:
        if arr.size == n_rows * 3:
            return arr.reshape(n_rows, 3)
        raise ValueError(f"Expected logits with 3 classes; got shape={arr.shape}")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected logits shape (n,3); got shape={arr.shape}")
    return arr


@dataclass(frozen=True)
class BustRegimeLabelerConfig:
    type: str = "ex108_compat"
    residual_threshold_f: float | None = None
    baseline_pred_source: str | None = None


class BustRegimeLabeler:
    def __init__(
        self,
        config: BustRegimeLabelerConfig,
        *,
        model_cols: list[str],
        target_col: str,
    ) -> None:
        self._config = config
        self._model_cols = list(model_cols)
        self._target_col = target_col
        self._threshold_a: float | None = None

    @property
    def threshold_a(self) -> float | None:
        return self._threshold_a

    def fit(self, train_df: pd.DataFrame) -> "BustRegimeLabeler":
        if self._config.type == "ex108_compat":
            ens_mean = train_df[self._model_cols].to_numpy(dtype=float).mean(axis=1)
            y = train_df[self._target_col].to_numpy(dtype=float)
            resid = y - ens_mean
            a = float(np.quantile(np.abs(resid), 0.60)) if len(resid) else 1.0
            self._threshold_a = a
            return self

        if self._config.type == "residual_threshold":
            if self._config.residual_threshold_f is None:
                raise ValueError("regime_labeler.residual_threshold_f must be set for residual_threshold")
            self._threshold_a = float(self._config.residual_threshold_f)
            return self

        raise ValueError(f"Unknown regime_labeler.type: {self._config.type}")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._threshold_a is None:
            raise RuntimeError("BustRegimeLabeler must be fit() before transform().")
        a = float(self._threshold_a)

        y = df[self._target_col].to_numpy(dtype=float)
        if self._config.type == "ex108_compat":
            ens_mean = df[self._model_cols].to_numpy(dtype=float).mean(axis=1)
        else:
            baseline_source = self._config.baseline_pred_source
            if not baseline_source:
                raise ValueError("regime_labeler.baseline_pred_source must be set for residual_threshold")
            if baseline_source == "ens_mean":
                ens_mean = df[self._model_cols].to_numpy(dtype=float).mean(axis=1)
            else:
                ens_mean = df[baseline_source].to_numpy(dtype=float)

        resid = y - ens_mean
        y_int = np.zeros(len(df), dtype=int)
        y_int[np.abs(resid) <= a] = 1
        y_int[resid > a] = 2
        return y_int


@dataclass(frozen=True)
class GateCalibrationConfig:
    method: str = "temperature_scaling"
    temperature_init: float = 1.0
    temperature_bounds: tuple[float, float] = (0.5, 10.0)
    optimizer: str = "lbfgs"
    max_iter: int = 200
    tol: float = 1e-7


class TemperatureScalerMulticlass:
    def __init__(self, config: GateCalibrationConfig) -> None:
        self._config = config
        self._temperature: float | None = None

    @property
    def temperature(self) -> float:
        if self._temperature is None:
            raise RuntimeError("TemperatureScalerMulticlass not fit.")
        return float(self._temperature)

    def fit(self, logits: np.ndarray, y_int: np.ndarray) -> float:
        if self._config.method != "temperature_scaling":
            raise ValueError(f"Unsupported gate_calibration.method: {self._config.method}")
        optimizer = self._config.optimizer.lower().replace("-", "").replace("_", "")
        if optimizer != "lbfgs":
            raise ValueError(f"Unsupported gate_calibration.optimizer: {self._config.optimizer}")

        Z = np.asarray(logits, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != 3:
            raise ValueError(f"Expected logits shape (n,3); got shape={Z.shape}")
        y = np.asarray(y_int, dtype=int)
        if len(y) != Z.shape[0]:
            raise ValueError("logits and labels length mismatch")
        if len(y) == 0:
            self._temperature = float(self._config.temperature_init)
            return float(self._temperature)

        bounds = self._config.temperature_bounds
        t0 = float(self._config.temperature_init)
        t0 = min(max(t0, float(bounds[0])), float(bounds[1]))

        def _loss_and_grad(t_arr: np.ndarray) -> tuple[float, np.ndarray]:
            t = float(t_arr[0])
            probs = softmax_temperature(Z, t)
            loss = multiclass_nll(probs, y)
            z_y = Z[np.arange(len(y)), y]
            p_dot_z = np.sum(probs * Z, axis=1)
            grad = float(np.mean((z_y - p_dot_z) / (t * t)))
            return loss, np.array([grad], dtype=float)

        def _loss(t_arr: np.ndarray) -> float:
            loss, _ = _loss_and_grad(t_arr)
            return loss

        def _grad(t_arr: np.ndarray) -> np.ndarray:
            _, grad = _loss_and_grad(t_arr)
            return grad

        res = minimize(
            _loss,
            x0=np.array([t0], dtype=float),
            jac=_grad,
            method="L-BFGS-B",
            bounds=[(float(bounds[0]), float(bounds[1]))],
            options={"maxiter": int(self._config.max_iter), "ftol": float(self._config.tol)},
        )
        t_hat = float(res.x[0])
        t_hat = min(max(t_hat, float(bounds[0])), float(bounds[1]))
        self._temperature = t_hat
        return t_hat

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax_temperature(np.asarray(logits, dtype=float), self.temperature)


@dataclass(frozen=True)
class OofGatingConfig:
    enabled: bool = True
    method: str = "expanding_time_blocks"
    n_folds: int = 5
    burnin_fraction: float = 0.20
    min_rows_per_fold: int = 100
    weight_floor: float = 0.02
    random_seed: int = 12345


@dataclass(frozen=True)
class OofGateResult:
    order: np.ndarray
    burnin_size: int
    fold_blocks: list[np.ndarray]
    oof_logits: np.ndarray
    oof_is_model_based: np.ndarray
    burnin_priors: np.ndarray


class OOFGateBuilder:
    def __init__(self, config: OofGatingConfig) -> None:
        self._config = config

    def build_oof_logits(
        self,
        *,
        train_df: pd.DataFrame,
        X_train: np.ndarray,
        y_regime: np.ndarray,
        build_gate_model,
    ) -> OofGateResult:
        if not self._config.enabled:
            raise ValueError("oof_gating.enabled must be true for RS-MoE official runs.")
        if self._config.method != "expanding_time_blocks":
            raise ValueError(f"Unsupported oof_gating.method: {self._config.method}")

        if len(train_df) != X_train.shape[0]:
            raise ValueError("train_df and X_train length mismatch")
        if len(train_df) != len(y_regime):
            raise ValueError("train_df and y_regime length mismatch")

        sort_cols = ["target_date_local", "station_id", "asof_utc"]
        sort_key = (
            train_df[sort_cols]
            .assign(
                target_date_local=pd.to_datetime(train_df["target_date_local"]),
                asof_utc=pd.to_datetime(train_df["asof_utc"]),
            )
            .sort_values(sort_cols, kind="mergesort")
        )
        order = train_df.index.get_indexer(sort_key.index)
        if np.any(order < 0):
            raise ValueError("Failed to compute stable train ordering for OOF gating.")

        n = int(len(order))
        burnin_size = int(np.floor(n * float(self._config.burnin_fraction)))
        burnin_size = max(min(burnin_size, n), 0)

        remaining = n - burnin_size
        n_folds = int(self._config.n_folds)
        if n_folds <= 0:
            raise ValueError("oof_gating.n_folds must be > 0")
        if remaining <= 0:
            raise ValueError("OOF gating has no rows after burn-in; reduce burnin_fraction.")

        fold_sizes = [remaining // n_folds] * n_folds
        for i in range(remaining % n_folds):
            fold_sizes[i] += 1
        fold_blocks: list[np.ndarray] = []
        cursor = burnin_size
        for size in fold_sizes:
            block = order[cursor : cursor + size]
            cursor += size
            fold_blocks.append(block)

        if any(len(block) < int(self._config.min_rows_per_fold) for block in fold_blocks):
            raise ValueError(
                "OOF gating fold too small for min_rows_per_fold. "
                f"fold_sizes={[len(b) for b in fold_blocks]} min_rows_per_fold={self._config.min_rows_per_fold}"
            )

        oof_logits = np.full((n, 3), np.nan, dtype=float)
        oof_is_model_based = np.zeros(n, dtype=bool)

        burnin_order = order[:burnin_size]
        if burnin_size > 0:
            burnin_labels = y_regime[burnin_order]
            priors = np.bincount(burnin_labels, minlength=3).astype(float)
            priors = priors / float(np.sum(priors)) if float(np.sum(priors)) > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            priors = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

        train_blocks: list[np.ndarray] = [burnin_order]
        for block_idx, pred_block in enumerate(fold_blocks):
            fit_idx = np.concatenate(train_blocks) if train_blocks else np.array([], dtype=int)
            if len(fit_idx) == 0:
                raise ValueError("OOF gating training set empty for fold.")

            gate_model = build_gate_model()
            gate_model.fit(X_train[fit_idx], y_regime[fit_idx])
            logits = gate_model.predict(X_train[pred_block], prediction_type="RawFormulaVal")
            logits = _ensure_2d_logits(logits, n_rows=len(pred_block))
            oof_logits[pred_block] = logits
            oof_is_model_based[pred_block] = True
            train_blocks.append(pred_block)

        return OofGateResult(
            order=order,
            burnin_size=burnin_size,
            fold_blocks=fold_blocks,
            oof_logits=oof_logits,
            oof_is_model_based=oof_is_model_based,
            burnin_priors=priors,
        )


@dataclass(frozen=True)
class ExpertsConfig:
    library: str = "xgboost"
    objective_variant: str = "absoluteerror"
    absoluteerror_params: dict[str, Any] | None = None
    quantile_median_params: dict[str, Any] | None = None


class RegimeSwitchingMixtureOfExpertsMeanModel:
    def __init__(
        self,
        *,
        regimes: tuple[str, str, str] = REGIMES_DEFAULT,
        gate_model: Any,
        expert_cool: Any,
        expert_normal: Any,
        expert_warm: Any,
        temperature: float,
    ) -> None:
        self.regimes = regimes
        self.gate_model = gate_model
        self.expert_cool = expert_cool
        self.expert_normal = expert_normal
        self.expert_warm = expert_warm
        self.temperature = float(temperature)

    def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        logits = self.gate_model.predict(X, prediction_type="RawFormulaVal")
        logits = _ensure_2d_logits(logits, n_rows=len(X))
        p = softmax_temperature(logits, self.temperature)
        mu_cool = np.asarray(self.expert_cool.predict(X), dtype=float)
        mu_normal = np.asarray(self.expert_normal.predict(X), dtype=float)
        mu_warm = np.asarray(self.expert_warm.predict(X), dtype=float)
        mu_hat = p[:, 0] * mu_cool + p[:, 1] * mu_normal + p[:, 2] * mu_warm
        assert_mixture_identity(
            p=p,
            mu_cool=mu_cool,
            mu_normal=mu_normal,
            mu_warm=mu_warm,
            mu_hat=mu_hat,
        )
        return {
            "logits": logits,
            "p": p,
            "mu_cool": mu_cool,
            "mu_normal": mu_normal,
            "mu_warm": mu_warm,
            "mu_hat": mu_hat,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_components(X)["mu_hat"]

    def save(
        self,
        path: Path,
        *,
        gate_calibration_payload: dict[str, Any],
    ) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)

        gate_path = path / "gate_model.cbm"
        self.gate_model.save_model(str(gate_path))

        joblib.dump(self.expert_cool, path / "expert_cool_model.joblib")
        joblib.dump(self.expert_normal, path / "expert_normal_model.joblib")
        joblib.dump(self.expert_warm, path / "expert_warm_model.joblib")

        calibration_path = path / "gate_calibration.json"
        calibration_path.write_text(
            json.dumps(gate_calibration_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> "RegimeSwitchingMixtureOfExpertsMeanModel":
        import joblib

        from catboost import CatBoostClassifier

        gate_path = path / "gate_model.cbm"
        if not gate_path.exists():
            raise FileNotFoundError(f"Missing gate model: {gate_path}")
        gate = CatBoostClassifier()
        gate.load_model(str(gate_path))

        expert_cool = joblib.load(path / "expert_cool_model.joblib")
        expert_normal = joblib.load(path / "expert_normal_model.joblib")
        expert_warm = joblib.load(path / "expert_warm_model.joblib")

        cal_payload = json.loads((path / "gate_calibration.json").read_text(encoding="utf-8"))
        temperature = float(cal_payload["temperature"])

        return RegimeSwitchingMixtureOfExpertsMeanModel(
            gate_model=gate,
            expert_cool=expert_cool,
            expert_normal=expert_normal,
            expert_warm=expert_warm,
            temperature=temperature,
        )


def feature_importance_top_k_xgb(model: Any, feature_names: list[str], k: int = 20) -> list[dict[str, Any]]:
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        pairs = []
        for key, value in scores.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < len(feature_names):
                    pairs.append((feature_names[idx], float(value)))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [{"feature": name, "importance": imp} for name, imp in pairs[:k]]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).ravel()
        pairs = list(zip(feature_names, values))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [{"feature": name, "importance": float(imp)} for name, imp in pairs[:k]]
    return []


def write_oof_gate_artifacts(
    run_dir: Path,
    *,
    train_df: pd.DataFrame,
    oof_logits: np.ndarray,
    oof_is_model_based: np.ndarray,
    y_regime: np.ndarray,
    oof_probs: np.ndarray,
    oof_probs_smoothed: np.ndarray,
) -> tuple[Path, Path]:
    ids = train_df[["station_id", "target_date_local", "asof_utc"]].copy()
    logits_df = ids.copy()
    logits_df["logit_cool"] = oof_logits[:, 0]
    logits_df["logit_normal"] = oof_logits[:, 1]
    logits_df["logit_warm"] = oof_logits[:, 2]
    logits_df["oof_is_model_based"] = oof_is_model_based.astype(bool)
    logits_df["y_regime"] = y_regime.astype(int)
    logits_path = run_dir / "oof_gate_logits_train.parquet"
    logits_df.to_parquet(logits_path, index=False, engine="pyarrow")

    probs_df = ids.copy()
    probs_df["p_cool_oof"] = oof_probs[:, 0]
    probs_df["p_normal_oof"] = oof_probs[:, 1]
    probs_df["p_warm_oof"] = oof_probs[:, 2]
    probs_df["p_cool_oof_smoothed"] = oof_probs_smoothed[:, 0]
    probs_df["p_normal_oof_smoothed"] = oof_probs_smoothed[:, 1]
    probs_df["p_warm_oof_smoothed"] = oof_probs_smoothed[:, 2]
    probs_path = run_dir / "oof_gate_probs_train.parquet"
    probs_df.to_parquet(probs_path, index=False, engine="pyarrow")

    return logits_path, probs_path


def _parse_version_tuple(version: str) -> tuple[int, int, int]:
    parts = []
    for token in version.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        if len(parts) >= 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


@dataclass(frozen=True)
class RsMoeFitResult:
    model: RegimeSwitchingMixtureOfExpertsMeanModel
    y_regime_train: np.ndarray
    labeler_threshold_a: float | None
    oof_logits: np.ndarray
    oof_is_model_based: np.ndarray
    burnin_priors: np.ndarray
    temperature: float
    oof_probs: np.ndarray
    oof_probs_smoothed: np.ndarray


def train_rs_moe(
    *,
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    feature_names: list[str],
    target_col: str,
    model_cols_for_labeler: list[str],
    regimes: list[str],
    regime_labeler: BustRegimeLabelerConfig,
    oof_gating: OofGatingConfig,
    gate_model_library: str,
    gate_model_params: dict[str, Any],
    gate_calibration: GateCalibrationConfig,
    experts: ExpertsConfig,
) -> RsMoeFitResult:
    if tuple(regimes) != REGIMES_DEFAULT:
        raise ValueError(f"rs_moe.regimes must be {list(REGIMES_DEFAULT)}; got {regimes}")
    if gate_model_library.lower() != "catboost":
        raise ValueError(f"Unsupported rs_moe.gate_model.library: {gate_model_library}")
    if experts.library.lower() != "xgboost":
        raise ValueError(f"Unsupported rs_moe.experts.library: {experts.library}")

    labeler = BustRegimeLabeler(
        regime_labeler,
        model_cols=model_cols_for_labeler,
        target_col=target_col,
    ).fit(train_df)
    y_regime_train = labeler.transform(train_df)

    from catboost import CatBoostClassifier

    def _build_gate() -> CatBoostClassifier:
        params = dict(gate_model_params)
        params.setdefault("loss_function", "MultiClass")
        return CatBoostClassifier(**params)

    oof_builder = OOFGateBuilder(oof_gating)
    oof = oof_builder.build_oof_logits(
        train_df=train_df,
        X_train=X_train,
        y_regime=y_regime_train,
        build_gate_model=_build_gate,
    )

    model_based_mask = oof.oof_is_model_based
    logits_model_based = oof.oof_logits[model_based_mask]
    y_model_based = y_regime_train[model_based_mask]
    temp_scaler = TemperatureScalerMulticlass(gate_calibration)
    temperature = temp_scaler.fit(logits_model_based, y_model_based)

    oof_probs = np.zeros_like(oof.oof_logits, dtype=float)
    if np.any(model_based_mask):
        oof_probs[model_based_mask] = temp_scaler.predict_proba(oof.oof_logits[model_based_mask])
    burnin_mask = ~model_based_mask
    if np.any(burnin_mask):
        oof_probs[burnin_mask] = oof.burnin_priors.reshape(1, 3)

    weight_floor = float(oof_gating.weight_floor)
    oof_probs_smoothed = (oof_probs + weight_floor) / (1.0 + 3.0 * weight_floor)

    import xgboost as xgb

    if experts.objective_variant == "quantile_median":
        if _parse_version_tuple(xgb.__version__) < (2, 0, 3):
            raise RuntimeError(
                f"xgboost>=2.0.3 required for reg:quantileerror; found {xgb.__version__}"
            )
        expert_params = dict(experts.quantile_median_params or {})
    elif experts.objective_variant == "absoluteerror":
        expert_params = dict(experts.absoluteerror_params or {})
    else:
        raise ValueError(f"Unsupported rs_moe.experts.objective_variant: {experts.objective_variant}")

    if "objective" not in expert_params:
        raise ValueError("Expert params must include an explicit objective.")

    def _fit_expert(sample_weight: np.ndarray) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(**expert_params)
        model.fit(X_train, train_df[target_col].to_numpy(dtype=float), sample_weight=sample_weight)
        return model

    expert_cool = _fit_expert(oof_probs_smoothed[:, 0])
    expert_normal = _fit_expert(oof_probs_smoothed[:, 1])
    expert_warm = _fit_expert(oof_probs_smoothed[:, 2])

    gate_full = _build_gate()
    gate_full.fit(X_train, y_regime_train)

    model = RegimeSwitchingMixtureOfExpertsMeanModel(
        gate_model=gate_full,
        expert_cool=expert_cool,
        expert_normal=expert_normal,
        expert_warm=expert_warm,
        temperature=temperature,
    )

    return RsMoeFitResult(
        model=model,
        y_regime_train=y_regime_train,
        labeler_threshold_a=labeler.threshold_a,
        oof_logits=oof.oof_logits,
        oof_is_model_based=oof.oof_is_model_based,
        burnin_priors=oof.burnin_priors,
        temperature=temperature,
        oof_probs=oof_probs,
        oof_probs_smoothed=oof_probs_smoothed,
    )
