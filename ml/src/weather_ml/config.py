"""Configuration helpers for the ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class DataConfig:
    csv_path: str
    dataset_schema_version: int = 1


@dataclass(frozen=True)
class ValidationConfig:
    strict_schema: bool = True
    forecast_min_f: float = -80.0
    forecast_max_f: float = 140.0
    spread_min_f: float = 0.0
    require_asof_not_after_target: bool = True


@dataclass(frozen=True)
class ValidationSplitConfig:
    enabled: bool = True
    val_start: date | None = None
    val_end: date | None = None


@dataclass(frozen=True)
class CvConfig:
    enabled: bool = True
    n_splits: int = 5
    gap_days: int = 2


@dataclass(frozen=True)
class SplitConfig:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    gap_dates: list[date] = field(default_factory=list)
    validation: ValidationSplitConfig = field(default_factory=ValidationSplitConfig)
    cv: CvConfig = field(default_factory=CvConfig)


@dataclass(frozen=True)
class ClimatologyConfig:
    enabled: bool = False
    label_lag_days: int = 2
    rolling_windows_days: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureConfig:
    base_features: list[str] = field(default_factory=list)
    ensemble_stats: bool = True
    pairwise_deltas: bool = True
    model_vs_ens_deltas: bool = True
    calendar: bool = True
    station_onehot: bool = True
    pairwise_pairs: list[list[str]] = field(default_factory=list)
    climatology: ClimatologyConfig = field(default_factory=ClimatologyConfig)


@dataclass(frozen=True)
class MeanModelConfig:
    candidates: list[str] = field(default_factory=list)
    primary: str = "ridge"
    param_grid: dict[str, dict[str, list[Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class SigmaModelConfig:
    method: str = "two_stage"
    primary: str = "ridge"
    param_grid: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    sigma_floor: float = 0.25
    eps: float = 1e-6


@dataclass(frozen=True)
class ModelConfig:
    mean: MeanModelConfig
    sigma: SigmaModelConfig


@dataclass(frozen=True)
class ArtifactConfig:
    root_dir: str = "artifacts"
    run_id: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class SeedConfig:
    global_seed: int = 1337
    force_single_thread: bool = True


@dataclass(frozen=True)
class DistributionConfig:
    support_min_f: int = -30
    support_max_f: int = 130


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    method: str = "isotonic"
    bins_to_calibrate: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class GlobalNormalCalibrationConfig:
    enabled: bool = False
    cal_start: date | None = None
    cal_end: date | None = None
    ddof: int = 1
    station_scope: list[str] | None = None


@dataclass(frozen=True)
class BaselineMedianCalibrationConfig:
    enabled: bool = False
    cal_start: date | None = None
    cal_end: date | None = None
    ddof: int = 1
    station_scope: list[str] | None = None
    forecast_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PostprocessConfig:
    global_normal_calibration: GlobalNormalCalibrationConfig = field(
        default_factory=GlobalNormalCalibrationConfig
    )
    baseline_median_calibration: BaselineMedianCalibrationConfig = field(
        default_factory=BaselineMedianCalibrationConfig
    )


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig
    features: FeatureConfig
    models: ModelConfig
    artifacts: ArtifactConfig
    seeds: SeedConfig
    distribution: DistributionConfig
    calibration: CalibrationConfig
    postprocess: PostprocessConfig


def load_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    data_raw = _require_section(raw, "data")
    split_raw = _require_section(raw, "split")
    features_raw = _require_section(raw, "features")
    models_raw = _require_section(raw, "models")

    data = DataConfig(
        csv_path=str(_require_value(data_raw, "csv_path")),
        dataset_schema_version=int(data_raw.get("dataset_schema_version", 1)),
    )

    validation_raw = raw.get("validation", {}) or {}
    validation = ValidationConfig(
        strict_schema=bool(validation_raw.get("strict_schema", True)),
        forecast_min_f=float(validation_raw.get("forecast_min_f", -80.0)),
        forecast_max_f=float(validation_raw.get("forecast_max_f", 140.0)),
        spread_min_f=float(validation_raw.get("spread_min_f", 0.0)),
        require_asof_not_after_target=bool(
            validation_raw.get("require_asof_not_after_target", True)
        ),
    )

    validation_split_raw = split_raw.get("validation", {}) or {}
    validation_split = ValidationSplitConfig(
        enabled=bool(validation_split_raw.get("enabled", True)),
        val_start=_parse_date(validation_split_raw.get("val_start")),
        val_end=_parse_date(validation_split_raw.get("val_end")),
    )

    cv_raw = split_raw.get("cv", {}) or {}
    cv = CvConfig(
        enabled=bool(cv_raw.get("enabled", True)),
        n_splits=int(cv_raw.get("n_splits", 5)),
        gap_days=int(cv_raw.get("gap_days", 2)),
    )

    split = SplitConfig(
        train_start=_parse_date(_require_value(split_raw, "train_start")),
        train_end=_parse_date(_require_value(split_raw, "train_end")),
        test_start=_parse_date(_require_value(split_raw, "test_start")),
        test_end=_parse_date(_require_value(split_raw, "test_end")),
        gap_dates=_parse_dates(split_raw.get("gap_dates", [])),
        validation=validation_split,
        cv=cv,
    )

    derived_raw = features_raw.get("derived") or features_raw
    climatology_raw = derived_raw.get("climatology", {}) or {}
    climatology = ClimatologyConfig(
        enabled=bool(climatology_raw.get("enabled", False)),
        label_lag_days=int(climatology_raw.get("label_lag_days", 2)),
        rolling_windows_days=[
            int(value) for value in climatology_raw.get("rolling_windows_days", [])
        ],
    )

    features = FeatureConfig(
        base_features=[str(value) for value in features_raw.get("base_features", [])],
        ensemble_stats=bool(derived_raw.get("ensemble_stats", True)),
        pairwise_deltas=bool(derived_raw.get("pairwise_deltas", True)),
        model_vs_ens_deltas=bool(derived_raw.get("model_vs_ens_deltas", True)),
        calendar=bool(derived_raw.get("calendar", True)),
        station_onehot=bool(derived_raw.get("station_onehot", True)),
        pairwise_pairs=[list(pair) for pair in derived_raw.get("pairwise_pairs", [])],
        climatology=climatology,
    )

    mean_raw = models_raw.get("mean", {}) or {}
    sigma_raw = models_raw.get("sigma", {}) or {}
    mean_primary = str(mean_raw.get("primary", "ridge"))
    mean_param_grid = _normalize_param_grid(mean_raw.get("param_grid", {}), mean_primary)
    mean = MeanModelConfig(
        candidates=[str(value) for value in mean_raw.get("candidates", [])],
        primary=mean_primary,
        param_grid=mean_param_grid,
    )
    sigma_primary = str(sigma_raw.get("primary", "ridge"))
    sigma_param_grid = _normalize_param_grid(
        sigma_raw.get("param_grid", {}), sigma_primary
    )
    sigma = SigmaModelConfig(
        method=str(sigma_raw.get("method", "two_stage")),
        primary=sigma_primary,
        param_grid=sigma_param_grid,
        sigma_floor=float(sigma_raw.get("sigma_floor", 0.25)),
        eps=float(sigma_raw.get("eps", 1e-6)),
    )
    models = ModelConfig(mean=mean, sigma=sigma)

    artifacts_raw = raw.get("artifacts", {}) or {}
    artifacts = ArtifactConfig(
        root_dir=str(artifacts_raw.get("root_dir", "artifacts")),
        run_id=artifacts_raw.get("run_id"),
        overwrite=bool(artifacts_raw.get("overwrite", False)),
    )

    seeds_raw = raw.get("seeds", {}) or {}
    seeds = SeedConfig(
        global_seed=int(seeds_raw.get("global_seed", 1337)),
        force_single_thread=bool(seeds_raw.get("force_single_thread", True)),
    )

    distribution_raw = raw.get("distribution", {}) or {}
    distribution = DistributionConfig(
        support_min_f=int(distribution_raw.get("support_min_f", -30)),
        support_max_f=int(distribution_raw.get("support_max_f", 130)),
    )

    calibration_raw = raw.get("calibration", {}) or {}
    calibration = CalibrationConfig(
        enabled=bool(calibration_raw.get("enabled", True)),
        method=str(calibration_raw.get("method", "isotonic")),
        bins_to_calibrate=[
            dict(item) for item in calibration_raw.get("bins_to_calibrate", [])
        ],
    )

    postprocess_raw = raw.get("postprocess", {}) or {}
    gnc_raw = postprocess_raw.get("global_normal_calibration", {}) or {}
    bmc_raw = postprocess_raw.get("baseline_median_calibration", {}) or {}
    postprocess = PostprocessConfig(
        global_normal_calibration=GlobalNormalCalibrationConfig(
            enabled=bool(gnc_raw.get("enabled", False)),
            cal_start=_parse_date(gnc_raw.get("cal_start")),
            cal_end=_parse_date(gnc_raw.get("cal_end")),
            ddof=int(gnc_raw.get("ddof", 1)),
            station_scope=_parse_station_scope(gnc_raw.get("station_scope", "ALL")),
        ),
        baseline_median_calibration=BaselineMedianCalibrationConfig(
            enabled=bool(bmc_raw.get("enabled", False)),
            cal_start=_parse_date(bmc_raw.get("cal_start")),
            cal_end=_parse_date(bmc_raw.get("cal_end")),
            ddof=int(bmc_raw.get("ddof", 1)),
            station_scope=_parse_station_scope(bmc_raw.get("station_scope", "ALL")),
            forecast_columns=[
                str(value) for value in bmc_raw.get("forecast_columns", [])
            ],
        ),
    )

    return TrainingConfig(
        data=data,
        validation=validation,
        split=split,
        features=features,
        models=models,
        artifacts=artifacts,
        seeds=seeds,
        distribution=distribution,
        calibration=calibration,
        postprocess=postprocess,
    )


def resolve_paths(config: TrainingConfig, *, repo_root: Path) -> TrainingConfig:
    csv_path = Path(config.data.csv_path)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    artifacts_root = Path(config.artifacts.root_dir)
    if not artifacts_root.is_absolute():
        artifacts_root = repo_root / artifacts_root
    return TrainingConfig(
        data=DataConfig(
            csv_path=str(csv_path),
            dataset_schema_version=config.data.dataset_schema_version,
        ),
        validation=config.validation,
        split=config.split,
        features=config.features,
        models=config.models,
        artifacts=ArtifactConfig(
            root_dir=str(artifacts_root),
            run_id=config.artifacts.run_id,
            overwrite=config.artifacts.overwrite,
        ),
        seeds=config.seeds,
        distribution=config.distribution,
        calibration=config.calibration,
        postprocess=config.postprocess,
    )


def _require_section(raw: dict, name: str) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: {name}")
    return section


def _require_value(raw: dict, name: str) -> Any:
    value = raw.get(name)
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return value


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception as exc:
        raise ValueError(f"Invalid date value: {value}") from exc


def _parse_dates(values: list[object]) -> list[date]:
    return [_parse_date(value) for value in values if value is not None]


def _parse_station_scope(value: object) -> list[str] | None:
    if isinstance(value, str):
        if value.strip().upper() == "ALL":
            return None
        return [value.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _normalize_param_grid(raw: Mapping[str, Any] | None, primary: str) -> dict[str, dict[str, list[Any]]]:
    if not raw:
        return {}
    if all(isinstance(value, list) for value in raw.values()):
        return {primary: {str(key): list(value) for key, value in raw.items()}}
    normalized: dict[str, dict[str, list[Any]]] = {}
    for model_name, params in raw.items():
        if not isinstance(params, Mapping):
            raise ValueError(f"Invalid param_grid for model {model_name}")
        normalized[str(model_name)] = {
            str(key): list(value) for key, value in params.items()
        }
    return normalized
