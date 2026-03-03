from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml


DEFAULT_QUANTILES = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]
INNER_FOLDS = [
    ("2007-12-31", "2008-01-01", "2010-12-31"),
    ("2010-12-31", "2011-01-01", "2013-12-31"),
    ("2013-12-31", "2014-01-01", "2016-12-31"),
    ("2016-12-31", "2017-01-01", "2019-12-31"),
    ("2019-12-31", "2020-01-01", "2021-12-31"),
]


@dataclass
class SplitConfig:
    train_start: str = "1973-01-01"
    train_end: str = "2021-12-31"
    dev_start: str = "2022-01-01"
    dev_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31"


@dataclass
class ModelConfig:
    quantiles: list[float] = field(default_factory=lambda: list(DEFAULT_QUANTILES))
    k_neighbors: int = 64
    random_seed: int = 42
    conformal_window: int = 365
    conformal_min_warmup: int = 120


@dataclass
class PipelineConfig:
    obs_csv: str
    truth_csv: str
    station_universe: str
    schema_profile: str | None = None
    bucket_config: str | None = None
    market_odds_file: str | None = None
    decision_stockholm_time: str = "19:00"
    output_dir: str = "artifacts/quantile_knn_conformal_KNYC_v1"
    skip_sanitization: bool = False
    force_rebuild_features: bool = False
    emit_global_diagnostics: bool = True
    allow_debug_failure: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    if path:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must be a mapping.")
        cfg = _update_dataclass(PipelineConfig(obs_csv="", truth_csv="", station_universe=""), raw)
    else:
        cfg = PipelineConfig(obs_csv="", truth_csv="", station_universe="")

    if overrides:
        cfg = _update_dataclass(cfg, overrides)

    missing = [k for k in ("obs_csv", "truth_csv", "station_universe", "output_dir") if not getattr(cfg, k)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    return cfg


def parse_date(value: str) -> date:
    return date.fromisoformat(value)
