"""Configuration objects for MOS dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import json
import pandas as pd


@dataclass(frozen=True)
class KnnViewConfig:
    name: str
    pool: str
    distance: str


@dataclass(frozen=True)
class MosDatasetConfig:
    station_id: str
    station_zoneid: str
    feature_version: str
    build_start_asof: date
    output_start_asof: date
    end_asof: date
    asof_rule: str = "target_minus_one_day_12z"
    asof_hour_utc: int = 12
    asof_minute_utc: int = 0
    climate_day_utc_offset_hours: int = -5
    obs_cutoff_lag_days: int = 1
    include_retrieved_at_guard: bool = True
    mos_runtime_policy: str = "latest_before_cutoff"
    mos_runtime_hour_utc: int = 12
    mos_runtime_minute_utc: int = 0
    asof_buckets_hours: list[int] | None = None
    truth_table: str = "station_daily_truth"
    models: list[str] = None
    variables: list[str] = None
    baseline_start: date | None = None
    baseline_end: date | None = None
    distance_calibration_start: date | None = None
    distance_calibration_end: date | None = None
    distance_features: list[str] = None
    distance_feature_weights: dict[str, float] | None = None
    missing_penalty: float = 1.0
    k: int = 90
    skip_knn: bool = False
    knn_views: list[KnnViewConfig] | None = None
    thresholds: list[int] | None = None
    tau_fixed: list[float] | None = None
    obs_windows_days: list[int] | None = None
    obs_slope_windows_days: list[int] | None = None
    bias_windows_days: list[int] | None = None
    output_root: str = "dataset"
    output_partition_yearly: bool = False
    use_time_step_mos: bool = False
    iem_minute_path: str | None = None
    iem_minute_station: str | None = None
    iem_minute_timezone: str | None = None
    iem_minute_time_col: str = "valid(UTC)"
    iem_minute_temp_col: str = "tmpf"
    iem_minute_last_hours: int = 6
    iem_minute_plateau_eps_f: float = 0.5
    iem_minute_drop_thr_f: float = 1.5
    iem_minute_auc_thresholds_f: list[float] | None = None

    def normalized(self) -> "MosDatasetConfig":
        models = [str(m).upper() for m in (self.models or [])]
        variables = [str(v).lower() for v in (self.variables or [])]
        distance_features = [str(v) for v in (self.distance_features or [])]
        thresholds = [int(v) for v in (self.thresholds or [])]
        tau_fixed = [float(v) for v in (self.tau_fixed or [])]
        obs_windows_days = [int(v) for v in (self.obs_windows_days or [])]
        obs_slope_windows_days = [int(v) for v in (self.obs_slope_windows_days or [])]
        bias_windows_days = [int(v) for v in (self.bias_windows_days or [])]
        iem_minute_auc_thresholds_f = [float(v) for v in (self.iem_minute_auc_thresholds_f or [])]
        distance_feature_weights = self.distance_feature_weights or {}
        knn_views = self.knn_views or []
        return MosDatasetConfig(
            station_id=str(self.station_id),
            station_zoneid=str(self.station_zoneid),
            feature_version=str(self.feature_version),
            build_start_asof=self.build_start_asof,
            output_start_asof=self.output_start_asof,
            end_asof=self.end_asof,
            asof_rule=str(self.asof_rule),
            asof_hour_utc=int(self.asof_hour_utc),
            asof_minute_utc=int(self.asof_minute_utc),
            climate_day_utc_offset_hours=int(self.climate_day_utc_offset_hours),
            obs_cutoff_lag_days=int(self.obs_cutoff_lag_days),
            include_retrieved_at_guard=bool(self.include_retrieved_at_guard),
            mos_runtime_policy=str(self.mos_runtime_policy),
            mos_runtime_hour_utc=int(self.mos_runtime_hour_utc),
            mos_runtime_minute_utc=int(self.mos_runtime_minute_utc),
            asof_buckets_hours=sorted(set(int(v) for v in (self.asof_buckets_hours or []))),
            truth_table=str(self.truth_table),
            models=sorted(set(models)),
            variables=sorted(set(variables)),
            baseline_start=self.baseline_start,
            baseline_end=self.baseline_end,
            distance_calibration_start=self.distance_calibration_start,
            distance_calibration_end=self.distance_calibration_end,
            distance_features=distance_features,
            distance_feature_weights=distance_feature_weights,
            missing_penalty=float(self.missing_penalty),
            k=int(self.k),
            skip_knn=bool(self.skip_knn),
            knn_views=knn_views,
            thresholds=thresholds,
            tau_fixed=tau_fixed,
            obs_windows_days=obs_windows_days,
            obs_slope_windows_days=obs_slope_windows_days,
            bias_windows_days=bias_windows_days,
            output_root=str(self.output_root),
            output_partition_yearly=bool(self.output_partition_yearly),
            use_time_step_mos=bool(self.use_time_step_mos),
            iem_minute_path=str(self.iem_minute_path) if self.iem_minute_path else None,
            iem_minute_station=str(self.iem_minute_station) if self.iem_minute_station else None,
            iem_minute_timezone=str(self.iem_minute_timezone) if self.iem_minute_timezone else None,
            iem_minute_time_col=str(self.iem_minute_time_col),
            iem_minute_temp_col=str(self.iem_minute_temp_col),
            iem_minute_last_hours=int(self.iem_minute_last_hours),
            iem_minute_plateau_eps_f=float(self.iem_minute_plateau_eps_f),
            iem_minute_drop_thr_f=float(self.iem_minute_drop_thr_f),
            iem_minute_auc_thresholds_f=iem_minute_auc_thresholds_f,
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        cfg = self.normalized()
        return {
            "station_id": cfg.station_id,
            "station_zoneid": cfg.station_zoneid,
            "feature_version": cfg.feature_version,
            "build_start_asof": cfg.build_start_asof.isoformat(),
            "output_start_asof": cfg.output_start_asof.isoformat(),
            "end_asof": cfg.end_asof.isoformat(),
            "asof_rule": cfg.asof_rule,
            "asof_hour_utc": cfg.asof_hour_utc,
            "asof_minute_utc": cfg.asof_minute_utc,
            "climate_day_utc_offset_hours": cfg.climate_day_utc_offset_hours,
            "obs_cutoff_lag_days": cfg.obs_cutoff_lag_days,
            "include_retrieved_at_guard": cfg.include_retrieved_at_guard,
            "mos_runtime_policy": cfg.mos_runtime_policy,
            "mos_runtime_hour_utc": cfg.mos_runtime_hour_utc,
            "mos_runtime_minute_utc": cfg.mos_runtime_minute_utc,
            "asof_buckets_hours": cfg.asof_buckets_hours,
            "truth_table": cfg.truth_table,
            "models": cfg.models,
            "variables": cfg.variables,
            "baseline_start": cfg.baseline_start.isoformat() if cfg.baseline_start else None,
            "baseline_end": cfg.baseline_end.isoformat() if cfg.baseline_end else None,
            "distance_calibration_start": cfg.distance_calibration_start.isoformat()
            if cfg.distance_calibration_start
            else None,
            "distance_calibration_end": cfg.distance_calibration_end.isoformat()
            if cfg.distance_calibration_end
            else None,
            "distance_features": cfg.distance_features,
            "distance_feature_weights": cfg.distance_feature_weights,
            "missing_penalty": cfg.missing_penalty,
            "k": cfg.k,
            "skip_knn": cfg.skip_knn,
            "knn_views": [
                {"name": view.name, "pool": view.pool, "distance": view.distance}
                for view in cfg.knn_views or []
            ],
            "thresholds": cfg.thresholds,
            "tau_fixed": cfg.tau_fixed,
            "obs_windows_days": cfg.obs_windows_days,
            "obs_slope_windows_days": cfg.obs_slope_windows_days,
            "bias_windows_days": cfg.bias_windows_days,
            "output_root": cfg.output_root,
            "output_partition_yearly": cfg.output_partition_yearly,
            "use_time_step_mos": cfg.use_time_step_mos,
            "iem_minute_path": cfg.iem_minute_path,
            "iem_minute_station": cfg.iem_minute_station,
            "iem_minute_timezone": cfg.iem_minute_timezone,
            "iem_minute_time_col": cfg.iem_minute_time_col,
            "iem_minute_temp_col": cfg.iem_minute_temp_col,
            "iem_minute_last_hours": cfg.iem_minute_last_hours,
            "iem_minute_plateau_eps_f": cfg.iem_minute_plateau_eps_f,
            "iem_minute_drop_thr_f": cfg.iem_minute_drop_thr_f,
            "iem_minute_auc_thresholds_f": cfg.iem_minute_auc_thresholds_f,
        }


def load_config(path: str | Path) -> MosDatasetConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _parse_config(data)


def _parse_config(raw: dict[str, Any]) -> MosDatasetConfig:
    def parse_date(value: Any) -> date:
        return pd.to_datetime(value).date()

    def parse_views(values: list[dict[str, Any]] | None) -> list[KnnViewConfig]:
        views: list[KnnViewConfig] = []
        for item in values or []:
            views.append(
                KnnViewConfig(
                    name=str(item.get("name")),
                    pool=str(item.get("pool")),
                    distance=str(item.get("distance")),
                )
            )
        return views

    return MosDatasetConfig(
        station_id=str(raw["station_id"]),
        station_zoneid=str(raw["station_zoneid"]),
        feature_version=str(raw["feature_version"]),
        build_start_asof=parse_date(raw["build_start_asof"]),
        output_start_asof=parse_date(raw["output_start_asof"]),
        end_asof=parse_date(raw["end_asof"]),
        asof_rule=str(raw.get("asof_rule", "target_minus_one_day_12z")),
        asof_hour_utc=int(raw.get("asof_hour_utc", 12)),
        asof_minute_utc=int(raw.get("asof_minute_utc", 0)),
        climate_day_utc_offset_hours=int(raw.get("climate_day_utc_offset_hours", -5)),
        obs_cutoff_lag_days=int(raw.get("obs_cutoff_lag_days", 1)),
        include_retrieved_at_guard=bool(raw.get("include_retrieved_at_guard", True)),
        mos_runtime_policy=str(raw.get("mos_runtime_policy", "latest_before_cutoff")),
        mos_runtime_hour_utc=int(raw.get("mos_runtime_hour_utc", 12)),
        mos_runtime_minute_utc=int(raw.get("mos_runtime_minute_utc", 0)),
        truth_table=str(raw.get("truth_table", "station_daily_truth")),
        asof_buckets_hours=list(raw.get("asof_buckets_hours", []) or []),
        models=list(raw.get("models", ["GFS", "NAM"])),
        variables=list(raw.get("variables", [])),
        baseline_start=parse_date(raw["baseline_start"]) if raw.get("baseline_start") else None,
        baseline_end=parse_date(raw["baseline_end"]) if raw.get("baseline_end") else None,
        distance_calibration_start=parse_date(raw["distance_calibration_start"])
        if raw.get("distance_calibration_start")
        else None,
        distance_calibration_end=parse_date(raw["distance_calibration_end"])
        if raw.get("distance_calibration_end")
        else None,
        distance_features=list(raw.get("distance_features", [])),
        distance_feature_weights=raw.get("distance_feature_weights", {}),
        missing_penalty=float(raw.get("missing_penalty", 1.0)),
        k=int(raw.get("k", 90)),
        skip_knn=bool(raw.get("skip_knn", False)),
        knn_views=parse_views(raw.get("knn_views")),
        thresholds=list(raw.get("thresholds", [])),
        tau_fixed=list(raw.get("tau_fixed", [0.8, 1.0, 1.6])),
        obs_windows_days=list(raw.get("obs_windows_days", [3, 7, 14, 30, 45, 90, 365])),
        obs_slope_windows_days=list(raw.get("obs_slope_windows_days", [7, 14, 30, 90])),
        bias_windows_days=list(raw.get("bias_windows_days", [7, 14, 30, 60, 90, 365])),
        output_root=str(raw.get("output_root", "dataset")),
        output_partition_yearly=bool(raw.get("output_partition_yearly", False)),
        use_time_step_mos=bool(raw.get("use_time_step_mos", False)),
        iem_minute_path=raw.get("iem_minute_path"),
        iem_minute_station=raw.get("iem_minute_station"),
        iem_minute_timezone=raw.get("iem_minute_timezone"),
        iem_minute_time_col=str(raw.get("iem_minute_time_col", "valid(UTC)")),
        iem_minute_temp_col=str(raw.get("iem_minute_temp_col", "tmpf")),
        iem_minute_last_hours=int(raw.get("iem_minute_last_hours", 6)),
        iem_minute_plateau_eps_f=float(raw.get("iem_minute_plateau_eps_f", 0.5)),
        iem_minute_drop_thr_f=float(raw.get("iem_minute_drop_thr_f", 1.5)),
        iem_minute_auc_thresholds_f=list(raw.get("iem_minute_auc_thresholds_f", [85.0, 88.0])),
    ).normalized()
