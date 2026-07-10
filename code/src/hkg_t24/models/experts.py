"""Expert definitions and chronological OOF prediction generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import mean

from hkg_t24.constants import EXPERT_IDS, PLACEHOLDER_REASON_CODES
from hkg_t24.features.matrix_builder import FeatureMatrixRow, FeatureValue
from hkg_t24.models.folds import FoldSpec, validate_folds


@dataclass(frozen=True)
class ExpertPrediction:
    target_date_hkt: date
    cutoff_id: str
    snapshot_id: str
    expert_id: str
    expert_scope: str
    fold_id: str
    prediction_tmax_c: float | None
    prediction_residual_c: float | None
    raw_anchor_tmax_c: float | None
    prediction_status: str
    placeholder_reason: str | None
    train_end_date: date | None
    test_start_date: date | None
    router_weight_cap: float
    feature_schema_version: str


def _number(value: FeatureValue) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return None


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rows_in_range(rows: Sequence[FeatureMatrixRow], start: date, end: date) -> list[FeatureMatrixRow]:
    return [row for row in rows if start <= row.target_date_hkt <= end and row.target_tmax_c is not None]


def _mean_residual(
    rows: Sequence[FeatureMatrixRow],
    anchor_getter: Callable[[FeatureMatrixRow], float | None],
) -> float | None:
    residuals = [
        float(row.target_tmax_c) - anchor
        for row in rows
        for anchor in [anchor_getter(row)]
        if row.target_tmax_c is not None and anchor is not None
    ]
    if not residuals:
        return None
    return mean(residuals)


def _prediction(
    row: FeatureMatrixRow,
    expert_id: str,
    fold: FoldSpec,
    *,
    expert_scope: str,
    prediction_tmax_c: float | None,
    prediction_residual_c: float | None,
    raw_anchor_tmax_c: float | None,
    prediction_status: str = "active",
    placeholder_reason: str | None = None,
    router_weight_cap: float,
) -> ExpertPrediction:
    return ExpertPrediction(
        target_date_hkt=row.target_date_hkt,
        cutoff_id=row.cutoff_id,
        snapshot_id=row.snapshot_id,
        expert_id=expert_id,
        expert_scope=expert_scope,
        fold_id=fold.fold_id,
        prediction_tmax_c=prediction_tmax_c,
        prediction_residual_c=prediction_residual_c,
        raw_anchor_tmax_c=raw_anchor_tmax_c,
        prediction_status=prediction_status,
        placeholder_reason=placeholder_reason,
        train_end_date=fold.train_end_date,
        test_start_date=fold.test_start_date,
        router_weight_cap=router_weight_cap,
        feature_schema_version=row.schema_version,
    )


def expert_feature_names(expert_id: str) -> tuple[str, ...]:
    """Return the feature prefixes/keys used by each Jira 002 expert."""
    if expert_id == "E0_OFFICIAL_RAW_ANCHOR":
        return ("official__forecast_max_c",)
    if expert_id == "E1_OFFICIAL_RESIDUAL":
        return ("official__", "target__", "calendar__", "online__official_raw__")
    if expert_id == "E2_TARGET_MEMORY":
        return ("target__", "calendar__")
    if expert_id == "E3_STATION_PROXY":
        return ("station__", "climate__", "official__forecast_max_c")
    if expert_id == "E4_GFS_MOS":
        return ("gfs__", "official__", "target__", "calendar__", "online__gfs_mos__")
    if expert_id == "E5_GEFS_ENSEMBLE":
        return ("gefsmean__", "gefsens__", "official__", "target__", "calendar__", "online__gefs_prob_mos__")
    if expert_id in set(EXPERT_IDS):
        return ("live_shadow_placeholder",)
    raise ValueError(f"Unknown expert_id: {expert_id}")


def _official_anchor(row: FeatureMatrixRow) -> float | None:
    return _number(row.features.get("official__forecast_max_c"))


def _target_lag2_anchor(row: FeatureMatrixRow) -> float | None:
    return _number(row.features.get("target__lag2_tmax_c"))


def _gfs_anchor(row: FeatureMatrixRow) -> float | None:
    return _number(row.features.get("gfs__center__tmax_c"))


def _gefs_anchor(row: FeatureMatrixRow) -> float | None:
    p50 = _number(row.features.get("gefsens__center__tmax_p50_c"))
    return p50 if p50 is not None else _number(row.features.get("gefsmean__center__tmax_c"))


def _proxy_anchor(row: FeatureMatrixRow) -> float | None:
    proxy = _number(row.features.get("station__network__tmax_mean_c"))
    if proxy is not None:
        return proxy
    return _official_anchor(row)


def _direct_shadow_anchor(row: FeatureMatrixRow, feature_names: Sequence[str]) -> float | None:
    for feature_name in feature_names:
        value = _number(row.features.get(feature_name))
        if value is not None:
            return value
    return None


def _residual_expert_predictions(
    *,
    expert_id: str,
    expert_scope: str,
    train_rows: Sequence[FeatureMatrixRow],
    test_rows: Sequence[FeatureMatrixRow],
    fold: FoldSpec,
    anchor_getter: Callable[[FeatureMatrixRow], float | None],
    residual_cap_c: float,
    router_weight_cap: float,
) -> list[ExpertPrediction]:
    correction = _mean_residual(train_rows, anchor_getter)
    predictions: list[ExpertPrediction] = []
    for row in test_rows:
        anchor = anchor_getter(row)
        if anchor is None or correction is None:
            predictions.append(
                _prediction(
                    row,
                    expert_id,
                    fold,
                    expert_scope=expert_scope,
                    prediction_tmax_c=None,
                    prediction_residual_c=None,
                    raw_anchor_tmax_c=anchor,
                    prediction_status="placeholder",
                    placeholder_reason="INSUFFICIENT_HISTORY",
                    router_weight_cap=0.0,
                )
            )
            continue
        capped = _clip(correction, -residual_cap_c, residual_cap_c)
        predictions.append(
            _prediction(
                row,
                expert_id,
                fold,
                expert_scope=expert_scope,
                prediction_tmax_c=anchor + capped,
                prediction_residual_c=capped,
                raw_anchor_tmax_c=anchor,
                router_weight_cap=router_weight_cap,
            )
        )
    return predictions


def generate_expert_oof_predictions(
    rows: Sequence[FeatureMatrixRow],
    folds: Sequence[FoldSpec],
    *,
    e1_promoted: bool = True,
    e3_proxy_enabled: bool = False,
) -> list[ExpertPrediction]:
    """Generate Jira 002 chronological OOF/direct/placeholder expert rows."""
    validate_folds(list(folds))
    predictions: list[ExpertPrediction] = []
    for fold in folds:
        train_rows = _rows_in_range(rows, fold.train_start_date, fold.train_end_date)
        test_rows = _rows_in_range(rows, fold.test_start_date, fold.test_end_date)
        for row in test_rows:
            anchor = _official_anchor(row)
            predictions.append(
                _prediction(
                    row,
                    "E0_OFFICIAL_RAW_ANCHOR",
                    fold,
                    expert_scope="strict",
                    prediction_tmax_c=anchor,
                    prediction_residual_c=None,
                    raw_anchor_tmax_c=anchor,
                    prediction_status="active" if anchor is not None else "placeholder",
                    placeholder_reason=None if anchor is not None else "NO_ELIGIBLE_ROWS_FOR_DATE",
                    router_weight_cap=1.0 if anchor is not None else 0.0,
                )
            )

        predictions.extend(
            _residual_expert_predictions(
                expert_id="E1_OFFICIAL_RESIDUAL",
                expert_scope="strict",
                train_rows=train_rows,
                test_rows=test_rows,
                fold=fold,
                anchor_getter=_official_anchor,
                residual_cap_c=0.7,
                router_weight_cap=0.8 if e1_promoted else 0.0,
            )
        )
        predictions.extend(
            _residual_expert_predictions(
                expert_id="E2_TARGET_MEMORY",
                expert_scope="strict",
                train_rows=train_rows,
                test_rows=test_rows,
                fold=fold,
                anchor_getter=_target_lag2_anchor,
                residual_cap_c=1.0,
                router_weight_cap=0.4,
            )
        )
        if e3_proxy_enabled:
            predictions.extend(
                _residual_expert_predictions(
                    expert_id="E3_STATION_PROXY",
                    expert_scope="proxy",
                    train_rows=train_rows,
                    test_rows=test_rows,
                    fold=fold,
                    anchor_getter=_proxy_anchor,
                    residual_cap_c=0.5,
                    router_weight_cap=0.0,
                )
            )
        else:
            for row in test_rows:
                predictions.append(
                    _prediction(
                        row,
                        "E3_STATION_PROXY",
                        fold,
                        expert_scope="proxy",
                        prediction_tmax_c=None,
                        prediction_residual_c=None,
                        raw_anchor_tmax_c=None,
                        prediction_status="placeholder",
                        placeholder_reason="NOT_PROMOTED",
                        router_weight_cap=0.0,
                    )
                )
        predictions.extend(
            _residual_expert_predictions(
                expert_id="E4_GFS_MOS",
                expert_scope="strict",
                train_rows=train_rows,
                test_rows=test_rows,
                fold=fold,
                anchor_getter=_gfs_anchor,
                residual_cap_c=1.0,
                router_weight_cap=0.4,
            )
        )
        predictions.extend(
            _residual_expert_predictions(
                expert_id="E5_GEFS_ENSEMBLE",
                expert_scope="strict",
                train_rows=train_rows,
                test_rows=test_rows,
                fold=fold,
                anchor_getter=_gefs_anchor,
                residual_cap_c=1.0,
                router_weight_cap=0.4,
            )
        )
        predictions.extend(_shadow_and_placeholder_predictions(test_rows, fold))
    return predictions


def _shadow_and_placeholder_predictions(
    rows: Sequence[FeatureMatrixRow],
    fold: FoldSpec,
) -> list[ExpertPrediction]:
    shadow_feature_map: Mapping[str, tuple[str, ...]] = {
        "E6_IFS_OPER_SHADOW": ("ifsoper__center__tmax_c",),
        "E7_IFS_ENS_SHADOW": ("ifsens__center__tmax_c",),
        "E8_AI_NWP_SHADOW": (
            "aifsoper__center__tmax_c",
            "aifsens__center__tmax_c",
            "aigfssfc__center__tmax_c",
            "graphcast__center__tmax_c",
            "fourcastnet__center__tmax_c",
        ),
        "E9_CWA_WRF_LIVE_SHADOW": ("cwawrf15__center__tmax_c",),
        "E10_DIAGNOSTIC_PROXY": (),
        "E11_ARWF_LIVE_SHADOW": ("arwf__center__tmax_c",),
    }
    predictions: list[ExpertPrediction] = []
    for row in rows:
        for expert_id, feature_names in shadow_feature_map.items():
            anchor = _direct_shadow_anchor(row, feature_names)
            placeholder = anchor is None or expert_id == "E10_DIAGNOSTIC_PROXY"
            reason = "SOURCE_TABLE_ABSENT" if placeholder else None
            if expert_id == "E10_DIAGNOSTIC_PROXY":
                reason = "NOT_PROMOTED"
            if reason is not None and reason not in PLACEHOLDER_REASON_CODES:
                raise ValueError(f"Unsupported placeholder reason: {reason}")
            predictions.append(
                _prediction(
                    row,
                    expert_id,
                    fold,
                    expert_scope="live_shadow" if expert_id != "E10_DIAGNOSTIC_PROXY" else "proxy",
                    prediction_tmax_c=None if placeholder else anchor,
                    prediction_residual_c=None,
                    raw_anchor_tmax_c=anchor,
                    prediction_status="placeholder" if placeholder else "active",
                    placeholder_reason=reason,
                    router_weight_cap=0.0,
                )
            )
    return predictions
