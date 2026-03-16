from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from math import sqrt
from typing import Iterable, Mapping, Sequence

from . import db
from .config import (
    EVALUATION_END_DATE,
    EVALUATION_START_DATE,
    FAMILY_WEIGHT_CAP,
    MIN_RMSE_FLOOR,
    MIN_TRAIN_DAYS,
    MODEL_WEIGHT_CAP,
    ROLLING_HALF_LIFE_DAYS,
    ROLLING_WINDOW_DAYS,
    STATION,
    isoformat_utc,
    utc_now,
)
from .model_catalog import eligible_specs_for_date

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12


@dataclass(frozen=True)
class ErrorRecord:
    settlement_date_local: date
    error_f: float


def _parse_date(value: object) -> date:
    return date.fromisoformat(str(value))


def _load_daily_model_tmax(
    connection,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, Mapping[str, object]]]:
    rows = connection.execute(
        """
        SELECT *
        FROM daily_model_tmax
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        ORDER BY settlement_date_local, model_code
        """,
        (
            STATION.station_id,
            start_date.isoformat(),
            end_date.isoformat(),
        ),
    ).fetchall()
    result: dict[date, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        result[_parse_date(row["settlement_date_local"])][str(row["model_code"])] = row
    return result


def _load_error_history(
    connection,
    start_date: date,
    end_date: date,
) -> dict[str, list[ErrorRecord]]:
    rows = connection.execute(
        """
        SELECT model_code, settlement_date_local, error_f
        FROM model_daily_errors
        WHERE station_id = ?
          AND settlement_date_local BETWEEN ? AND ?
        ORDER BY model_code, settlement_date_local
        """,
        (
            STATION.station_id,
            start_date.isoformat(),
            end_date.isoformat(),
        ),
    ).fetchall()
    result: dict[str, list[ErrorRecord]] = defaultdict(list)
    for row in rows:
        result[str(row["model_code"])].append(
            ErrorRecord(
                settlement_date_local=_parse_date(row["settlement_date_local"]),
                error_f=float(row["error_f"]),
            )
        )
    return result


def _training_records(
    history: Sequence[ErrorRecord],
    target_date: date,
) -> list[ErrorRecord]:
    window_start = target_date - timedelta(days=ROLLING_WINDOW_DAYS)
    return [
        record
        for record in history
        if window_start <= record.settlement_date_local < target_date
    ]


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        numerator += value * weight
        denominator += weight
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _ew_stats(
    records: Sequence[ErrorRecord],
    target_date: date,
) -> tuple[int, date | None, date | None, float | None, float | None, float | None]:
    if not records:
        return 0, None, None, None, None, None
    weighted_errors: list[tuple[float, float]] = []
    for record in records:
        age_days = (target_date - record.settlement_date_local).days
        decay_weight = 0.5 ** (age_days / ROLLING_HALF_LIFE_DAYS)
        weighted_errors.append((record.error_f, decay_weight))
    ew_bias_f = _weighted_mean(weighted_errors)
    if ew_bias_f is None:
        return 0, None, None, None, None, None
    weighted_abs = [
        (abs(record.error_f - ew_bias_f), 0.5 ** ((target_date - record.settlement_date_local).days / ROLLING_HALF_LIFE_DAYS))
        for record in records
    ]
    weighted_sq = [
        ((record.error_f - ew_bias_f) ** 2, 0.5 ** ((target_date - record.settlement_date_local).days / ROLLING_HALF_LIFE_DAYS))
        for record in records
    ]
    ew_mae_f = _weighted_mean(weighted_abs)
    ew_rmse_sq = _weighted_mean(weighted_sq)
    ew_rmse_f = sqrt(ew_rmse_sq) if ew_rmse_sq is not None else None
    return (
        len(records),
        min(record.settlement_date_local for record in records),
        max(record.settlement_date_local for record in records),
        ew_bias_f,
        ew_mae_f,
        ew_rmse_f,
    )


def _normalize_positive_weights(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {key: value for key, value in weights.items() if value > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {key: 0.0 for key in weights}
    return {key: positive.get(key, 0.0) / total for key in weights}


def _apply_model_cap(
    weights: Mapping[str, float],
    cap: float = MODEL_WEIGHT_CAP,
) -> tuple[dict[str, float], set[str]]:
    normalized = _normalize_positive_weights(weights)
    active_models = {model_code: weight for model_code, weight in normalized.items() if weight > 0.0}
    if not active_models or len(active_models) * cap + EPSILON < 1.0:
        return normalized, set()
    fixed: dict[str, float] = {}
    remaining = set(active_models)
    capped_models: set[str] = set()
    while remaining:
        remaining_mass = 1.0 - sum(fixed.values())
        if remaining_mass <= EPSILON:
            break
        base_total = sum(active_models[model_code] for model_code in remaining)
        if base_total <= EPSILON:
            break
        provisional = {
            model_code: active_models[model_code] / base_total * remaining_mass
            for model_code in remaining
        }
        over_limit = {
            model_code: weight
            for model_code, weight in provisional.items()
            if weight > cap + EPSILON
        }
        if not over_limit:
            fixed.update(provisional)
            break
        for model_code in over_limit:
            fixed[model_code] = cap
            capped_models.add(model_code)
            remaining.remove(model_code)
    final_weights = {model_code: fixed.get(model_code, 0.0) for model_code in normalized}
    total = sum(final_weights.values())
    if total > 0.0:
        final_weights = {model_code: weight / total for model_code, weight in final_weights.items()}
    return final_weights, capped_models


def _apply_family_cap(
    model_weights: Mapping[str, float],
    families: Mapping[str, str],
    cap: float = FAMILY_WEIGHT_CAP,
) -> tuple[dict[str, float], set[str]]:
    active_models = {model_code: weight for model_code, weight in model_weights.items() if weight > 0.0}
    if not active_models:
        return dict(model_weights), set()
    family_totals: dict[str, float] = defaultdict(float)
    for model_code, weight in active_models.items():
        family_totals[families[model_code]] += weight
    active_families = {family: weight for family, weight in family_totals.items() if weight > 0.0}
    if len(active_families) <= 1 or len(active_families) * cap + EPSILON < 1.0:
        return dict(model_weights), set()
    over_limit = {
        family: weight
        for family, weight in active_families.items()
        if weight > cap + EPSILON
    }
    if not over_limit:
        return dict(model_weights), set()
    adjusted_family_totals = dict(active_families)
    excess = 0.0
    capped_families: set[str] = set()
    for family, weight in over_limit.items():
        adjusted_family_totals[family] = cap
        excess += weight - cap
        capped_families.add(family)
    receivers = {
        family: weight
        for family, weight in adjusted_family_totals.items()
        if family not in capped_families and weight < cap - EPSILON
    }
    total_capacity = sum(cap - weight for weight in receivers.values())
    if total_capacity <= EPSILON:
        return dict(model_weights), set()
    for family, weight in receivers.items():
        capacity = cap - weight
        adjusted_family_totals[family] = weight + excess * (capacity / total_capacity)
    adjusted_weights = dict(model_weights)
    for family, original_total in active_families.items():
        adjusted_total = adjusted_family_totals.get(family, 0.0)
        scale = adjusted_total / original_total if original_total > EPSILON else 0.0
        for model_code, weight in model_weights.items():
            if families.get(model_code) == family and weight > 0.0:
                adjusted_weights[model_code] = weight * scale
    total = sum(weight for weight in adjusted_weights.values() if weight > 0.0)
    if total > 0.0:
        adjusted_weights = {
            model_code: (weight / total if weight > 0.0 else 0.0)
            for model_code, weight in adjusted_weights.items()
        }
    return adjusted_weights, capped_families


def compute_daily_model_weights(
    connection,
    *,
    start_date: date = EVALUATION_START_DATE,
    end_date: date = EVALUATION_END_DATE,
    include_live_only: bool = False,
) -> list[dict[str, object]]:
    tmax_by_date = _load_daily_model_tmax(connection, start_date, end_date)
    history_start = start_date - timedelta(days=ROLLING_WINDOW_DAYS)
    error_history = _load_error_history(connection, history_start, end_date - timedelta(days=1))
    created_at_utc = isoformat_utc(utc_now())
    rows_to_persist: list[dict[str, object]] = []
    for offset in range((end_date - start_date).days + 1):
        settlement_date_local = start_date + timedelta(days=offset)
        date_rows = tmax_by_date.get(settlement_date_local, {})
        provisional_rows: list[dict[str, object]] = []
        raw_weights: dict[str, float] = {}
        model_families: dict[str, str] = {}
        for spec in eligible_specs_for_date(
            settlement_date_local,
            include_live_only=include_live_only,
        ):
            tmax_row = date_rows.get(spec.model_code)
            selected_raw_tmax_f = (
                float(tmax_row["selected_raw_tmax_f"])
                if tmax_row is not None and tmax_row["selected_raw_tmax_f"] is not None
                else None
            )
            history_records = _training_records(error_history.get(spec.model_code, ()), settlement_date_local)
            (
                train_n_days,
                train_start_date,
                train_end_date,
                ew_bias_f,
                ew_mae_f,
                ew_rmse_f,
            ) = _ew_stats(history_records, settlement_date_local)
            included_in_blend = 1
            exclusion_reason: str | None = None
            bias_corrected_tmax_f = selected_raw_tmax_f
            raw_weight: float | None = None
            if selected_raw_tmax_f is None:
                included_in_blend = 0
                exclusion_reason = "no_forecast"
            elif train_n_days < MIN_TRAIN_DAYS:
                included_in_blend = 0
                exclusion_reason = "insufficient_history"
            else:
                bias_correction = ew_bias_f or 0.0
                bias_corrected_tmax_f = selected_raw_tmax_f - bias_correction
                raw_weight = 1.0 / max(float(ew_rmse_f or 0.0), MIN_RMSE_FLOOR) ** 2
                raw_weights[spec.model_code] = raw_weight
                model_families[spec.model_code] = spec.family
            provisional_rows.append(
                {
                    "station_id": STATION.station_id,
                    "settlement_date_local": settlement_date_local.isoformat(),
                    "model_code": spec.model_code,
                    "family": spec.family,
                    "train_start_date": train_start_date.isoformat() if train_start_date else None,
                    "train_end_date": train_end_date.isoformat() if train_end_date else None,
                    "train_n_days": train_n_days,
                    "ew_bias_f": ew_bias_f,
                    "ew_mae_f": ew_mae_f,
                    "ew_rmse_f": ew_rmse_f,
                    "bias_corrected_tmax_f": bias_corrected_tmax_f,
                    "raw_weight": raw_weight,
                    "model_cap_applied": 0,
                    "family_cap_applied": 0,
                    "final_weight": 0.0,
                    "included_in_blend": included_in_blend,
                    "exclusion_reason": exclusion_reason,
                    "created_at_utc": created_at_utc,
                }
            )
        model_capped_weights, capped_models = _apply_model_cap(raw_weights, MODEL_WEIGHT_CAP)
        family_capped_weights, capped_families = _apply_family_cap(
            model_capped_weights,
            model_families,
            FAMILY_WEIGHT_CAP,
        )
        for row in provisional_rows:
            model_code = str(row["model_code"])
            if int(row["included_in_blend"]) == 1:
                row["final_weight"] = float(family_capped_weights.get(model_code, 0.0))
                row["model_cap_applied"] = int(model_code in capped_models)
                row["family_cap_applied"] = int(row["family"] in capped_families)
            rows_to_persist.append(row)
        if (offset + 1) % 60 == 0 or settlement_date_local == end_date:
            LOGGER.info(
                "Computed daily_model_weights progress date=%s rows=%d",
                settlement_date_local,
                len(rows_to_persist),
            )
    db.delete_range_rows(
        connection,
        "daily_model_weights",
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    db.commit(connection)
    db.replace_daily_model_weights(connection, rows_to_persist)
    db.commit(connection)
    LOGGER.info(
        "Persisted daily_model_weights rows=%d range=%s..%s include_live_only=%s",
        len(rows_to_persist),
        start_date,
        end_date,
        include_live_only,
    )
    return rows_to_persist
