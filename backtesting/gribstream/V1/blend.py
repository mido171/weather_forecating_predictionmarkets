from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping

from . import db
from .config import MIN_TRAIN_DAYS, STATION, isoformat_utc, settlement_asof_utc, utc_now

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionRunResult:
    prediction_rows: list[dict[str, object]]
    component_rows: list[dict[str, object]]
    previews_by_date: dict[date, dict[str, object]]


def _parse_date(value: object) -> date:
    return date.fromisoformat(str(value))


def _load_weights_by_date(
    connection,
    start_date: date,
    end_date: date,
) -> dict[date, list[Mapping[str, object]]]:
    rows = connection.execute(
        """
        SELECT w.*,
               t.selected_raw_tmax_f
        FROM daily_model_weights w
        LEFT JOIN daily_model_tmax t
          ON t.station_id = w.station_id
         AND t.settlement_date_local = w.settlement_date_local
         AND t.model_code = w.model_code
        WHERE w.station_id = ?
          AND w.settlement_date_local BETWEEN ? AND ?
        ORDER BY w.settlement_date_local, w.model_code
        """,
        (
            STATION.station_id,
            start_date.isoformat(),
            end_date.isoformat(),
        ),
    ).fetchall()
    grouped: dict[date, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_parse_date(row["settlement_date_local"]), []).append(row)
    return grouped


def _load_truth_by_date(
    connection,
    start_date: date,
    end_date: date,
) -> dict[date, Mapping[str, object]]:
    rows = db.load_truth_rows(
        connection,
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    return {_parse_date(row["settlement_date_local"]): row for row in rows}


def _eligible_rows(rows: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row["selected_raw_tmax_f"] is not None
        and row["bias_corrected_tmax_f"] is not None
        and int(row["train_n_days"]) >= MIN_TRAIN_DAYS
    ]


def _normalized_raw_weight_rows(rows: list[Mapping[str, object]]) -> list[tuple[Mapping[str, object], float]]:
    positive = [
        (row, float(row["raw_weight"]))
        for row in rows
        if row["raw_weight"] is not None and float(row["raw_weight"]) > 0.0
    ]
    total = sum(weight for _, weight in positive)
    if total <= 0.0:
        return []
    return [(row, weight / total) for row, weight in positive]


def _equal_weight_blend(rows: list[Mapping[str, object]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["bias_corrected_tmax_f"]) for row in rows) / float(len(rows))


def _inverse_rmse_blend(rows: list[Mapping[str, object]]) -> float | None:
    normalized = _normalized_raw_weight_rows(rows)
    if not normalized:
        return None
    return sum(float(row["bias_corrected_tmax_f"]) * weight for row, weight in normalized)


def _family_capped_blend(rows: list[Mapping[str, object]]) -> float | None:
    active = [
        row
        for row in rows
        if row["final_weight"] is not None and float(row["final_weight"]) > 0.0
    ]
    if not active:
        return None
    return sum(float(row["bias_corrected_tmax_f"]) * float(row["final_weight"]) for row in active)


def _single_model_prediction(
    rows: list[Mapping[str, object]],
    model_code: str,
) -> float | None:
    for row in rows:
        if str(row["model_code"]) == model_code and row["bias_corrected_tmax_f"] is not None:
            return float(row["bias_corrected_tmax_f"])
    return None


def _best_single_model(
    rows: list[Mapping[str, object]],
    actual_tmax_f: float | None,
) -> tuple[str | None, float | None]:
    if actual_tmax_f is None:
        return None, None
    candidates = [
        (str(row["model_code"]), float(row["bias_corrected_tmax_f"]))
        for row in rows
        if row["bias_corrected_tmax_f"] is not None
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: (abs(item[1] - actual_tmax_f), item[0]))


def compute_daily_predictions(
    connection,
    *,
    start_date: date,
    end_date: date,
    require_truth: bool = True,
) -> PredictionRunResult:
    weights_by_date = _load_weights_by_date(connection, start_date, end_date)
    truth_by_date = _load_truth_by_date(connection, start_date, end_date)
    prediction_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    previews_by_date: dict[date, dict[str, object]] = {}
    created_at_utc = isoformat_utc(utc_now())
    db.delete_range_rows(
        connection,
        "daily_prediction_components",
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    db.delete_range_rows(
        connection,
        "daily_predictions",
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    db.commit(connection)
    for offset in range((end_date - start_date).days + 1):
        settlement_date_local = start_date + timedelta(days=offset)
        date_rows = list(weights_by_date.get(settlement_date_local, ()))
        if not date_rows:
            continue
        actual_row = truth_by_date.get(settlement_date_local)
        if require_truth and actual_row is None:
            raise ValueError(f"Missing truth row for {settlement_date_local}")
        actual_tmax_f = float(actual_row["actual_tmax_f"]) if actual_row is not None else None
        eligible = _eligible_rows(date_rows)
        family_capped_blend_f = _family_capped_blend(eligible)
        preview_row = {
            "station_id": STATION.station_id,
            "settlement_date_local": settlement_date_local.isoformat(),
            "as_of_utc": isoformat_utc(settlement_asof_utc(settlement_date_local)),
            "actual_tmax_f": actual_tmax_f,
            "equal_weight_blend_f": _equal_weight_blend(eligible),
            "inverse_rmse_blend_f": _inverse_rmse_blend(eligible),
            "family_capped_blend_f": family_capped_blend_f,
            "nbm_only_f": _single_model_prediction(date_rows, "nbm"),
            "hrrr_only_f": _single_model_prediction(date_rows, "hrrr"),
            "rap_only_f": _single_model_prediction(date_rows, "rap"),
            "gfs_only_f": _single_model_prediction(date_rows, "gfs"),
            "best_single_model_code": None,
            "best_single_model_pred_f": None,
            "family_capped_error_f": (
                family_capped_blend_f - actual_tmax_f
                if family_capped_blend_f is not None and actual_tmax_f is not None
                else None
            ),
            "family_capped_abs_error_f": None,
            "created_at_utc": created_at_utc,
        }
        best_model_code, best_model_pred_f = _best_single_model(date_rows, actual_tmax_f)
        preview_row["best_single_model_code"] = best_model_code
        preview_row["best_single_model_pred_f"] = best_model_pred_f
        preview_row["family_capped_abs_error_f"] = (
            abs(float(preview_row["family_capped_error_f"]))
            if preview_row["family_capped_error_f"] is not None
            else None
        )
        previews_by_date[settlement_date_local] = dict(preview_row)
        if actual_tmax_f is not None:
            prediction_rows.append(preview_row)
        for row in date_rows:
            bias_corrected_tmax_f = (
                float(row["bias_corrected_tmax_f"])
                if row["bias_corrected_tmax_f"] is not None
                else None
            )
            final_weight = float(row["final_weight"]) if row["final_weight"] is not None else 0.0
            component_rows.append(
                {
                    "station_id": STATION.station_id,
                    "settlement_date_local": settlement_date_local.isoformat(),
                    "model_code": str(row["model_code"]),
                    "family": str(row["family"]),
                    "selected_raw_tmax_f": (
                        float(row["selected_raw_tmax_f"])
                        if row["selected_raw_tmax_f"] is not None
                        else None
                    ),
                    "bias_corrected_tmax_f": bias_corrected_tmax_f,
                    "final_weight": final_weight,
                    "weighted_contribution_f": (
                        bias_corrected_tmax_f * final_weight
                        if bias_corrected_tmax_f is not None
                        else None
                    ),
                    "created_at_utc": created_at_utc,
                }
            )
    db.replace_daily_prediction_components(connection, component_rows)
    db.replace_daily_predictions(connection, prediction_rows)
    db.commit(connection)
    LOGGER.info(
        "Persisted daily predictions rows=%d components=%d range=%s..%s",
        len(prediction_rows),
        len(component_rows),
        start_date,
        end_date,
    )
    return PredictionRunResult(
        prediction_rows=prediction_rows,
        component_rows=component_rows,
        previews_by_date=previews_by_date,
    )
