from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Mapping, Sequence

from . import db
from .config import (
    STATION,
    isoformat_utc,
    local_day_window_utc,
    parse_utc,
    settlement_asof_utc,
    utc_now,
)
from .model_catalog import ModelSpec, VariableSpec, eligible_specs_for_date

LOGGER = logging.getLogger(__name__)


def _normalized_text(value: object | None) -> str:
    return str(value or "").strip()


def _variable_matches(row: Mapping[str, object], variable: VariableSpec | None) -> bool:
    if variable is None:
        return False
    return (
        _normalized_text(row.get("variable_name")) == variable.name
        and _normalized_text(row.get("variable_level")) == variable.level
        and _normalized_text(row.get("variable_info")) == variable.info
    )


def _latest_revision_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    latest_by_key: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in rows:
        key = (
            _normalized_text(row.get("forecasted_time_utc")),
            _normalized_text(row.get("variable_name")),
            _normalized_text(row.get("variable_level")),
            _normalized_text(row.get("variable_info")),
            -1 if row.get("member") is None else int(row["member"]),
        )
        existing = latest_by_key.get(key)
        if existing is None:
            latest_by_key[key] = row
            continue
        current_forecasted_at = parse_utc(_normalized_text(row.get("forecasted_at_utc")))
        existing_forecasted_at = parse_utc(_normalized_text(existing.get("forecasted_at_utc")))
        current_id = int(row.get("id") or 0)
        existing_id = int(existing.get("id") or 0)
        if (current_forecasted_at, current_id) >= (existing_forecasted_at, existing_id):
            latest_by_key[key] = row
    return sorted(
        latest_by_key.values(),
        key=lambda row: (
            _normalized_text(row.get("forecasted_time_utc")),
            _normalized_text(row.get("forecasted_at_utc")),
            _normalized_text(row.get("variable_name")),
            _normalized_text(row.get("variable_level")),
            _normalized_text(row.get("variable_info")),
            -1 if row.get("member") is None else int(row["member"]),
        ),
    )


def _daily_max_f(rows: Sequence[Mapping[str, object]]) -> float | None:
    values = [float(row["value_f"]) for row in rows if row.get("value_f") is not None]
    if not values:
        return None
    return max(values)


def _snapshot_points(rows: Sequence[Mapping[str, object]]) -> list[tuple[datetime, float]]:
    by_time: dict[datetime, float] = {}
    for row in rows:
        forecasted_time = parse_utc(_normalized_text(row.get("forecasted_time_utc")))
        by_time[forecasted_time] = float(row["value_f"])
    return sorted(by_time.items(), key=lambda item: item[0])


def _median_snapshot_cadence_hours(points: Sequence[tuple[datetime, float]]) -> float | None:
    if len(points) < 2:
        return None
    deltas = [
        (points[index][0] - points[index - 1][0]).total_seconds() / 3600.0
        for index in range(1, len(points))
        if points[index][0] > points[index - 1][0]
    ]
    if not deltas:
        return None
    return float(statistics.median(deltas))


def _interpolated_value(
    left: tuple[datetime, float],
    right: tuple[datetime, float],
    target: datetime,
) -> float:
    if target <= left[0]:
        return left[1]
    if target >= right[0]:
        return right[1]
    span_seconds = (right[0] - left[0]).total_seconds()
    if span_seconds <= 0.0:
        return left[1]
    ratio = (target - left[0]).total_seconds() / span_seconds
    return left[1] + ratio * (right[1] - left[1])


def _interpolated_snapshot_tmax(
    rows: Sequence[Mapping[str, object]],
    local_day_start_utc: datetime,
    local_day_end_utc: datetime,
) -> tuple[float | None, float | None]:
    points = _snapshot_points(rows)
    median_cadence_hours = _median_snapshot_cadence_hours(points)
    if median_cadence_hours is None or median_cadence_hours <= 3.0 or len(points) < 2:
        return None, median_cadence_hours
    hourly_times: list[datetime] = []
    current = local_day_start_utc
    while current < local_day_end_utc:
        hourly_times.append(current)
        current += timedelta(hours=1)
    interpolated_values: list[float] = []
    point_index = 0
    for target_time in hourly_times:
        if target_time < points[0][0] or target_time > points[-1][0]:
            continue
        while point_index + 1 < len(points) and points[point_index + 1][0] < target_time:
            point_index += 1
        left = points[point_index]
        if left[0] == target_time:
            interpolated_values.append(left[1])
            continue
        if point_index + 1 >= len(points):
            continue
        right = points[point_index + 1]
        if target_time > right[0]:
            continue
        interpolated_values.append(_interpolated_value(left, right, target_time))
    if not interpolated_values:
        return None, median_cadence_hours
    return max(interpolated_values), median_cadence_hours


def select_raw_tmax(
    *,
    native_tmax_f: float | None,
    interpolated_tmax_f: float | None,
    snapshot_tmax_f: float | None,
) -> tuple[float | None, str]:
    if native_tmax_f is not None:
        return native_tmax_f, "native_tmax"
    if interpolated_tmax_f is not None:
        return interpolated_tmax_f, "interpolated_snapshot_tmax"
    if snapshot_tmax_f is not None:
        return snapshot_tmax_f, "snapshot_tmax"
    return None, "unavailable"


def _truth_window_by_date(
    connection,
    start_date: date,
    end_date: date,
) -> dict[date, Mapping[str, object]]:
    truth_rows = db.load_truth_rows(
        connection,
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    return {
        date.fromisoformat(_normalized_text(row["settlement_date_local"])): row
        for row in truth_rows
    }


def _raw_rows_by_model_for_date(
    connection,
    settlement_date_local: date,
) -> dict[str, list[Mapping[str, object]]]:
    rows = connection.execute(
        """
        SELECT *
        FROM gribstream_raw_forecasts
        WHERE station_id = ?
          AND settlement_date_local = ?
        ORDER BY model_code, forecasted_time_utc, forecasted_at_utc, variable_name,
                 variable_level, IFNULL(variable_info, ''), IFNULL(member, -1), id
        """,
        (STATION.station_id, settlement_date_local.isoformat()),
    ).fetchall()
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_normalized_text(row["model_code"])].append(row)
    return grouped


def _derive_row(
    spec: ModelSpec,
    settlement_date_local: date,
    local_day_start_utc: datetime,
    local_day_end_utc: datetime,
    raw_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    as_of_utc = isoformat_utc(settlement_asof_utc(settlement_date_local))
    latest_rows = _latest_revision_rows(raw_rows)
    snapshot_rows = [row for row in latest_rows if _variable_matches(row, spec.snapshot_var)]
    native_rows = [
        row
        for row in latest_rows
        if _variable_matches(row, spec.native_tmax_for_date(settlement_date_local))
    ]
    native_tmax_f = _daily_max_f(native_rows)
    snapshot_tmax_f = _daily_max_f(snapshot_rows)
    interpolated_tmax_f, median_cadence_hours = _interpolated_snapshot_tmax(
        snapshot_rows,
        local_day_start_utc,
        local_day_end_utc,
    )
    selected_raw_tmax_f, selected_method = select_raw_tmax(
        native_tmax_f=native_tmax_f,
        interpolated_tmax_f=interpolated_tmax_f,
        snapshot_tmax_f=snapshot_tmax_f,
    )
    notes_parts = [
        f"raw_rows={len(raw_rows)}",
        f"latest_rows={len(latest_rows)}",
        f"snapshot_rows={len(snapshot_rows)}",
        f"native_rows={len(native_rows)}",
    ]
    if median_cadence_hours is not None:
        notes_parts.append(f"snapshot_median_cadence_hours={median_cadence_hours:.2f}")
    if selected_method == "unavailable":
        notes_parts.append("reason=no_matching_forecast_rows")
    return {
        "station_id": STATION.station_id,
        "settlement_date_local": settlement_date_local.isoformat(),
        "model_code": spec.model_code,
        "family": spec.family,
        "as_of_utc": as_of_utc,
        "local_day_start_utc": isoformat_utc(local_day_start_utc),
        "local_day_end_utc": isoformat_utc(local_day_end_utc),
        "native_tmax_f": native_tmax_f,
        "snapshot_tmax_f": snapshot_tmax_f,
        "interpolated_tmax_f": interpolated_tmax_f,
        "selected_raw_tmax_f": selected_raw_tmax_f,
        "selected_method": selected_method,
        "snapshot_row_count": len(snapshot_rows),
        "native_row_count": len(native_rows),
        "model_available": int(selected_raw_tmax_f is not None),
        "notes": "; ".join(notes_parts),
        "created_at_utc": isoformat_utc(utc_now()),
    }


def derive_daily_model_tmax(
    connection,
    *,
    start_date: date,
    end_date: date,
    include_live_only: bool = False,
    require_truth: bool = True,
) -> list[dict[str, object]]:
    truth_windows = _truth_window_by_date(connection, start_date, end_date)
    rows_to_persist: list[dict[str, object]] = []
    for settlement_date_local in (
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ):
        truth_row = truth_windows.get(settlement_date_local)
        if truth_row is not None:
            local_day_start_utc = parse_utc(_normalized_text(truth_row["local_day_start_utc"]))
            local_day_end_utc = parse_utc(_normalized_text(truth_row["local_day_end_utc"]))
        else:
            if require_truth:
                raise ValueError(
                    f"Missing truth row for {settlement_date_local}. Run fetch-truth first."
                )
            local_day_start_utc, local_day_end_utc = local_day_window_utc(
                settlement_date_local,
                STATION.timezone_name,
            )
        raw_rows_by_model = _raw_rows_by_model_for_date(connection, settlement_date_local)
        for spec in eligible_specs_for_date(
            settlement_date_local,
            include_live_only=include_live_only,
        ):
            rows_to_persist.append(
                _derive_row(
                    spec,
                    settlement_date_local,
                    local_day_start_utc,
                    local_day_end_utc,
                    raw_rows_by_model.get(spec.model_code, ()),
                )
            )
        if len(rows_to_persist) % 500 == 0:
            LOGGER.info(
                "Derived daily_model_tmax progress date=%s rows=%d",
                settlement_date_local,
                len(rows_to_persist),
            )
    db.delete_range_rows(
        connection,
        "daily_model_tmax",
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    db.commit(connection)
    db.replace_daily_model_tmax(connection, rows_to_persist)
    db.commit(connection)
    LOGGER.info(
        "Derived daily_model_tmax rows=%d range=%s..%s include_live_only=%s",
        len(rows_to_persist),
        start_date,
        end_date,
        include_live_only,
    )
    return rows_to_persist


def derive_model_daily_errors(
    connection,
    *,
    start_date: date,
    end_date: date,
) -> list[dict[str, object]]:
    joined_rows = connection.execute(
        """
        SELECT d.station_id,
               d.settlement_date_local,
               d.model_code,
               d.selected_raw_tmax_f,
               t.actual_tmax_f
        FROM daily_model_tmax d
        JOIN nws_daily_settlements t
          ON t.station_id = d.station_id
         AND t.settlement_date_local = d.settlement_date_local
        WHERE d.station_id = ?
          AND d.settlement_date_local BETWEEN ? AND ?
          AND d.selected_raw_tmax_f IS NOT NULL
        ORDER BY d.settlement_date_local, d.model_code
        """,
        (
            STATION.station_id,
            start_date.isoformat(),
            end_date.isoformat(),
        ),
    ).fetchall()
    rows_to_persist: list[dict[str, object]] = []
    created_at_utc = isoformat_utc(utc_now())
    for row in joined_rows:
        selected_raw_tmax_f = float(row["selected_raw_tmax_f"])
        actual_tmax_f = float(row["actual_tmax_f"])
        error_f = selected_raw_tmax_f - actual_tmax_f
        rows_to_persist.append(
            {
                "station_id": STATION.station_id,
                "settlement_date_local": _normalized_text(row["settlement_date_local"]),
                "model_code": _normalized_text(row["model_code"]),
                "selected_raw_tmax_f": selected_raw_tmax_f,
                "actual_tmax_f": actual_tmax_f,
                "error_f": error_f,
                "abs_error_f": abs(error_f),
                "squared_error_f": error_f * error_f,
                "created_at_utc": created_at_utc,
            }
        )
    db.delete_range_rows(
        connection,
        "model_daily_errors",
        STATION.station_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    db.commit(connection)
    db.replace_model_daily_errors(connection, rows_to_persist)
    db.commit(connection)
    LOGGER.info(
        "Derived model_daily_errors rows=%d range=%s..%s",
        len(rows_to_persist),
        start_date,
        end_date,
    )
    return rows_to_persist


def derive_daily_products(
    connection,
    *,
    start_date: date,
    end_date: date,
    include_live_only: bool = False,
    require_truth: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tmax_rows = derive_daily_model_tmax(
        connection,
        start_date=start_date,
        end_date=end_date,
        include_live_only=include_live_only,
        require_truth=require_truth,
    )
    error_rows = derive_model_daily_errors(
        connection,
        start_date=start_date,
        end_date=end_date,
    ) if require_truth else []
    return tmax_rows, error_rows
