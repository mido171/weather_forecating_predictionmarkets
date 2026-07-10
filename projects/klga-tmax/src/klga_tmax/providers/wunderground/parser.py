from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from klga_tmax.constants import TARGET_TZ
from klga_tmax.registry.cutoffs import target_local_day_window_utc
from klga_tmax.providers.wunderground.models import (
    ParsedWundergroundResponse,
    WundergroundDailyActual,
    WundergroundIntradayObservation,
    WundergroundRawDayResponse,
)

DAILY_LABEL_METHOD_COMPUTED = "hourly_temp_max"


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _bounded_float_value(value: Any, minimum: float, maximum: float) -> float | None:
    parsed = _float_value(value)
    if parsed is None or not (minimum <= parsed <= maximum):
        return None
    return parsed


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _sum(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean)


def _max(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return max(clean)


def _min(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return min(clean)


def _dominant_wind_direction(values: Iterable[float | None]) -> float | None:
    clean = [float(value) % 360.0 for value in values if value is not None]
    if not clean:
        return None
    sin_sum = sum(math.sin(math.radians(value)) for value in clean)
    cos_sum = sum(math.cos(math.radians(value)) for value in clean)
    if sin_sum == 0.0 and cos_sum == 0.0:
        return None
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0


def _quality_for_daily(
    *,
    daily_high_f: int | None,
    daily_low_f: int | None,
    observation_count: int,
    duplicate_timestamp_count: int,
) -> tuple[str, str | None, str, dict[str, Any]]:
    notes: list[str] = []
    note_json: dict[str, Any] = {
        "daily_high_source": DAILY_LABEL_METHOD_COMPUTED,
        "observation_count": observation_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
    }
    if daily_high_f is None:
        notes.append("missing_daily_high")
    elif not (-30 <= daily_high_f <= 120):
        notes.append("daily_high_out_of_range")
    if daily_low_f is not None and not (-40 <= daily_low_f <= 110):
        notes.append("daily_low_out_of_range")
    if daily_high_f is not None and daily_low_f is not None and daily_high_f < daily_low_f:
        notes.append("daily_high_below_daily_low")
    if observation_count == 0:
        notes.append("no_hourly_temperature_observations")
    elif observation_count < 18:
        notes.append("incomplete_hourly_temperature_coverage")
    if duplicate_timestamp_count:
        notes.append("duplicate_observation_timestamps_removed")
    status = "accepted" if not notes else "suspect"
    quality = "ok" if not notes else "suspect"
    note_json["notes"] = notes
    return quality, ("; ".join(notes) if notes else None), status, note_json


def _quality_for_intraday(row: dict[str, Any]) -> tuple[str, str | None]:
    notes: list[str] = []
    temp = _float_value(row.get("temp"))
    dewpoint = _float_value(row.get("dewPt"))
    humidity = _float_value(row.get("rh"))
    wind_speed = _float_value(row.get("wspd"))
    precip = _float_value(row.get("precip_hrly"))
    if temp is not None and not (-40 <= temp <= 130):
        notes.append("temp_out_of_range")
    if dewpoint is not None and not (-80 <= dewpoint <= 90):
        notes.append("dewpoint_out_of_range")
    if humidity is not None and not (0 <= humidity <= 100):
        notes.append("humidity_out_of_range")
    if wind_speed is not None and not (0 <= wind_speed <= 150):
        notes.append("wind_speed_out_of_range")
    if precip is not None and not (0 <= precip <= 20):
        notes.append("precipitation_out_of_range")
    return ("suspect", "; ".join(notes)) if notes else ("ok", None)


def _candidate_high(row: dict[str, Any]) -> tuple[int | None, str | None]:
    temp = _int_value(row.get("temp"))
    if temp is not None and -30 <= temp <= 120:
        return temp, DAILY_LABEL_METHOD_COMPUTED
    return None, None


def _candidate_low(row: dict[str, Any]) -> int | None:
    temp = _int_value(row.get("temp"))
    if temp is not None and -40 <= temp <= 110:
        return temp
    return None


def _hourly_observation_evidence(
    *,
    row: dict[str, Any],
    observed_local: datetime,
    observed_utc: datetime,
) -> dict[str, Any]:
    return {
        "observation_time_local": observed_local.isoformat(),
        "observation_time_utc": observed_utc.isoformat(),
        "temp_f": _bounded_float_value(row.get("temp"), -40, 130),
        "dewpoint_f": _bounded_float_value(row.get("dewPt"), -80, 90),
        "humidity_pct": _bounded_float_value(row.get("rh"), 0, 100),
        "wind_speed_mph": _bounded_float_value(row.get("wspd"), 0, 150),
        "wind_gust_mph": _float_value(row.get("gust")),
        "wind_direction_deg": _float_value(row.get("wdir")),
        "pressure_in": _float_value(row.get("pressure")),
        "precipitation_in": _bounded_float_value(row.get("precip_hrly"), 0, 20),
        "condition_text": row.get("wx_phrase") or row.get("terse_phrase"),
    }


def _provider_temp_diagnostics(
    *,
    row: dict[str, Any],
    observed_local: datetime,
    observed_utc: datetime,
    field_name: str,
) -> dict[str, Any] | None:
    if row.get(field_name) is None:
        return None
    return {
        "observation_time_local": observed_local.isoformat(),
        "observation_time_utc": observed_utc.isoformat(),
        field_name: _float_value(row.get(field_name)),
        "actual_temp_f": _bounded_float_value(row.get("temp"), -40, 130),
    }


def parse_wunderground_response(
    response: WundergroundRawDayResponse,
    *,
    canonical_station_id: str,
    timezone_name: str = TARGET_TZ,
    intraday_lag_minutes: int = 90,
) -> ParsedWundergroundResponse:
    payload = response.payload_json or {}
    observations_raw = payload.get("observations")
    if not isinstance(observations_raw, list):
        return ParsedWundergroundResponse(daily_actuals=(), intraday_observations=())

    target_zone = ZoneInfo(timezone_name)
    intraday_rows: list[WundergroundIntradayObservation] = []
    grouped_raw_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    grouped_evidence_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    grouped_provider_max_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    grouped_provider_min_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    grouped_high_times: dict[Any, list[str]] = defaultdict(list)
    duplicate_counts: dict[Any, int] = defaultdict(int)
    seen_observation_times: set[datetime] = set()

    for item in observations_raw:
        if not isinstance(item, dict):
            continue
        valid_time_gmt = item.get("valid_time_gmt")
        if valid_time_gmt is None:
            continue
        try:
            observed_utc = datetime.fromtimestamp(int(valid_time_gmt), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            continue
        observed_local = observed_utc.astimezone(target_zone)
        local_date = observed_local.date()
        if local_date < response.start_local_date or local_date > response.end_local_date:
            continue
        if observed_utc in seen_observation_times:
            duplicate_counts[local_date] += 1
            continue
        seen_observation_times.add(observed_utc)
        provider_available = observed_utc + timedelta(minutes=intraday_lag_minutes)
        quality_flag, quality_note = _quality_for_intraday(item)
        intraday_rows.append(
            WundergroundIntradayObservation(
                station_id=canonical_station_id,
                wunderground_station_id=response.wunderground_station_id,
                weathercom_location_id=response.weathercom_location_id,
                observation_time_local=observed_local,
                observation_time_utc=observed_utc,
                local_date=local_date,
                timezone_name=timezone_name,
                temp_f=_bounded_float_value(item.get("temp"), -40, 130),
                dewpoint_f=_bounded_float_value(item.get("dewPt"), -80, 90),
                humidity_pct=_bounded_float_value(item.get("rh"), 0, 100),
                wind_speed_mph=_bounded_float_value(item.get("wspd"), 0, 150),
                wind_gust_mph=_float_value(item.get("gust")),
                wind_direction_deg=_float_value(item.get("wdir")),
                pressure_in=_float_value(item.get("pressure")),
                precipitation_in=_bounded_float_value(item.get("precip_hrly"), 0, 20),
                condition_text=item.get("wx_phrase") or item.get("terse_phrase"),
                cloud_cover_text=item.get("clds"),
                uv_index=_float_value(item.get("uv_index")),
                solar_radiation=_float_value(item.get("solar_radiation")),
                raw_observation_json=item,
                provider_available_at_utc=provider_available,
                quality_flag=quality_flag,
                quality_note=quality_note,
            )
        )
        grouped_raw_rows[local_date].append(item)
        grouped_evidence_rows[local_date].append(
            _hourly_observation_evidence(
                row=item,
                observed_local=observed_local,
                observed_utc=observed_utc,
            )
        )
        max_diag = _provider_temp_diagnostics(
            row=item,
            observed_local=observed_local,
            observed_utc=observed_utc,
            field_name="max_temp",
        )
        if max_diag is not None:
            grouped_provider_max_rows[local_date].append(max_diag)
        min_diag = _provider_temp_diagnostics(
            row=item,
            observed_local=observed_local,
            observed_utc=observed_utc,
            field_name="min_temp",
        )
        if min_diag is not None:
            grouped_provider_min_rows[local_date].append(min_diag)

    daily_rows: list[WundergroundDailyActual] = []
    for local_date in sorted(grouped_raw_rows):
        day_rows = grouped_raw_rows[local_date]
        high_candidates = [_candidate_high(row) for row in day_rows]
        high_candidates = [candidate for candidate in high_candidates if candidate[0] is not None]
        if high_candidates:
            daily_high_f, high_source_field = max(high_candidates, key=lambda candidate: candidate[0] or -999)
        else:
            daily_high_f, high_source_field = None, None
        daily_low_float = _min(_candidate_low(row) for row in day_rows)
        daily_low_f = int(round(daily_low_float)) if daily_low_float is not None else None
        if daily_high_f is not None:
            for evidence in grouped_evidence_rows[local_date]:
                temp = evidence.get("temp_f")
                if temp is not None and int(round(float(temp))) == daily_high_f:
                    grouped_high_times[local_date].append(str(evidence["observation_time_local"]))
        daily_avg_temp_f = _mean(_bounded_float_value(row.get("temp"), -40, 130) for row in day_rows)
        daily_high_dewpoint_f = _max(_bounded_float_value(row.get("dewPt"), -80, 90) for row in day_rows)
        daily_low_dewpoint_f = _min(_bounded_float_value(row.get("dewPt"), -80, 90) for row in day_rows)
        local_day_start_utc, local_day_end_utc = target_local_day_window_utc(local_date)
        provider_available = local_day_end_utc + timedelta(hours=24)
        quality_flag, quality_note, validation_status, validation_notes_json = _quality_for_daily(
            daily_high_f=daily_high_f,
            daily_low_f=daily_low_f,
            observation_count=len(day_rows),
            duplicate_timestamp_count=duplicate_counts.get(local_date, 0),
        )
        daily_rows.append(
            WundergroundDailyActual(
                station_id=canonical_station_id,
                wunderground_station_id=response.wunderground_station_id,
                weathercom_location_id=response.weathercom_location_id,
                local_date=local_date,
                timezone_name=timezone_name,
                local_day_start_utc=local_day_start_utc,
                local_day_end_utc=local_day_end_utc,
                daily_high_f=daily_high_f,
                settlement_high_f_whole=daily_high_f,
                daily_low_f=daily_low_f,
                daily_avg_temp_f=daily_avg_temp_f,
                daily_high_dewpoint_f=daily_high_dewpoint_f,
                daily_low_dewpoint_f=daily_low_dewpoint_f,
                daily_precipitation_in=_sum(
                    _bounded_float_value(row.get("precip_hrly"), 0, 20) for row in day_rows
                ),
                daily_max_wind_speed_mph=_max(
                    _bounded_float_value(row.get("wspd"), 0, 150) for row in day_rows
                ),
                daily_max_wind_gust_mph=_max(_float_value(row.get("gust")) for row in day_rows),
                daily_avg_wind_speed_mph=_mean(
                    _bounded_float_value(row.get("wspd"), 0, 150) for row in day_rows
                ),
                daily_dominant_wind_direction_deg=_dominant_wind_direction(
                    _float_value(row.get("wdir")) for row in day_rows
                ),
                label_method=DAILY_LABEL_METHOD_COMPUTED if daily_high_f is not None else None,
                daily_high_source_field=high_source_field,
                provider_available_at_utc=provider_available,
                source_daily_summary_json={},
                raw_daily_json={
                    "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    "observation_count": len(day_rows),
                    "daily_high_source_field": high_source_field,
                    "daily_high_rule": "max bounded hourly temp field only; provider max_temp ignored",
                    "provider_max_temp_values_json": grouped_provider_max_rows[local_date],
                    "provider_min_temp_values_json": grouped_provider_min_rows[local_date],
                },
                high_observation_times_local_json=grouped_high_times[local_date],
                hourly_observations_json=grouped_evidence_rows[local_date],
                provider_max_temp_values_json=grouped_provider_max_rows[local_date],
                provider_min_temp_values_json=grouped_provider_min_rows[local_date],
                validation_status=validation_status,
                validation_notes_json=validation_notes_json,
                observations_count=len(day_rows),
                quality_flag=quality_flag,
                quality_note=quality_note,
            )
        )

    return ParsedWundergroundResponse(
        daily_actuals=tuple(daily_rows),
        intraday_observations=tuple(sorted(intraday_rows, key=lambda row: row.observation_time_utc)),
    )
