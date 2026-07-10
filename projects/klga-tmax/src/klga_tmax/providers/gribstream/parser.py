from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
from typing import Any

from klga_tmax.ingestion.hash_keys import canonical_json, sha256_hex
from klga_tmax.providers.gribstream.models import (
    GribStreamChunk,
    GribStreamParsedValue,
    GribStreamRawResponse,
    ParsedGribStreamResponse,
    ResolvedSelector,
)
from klga_tmax.providers.gribstream.plan import as_of_utc, cutoff_utc, model_spec_by_id, target_date_for_valid_time
from klga_tmax.registry.station_universe import coordinate_tier


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _selector_by_alias(chunk: GribStreamChunk) -> dict[str, ResolvedSelector]:
    return {selector.alias: selector for selector in chunk.selectors}


def _coordinates_by_index(chunk: GribStreamChunk) -> dict[int, tuple[str, float, float]]:
    points = coordinate_tier(chunk.coordinate_tier)
    return {idx: (point.grid_point_id, float(point.lat), float(point.lon)) for idx, point in enumerate(points)}


def _coordinate_for_row(
    *,
    row: dict[str, Any],
    chunk: GribStreamChunk,
    coordinates: dict[int, tuple[str, float, float]],
) -> tuple[str, float, float]:
    point_index = _first_present(row, ("pointIndex", "point_index", "locationIndex", "location_index", "coordinateIndex", "coordinate_index"))
    if point_index is not None:
        try:
            return coordinates[int(point_index)]
        except (KeyError, TypeError, ValueError):
            pass
    lat = _float_or_none(_first_present(row, ("lat", "latitude")))
    lon = _float_or_none(_first_present(row, ("lon", "lng", "longitude")))
    if lat is not None and lon is not None:
        for grid_point_id, point_lat, point_lon in coordinates.values():
            if abs(point_lat - lat) < 0.0001 and abs(point_lon - lon) < 0.0001:
                return grid_point_id, point_lat, point_lon
    return coordinates[0]


def _value_fields(row: dict[str, Any], aliases: tuple[str, ...]) -> list[tuple[str, Any, str | None]]:
    values: list[tuple[str, Any, str | None]] = []
    nested_values = row.get("values")
    if isinstance(nested_values, dict):
        for alias in aliases:
            if alias in nested_values:
                values.append((alias, nested_values[alias], None))
    for alias in aliases:
        if alias in row:
            values.append((alias, row[alias], None))
        unit_key = f"{alias}_unit"
        if unit_key in row and alias in row:
            values[-1] = (alias, row[alias], str(row[unit_key]))
    return values


def _member_for_row(row: dict[str, Any]) -> str:
    raw_member = _first_present(row, ("member", "ensemble_member", "member_id"))
    if raw_member is None:
        return "deterministic"
    text = str(raw_member).strip()
    return text if text else "deterministic"


def _target_date_for_row(chunk: GribStreamChunk, forecasted_time: datetime) -> date:
    if chunk.model_id == "nbmqmd" or chunk.fetch_shape == "nbm_tmax_native":
        target_date = forecasted_time.date() - timedelta(days=1)
        if chunk.target_start_date <= target_date <= chunk.target_end_date:
            return target_date
        if chunk.target_start_date == chunk.target_end_date:
            return chunk.target_start_date
    if chunk.model_id == "rtma" and chunk.target_start_date == chunk.target_end_date:
        return chunk.target_start_date
    target_date = target_date_for_valid_time(forecasted_time)
    if target_date < chunk.target_start_date or target_date > chunk.target_end_date:
        if forecasted_time.hour == 0:
            target_date = target_date_for_valid_time(forecasted_time) - timedelta(days=1)
        if target_date < chunk.target_start_date or target_date > chunk.target_end_date:
            target_date = chunk.target_start_date
    return target_date


def _missing_response_gaps(
    chunk: GribStreamChunk,
    *,
    observed_keys: set[tuple[str, str, str, str]],
    row_count_raw: int,
) -> tuple[dict[str, Any], ...]:
    if not observed_keys:
        return (
            {
                "model_id": chunk.model_id,
                "target_start_date": chunk.target_start_date,
                "target_end_date": chunk.target_end_date,
                "cutoff_id": chunk.cutoff_id,
                "gap_type": "empty_response",
                "gap_reason": "HTTP 200 response contained no parsed GribStream values",
                "request_sha256": chunk.request_sha256,
                "valid_times_utc": [valid.isoformat() for valid in chunk.valid_times_utc],
                "selector_aliases": [selector.alias for selector in chunk.selectors],
                "members": list(chunk.members),
                "row_count_raw": row_count_raw,
            },
        )

    points = coordinate_tier(chunk.coordinate_tier)
    aliases = tuple(selector.alias for selector in chunk.selectors)
    observed_members = {member for _, _, member, _ in observed_keys}
    expected_members = tuple(str(member) for member in chunk.members) if chunk.members else tuple(sorted(observed_members))
    gaps: list[dict[str, Any]] = []
    for point in points:
        for alias in aliases:
            selector = next(item for item in chunk.selectors if item.alias == alias)
            for member in expected_members:
                missing_times = [
                    valid.isoformat()
                    for valid in chunk.valid_times_utc
                    if (point.grid_point_id, valid.isoformat(), member, alias) not in observed_keys
                ]
                if not missing_times:
                    continue
                gaps.append(
                    {
                        "model_id": chunk.model_id,
                        "target_start_date": chunk.target_start_date,
                        "target_end_date": chunk.target_end_date,
                        "cutoff_id": chunk.cutoff_id,
                        "grid_point_id": point.grid_point_id,
                        "variable_alias": alias,
                        "variable_name": selector.variable_name,
                        "member": member,
                        "gap_type": f"missing_{chunk.endpoint_type}_value",
                        "gap_reason": f"missing {len(missing_times)} expected valid time(s) for selector/member/coordinate",
                        "request_sha256": chunk.request_sha256,
                        "missing_valid_times_utc": missing_times,
                        "expected_valid_time_count": len(chunk.valid_times_utc),
                        "row_count_raw": row_count_raw,
                    }
                )
    return tuple(gaps)


def parse_gribstream_response(response: GribStreamRawResponse) -> ParsedGribStreamResponse:
    if not response.success:
        return ParsedGribStreamResponse(values=(), row_count_raw=0)
    chunk = response.chunk
    selectors = _selector_by_alias(chunk)
    coordinates = _coordinates_by_index(chunk)
    expected_pairs = {
        (
            run_time.astimezone(timezone.utc),
            valid_time.astimezone(timezone.utc),
        )
        for run_time, valid_time in chunk.expected_run_valid_pairs_utc
    }
    spec = model_spec_by_id(chunk.model_id)
    values: list[GribStreamParsedValue] = []
    observed_keys: set[tuple[str, str, str, str]] = set()
    row_count = 0
    raw_path = Path(response.raw_storage_uri)
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row_count += 1
            row = json.loads(stripped)
            if not isinstance(row, dict):
                continue
            forecasted_time = _parse_datetime(
                _first_present(row, ("forecasted_time", "valid_time", "validTime", "time", "timestamp"))
            )
            forecasted_at = _parse_datetime(
                _first_present(row, ("forecasted_at", "run_time", "runTime", "reference_time", "referenceTime", "init_time"))
            )
            if forecasted_time is None:
                continue
            if forecasted_at is None:
                fallback_target_date = _target_date_for_row(chunk, forecasted_time)
                forecasted_at = chunk.as_of_utc or cutoff_utc(fallback_target_date, cutoff_id=chunk.cutoff_id)
            if expected_pairs and (
                forecasted_at.astimezone(timezone.utc),
                forecasted_time.astimezone(timezone.utc),
            ) not in expected_pairs:
                continue
            target_date = _target_date_for_row(chunk, forecasted_time)
            row_as_of = as_of_utc(target_date, spec, cutoff_id=chunk.cutoff_id)
            grid_point_id, lat, lon = _coordinate_for_row(row=row, chunk=chunk, coordinates=coordinates)
            member = _member_for_row(row)
            index_updated_at = _parse_datetime(_first_present(row, ("index_updated_at", "indexUpdatedAt")))
            effective_available_at = row_as_of or response.retrieved_at_utc
            availability_method = "manual_override" if chunk.model_id == "urma" else "conservative_lag_rule"
            quality_note = "retrospective_only_not_pre_target_live_evidence" if chunk.model_id == "urma" else None
            row_cutoff = cutoff_utc(target_date, cutoff_id=chunk.cutoff_id)
            forecast_hour = (forecasted_time - forecasted_at).total_seconds() / 3600.0
            for alias, raw_value, unit in _value_fields(row, tuple(selectors.keys())):
                selector = selectors[alias]
                value = _float_or_none(raw_value)
                observed_keys.add((grid_point_id, forecasted_time.isoformat(), member, alias))
                hash_payload = {
                    "model_id": chunk.model_id,
                    "target_date": target_date.isoformat(),
                    "cutoff_id": chunk.cutoff_id,
                    "grid_point_id": grid_point_id,
                    "forecasted_at_utc": forecasted_at.isoformat(),
                    "forecasted_time_utc": forecasted_time.isoformat(),
                    "member": member,
                    "alias": alias,
                    "value": value,
                    "request_sha256": chunk.request_sha256,
                }
                values.append(
                    GribStreamParsedValue(
                        model_id=chunk.model_id,
                        endpoint_type=chunk.endpoint_type,
                        target_date=target_date,
                        cutoff_id=chunk.cutoff_id,
                        cutoff_utc=row_cutoff,
                        as_of_utc=row_as_of,
                        coordinate_tier=chunk.coordinate_tier,
                        grid_point_id=grid_point_id,
                        lat=lat,
                        lon=lon,
                        forecasted_at_utc=forecasted_at,
                        forecasted_time_utc=forecasted_time,
                        forecast_hour=forecast_hour,
                        member=member,
                        variable_alias=alias,
                        variable_name=selector.variable_name,
                        variable_level=selector.variable_level,
                        variable_info=selector.variable_info,
                        unit_original=unit or selector.unit_hint,
                        value_original=value,
                        unit_canonical=unit or selector.unit_hint,
                        value_canonical=value,
                        index_updated_at_utc=index_updated_at,
                        provider_available_at_utc=effective_available_at,
                        effective_available_at_utc=effective_available_at,
                        availability_method=availability_method,
                        raw_row_hash=sha256_hex(canonical_json(hash_payload)),
                        raw_row_json=row,
                        quality_note=quality_note,
                    )
                )
    return ParsedGribStreamResponse(
        values=tuple(values),
        row_count_raw=row_count,
        gaps=_missing_response_gaps(chunk, observed_keys=observed_keys, row_count_raw=row_count),
    )
