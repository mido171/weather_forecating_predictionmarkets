from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.constants import (
    FEATURE_SET_NAME,
    FEATURE_VERSION,
    FORMULA_CONTRACT_HASH,
    STALE_SOURCE_MAX_AGE_HOURS,
)
from klga_tmax.db.migrations_check import inspect_contract
from klga_tmax.db.normalize_acquisition import normalize_acquisition
from klga_tmax.features.aliases import resolve_feature_alias
from klga_tmax.features.context import MaterializationContext
from klga_tmax.features.leakage import (
    FeatureSourceTrace,
    assert_daily_high_label_history_safe,
    validate_feature_trace_for_cutoff,
)
from klga_tmax.registry.materialize_targets import materialize_target_instances
from klga_tmax.registry.seed import seed_all
from klga_tmax.utils.git import current_git_sha


@dataclass(frozen=True)
class FeatureRow:
    family: str
    name: str
    value: float | None
    available: bool
    unit: str | None = None
    max_available_at_utc: datetime | None = None
    latest_valid_time_utc: datetime | None = None
    latest_run_time_utc: datetime | None = None
    source_trace: dict[str, Any] | None = None


def materialize_features(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
    feature_version: str = FEATURE_VERSION,
    replace: bool = False,
) -> dict[str, int]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    inspection = inspect_contract(connection)
    if not inspection.ok:
        raise RuntimeError("; ".join(inspection.failures))

    seed_all(connection)
    _ensure_feature_version(connection, feature_version=feature_version)
    normalized = normalize_acquisition(
        connection,
        start_date=start_date - timedelta(days=45),
        end_date=end_date,
        cutoff_id=cutoff_id,
        observation_start_date=start_date,
        mos_start_date=start_date,
    )
    target_rows = materialize_target_instances(
        connection,
        start_date=start_date,
        end_date=end_date,
        replace=False,
    )
    feature_version_id = _feature_version_id(connection, feature_version=feature_version)
    targets = _load_targets(connection, start_date=start_date, end_date=end_date, cutoff_id=cutoff_id)
    if replace and targets:
        for target in targets:
            connection.execute(
                text(
                    """
                    DELETE FROM gold.feature_values
                    WHERE target_instance_id = :target_instance_id
                      AND feature_build_version = :feature_version
                    """
                ),
                {
                    "target_instance_id": target["target_instance_id"],
                    "feature_version": feature_version,
                },
            )

    feature_count = 0
    matrix_count = 0
    late_feature_count = 0
    for target in targets:
        context = MaterializationContext(
            target_date=target["target_date"],
            cutoff_id=target["cutoff_id"],
            cutoff_utc=target["cutoff_utc"],
            local_day_start_utc=target["local_day_start_utc"],
            local_day_end_utc=target["local_day_end_utc"],
            feature_version=feature_version,
            mode="materialize",
        )
        rows = build_features_for_target(connection, context=context)
        feature_count += _upsert_feature_values(
            connection,
            target_instance_id=target["target_instance_id"],
            context=context,
            features=rows,
        )
        matrix_count += _upsert_feature_matrix(
            connection,
            target=target,
            feature_version_id=feature_version_id,
            feature_version=feature_version,
        )
        late_feature_count += _late_feature_count(
            connection,
            target_instance_id=target["target_instance_id"],
            cutoff_utc=context.cutoff_utc,
            feature_version=feature_version,
        )
    if late_feature_count:
        raise RuntimeError(f"feature materialization produced {late_feature_count} late features")

    return {
        **{f"normalized.{key}": value for key, value in normalized.items()},
        "gold.target_instances_inserted": target_rows,
        "gold.feature_values_upserted": feature_count,
        "gold.feature_matrix_upserted": matrix_count,
        "late_features": late_feature_count,
    }


def build_features_for_target(
    connection: Connection,
    *,
    context: MaterializationContext,
) -> list[FeatureRow]:
    features: list[FeatureRow] = []
    features.extend(_calendar_features(context))
    features.extend(_wunderground_history_features(connection, context=context))
    features.extend(_station_actual_features(connection, context=context))
    features.extend(_observation_features(connection, context=context))
    features.extend(_mos_features(connection, context=context))
    features.extend(_grib_features(connection, context=context))
    features.extend(_risk_and_regime_features(features, context=context))
    return [_canonical_feature(feature) for feature in features]


def _load_targets(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    cutoff_id: str | None,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        text(
            """
            SELECT
                target_instance_id,
                target_date,
                cutoff_id,
                cutoff_utc,
                local_day_start_utc,
                local_day_end_utc,
                settlement_high_f_whole,
                label_available,
                label_revision_sensitive
            FROM gold.target_instances
            WHERE target_date BETWEEN :start_date AND :end_date
              AND (CAST(:cutoff_id AS text) IS NULL OR cutoff_id = CAST(:cutoff_id AS text))
            ORDER BY target_date, cutoff_id
            """
        ),
        {"start_date": start_date, "end_date": end_date, "cutoff_id": cutoff_id},
    ).mappings().all()
    return [dict(row) for row in rows]


def _ensure_feature_version(connection: Connection, *, feature_version: str) -> None:
    connection.execute(
        text(
            """
            INSERT INTO registry.feature_versions (
                feature_set_name,
                feature_version,
                source_code_git_sha,
                formula_contract_hash,
                feature_names
            )
            VALUES (
                :feature_set_name,
                :feature_version,
                :source_code_git_sha,
                :formula_contract_hash,
                :feature_names
            )
            ON CONFLICT (feature_set_name, feature_version) DO NOTHING
            """
        ),
        {
            "feature_set_name": FEATURE_SET_NAME,
            "feature_version": feature_version,
            "source_code_git_sha": current_git_sha(),
            "formula_contract_hash": FORMULA_CONTRACT_HASH,
            "feature_names": [],
        },
    )


def _feature_version_id(connection: Connection, *, feature_version: str) -> str:
    row = connection.execute(
        text(
            """
            SELECT feature_version_id
            FROM registry.feature_versions
            WHERE feature_set_name = :feature_set_name
              AND feature_version = :feature_version
            """
        ),
        {"feature_set_name": FEATURE_SET_NAME, "feature_version": feature_version},
    ).mappings().one()
    return str(row["feature_version_id"])


def _calendar_features(context: MaterializationContext) -> list[FeatureRow]:
    doy = context.target_date.timetuple().tm_yday
    radians = 2.0 * math.pi * doy / 366.0
    return [
        FeatureRow("calendar", "calendar_month", float(context.target_date.month), True),
        FeatureRow("calendar", "calendar_day_of_year", float(doy), True),
        FeatureRow("calendar", "calendar_sin_day_of_year", math.sin(radians), True),
        FeatureRow("calendar", "calendar_cos_day_of_year", math.cos(radians), True),
    ]


def _wunderground_history_features(
    connection: Connection,
    *,
    context: MaterializationContext,
) -> list[FeatureRow]:
    max_label_date = context.target_date - timedelta(days=2)
    assert_daily_high_label_history_safe(
        target_date=context.target_date,
        label_dates_used=[max_label_date],
        feature_name="wu_history_features",
    )
    rows = connection.execute(
        text(
            """
            SELECT
                local_date AS target_date,
                tmax_f AS high_temp_f,
                settlement_available_at_utc AS source_available_at_utc,
                station_id || ':' || local_date::text AS source_record_id
            FROM public.wunderground_daily_tmax
            WHERE station_id = 'KLGA'
              AND validation_status IN ('accepted','manual_confirmed')
              AND tmax_f IS NOT NULL
              AND local_date <= :max_label_date
              AND local_date >= :max_label_date - interval '45 days'
              AND settlement_available_at_utc <= :cutoff_utc
            ORDER BY local_date DESC
            """
        ),
        {"max_label_date": max_label_date, "cutoff_utc": context.cutoff_utc},
    ).mappings().all()
    values = [float(row["high_temp_f"]) for row in rows if row["high_temp_f"] is not None]
    traces = [
        FeatureSourceTrace(
            "wunderground",
            str(row["source_record_id"] or row["target_date"]),
            row["source_available_at_utc"],
        )
        for row in rows
        if row["source_available_at_utc"] is not None
    ]
    validate_feature_trace_for_cutoff(cutoff_utc=context.cutoff_utc, source_trace=traces)
    latest_available = max((trace.effective_available_at_utc for trace in traces), default=None)
    trace_json = {
        "source_table": "public.wunderground_daily_tmax",
        "max_label_date": max_label_date.isoformat(),
        "rows": len(rows),
    }
    features = [
        FeatureRow(
            "wunderground_history",
            "wu_history_tmax_count_45d",
            float(len(values)),
            True,
            "count",
            latest_available,
            source_trace=trace_json,
        )
    ]
    for window in (3, 7, 14, 30):
        window_values = values[:window]
        features.append(
            FeatureRow(
                "wunderground_history",
                f"wu_history_tmax_mean_{window}d_f",
                _mean(window_values),
                bool(window_values),
                "F",
                latest_available,
                source_trace=trace_json,
            )
        )
    if values:
        features.extend(
            [
                FeatureRow(
                    "wunderground_history",
                    "wu_history_tmax_lag_2d_f",
                    values[0],
                    True,
                    "F",
                    latest_available,
                    source_trace=trace_json,
                ),
                FeatureRow(
                    "wunderground_history",
                    "wu_history_tmax_std_14d_f",
                    _std(values[:14]),
                    len(values[:14]) >= 2,
                    "F",
                    latest_available,
                    source_trace=trace_json,
                ),
            ]
        )
    clim = connection.execute(
        text(
            """
            SELECT
                avg(tmax_f)::double precision AS mean_f,
                stddev_samp(tmax_f)::double precision AS std_f,
                count(*)::integer AS count_rows,
                max(settlement_available_at_utc) AS max_available
            FROM public.wunderground_daily_tmax
            WHERE station_id = 'KLGA'
              AND validation_status IN ('accepted','manual_confirmed')
              AND tmax_f IS NOT NULL
              AND local_date < :target_date
              AND settlement_available_at_utc <= :cutoff_utc
              AND abs(EXTRACT(DOY FROM local_date)::int - :doy) <= 15
            """
        ),
        {
            "target_date": context.target_date,
            "cutoff_utc": context.cutoff_utc,
            "doy": context.target_date.timetuple().tm_yday,
        },
    ).mappings().one()
    features.extend(
        [
            FeatureRow(
                "climatology",
                "climatology_wu_tmax_mean_31d_f",
                clim["mean_f"],
                clim["mean_f"] is not None,
                "F",
                clim["max_available"],
                source_trace={"source_table": "public.wunderground_daily_tmax", "doy_window": 31},
            ),
            FeatureRow(
                "climatology",
                "climatology_wu_tmax_std_31d_f",
                clim["std_f"],
                clim["std_f"] is not None,
                "F",
                clim["max_available"],
                source_trace={"source_table": "public.wunderground_daily_tmax", "doy_window": 31},
            ),
            FeatureRow(
                "climatology",
                "climatology_wu_tmax_count_31d",
                float(clim["count_rows"]),
                True,
                "count",
                clim["max_available"],
                source_trace={"source_table": "public.wunderground_daily_tmax", "doy_window": 31},
            ),
        ]
    )
    return features


def _station_actual_features(
    connection: Connection,
    *,
    context: MaterializationContext,
) -> list[FeatureRow]:
    max_label_date = context.target_date - timedelta(days=2)
    row = connection.execute(
        text(
            """
            SELECT
                avg(tmax_f)::double precision AS nearby_mean_f,
                max(tmax_f)::double precision AS nearby_max_f,
                min(tmax_f)::double precision AS nearby_min_f,
                count(*)::integer AS count_rows,
                max(settlement_available_at_utc) AS max_available
            FROM public.wunderground_daily_tmax
            WHERE local_date = :max_label_date
              AND station_id <> 'KLGA'
              AND validation_status IN ('accepted','manual_confirmed')
              AND tmax_f IS NOT NULL
              AND settlement_available_at_utc <= :cutoff_utc
            """
        ),
        {"max_label_date": max_label_date, "cutoff_utc": context.cutoff_utc},
    ).mappings().one()
    trace = {"source_table": "public.wunderground_daily_tmax", "label_date": max_label_date.isoformat()}
    return [
        FeatureRow("station_daily_actuals", "station_actuals_nearby_tmax_mean_lag2_f", row["nearby_mean_f"], row["nearby_mean_f"] is not None, "F", row["max_available"], source_trace=trace),
        FeatureRow("station_daily_actuals", "station_actuals_nearby_tmax_max_lag2_f", row["nearby_max_f"], row["nearby_max_f"] is not None, "F", row["max_available"], source_trace=trace),
        FeatureRow("station_daily_actuals", "station_actuals_nearby_tmax_min_lag2_f", row["nearby_min_f"], row["nearby_min_f"] is not None, "F", row["max_available"], source_trace=trace),
        FeatureRow("station_daily_actuals", "station_actuals_nearby_count_lag2", float(row["count_rows"]), True, "count", row["max_available"], source_trace=trace),
    ]


def _observation_features(
    connection: Connection,
    *,
    context: MaterializationContext,
) -> list[FeatureRow]:
    rows = connection.execute(
        text(
            """
            SELECT
                d.station_id,
                (obs->>'observation_time_utc')::timestamptz AS observation_time_utc,
                NULLIF(obs->>'temp_f', 'null')::double precision AS temp_f,
                NULLIF(obs->>'dewpoint_f', 'null')::double precision AS dewpoint_f,
                NULLIF(obs->>'wind_speed_mph', 'null')::double precision AS wind_speed_mph,
                NULLIF(obs->>'wind_direction_deg', 'null')::double precision AS wind_direction_deg,
                d.settlement_available_at_utc AS effective_available_at_utc
            FROM public.wunderground_daily_tmax d
            CROSS JOIN LATERAL jsonb_array_elements(d.hourly_observations_json) obs
            WHERE d.validation_status IN ('accepted','manual_confirmed')
              AND d.settlement_available_at_utc <= :cutoff_utc
              AND (obs->>'observation_time_utc')::timestamptz <= :cutoff_utc
              AND (obs->>'observation_time_utc')::timestamptz >= :cutoff_utc - interval '6 hours'
              AND d.station_id IN ('KLGA','KNYC','KJFK','KEWR','KTEB')
            ORDER BY observation_time_utc DESC
            """
        ),
        {"cutoff_utc": context.cutoff_utc},
    ).mappings().all()
    latest_by_station: dict[str, Any] = {}
    for row in rows:
        latest_by_station.setdefault(row["station_id"], row)
    klga = latest_by_station.get("KLGA")
    temps = [float(row["temp_f"]) for row in latest_by_station.values() if row["temp_f"] is not None]
    latest_available = max((row["effective_available_at_utc"] for row in latest_by_station.values()), default=None)
    trace = {"source_table": "public.wunderground_daily_tmax.hourly_observations_json", "lookback_hours": 6}
    return [
        FeatureRow("station_observations", "obs_klga_latest_temp_f", _row_float(klga, "temp_f"), klga is not None and klga["temp_f"] is not None, "F", latest_available, _row_dt(klga, "observation_time_utc"), source_trace=trace),
        FeatureRow("station_observations", "obs_klga_latest_dewpoint_f", _row_float(klga, "dewpoint_f"), klga is not None and klga["dewpoint_f"] is not None, "F", latest_available, _row_dt(klga, "observation_time_utc"), source_trace=trace),
        FeatureRow("station_observations", "obs_nearby_latest_temp_mean_f", _mean(temps), bool(temps), "F", latest_available, source_trace=trace),
        FeatureRow("station_observations", "obs_nearby_latest_temp_count", float(len(temps)), True, "count", latest_available, source_trace=trace),
    ]


def _mos_features(connection: Connection, *, context: MaterializationContext) -> list[FeatureRow]:
    rows = connection.execute(
        text(
            """
            SELECT feature_vector_json, feature_trace_json, source_feature_count
            FROM gold.iem_mos_feature_matrix_v1
            WHERE target_date = :target_date
              AND cutoff_id = :cutoff_id
            """
        ),
        {"target_date": context.target_date, "cutoff_id": context.cutoff_id},
    ).mappings().first()
    max_available = connection.execute(
        text(
            """
            SELECT max(max_source_available_at_utc) AS max_available
            FROM gold.iem_mos_daily_features
            WHERE target_date = :target_date
              AND cutoff_id = :cutoff_id
              AND max_source_available_at_utc <= :cutoff_utc
            """
        ),
        {
            "target_date": context.target_date,
            "cutoff_id": context.cutoff_id,
            "cutoff_utc": context.cutoff_utc,
        },
    ).mappings().one()["max_available"]
    features: list[FeatureRow] = []
    vector = dict(rows["feature_vector_json"] or {}) if rows else {}
    for name, value in sorted(vector.items()):
        numeric = _coerce_float(value)
        features.append(
            FeatureRow(
                "mos_guidance",
                resolve_feature_alias(name),
                numeric,
                numeric is not None,
                "F" if name.endswith("_f") else None,
                max_available,
                source_trace={"source_table": "gold.iem_mos_feature_matrix_v1", "source_feature_count": rows["source_feature_count"] if rows else 0},
            )
        )
    tmax_values = [
        _coerce_float(value)
        for name, value in vector.items()
        if name.endswith("_tmax_f") and _coerce_float(value) is not None
    ]
    tmax_values = [value for value in tmax_values if value is not None]
    features.extend(
        [
            FeatureRow("mos_guidance", "mos_guidance_tmax_mean_f", _mean(tmax_values), bool(tmax_values), "F", max_available, source_trace={"source_table": "gold.iem_mos_feature_matrix_v1"}),
            FeatureRow("mos_guidance", "mos_guidance_tmax_std_f", _std(tmax_values), len(tmax_values) >= 2, "F", max_available, source_trace={"source_table": "gold.iem_mos_feature_matrix_v1"}),
            FeatureRow("mos_guidance", "mos_guidance_tmax_count", float(len(tmax_values)), True, "count", max_available, source_trace={"source_table": "gold.iem_mos_feature_matrix_v1"}),
        ]
    )
    return features


def _grib_features(connection: Connection, *, context: MaterializationContext) -> list[FeatureRow]:
    rows = connection.execute(
        text(
            """
            SELECT
                feature_family,
                feature_name,
                feature_value,
                feature_unit,
                feature_available,
                source_latest_valid_time_utc,
                source_latest_run_time_utc,
                source_age_hours,
                source_latency_minutes,
                max_source_available_at_utc,
                source_trace_json
            FROM gold.feature_values
            WHERE target_instance_id = (
                SELECT target_instance_id
                FROM gold.target_instances
                WHERE target_date = :target_date
                  AND cutoff_id = :cutoff_id
            )
              AND feature_build_version <> :feature_version
              AND (max_source_available_at_utc IS NULL OR max_source_available_at_utc <= :cutoff_utc)
            """
        ),
        {
            "target_date": context.target_date,
            "cutoff_id": context.cutoff_id,
            "feature_version": context.feature_version,
            "cutoff_utc": context.cutoff_utc,
        },
    ).mappings().all()
    features = [
        FeatureRow(
            row["feature_family"] or "gribstream",
            resolve_feature_alias(row["feature_name"]),
            _coerce_float(row["feature_value"]),
            bool(row["feature_available"]),
            row["feature_unit"],
            row["max_source_available_at_utc"],
            row["source_latest_valid_time_utc"],
            row["source_latest_run_time_utc"],
            row["source_trace_json"],
        )
        for row in rows
    ]
    tmax_values = [
        feature.value
        for feature in features
        if feature.value is not None
        and ("tmax" in feature.name or "peak_window_max_tmp" in feature.name or "tmp_peak_window_max" in feature.name)
    ]
    latest_available = max((feature.max_available_at_utc for feature in features if feature.max_available_at_utc), default=None)
    features.extend(
        [
            FeatureRow("gribstream", "gribstream_tmax_mean_f", _mean(tmax_values), bool(tmax_values), "F", latest_available, source_trace={"source_table": "gold.feature_values"}),
            FeatureRow("gribstream", "gribstream_tmax_std_f", _std(tmax_values), len(tmax_values) >= 2, "F", latest_available, source_trace={"source_table": "gold.feature_values"}),
            FeatureRow("gribstream", "gribstream_tmax_count", float(len(tmax_values)), True, "count", latest_available, source_trace={"source_table": "gold.feature_values"}),
        ]
    )
    return features


def _risk_and_regime_features(
    features: list[FeatureRow],
    *,
    context: MaterializationContext,
) -> list[FeatureRow]:
    vector = {feature.name: feature.value for feature in features}
    mos_mean = vector.get("mos_guidance_tmax_mean_f")
    grib_mean = vector.get("gribstream_tmax_mean_f")
    clim_mean = vector.get("climatology_wu_tmax_mean_31d_f")
    obs_temp = vector.get("obs_klga_latest_temp_f")
    disagreement_values = [value for value in (mos_mean, grib_mean, clim_mean) if value is not None]
    disagreement = _std(disagreement_values) if len(disagreement_values) >= 2 else None
    spread = _max_present(vector.get("mos_guidance_tmax_std_f"), vector.get("gribstream_tmax_std_f"))
    heat = _bool_float(mos_mean is not None and mos_mean >= 90.0)
    cool = _bool_float(mos_mean is not None and mos_mean <= 55.0)
    stale_flags = []
    nowish = context.cutoff_utc
    for family in ("wunderground", "mos_guidance", "gribstream"):
        max_age = STALE_SOURCE_MAX_AGE_HOURS.get(family, 24.0)
        latest = max(
            (feature.max_available_at_utc for feature in features if feature.family.startswith(family) and feature.max_available_at_utc),
            default=None,
        )
        stale_flags.append(latest is None or ((nowish - latest).total_seconds() / 3600.0) > max_age)
    sea_breeze = _score_any(
        [
            vector.get("obs_klga_latest_temp_f") is not None and obs_temp is not None and obs_temp >= 75.0,
            vector.get("obs_klga_latest_dewpoint_f") is not None and (vector.get("obs_klga_latest_dewpoint_f") or 0) >= 60.0,
            disagreement is not None and disagreement >= 3.0,
        ]
    )
    backdoor = _score_any([cool == 1.0, disagreement is not None and disagreement >= 4.0])
    cloud_bust = _score_any([spread is not None and spread >= 4.0])
    storm = _score_any([vector.get("mos_guidance_tmax_std_f") is not None and (vector.get("mos_guidance_tmax_std_f") or 0) >= 5.0])
    return [
        FeatureRow("model_disagreement", "model_disagreement_tmax_std_f", disagreement, disagreement is not None, "F"),
        FeatureRow("model_disagreement", "ensemble_spread_proxy_f", spread, spread is not None, "F"),
        FeatureRow("regime", "regime_warm_flag", heat, True),
        FeatureRow("regime", "regime_cool_flag", cool, True),
        FeatureRow("regime", "regime_high_model_disagreement_flag", _bool_float(disagreement is not None and disagreement >= 4.0), True),
        FeatureRow("regime", "regime_high_ensemble_spread_flag", _bool_float(spread is not None and spread >= 5.0), True),
        FeatureRow("staleness", "stale_critical_source_count", float(sum(stale_flags)), True, "count"),
        FeatureRow("risk", "risk_sea_breeze_final_score", sea_breeze, True),
        FeatureRow("risk", "risk_sea_breeze_input_count", 3.0, True, "count"),
        FeatureRow("risk", "risk_backdoor_front_final_score", backdoor, True),
        FeatureRow("risk", "risk_backdoor_front_input_count", 2.0, True, "count"),
        FeatureRow("risk", "risk_marine_layer_final_score", sea_breeze, True),
        FeatureRow("risk", "risk_marine_layer_input_count", 2.0, True, "count"),
        FeatureRow("risk", "risk_cloud_bust_final_score", cloud_bust, True),
        FeatureRow("risk", "risk_cloud_bust_input_count", 1.0, True, "count"),
        FeatureRow("risk", "risk_storm_outflow_final_score", storm, True),
        FeatureRow("risk", "risk_storm_outflow_input_count", 1.0, True, "count"),
    ]


def _upsert_feature_values(
    connection: Connection,
    *,
    target_instance_id: str,
    context: MaterializationContext,
    features: list[FeatureRow],
) -> int:
    rows = []
    for feature in features:
        if feature.max_available_at_utc is not None:
            validate_feature_trace_for_cutoff(
                cutoff_utc=context.cutoff_utc,
                source_trace=[
                    FeatureSourceTrace(
                        feature.family,
                        feature.name,
                        feature.max_available_at_utc,
                    )
                ],
            )
        age_hours = None
        if feature.max_available_at_utc is not None:
            age_hours = (context.cutoff_utc - feature.max_available_at_utc).total_seconds() / 3600.0
        rows.append(
            {
                "target_instance_id": target_instance_id,
                "feature_family": feature.family,
                "feature_name": feature.name,
                "feature_value": feature.value,
                "feature_unit": feature.unit,
                "feature_available": feature.available and feature.value is not None,
                "source_latest_valid_time_utc": feature.latest_valid_time_utc,
                "source_latest_run_time_utc": feature.latest_run_time_utc,
                "source_age_hours": age_hours,
                "source_latency_minutes": None,
                "feature_build_version": context.feature_version,
                "max_source_available_at_utc": feature.max_available_at_utc,
                "source_trace_json": json.dumps(feature.source_trace or {}, sort_keys=True, default=str),
            }
        )
    if not rows:
        return 0
    connection.execute(
        text(
            """
            INSERT INTO gold.feature_values (
                target_instance_id,
                feature_family,
                feature_name,
                feature_value,
                feature_unit,
                feature_available,
                source_latest_valid_time_utc,
                source_latest_run_time_utc,
                source_age_hours,
                source_latency_minutes,
                feature_build_version,
                max_source_available_at_utc,
                source_trace_json
            )
            VALUES (
                :target_instance_id,
                :feature_family,
                :feature_name,
                :feature_value,
                :feature_unit,
                :feature_available,
                :source_latest_valid_time_utc,
                :source_latest_run_time_utc,
                :source_age_hours,
                :source_latency_minutes,
                :feature_build_version,
                :max_source_available_at_utc,
                CAST(:source_trace_json AS jsonb)
            )
            ON CONFLICT (target_instance_id, feature_name, feature_build_version)
            DO UPDATE SET
                feature_family = EXCLUDED.feature_family,
                feature_value = EXCLUDED.feature_value,
                feature_unit = EXCLUDED.feature_unit,
                feature_available = EXCLUDED.feature_available,
                source_latest_valid_time_utc = EXCLUDED.source_latest_valid_time_utc,
                source_latest_run_time_utc = EXCLUDED.source_latest_run_time_utc,
                source_age_hours = EXCLUDED.source_age_hours,
                source_latency_minutes = EXCLUDED.source_latency_minutes,
                max_source_available_at_utc = EXCLUDED.max_source_available_at_utc,
                source_trace_json = EXCLUDED.source_trace_json
            """
        ),
        rows,
    )
    return len(rows)


def _upsert_feature_matrix(
    connection: Connection,
    *,
    target: dict[str, Any],
    feature_version_id: str,
    feature_version: str,
) -> int:
    rows = connection.execute(
        text(
            """
            SELECT feature_name, feature_value, feature_available, source_trace_json
            FROM gold.feature_values
            WHERE target_instance_id = :target_instance_id
              AND feature_build_version = :feature_version
            ORDER BY feature_name
            """
        ),
        {"target_instance_id": target["target_instance_id"], "feature_version": feature_version},
    ).mappings().all()
    vector = {row["feature_name"]: row["feature_value"] for row in rows}
    availability = {row["feature_name"]: bool(row["feature_available"]) for row in rows}
    result = connection.execute(
        text(
            """
            INSERT INTO gold.feature_matrix (
                target_instance_id,
                feature_version_id,
                feature_vector_json,
                feature_availability_json,
                label_high_temp_f,
                label_available,
                label_revision_sensitive
            )
            VALUES (
                :target_instance_id,
                :feature_version_id,
                CAST(:feature_vector_json AS jsonb),
                CAST(:feature_availability_json AS jsonb),
                :label_high_temp_f,
                :label_available,
                :label_revision_sensitive
            )
            ON CONFLICT (target_instance_id, feature_version_id)
            DO UPDATE SET
                feature_vector_json = EXCLUDED.feature_vector_json,
                feature_availability_json = EXCLUDED.feature_availability_json,
                label_high_temp_f = EXCLUDED.label_high_temp_f,
                label_available = EXCLUDED.label_available,
                label_revision_sensitive = EXCLUDED.label_revision_sensitive
            """
        ),
        {
            "target_instance_id": target["target_instance_id"],
            "feature_version_id": feature_version_id,
            "feature_vector_json": json.dumps(vector, sort_keys=True, default=str),
            "feature_availability_json": json.dumps(availability, sort_keys=True, default=str),
            "label_high_temp_f": target["settlement_high_f_whole"] if target["label_available"] else None,
            "label_available": bool(target["label_available"]),
            "label_revision_sensitive": bool(target["label_revision_sensitive"]),
        },
    )
    _update_feature_version_names(connection, sorted(vector), feature_version=feature_version)
    return result.rowcount or 0


def _update_feature_version_names(
    connection: Connection,
    feature_names: list[str],
    *,
    feature_version: str,
) -> None:
    connection.execute(
        text(
            """
            UPDATE registry.feature_versions
            SET feature_names = (
                SELECT ARRAY(
                    SELECT DISTINCT unnest(feature_names || CAST(:feature_names AS text[]))
                    ORDER BY 1
                )
            )
            WHERE feature_set_name = :feature_set_name
              AND feature_version = :feature_version
            """
        ),
        {
            "feature_set_name": FEATURE_SET_NAME,
            "feature_version": feature_version,
            "feature_names": feature_names,
        },
    )


def _late_feature_count(
    connection: Connection,
    *,
    target_instance_id: str,
    cutoff_utc: datetime,
    feature_version: str,
) -> int:
    return int(
        connection.execute(
            text(
                """
                SELECT count(*)
                FROM gold.feature_values
                WHERE target_instance_id = :target_instance_id
                  AND feature_build_version = :feature_version
                  AND max_source_available_at_utc > :cutoff_utc
                """
            ),
            {
                "target_instance_id": target_instance_id,
                "feature_version": feature_version,
                "cutoff_utc": cutoff_utc,
            },
        ).scalar_one()
    )


def _canonical_feature(feature: FeatureRow) -> FeatureRow:
    return FeatureRow(
        family=feature.family,
        name=resolve_feature_alias(feature.name).lower(),
        value=feature.value,
        available=feature.available,
        unit=feature.unit,
        max_available_at_utc=_as_utc(feature.max_available_at_utc),
        latest_valid_time_utc=_as_utc(feature.latest_valid_time_utc),
        latest_run_time_utc=_as_utc(feature.latest_run_time_utc),
        source_trace=feature.source_trace,
    )


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _row_float(row: Any, key: str) -> float | None:
    if row is None or row[key] is None:
        return None
    return float(row[key])


def _row_dt(row: Any, key: str) -> datetime | None:
    if row is None:
        return None
    return row[key]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _std(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))


def _max_present(*values: float | None) -> float | None:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _bool_float(value: bool) -> float:
    return 1.0 if value else 0.0


def _score_any(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)
