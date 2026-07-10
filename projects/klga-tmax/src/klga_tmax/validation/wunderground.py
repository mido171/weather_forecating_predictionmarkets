from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.db.migrations_check import ContractInspection, inspect_contract
from klga_tmax.registry.station_universe import MANDATORY_STATION_REGISTRY


CONTRACT_START_DATE = date(1973, 1, 1)


def validate_wunderground(connection: Connection) -> ContractInspection:
    result = inspect_contract(connection)

    station_rows = connection.execute(
        text(
            """
            SELECT count(*)
            FROM registry.station_registry
            WHERE role <> 'gridded_pseudo_point'
              AND wunderground_station_id IS NOT NULL
            """
        )
    ).scalar_one()
    expected_station_rows = len([row for row in MANDATORY_STATION_REGISTRY if row.wunderground_station_id])
    if int(station_rows) != expected_station_rows:
        result.failures.append(
            f"Wunderground fetchable station count expected {expected_station_rows}; observed {station_rows}"
        )

    bad_source = connection.execute(
        text(
            """
            SELECT count(*)
            FROM public.wunderground_daily_tmax
            WHERE daily_high_source <> 'hourly_temp_max'
            """
        )
    ).scalar_one()
    if int(bad_source):
        result.failures.append(f"{bad_source} WU truth rows do not use hourly_temp_max")

    bad_tmax = connection.execute(
        text(
            """
            SELECT count(*)
            FROM public.wunderground_daily_tmax
            WHERE validation_status IN ('accepted','manual_confirmed')
              AND (
                    tmax_f IS NULL
                 OR tmin_f IS NULL
                 OR tmax_f < tmin_f
                 OR tmax_f < -30
                 OR tmax_f > 120
                 OR tmin_f < -40
                 OR tmin_f > 110
              )
            """
        )
    ).scalar_one()
    if int(bad_tmax):
        result.failures.append(f"{bad_tmax} accepted WU truth rows violate Tmax/Tmin bounds")

    mismatched_hourly_max = connection.execute(
        text(
            """
            WITH hourly AS (
                SELECT
                    station_id,
                    local_date,
                    max(round((item->>'temp_f')::numeric))::integer AS hourly_max_f,
                    min(round((item->>'temp_f')::numeric))::integer AS hourly_min_f
                FROM public.wunderground_daily_tmax d
                CROSS JOIN LATERAL jsonb_array_elements(d.hourly_observations_json) item
                WHERE d.validation_status IN ('accepted','manual_confirmed')
                  AND item ? 'temp_f'
                  AND item->>'temp_f' IS NOT NULL
                  AND item->>'temp_f' <> 'null'
                GROUP BY station_id, local_date
            )
            SELECT count(*)
            FROM public.wunderground_daily_tmax d
            JOIN hourly h
              ON h.station_id = d.station_id
             AND h.local_date = d.local_date
            WHERE d.validation_status IN ('accepted','manual_confirmed')
              AND (d.tmax_f <> h.hourly_max_f OR d.tmin_f <> h.hourly_min_f)
            """
        )
    ).scalar_one()
    if int(mismatched_hourly_max):
        result.failures.append(
            f"{mismatched_hourly_max} accepted WU truth rows do not match stored hourly temp extrema"
        )

    canary = connection.execute(
        text(
            """
            SELECT tmax_f, tmin_f, validation_status, daily_high_source
            FROM public.wunderground_daily_tmax
            WHERE station_id = 'KLGA'
              AND local_date = '2026-05-21'::date
            """
        )
    ).mappings().first()
    if canary is not None:
        if (
            int(canary["tmax_f"] or -999) != 66
            or int(canary["tmin_f"] or -999) != 56
            or canary["daily_high_source"] != "hourly_temp_max"
            or canary["validation_status"] not in {"accepted", "manual_confirmed"}
        ):
            result.failures.append(
                "KLGA 2026-05-21 canary row must be tmax_f=66, tmin_f=56, source=hourly_temp_max"
            )

    row_count = connection.execute(text("SELECT count(*) FROM public.wunderground_daily_tmax")).scalar_one()
    accepted_rows = connection.execute(
        text(
            """
            SELECT count(*)
            FROM public.wunderground_daily_tmax
            WHERE validation_status IN ('accepted','manual_confirmed')
            """
        )
    ).scalar_one()
    klga_contract_rows = connection.execute(
        text(
            """
            SELECT count(*)
            FROM public.wunderground_daily_tmax
            WHERE station_id = 'KLGA'
              AND local_date >= :contract_start_date
              AND validation_status IN ('accepted','manual_confirmed')
              AND tmax_f IS NOT NULL
            """
        ),
        {"contract_start_date": CONTRACT_START_DATE},
    ).scalar_one()

    result.details.update(
        {
            "wunderground_fetchable_stations": int(station_rows),
            "wunderground_daily_tmax_rows": int(row_count),
            "wunderground_daily_tmax_accepted_rows": int(accepted_rows),
            "wu_klga_contract_rows_since_1973": int(klga_contract_rows),
            "wu_contract_start_date": CONTRACT_START_DATE.isoformat(),
        }
    )
    return result
