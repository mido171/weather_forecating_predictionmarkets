from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.registry.cutoffs import materialized_cutoff_rows


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def materialize_target_instances(
    connection: Connection,
    *,
    start_date: date,
    end_date: date,
    replace: bool,
) -> int:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    if replace:
        connection.execute(
            text(
                """
                DELETE FROM gold.target_instances
                WHERE target_date BETWEEN :start_date AND :end_date
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        )

    inserted = 0
    for target_date in iter_dates(start_date, end_date):
        for row in materialized_cutoff_rows(target_date):
            result = connection.execute(
                text(
                    """
                    INSERT INTO gold.target_instances (
                        target_date,
                        cutoff_id,
                        cutoff_utc,
                        target_station_id,
                        local_day_start_utc,
                        local_day_end_utc,
                        settlement_high_f_whole,
                        settlement_high_available_at_utc,
                        label_available,
                        label_revision_sensitive
                    )
                    SELECT
                        :target_date,
                        :cutoff_id,
                        :cutoff_utc,
                        'KLGA',
                        :local_day_start_utc,
                        :local_day_end_utc,
                        actual.tmax_f,
                        actual.settlement_available_at_utc,
                        actual.tmax_f IS NOT NULL
                            AND actual.settlement_available_at_utc <= :cutoff_utc,
                        true
                    FROM (SELECT 1) anchor
                    LEFT JOIN public.wunderground_daily_tmax actual
                      ON actual.local_date = :target_date
                     AND actual.station_id = 'KLGA'
                     AND actual.validation_status IN ('accepted','manual_confirmed')
                    ON CONFLICT (target_date, cutoff_id)
                    DO UPDATE SET
                        settlement_high_f_whole = EXCLUDED.settlement_high_f_whole,
                        settlement_high_available_at_utc = EXCLUDED.settlement_high_available_at_utc,
                        label_available = EXCLUDED.label_available,
                        label_revision_sensitive = EXCLUDED.label_revision_sensitive
                    """
                ),
                row,
            )
            inserted += result.rowcount or 0
    return inserted
