from __future__ import annotations

from datetime import date, timezone

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.db.migrations_check import ContractInspection, inspect_contract
from klga_tmax.registry.cutoffs import (
    CANONICAL_CUTOFFS,
    cutoff_timestamp_utc,
    sample_dst_and_non_dst_dates,
)

EXPECTED_2026_06_28_CUTOFF_UTC = {
    "T_MINUS_1_STOCKHOLM_1500": "2026-06-27T13:00:00+00:00",
    "T_MINUS_1_STOCKHOLM_1915": "2026-06-27T17:15:00+00:00",
    "T_MINUS_1_STOCKHOLM_2230": "2026-06-27T20:30:00+00:00",
    "PRE_LOCAL_DAY_NYC_2350": "2026-06-28T03:50:00+00:00",
    "T_MINUS_1_2045UTC": "2026-06-27T20:45:00+00:00",
    "T_1245UTC": "2026-06-28T12:45:00+00:00",
}


def validate_foundation(connection: Connection) -> ContractInspection:
    result = inspect_contract(connection)

    for target_date in sample_dst_and_non_dst_dates():
        for cutoff in CANONICAL_CUTOFFS:
            cutoff_utc = cutoff_timestamp_utc(target_date, cutoff)
            if cutoff_utc.tzinfo is None or cutoff_utc.utcoffset() is None:
                result.failures.append(
                    f"cutoff {cutoff.cutoff_id} for {target_date.isoformat()} is naive"
                )
            if cutoff_utc.tzinfo is not timezone.utc:
                result.failures.append(
                    f"cutoff {cutoff.cutoff_id} for {target_date.isoformat()} is not UTC"
                )

    for cutoff in CANONICAL_CUTOFFS:
        observed = cutoff_timestamp_utc(date(2026, 6, 28), cutoff).isoformat()
        if observed != EXPECTED_2026_06_28_CUTOFF_UTC[cutoff.cutoff_id]:
            result.failures.append(
                f"2026-06-28 {cutoff.cutoff_id} expected "
                f"{EXPECTED_2026_06_28_CUTOFF_UTC[cutoff.cutoff_id]} observed {observed}"
            )

    late_feature_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM gold.feature_values fv
            JOIN gold.target_instances ti
              ON ti.target_instance_id = fv.target_instance_id
            WHERE fv.feature_available = true
              AND fv.max_source_available_at_utc IS NOT NULL
              AND fv.max_source_available_at_utc > ti.cutoff_utc
            """
        )
    ).scalar_one()
    if late_feature_count:
        result.failures.append(
            f"{late_feature_count} gold feature rows have source availability after cutoff"
        )

    target_instance_count = connection.execute(
        text("SELECT count(*) FROM gold.target_instances")
    ).scalar_one()
    result.details["target_instance_rows"] = int(target_instance_count)
    result.details["late_feature_rows"] = int(late_feature_count)
    return result
