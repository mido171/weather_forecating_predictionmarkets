from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.registry.cutoffs import CANONICAL_CUTOFFS


def seed_cutoffs(connection: Connection) -> int:
    rows_changed = 0
    for cutoff in CANONICAL_CUTOFFS:
        result = connection.execute(
            text(
                """
                INSERT INTO registry.cutoffs (
                    cutoff_id,
                    cutoff_order,
                    timezone_name,
                    local_time,
                    target_day_offset,
                    description,
                    active
                )
                VALUES (
                    :cutoff_id,
                    :cutoff_order,
                    :timezone_name,
                    :local_time,
                    :target_day_offset,
                    :description,
                    true
                )
                ON CONFLICT (cutoff_id) DO UPDATE SET
                    cutoff_order = EXCLUDED.cutoff_order,
                    timezone_name = EXCLUDED.timezone_name,
                    local_time = EXCLUDED.local_time,
                    target_day_offset = EXCLUDED.target_day_offset,
                    description = EXCLUDED.description,
                    active = true
                """
            ),
            {
                "cutoff_id": cutoff.cutoff_id,
                "cutoff_order": cutoff.cutoff_order,
                "timezone_name": cutoff.timezone_name,
                "local_time": cutoff.local_time,
                "target_day_offset": cutoff.target_day_offset,
                "description": cutoff.description,
            },
        )
        rows_changed += result.rowcount or 0
    return rows_changed
