from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from hkg_tmax.analysis_contracts import (
    PointInTimeEligibilityError,
    hko_tminus1_15_cutoff,
    validate_point_in_time_rows,
)

HKT = ZoneInfo("Asia/Hong_Kong")


def test_cutoff_is_tminus1_1500_hkt() -> None:
    cutoff = hko_tminus1_15_cutoff(datetime(2026, 6, 20, 0, 0, tzinfo=HKT))

    assert cutoff.isoformat() == "2026-06-19T15:00:00+08:00"


def test_point_in_time_validator_accepts_operational_row_before_cutoff() -> None:
    validate_point_in_time_rows(
        [
            {
                "role": "OPERATIONAL_POINT_IN_TIME",
                "available_at": "2026-06-19T14:59:00+08:00",
                "target_derived": "false",
            }
        ],
        cutoff_hkt=datetime(2026, 6, 19, 15, 0, tzinfo=HKT),
    )


@pytest.mark.parametrize(
    "row",
    [
        {"role": "OPERATIONAL_POINT_IN_TIME", "available_at": "2026-06-19T15:00:01+08:00"},
        {"role": "RETROSPECTIVE_MECHANISM_ONLY", "available_at": "2026-06-19T14:00:00+08:00"},
        {"role": "TARGET_ONLY", "available_at": "2026-06-19T14:00:00+08:00"},
        {
            "role": "OPERATIONAL_POINT_IN_TIME",
            "available_at": "2026-06-19T14:00:00+08:00",
            "target_derived": "true",
        },
    ],
)
def test_point_in_time_validator_rejects_adversarial_rows(row: dict[str, str]) -> None:
    with pytest.raises(PointInTimeEligibilityError):
        validate_point_in_time_rows(
            [row],
            cutoff_hkt=datetime(2026, 6, 19, 15, 0, tzinfo=HKT),
        )

