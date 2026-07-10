from __future__ import annotations

from datetime import datetime, timezone

import pytest

from klga_tmax.features.leakage import (
    FeatureSourceTrace,
    TraceAvailabilityViolation,
    validate_feature_trace_for_cutoff,
)


def test_feature_trace_at_or_before_cutoff_is_accepted() -> None:
    cutoff = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)
    validate_feature_trace_for_cutoff(
        cutoff_utc=cutoff,
        source_trace=[
            FeatureSourceTrace("iem_mos", "MAVLGA:2026062700", datetime(2026, 6, 27, 12, 59, tzinfo=timezone.utc)),
            FeatureSourceTrace("iem_mos", "MAVLGA:2026062700", cutoff),
        ],
    )


def test_feature_trace_after_cutoff_is_rejected() -> None:
    cutoff = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)
    with pytest.raises(TraceAvailabilityViolation):
        validate_feature_trace_for_cutoff(
            cutoff_utc=cutoff,
            source_trace=[
                FeatureSourceTrace("iem_mos", "MAVLGA:2026062700", datetime(2026, 6, 27, 13, 0, 1, tzinfo=timezone.utc)),
            ],
        )
