from __future__ import annotations

from datetime import datetime, timedelta, timezone

from klga_tmax.ingestion.eligibility import AvailabilityInput, effective_available_at_utc, is_cutoff_eligible


def test_availability_boundary_is_inclusive_at_cutoff() -> None:
    cutoff = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)
    assert is_cutoff_eligible(
        AvailabilityInput(cutoff_utc=cutoff, our_ingested_at_utc=datetime(2026, 6, 27, 12, 59, tzinfo=timezone.utc))
    )
    assert is_cutoff_eligible(
        AvailabilityInput(cutoff_utc=cutoff, our_ingested_at_utc=datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc))
    )
    assert not is_cutoff_eligible(
        AvailabilityInput(cutoff_utc=cutoff, our_ingested_at_utc=datetime(2026, 6, 27, 13, 0, 1, tzinfo=timezone.utc))
    )


def test_provider_availability_is_used_when_ingest_timestamp_is_absent() -> None:
    cutoff = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)
    observed = effective_available_at_utc(
        AvailabilityInput(
            cutoff_utc=cutoff,
            provider_available_at_utc=datetime(2026, 6, 27, 12, 45, tzinfo=timezone.utc),
        )
    )
    assert observed == datetime(2026, 6, 27, 12, 45, tzinfo=timezone.utc)


def test_conservative_lag_falls_back_to_run_time() -> None:
    cutoff = datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc)
    observed = effective_available_at_utc(
        AvailabilityInput(
            cutoff_utc=cutoff,
            run_time_utc=datetime(2026, 6, 27, 10, 0, tzinfo=timezone.utc),
            conservative_lag=timedelta(hours=2, minutes=15),
        )
    )
    assert observed == datetime(2026, 6, 27, 12, 15, tzinfo=timezone.utc)
