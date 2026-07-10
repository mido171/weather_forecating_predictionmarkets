from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from klga_tmax.providers.polymarket.cutoff_analysis import (
    ArtifactPaths,
    PolymarketPublicClient,
    RequestResult,
    candidate_cutoffs,
    model_availability_score,
    parse_bucket,
    select_recommendation,
)


class NeverEndingCursorClient(PolymarketPublicClient):
    def __init__(self, *, paths: ArtifactPaths) -> None:
        super().__init__(
            paths=paths,
            manifest=[],
            use_cache=False,
            sleep_seconds=0.0,
            max_event_pages=2,
        )
        self.calls = 0

    def request_json(self, *args, **kwargs) -> RequestResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        return RequestResult(
            payload={"events": [], "next_cursor": "still-more"},
            request_sha256="test-only",
            status_code=200,
            row_count=0,
            cache_hit=False,
        )


def test_parse_bucket_handles_common_nyc_tmax_labels() -> None:
    assert parse_bucket("73°F or below") == {"label": "73F or below", "lower_f": None, "upper_f": 73}
    assert parse_bucket("74-75°F") == {"label": "74-75F", "lower_f": 74, "upper_f": 75}
    assert parse_bucket("92°F or higher") == {"label": "92F or higher", "lower_f": 92, "upper_f": None}


def test_candidate_grid_contains_canonical_baseline() -> None:
    candidates = candidate_cutoffs(date(2026, 6, 28))
    baseline = [candidate for candidate in candidates if candidate["candidate_id"] == "T_MINUS_1_2045UTC"]
    assert len(baseline) == 1
    assert baseline[0]["cutoff_utc"] == datetime(2026, 6, 27, 20, 45, tzinfo=timezone.utc)


def test_event_pagination_fails_closed_at_hard_page_budget(tmp_path) -> None:
    client = NeverEndingCursorClient(paths=ArtifactPaths.create(tmp_path / "artifacts"))

    with pytest.raises(RuntimeError, match="hard page budget"):
        client.fetch_nyc_tmax_events(
            start_date=date(2026, 6, 27),
            end_date=date(2026, 6, 28),
        )

    assert client.calls == 2


def test_model_score_improves_for_later_safe_cutoff() -> None:
    early = model_availability_score(datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc), date(2026, 6, 28))
    baseline = model_availability_score(datetime(2026, 6, 27, 20, 45, tzinfo=timezone.utc), date(2026, 6, 28))
    assert baseline["available_model_count"] >= early["available_model_count"]
    assert baseline["model_score"] > early["model_score"]


def test_recommendation_uses_eligible_latest_near_best_model_score() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "candidate_id": "T_MINUS_1_1200UTC",
                "relative_day": -1,
                "cutoff_time_utc": "12:00:00",
                "tradable_rate": 1.0,
                "pre_explosion_rate": 1.0,
                "model_score": 0.8,
                "model_score_normalized": 0.5,
                "median_remaining_move": 0.4,
            },
            {
                "candidate_id": "T_MINUS_1_2045UTC",
                "relative_day": -1,
                "cutoff_time_utc": "20:45:00",
                "tradable_rate": 1.0,
                "pre_explosion_rate": 0.8,
                "model_score": 1.0,
                "model_score_normalized": 1.0,
                "median_remaining_move": 0.3,
            },
            {
                "candidate_id": "T_1200UTC",
                "relative_day": 0,
                "cutoff_time_utc": "12:00:00",
                "tradable_rate": 1.0,
                "pre_explosion_rate": 0.2,
                "model_score": 1.2,
                "model_score_normalized": 1.0,
                "median_remaining_move": 0.1,
            },
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "guardrail": 0.70,
                "selected_candidate_id": "T_MINUS_1_2045UTC",
                "tradable_rate": 1.0,
                "pre_explosion_rate": 0.8,
                "model_score_normalized": 1.0,
                "median_remaining_move": 0.3,
                "available_model_count": 10,
            }
        ]
    )
    recommendation = select_recommendation(aggregate, sensitivity)
    assert recommendation["selected_candidate"]["candidate_id"] == "T_MINUS_1_2045UTC"
