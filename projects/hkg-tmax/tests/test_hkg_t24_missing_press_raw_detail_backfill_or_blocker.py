from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_missing_press_raw_detail_backfill_or_blocker import (
    coverage_by_year_product,
    missing_candidates,
    priority_sample,
    probe_blocker_status,
)


def test_coverage_by_year_product_marks_partial_and_zero_success() -> None:
    frame = pd.DataFrame(
        {
            "index_year": [2007, 2007, 2008],
            "product_type": ["local", "local", "local"],
            "url": ["a", "b", "c"],
            "has_successful_raw_detail": [True, False, False],
            "has_any_attempt": [True, True, False],
        }
    )

    coverage = coverage_by_year_product(frame)

    row_2007 = coverage[coverage["index_year"].eq(2007)].iloc[0]
    row_2008 = coverage[coverage["index_year"].eq(2008)].iloc[0]
    assert int(row_2007["candidate_urls"]) == 2
    assert int(row_2007["missing_success_urls"]) == 1
    assert row_2007["status"] == "partial"
    assert int(row_2008["never_attempted_urls"]) == 1
    assert row_2008["status"] == "no_successful_raw_detail"


def test_missing_candidates_filters_years_products_and_success() -> None:
    frame = pd.DataFrame(
        {
            "index_date": pd.to_datetime(["2007-01-01", "2008-01-01", "2008-01-02"]),
            "index_year": [2007, 2008, 2008],
            "product_type": ["local", "local", "tc"],
            "title": ["a", "b", "c"],
            "url": ["a", "b", "c"],
            "has_successful_raw_detail": [False, False, False],
            "retrieval_attempts": [0, 1, 0],
            "last_attempted_at_utc": ["", "x", ""],
            "max_status_code": [None, 0, None],
            "sample_error": ["", "err", ""],
        }
    )

    missing = missing_candidates(frame, start_year=2008, end_year=2008, product_types={"local"})

    assert missing["url"].to_list() == ["b"]
    assert missing["index_date"].to_list() == ["2008-01-01"]


def test_priority_sample_takes_per_year_head() -> None:
    missing = pd.DataFrame(
        {
            "index_year": [2008, 2008, 2008, 2009],
            "url": ["a", "b", "c", "d"],
        }
    )

    sample = priority_sample(missing, per_year=2)

    assert sample["url"].to_list() == ["a", "b", "d"]


def test_probe_blocker_status_classifies_socket_permission_error() -> None:
    probe = pd.DataFrame(
        {
            "ok": [False],
            "error": ["[WinError 10013] An attempt was made to access a socket in a way forbidden by permissions"],
        }
    )

    assert probe_blocker_status(probe) == "network_socket_blocked"
