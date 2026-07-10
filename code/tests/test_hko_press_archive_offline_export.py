from __future__ import annotations

import pandas as pd

from scripts.export_hko_press_archive_offline import build_coverage, normalize_forecast_days


def test_normalize_forecast_days_marks_invalid_temperature_rows() -> None:
    raw = pd.DataFrame(
        {
            "bulletin_id": ["a", "b", "c"],
            "source": ["info_gov", "info_gov", "info_gov"],
            "source_url": ["https://example/a", "https://example/b", "https://example/c"],
            "product_type": ["5day", "5day", "local"],
            "issue_at_hkt": [
                "2001-01-01T12:00:00+08:00",
                "2001-01-01T12:00:00+08:00",
                "2001-01-01T12:00:00+08:00",
            ],
            "target_date": ["2001-01-02", "2001-01-03", "2001-01-04"],
            "forecast_min_c": [18, 30, None],
            "forecast_max_c": [23, 25, 24],
            "rh_min_pct": [50, None, None],
            "rh_max_pct": [80, None, None],
            "wind_text": ["east", "", ""],
            "weather_text": ["fine", "", ""],
            "raw_sha256": ["0" * 64, "1" * 64, "2" * 64],
            "raw_path": ["a.html", "b.html", "c.html"],
            "candidate_index_date": ["2001-01-01", "2001-01-01", "2001-01-01"],
            "candidate_title": ["A", "B", "C"],
            "candidate_product_type": ["5day", "5day", "local"],
            "retrieval_id": [1, 2, 3],
            "attempted_at_utc": ["2026-01-01T00:00:00Z"] * 3,
        }
    )

    normalized = normalize_forecast_days(raw, exported_at_utc="2026-01-01T00:00:00Z")

    assert normalized["temperature_row_valid"].to_list() == [True, False, True]
    assert normalized["scoreable_row_valid"].to_list() == [True, False, True]
    assert normalized["source_id"].eq("hko_info_gov_press_weather_forecast_archive").all()
    assert normalized["operational_input_allowed"].all()


def test_normalize_forecast_days_excludes_implausible_target_dates_from_scoreable_rows() -> None:
    raw = pd.DataFrame(
        {
            "bulletin_id": ["bad", "good"],
            "source": ["info_gov", "info_gov"],
            "source_url": ["https://example/bad", "https://example/good"],
            "product_type": ["local", "local"],
            "issue_at_hkt": ["2002-11-02T12:00:00+08:00", "2002-11-02T12:00:00+08:00"],
            "target_date": ["1990-11-03", "2002-11-03"],
            "forecast_min_c": [20, 20],
            "forecast_max_c": [25, 25],
            "rh_min_pct": [None, None],
            "rh_max_pct": [None, None],
            "wind_text": ["", ""],
            "weather_text": ["", ""],
            "raw_sha256": ["0" * 64, "1" * 64],
            "raw_path": ["bad.html", "good.html"],
            "candidate_index_date": ["2002-11-02", "2002-11-02"],
            "candidate_title": ["Bad", "Good"],
            "candidate_product_type": ["local", "local"],
            "retrieval_id": [1, 2],
            "attempted_at_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        }
    )

    normalized = normalize_forecast_days(raw, exported_at_utc="2026-01-01T00:00:00Z")

    assert normalized["temperature_row_valid"].to_list() == [True, True]
    assert normalized["target_date_plausible"].to_list() == [False, True]
    assert normalized["scoreable_row_valid"].to_list() == [False, True]
    assert normalized["target_issue_lead_days"].to_list() == [-4382, 1]


def test_build_coverage_exposes_candidate_detail_gaps() -> None:
    candidates = pd.DataFrame(
        {
            "source": ["info_gov", "info_gov", "info_gov"],
            "index_date": ["2001-01-01", "2001-01-02", "2003-01-01"],
            "title": ["A", "B", "C"],
            "product_type": ["local", "local", "7day"],
            "url": ["https://example/a", "https://example/b", "https://example/c"],
            "discovered_at_utc": ["2026-01-01T00:00:00Z"] * 3,
        }
    )
    retrievals = pd.DataFrame(
        {
            "raw_path": ["a.html"],
            "index_date": ["2001-01-01"],
            "candidate_product_type": ["local"],
        }
    )
    forecast_days = pd.DataFrame(
        {
            "candidate_index_date": ["2001-01-01"],
            "candidate_product_type": ["local"],
            "bulletin_id": ["a"],
            "target_date": [pd.Timestamp("2001-01-02")],
            "temperature_row_valid": [True],
            "scoreable_row_valid": [True],
        }
    )

    coverage, missing = build_coverage(candidates, retrievals, forecast_days)

    local_2001 = coverage[(coverage["index_year"].eq(2001)) & (coverage["product_type"].eq("local"))].iloc[0]
    assert int(local_2001["candidate_count"]) == 2
    assert int(local_2001["raw_detail_count"]) == 1
    assert int(local_2001["raw_detail_gap"]) == 1
    assert int(local_2001["scoreable_temperature_day_count"]) == 1
    assert set(missing["index_year"]) == {2001, 2003}
