from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_press_archive_raw_detail_gap_audit import (
    build_year_product_coverage,
    count_raw_html_files_by_year,
    scoreable_forecast_day_flags,
)


def test_build_year_product_coverage_separates_candidates_raw_and_parsed() -> None:
    candidates = pd.DataFrame(
        {
            "source": ["info_gov", "info_gov", "info_gov"],
            "index_date": ["2004-01-01", "2004-01-02", "2005-01-01"],
            "title": ["A", "B", "C"],
            "product_type": ["local", "local", "local"],
            "url": ["https://example/a", "https://example/b", "https://example/c"],
            "discovered_at_utc": ["2026-01-01T00:00:00Z"] * 3,
        }
    )
    retrievals = pd.DataFrame(
        {
            "source": ["info_gov_bulletin"],
            "url": ["https://example/a"],
            "attempted_at_utc": ["2026-01-01T00:00:00Z"],
            "status_code": [200],
            "error": [None],
            "content_sha256": ["0" * 64],
            "raw_path": ["a.html"],
        }
    )
    forecast_days = pd.DataFrame(
        {
            "bulletin_id": ["a"],
            "source": ["info_gov"],
            "source_url": ["https://example/a"],
            "product_type": ["local"],
            "issue_at_hkt": ["2004-01-01T12:00:00+08:00"],
            "target_date": ["2004-01-02"],
            "forecast_min_c": [18.0],
            "forecast_max_c": [23.0],
        }
    )

    coverage, missing, no_raw = build_year_product_coverage(candidates, retrievals, forecast_days)

    local_2004 = coverage[(coverage["index_year"].eq(2004)) & coverage["product_type"].eq("local")].iloc[0]
    local_2005 = coverage[(coverage["index_year"].eq(2005)) & coverage["product_type"].eq("local")].iloc[0]
    assert int(local_2004["candidate_count"]) == 2
    assert int(local_2004["raw_url_count"]) == 1
    assert int(local_2004["scoreable_rows"]) == 1
    assert int(local_2005["candidate_count"]) == 1
    assert int(local_2005["raw_url_count"]) == 0
    assert set(missing["index_year"]) == {2004, 2005}
    assert no_raw["index_year"].to_list() == [2005]


def test_scoreable_forecast_day_flags_blocks_implausible_target_date() -> None:
    forecast_days = pd.DataFrame(
        {
            "bulletin_id": ["bad", "good"],
            "source": ["info_gov", "info_gov"],
            "source_url": ["https://example/bad", "https://example/good"],
            "product_type": ["local", "local"],
            "issue_at_hkt": ["2004-01-01T12:00:00+08:00", "2004-01-01T12:00:00+08:00"],
            "target_date": ["1990-01-01", "2004-01-02"],
            "forecast_min_c": [18.0, 18.0],
            "forecast_max_c": [23.0, 23.0],
        }
    )

    flagged = scoreable_forecast_day_flags(forecast_days)

    assert flagged["temperature_row_valid"].to_list() == [True, True]
    assert flagged["target_date_plausible"].to_list() == [False, True]
    assert flagged["scoreable_row_valid"].to_list() == [False, True]


def test_count_raw_html_files_by_year_counts_only_html(tmp_path) -> None:
    raw_root = tmp_path / "raw" / "info_gov_bulletin"
    year_dir = raw_root / "2004" / "01" / "01"
    year_dir.mkdir(parents=True)
    (year_dir / "a.html").write_text("one", encoding="utf-8")
    (year_dir / "a.html.metadata.json").write_text("{}", encoding="utf-8")
    (year_dir / "b.html").write_text("two", encoding="utf-8")

    counts = count_raw_html_files_by_year(raw_root)

    assert counts.loc[0, "raw_file_year"] == 2004
    assert counts.loc[0, "html_file_count"] == 2
    assert counts.loc[0, "total_bytes"] == 6
