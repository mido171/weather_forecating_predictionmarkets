from __future__ import annotations

import sqlite3

import pandas as pd

import scripts.run_hkg_t24_beastmode_signal_discovery as discovery


def test_inspect_press_archive_reports_repo_local_scoreable_export(tmp_path, monkeypatch) -> None:
    export_path = tmp_path / "hko_press_archive_temperature_forecast_days.parquet"
    pd.DataFrame(
        {
            "target_date": ["2001-01-02"],
            "issue_at_hkt": ["2001-01-01T12:00:00+08:00"],
            "forecast_min_c": [18.0],
            "forecast_max_c": [24.0],
        }
    ).to_parquet(export_path, index=False)

    db_path = tmp_path / "archive.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE candidates (
                source TEXT,
                index_date TEXT,
                title TEXT,
                product_type TEXT,
                url TEXT,
                discovered_at_utc TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO candidates VALUES (
                'info_gov',
                '2001-01-01',
                'Weather forecast',
                'local',
                'https://example/a',
                '2026-01-01T00:00:00Z'
            )
            """
        )

    monkeypatch.setattr(discovery, "PRESS_FORECAST_EXPORT_PATH", export_path)
    features = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2001-01-02"]),
            "target_tmax_c": [23.0],
            "season": ["DJF"],
            "month": [1],
        }
    )

    coverage, joined, scores = discovery.inspect_press_archive(db_path, features)

    assert "hko_press_archive_offline_export.temperature_forecast_days" in set(coverage["source_id"])
    assert len(joined) == 1
    assert not scores.empty
