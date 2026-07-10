from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_hkg_t24_0082_refreshed_archive_replay_readiness_audit import (
    build_leakage_audit,
    coverage_row,
    date_set,
    dependency_gap_rows,
    missing_official_dates,
)


def test_coverage_row_marks_partial_artifact_as_stale() -> None:
    official = pd.DataFrame({"target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])})
    artifact = pd.DataFrame({"target_date": pd.to_datetime(["2020-01-01", "2020-01-03"])})

    row = coverage_row(
        artifact_id="test_artifact",
        frame=artifact,
        official_dates=date_set(official),
        path=Path("dummy.csv"),
        role="test",
        replay_required=True,
    )

    assert row["replay_status"] == "stale_partial_frame"
    assert row["official_days_covered"] == 2
    assert row["official_days_missing"] == 1
    assert row["official_coverage_ratio"] == 2 / 3


def test_missing_official_dates_preserves_source_family() -> None:
    official = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press_archive", "press_archive", "rss_archive"],
        }
    )
    artifact = pd.DataFrame({"target_date": pd.to_datetime(["2020-01-01"])})

    missing = missing_official_dates(official, artifact)

    assert missing.to_dict("records") == [
        {"target_date": "2020-01-02", "forecast_source_family": "press_archive"},
        {"target_date": "2020-01-03", "forecast_source_family": "rss_archive"},
    ]


def test_leakage_audit_flags_confirmation_rows() -> None:
    frames = {
        "safe": pd.DataFrame({"target_date": pd.to_datetime(["2023-12-31"])}),
        "unsafe": pd.DataFrame({"target_date": pd.to_datetime(["2024-01-01"])}),
    }

    audit = build_leakage_audit(frames).set_index("artifact_id")

    assert audit.loc["safe", "status"] == "PASS"
    assert audit.loc["unsafe", "status"] == "FAIL"
    assert audit.loc["unsafe", "confirmation_rows"] == 1


def test_dependency_gap_rows_requires_regeneration_for_stale_required_artifact() -> None:
    coverage = pd.DataFrame(
        [
            {
                "artifact_id": "stale",
                "replay_status": "stale_partial_frame",
                "replay_required": True,
                "official_days_missing": 5,
                "unique_target_days": 10,
            }
        ]
    )

    gaps = dependency_gap_rows(coverage)

    assert bool(gaps.loc[0, "required_for_0081_replay"]) is True
    assert "regenerate" in gaps.loc[0, "required_action"]
