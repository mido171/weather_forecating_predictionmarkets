from __future__ import annotations

from datetime import time

import pandas as pd

from hkg_tmax.data.forecast_anchor import (
    CutoffProfile,
    build_cutoff_frame,
    cutoff_timestamps,
    select_forecast_anchors,
)
from hkg_tmax.data.target_history_features import build_target_history_for_dates
from hkg_tmax.features.leakage_guards import leakage_audit_payload


def test_cutoff_timestamps_are_target_minus_one_hkt_and_utc() -> None:
    cutoff_hkt, cutoff_utc = cutoff_timestamps(
        pd.Timestamp("2026-07-05"),
        CutoffProfile("tminus1_2359", time(23, 59)),
    )

    assert str(cutoff_hkt) == "2026-07-04 23:59:00+08:00"
    assert str(cutoff_utc) == "2026-07-04 15:59:00+00:00"


def test_forecast_anchor_selects_latest_info_gov_issue_before_cutoff() -> None:
    targets = pd.DataFrame(
        {
            "target_date": [pd.Timestamp("2024-06-02")],
            "y_true_c": [32.0],
            "label_source": ["sealed_confirmation"],
        }
    )
    cutoff_rows = build_cutoff_frame(
        targets,
        profiles=(CutoffProfile("tminus1_1800", time(18, 0)),),
    )
    forecasts = pd.DataFrame(
        [
            {
                "target_date": pd.Timestamp("2024-06-02"),
                "issue_at_hkt": pd.Timestamp("2024-06-01 16:30:00"),
                "issue_at_utc": pd.Timestamp("2024-06-01 08:30:00", tz="UTC"),
                "forecast_min_c": 26.0,
                "forecast_max_c": 30.0,
                "forecast_midpoint_c": 28.0,
                "full_text": "Sunny periods. Very hot during the day.",
                "raw_sha256": "before",
                "source_url": "https://www.info.gov.hk/gia/wr/202406/01/P202406010001.htm",
            },
            {
                "target_date": pd.Timestamp("2024-06-02"),
                "issue_at_hkt": pd.Timestamp("2024-06-01 19:00:00"),
                "issue_at_utc": pd.Timestamp("2024-06-01 11:00:00", tz="UTC"),
                "forecast_min_c": 27.0,
                "forecast_max_c": 35.0,
                "forecast_midpoint_c": 31.0,
                "full_text": "Post-cutoff hotter update.",
                "raw_sha256": "after",
                "source_url": "https://www.info.gov.hk/gia/wr/202406/01/P202406010999.htm",
            },
        ]
    )

    selected = select_forecast_anchors(cutoff_rows, forecasts)

    assert selected.loc[0, "forecast_selector_status"] == "selected"
    assert selected.loc[0, "anchor_forecast_max_c"] == 30.0
    assert selected.loc[0, "eligible_forecast_count"] == 1
    assert selected.loc[0, "anchor_raw_sha256"] == "before"


def test_target_history_uses_lag2_floor_and_does_not_create_lag1() -> None:
    history = pd.DataFrame(
        {
            "local_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "target_tmax_c": [20.0 + idx for idx in range(10)],
        }
    )

    features = build_target_history_for_dates(pd.Series([pd.Timestamp("2024-01-06")]), history)

    assert features.loc[0, "target_history_max_source_date"] == pd.Timestamp("2024-01-04")
    assert features.loc[0, "target_lag2_tmax_c"] == 23.0
    assert features.loc[0, "target_lag3_tmax_c"] == 22.0
    assert "target_lag1_tmax_c" not in features.columns


def test_leakage_audit_flags_post_cutoff_and_sub_lag2_predictors() -> None:
    matrix = pd.DataFrame(
        {
            "target_date": [pd.Timestamp("2024-06-02")],
            "cutoff_at_utc": [pd.Timestamp("2024-06-01 15:59:00", tz="UTC")],
            "anchor_issue_at_utc": [pd.Timestamp("2024-06-01 16:00:00", tz="UTC")],
            "latest_hourly_dispatch_at_utc_used": [pd.Timestamp("2024-06-01 16:01:00", tz="UTC")],
            "target_history_max_source_date": [pd.Timestamp("2024-06-01")],
        }
    )
    lineage = pd.DataFrame(
        [
            {
                "feature_name": "unsafe_lag1",
                "source_table": "feature_safe.hko_target_history_pre2024",
                "uses_target_label_boolean": True,
                "minimum_lag_days": 1,
            },
            {
                "feature_name": "raw_daily_extract_payload",
                "source_table": "raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da",
                "uses_target_label_boolean": False,
                "minimum_lag_days": None,
            },
        ]
    )

    payload = leakage_audit_payload(matrix, lineage)
    failures = {check["check_name"]: check["violation_count"] for check in payload["checks"]}

    assert payload["status"] == "fail"
    assert failures["forecast_anchor_issue_at_or_before_cutoff"] == 1
    assert failures["hourly_dispatch_at_or_before_cutoff"] == 1
    assert failures["target_history_lag2_floor"] == 1
    assert failures["target_label_predictor_minimum_lag_days"] == 1
    assert failures["raw_daily_extract_not_used_as_predictor"] == 1
