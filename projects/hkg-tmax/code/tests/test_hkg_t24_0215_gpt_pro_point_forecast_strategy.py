from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("exp0215", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cutoff_timestamp_is_t_minus_one_hkt_clock_time():
    mod = load_module()

    assert mod.cutoff_timestamp_hkt(pd.Timestamp("2023-08-15"), "23:59") == pd.Timestamp("2023-08-14 23:59")
    assert mod.cutoff_timestamp_hkt(pd.Timestamp("2023-08-15"), "17:00") == pd.Timestamp("2023-08-14 17:00")


def test_fast_forecast_selector_excludes_post_cutoff_issues_and_keeps_revision_path():
    mod = load_module()
    labels = pd.DataFrame({"target_date": [pd.Timestamp("2023-07-10")]})
    forecasts = pd.DataFrame(
        [
            {
                "target_date": pd.Timestamp("2023-07-10"),
                "issue_at_hkt": pd.Timestamp("2023-07-09 16:45"),
                "issue_at_utc": pd.Timestamp("2023-07-09 08:45", tz="UTC"),
                "forecast_min_c": 27.0,
                "forecast_max_c": 30.0,
                "forecast_midpoint_c": 28.5,
                "target_date_confidence": "high",
                "parse_status": "ok",
                "stale_snapshot_flag": False,
                "stale_hours": 0.0,
                "title": "Local weather forecast",
                "temperature_text": "Hot with sunny periods.",
                "full_text": "Hot with sunny periods.",
            },
            {
                "target_date": pd.Timestamp("2023-07-10"),
                "issue_at_hkt": pd.Timestamp("2023-07-09 17:45"),
                "issue_at_utc": pd.Timestamp("2023-07-09 09:45", tz="UTC"),
                "forecast_min_c": 28.0,
                "forecast_max_c": 31.0,
                "forecast_midpoint_c": 29.5,
                "target_date_confidence": "high",
                "parse_status": "ok",
                "stale_snapshot_flag": False,
                "stale_hours": 0.0,
                "title": "Local weather forecast",
                "temperature_text": "Very hot with showers later.",
                "full_text": "Very hot with showers later.",
            },
            {
                "target_date": pd.Timestamp("2023-07-10"),
                "issue_at_hkt": pd.Timestamp("2023-07-09 19:00"),
                "issue_at_utc": pd.Timestamp("2023-07-09 11:00", tz="UTC"),
                "forecast_min_c": 29.0,
                "forecast_max_c": 33.0,
                "forecast_midpoint_c": 31.0,
                "target_date_confidence": "high",
                "parse_status": "ok",
                "stale_snapshot_flag": False,
                "stale_hours": 0.0,
                "title": "Local weather forecast",
                "temperature_text": "Post cutoff row.",
                "full_text": "Post cutoff row.",
            },
        ]
    )

    selected = mod.select_forecast_features_fast(labels, forecasts, "18:00").iloc[0]

    assert bool(selected["official_available_before_cutoff"])
    assert selected["latest_issue_at_hkt"] == pd.Timestamp("2023-07-09 17:45")
    assert selected["forecast_max_c_latest"] == 31.0
    assert selected["n_issues_before_cutoff"] == 2
    assert selected["max_delta_latest_minus_first"] == 1.0
    assert bool(selected["has_1645_issue_before_cutoff"])
    assert bool(selected["is_1745_family"])
    assert bool(selected["txt_showers"])


def test_leakage_row_audit_accepts_latest_issue_column_and_tminus2_contract():
    mod = load_module()
    frame = pd.DataFrame(
        {
            "cutoff": ["23:59"],
            "target_date": [pd.Timestamp("2023-07-10")],
            "asof_cutoff_hkt": [pd.Timestamp("2023-07-09 23:59")],
            "official_available_before_cutoff": [True],
            "latest_issue_at_hkt": [pd.Timestamp("2023-07-09 21:45")],
            "target_tmax_lag_1": [pd.NA],
        }
    )

    audit = mod.leakage_row_audit(frame)

    assert set(audit["status"]) == {"pass"}
    assert int(audit["failed_rows"].sum()) == 0
