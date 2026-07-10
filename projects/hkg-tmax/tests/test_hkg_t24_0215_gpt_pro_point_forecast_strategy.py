from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from functools import cache, lru_cache
from pathlib import Path

import pandas as pd
import pytest

from hkg_tmax.paths import find_project_root

PROJECT_ROOT = find_project_root(Path(__file__))
SCRIPT = PROJECT_ROOT / "scripts" / "run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py"
TACTICAL_SCRIPT_NAMES = (
    "audit_tactical_gribstream_deep_sanity.py",
    "reset_tactical_gribstream_store.py",
    "run_tactical_gribstream_batch_smoke.py",
    "run_tactical_gribstream_first_week.py",
    "run_tactical_gribstream_h24n_smoke.py",
)


@lru_cache(maxsize=1)
def load_module():
    spec = importlib.util.spec_from_file_location("exp0215", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@cache
def load_named_script(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script_path.stem}", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_0215_uses_canonical_hkg_t24_campaign_root():
    mod = load_module()

    assert mod.EXP_DIR == (
        PROJECT_ROOT
        / "experiments"
        / "campaigns"
        / "hkg-t24"
        / "0215_gpt_pro_point_forecast_strategy"
    )


@pytest.mark.parametrize("script_name", TACTICAL_SCRIPT_NAMES)
def test_0214_producers_use_canonical_hkg_t24_campaign_root(script_name: str):
    mod = load_named_script(script_name)

    assert mod.EXPERIMENT_ROOT == (
        PROJECT_ROOT
        / "experiments"
        / "campaigns"
        / "hkg-t24"
        / "0214_tactical_h24n_gribstream_backfill"
    )


def test_initialize_0215_writes_one_readme_and_machine_metadata(tmp_path, monkeypatch):
    mod = load_module()
    experiment_root = tmp_path / "0215_gpt_pro_point_forecast_strategy"
    monkeypatch.setattr(mod, "EXP_DIR", experiment_root)

    mod.initialize_experiment_folder()

    markdown_paths = list(experiment_root.rglob("*.md"))
    assert markdown_paths == [experiment_root / "README.md"]
    readme = markdown_paths[0].read_text(encoding="utf-8")
    assert "## Hypothesis" in readme
    assert "## As-of contract" in readme
    assert "## Protocol" in readme
    assert "## Reproduce" in readme
    assert "No results have been written yet" in readme

    run_config = json.loads((experiment_root / "run_config.json").read_text(encoding="utf-8"))
    status = json.loads((experiment_root / "status.json").read_text(encoding="utf-8"))
    assert run_config["experiment_id"] == "0215"
    assert status["status"] == "initialized"

    selected_metadata = {
        "selected_cutoff": "23:59",
        "selected_model_id": "test_model",
        "selection_rule": "offline regression fixture",
        "official_rows_only_score": {
            "first_date": "2020-01-01",
            "last_date": "2023-12-31",
            "n": 100,
            "mae": 0.50,
            "rmse": 0.70,
            "median_abs_error": 0.40,
            "p90_abs_error": 1.00,
            "bias": 0.01,
        },
        "raw_official_baseline_same_cutoff": {"mae": 0.60, "rmse": 0.80},
        "promotion_gates": {"offline_test_gate": True},
    }
    mod.write_final_report(
        {"generated_at_utc": "2026-07-10T00:00:00Z"},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        selected_metadata,
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )

    markdown_paths = list(experiment_root.rglob("*.md"))
    assert markdown_paths == [experiment_root / "README.md"]
    completed_readme = markdown_paths[0].read_text(encoding="utf-8")
    assert "Complete. Validation artifacts" in completed_readme
    assert "## Conclusion" in completed_readme

    mod.initialize_experiment_folder()
    assert markdown_paths[0].read_text(encoding="utf-8") == completed_readme


def test_deep_sanity_audit_writes_machine_json_only(tmp_path, monkeypatch):
    mod = load_named_script("audit_tactical_gribstream_deep_sanity.py")
    output_name = "offline_test"
    output_dir = tmp_path / output_name
    output_dir.mkdir(parents=True)
    (output_dir / "progress.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(mod, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: Namespace(
            database_url="postgresql://offline", output_name=output_name, skip_file_hash=True
        ),
    )
    monkeypatch.setattr(mod, "read_csv_summary", lambda _path: {})
    monkeypatch.setattr(mod, "read_api_log", lambda _path: {"http_error_count": 0})
    monkeypatch.setattr(
        mod,
        "audit_db",
        lambda _database_url, _skip_file_hash: {
            "table_counts": {"forecast_wide": {"rows": 0}},
            "forecast_counts_by_source_dataset": [],
            "full_raw_file_checks": {
                "missing_files": [],
                "size_mismatch": [],
                "sha256_mismatch": [],
            },
        },
    )

    mod.main()

    assert (output_dir / "deep_sanity_audit_20260625.json").is_file()
    assert list(tmp_path.rglob("*.md")) == []


def test_cutoff_timestamp_is_t_minus_one_hkt_clock_time():
    mod = load_module()

    assert mod.cutoff_timestamp_hkt(pd.Timestamp("2023-08-15"), "23:59") == pd.Timestamp(
        "2023-08-14 23:59"
    )
    assert mod.cutoff_timestamp_hkt(pd.Timestamp("2023-08-15"), "17:00") == pd.Timestamp(
        "2023-08-14 17:00"
    )


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
