from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = REPO_ROOT / "experiments" / "0104_canonical_frame_evidence_registry" / "run.py"


def load_run_module():
    spec = importlib.util.spec_from_file_location("hkg_t24_0104_run", RUN_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_row_hash_is_stable_and_sensitive_to_key_parts() -> None:
    module = load_run_module()

    first = module.make_row_hash("official", "2023-01-01", "rss_archive", "v1")
    second = module.make_row_hash("official", "2023-01-01", "rss_archive", "v1")
    changed = module.make_row_hash("official", "2023-01-01", "press_archive", "v1")

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_score_predictions_reports_standard_error_metrics() -> None:
    module = load_run_module()
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "target_tmax_c": [20.0, 22.0, 24.0],
            "prediction_c": [21.0, 21.0, 27.0],
        }
    )

    scored = module.score_predictions(frame, "prediction_c")

    assert scored["n"] == 3
    assert math.isclose(float(scored["mae"]), 5.0 / 3.0)
    assert math.isclose(float(scored["bias"]), 1.0)
    assert scored["first_date"] == "2023-01-01"
    assert scored["last_date"] == "2023-01-03"


def test_compute_date_gaps_finds_calendar_holes() -> None:
    module = load_run_module()

    gaps = module.compute_date_gaps(
        pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"]),
        "F-SAMPLE",
    )

    assert len(gaps) == 1
    assert gaps.iloc[0]["frame_id"] == "F-SAMPLE"
    assert gaps.iloc[0]["gap_start"] == "2023-01-03"
    assert gaps.iloc[0]["gap_end"] == "2023-01-04"
    assert int(gaps.iloc[0]["missing_days"]) == 2
