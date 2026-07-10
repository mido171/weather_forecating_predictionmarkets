from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_spring_transition_pressure_dew_specialist import (
    SpringSpec,
    compute_spring_correction,
    score_candidates,
    spring_phase,
)


def test_spring_phase_labels_target_month_halves() -> None:
    assert spring_phase(pd.Timestamp("2020-03-01")) == "early_mar"
    assert spring_phase(pd.Timestamp("2020-03-31")) == "late_mar"
    assert spring_phase(pd.Timestamp("2020-04-10")) == "early_apr"
    assert spring_phase(pd.Timestamp("2020-04-20")) == "late_apr"
    assert spring_phase(pd.Timestamp("2020-05-15")) == "early_may"
    assert spring_phase(pd.Timestamp("2020-05-16")) == "late_may"
    assert spring_phase(pd.Timestamp("2020-06-01")) == "outside_spring"


def test_spring_correction_excludes_current_row() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-03"]),
            "spring_month_bucket": ["march", "march", "march"],
            "spring_target_window": [True, True, True],
            "residual_to_add_c": [1.0, 3.0, 90.0],
        }
    )
    spec = SpringSpec(
        "test",
        ("spring_month_bucket",),
        min_prior_rows=1,
        shrinkage=0.0,
        cap_c=100.0,
    )

    corrections, prior_rows, raw_means = compute_spring_correction(frame, spec)

    assert corrections.tolist() == [0.0, 1.0, 2.0]
    assert prior_rows.tolist() == [0, 1, 2]
    assert raw_means[0] != raw_means[0]


def test_score_candidates_requires_all_post2006_folds_to_improve() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2007-03-01", "2013-03-01", "2019-03-01", "2019-06-01"]),
            "target_tmax_c": [20.0, 20.0, 20.0, 20.0],
            "global_bias_repaired_prediction_c": [18.0, 18.0, 18.0, 20.0],
            "candidate_prediction_c": [20.0, 20.0, 17.0, 20.0],
            "spring_residual_correction_c": [2.0, 2.0, -1.0, 0.0],
            "spring_target_window": [True, True, True, False],
            "fold_id": ["fold_2006_2011", "fold_2012_2017", "fold_2018_2023", "fold_2018_2023"],
            "correction_id": ["candidate", "candidate", "candidate", "candidate"],
            "group_columns": ["test", "test", "test", "test"],
            "window_days": [float("nan"), float("nan"), float("nan"), float("nan")],
            "min_prior_rows": [1, 1, 1, 1],
            "shrinkage": [0.0, 0.0, 0.0, 0.0],
            "cap_c": [2.0, 2.0, 2.0, 2.0],
        }
    )

    scoreboard, folds = score_candidates(predictions)

    assert not bool(scoreboard.iloc[0]["promotion_gate_passed"])
    assert scoreboard.iloc[0]["post2006_folds_improved"] == 2
    assert len(folds) == 3
