from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_residual_specialist_design_queue import (
    build_design_queue,
    build_test_protocol,
    leakage_audit,
)


def sample_failure_rank() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_name": "fold",
                "fold_id": "fold_2018_2023",
                "n": 2190,
                "mae": 1.58,
                "bias": -0.87,
            },
            {
                "analysis_name": "season_x_heat",
                "season": "DJF",
                "heat_bucket_pre2000_target": "mid",
                "n": 374,
                "mae": 2.60,
                "bias": -2.55,
            },
            {
                "analysis_name": "season_x_heat",
                "season": "JJA",
                "heat_bucket_pre2000_target": "mid",
                "n": 263,
                "mae": 2.17,
                "bias": 2.14,
            },
            {
                "analysis_name": "month",
                "month": "2.0",
                "n": 677,
                "mae": 1.97,
                "bias": -0.88,
            },
            {
                "analysis_name": "season",
                "season": "MAM",
                "n": 2208,
                "mae": 1.64,
                "bias": -0.24,
            },
            {
                "analysis_name": "pressure_pair_spread_bucket",
                "pressure_pair_spread_bucket": "high",
                "n": 2849,
                "mae": 1.46,
                "bias": -0.25,
            },
        ]
    )


def test_design_queue_blocks_diagnostic_target_heat_candidates() -> None:
    feature_corr = pd.DataFrame(
        [
            {
                "feature": "stat_590960_99999_air_temperature_c_latest_before_1500",
                "n_abs_error_corr": 8743,
                "corr_abs_error": -0.09,
            }
        ]
    )

    queue = build_design_queue(sample_failure_rank(), feature_corr)

    heat_rows = queue[queue["diagnostic_inputs_forbidden_in_model"].str.contains("target heat bucket", case=False)]
    assert not heat_rows.empty
    assert heat_rows["leakage_status"].eq("requires_proxy_validation").all()


def test_ready_queue_rows_require_fold_local_protocol() -> None:
    queue = build_design_queue(sample_failure_rank(), pd.DataFrame())
    protocol = build_test_protocol(queue)
    audit = leakage_audit(queue)

    assert queue["ready_for_training_now"].sum() > 0
    assert protocol["split_policy"].str.contains("OOF").all()
    assert audit["passed"].astype(bool).all()
