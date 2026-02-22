from datetime import date

import pandas as pd

from weather_ml.mos_config import KnnViewConfig
from weather_ml.mos_knn import compute_knn_features, prepare_distance_data


def test_knn_basic_counts_and_weights():
    df = pd.DataFrame(
        {
            "asof_date_local": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "base_tmax_blend": [70.0, 72.0, 74.0],
            "y_actual_tmax_f": [71.0, 73.0, 76.0],
        }
    )
    df["target_date_local"] = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]

    calib_mask = pd.Series([True, True, True]).to_numpy()
    distance = prepare_distance_data(
        df,
        ["base_tmax_blend"],
        calib_mask,
        missing_penalty=1.0,
        weight_map=None,
    )
    views = [KnnViewConfig("v0", "full", "l2")]
    knn_df, meta = compute_knn_features(
        df,
        distance,
        views,
        k=2,
        thresholds=[70],
        tau_fixed=[1.0],
        season_window=45,
        label_lag_days=0,
        consistency_features=["base_tmax_blend"],
        base_col="base_tmax_blend",
        target_col="y_actual_tmax_f",
    )

    assert knn_df.loc[2, "knn_v0_candidate_count"] == 2.0
    assert knn_df.loc[2, "knn_v0_k_used"] == 2.0
    assert abs(knn_df.loc[2, "knn_v0_weight_norm_sum"] - 1.0) < 1e-6
    assert meta["knn_zero_candidates"] == 1  # first row has zero candidates
