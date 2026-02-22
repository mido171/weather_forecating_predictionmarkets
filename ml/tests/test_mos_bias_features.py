from datetime import date

import pandas as pd

from weather_ml.mos_bias_features import compute_bias_features
from weather_ml.mos_config import MosDatasetConfig


def test_bias_best_model_selection():
    df = pd.DataFrame(
        {
            "target_date_local": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "y_actual_tmax_f": [10.0, 12.0, 14.0],
            "base_tmax_gfs": [10.0, 12.0, 14.0],
            "base_tmax_nam": [0.0, 0.0, 0.0],
            "base_tmax_blend": [5.0, 6.0, 7.0],
        }
    )
    cfg = MosDatasetConfig(
        station_id="KMIA",
        station_zoneid="America/New_York",
        feature_version="test",
        build_start_asof=date(2007, 1, 1),
        output_start_asof=date(2010, 1, 1),
        end_asof=date(2010, 1, 10),
        obs_cutoff_lag_days=0,
        models=["GFS", "NAM"],
        variables=[],
        bias_windows_days=[2],
    ).normalized()

    out = compute_bias_features(df, cfg)
    assert out.loc[2, "bias_best_is_gfs_2"] == 1.0
