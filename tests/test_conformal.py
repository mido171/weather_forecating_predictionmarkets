import pandas as pd

from pipelines.quantile_knn_conformal.conformal import apply_rolling_conformal, init_conformal_state


def test_conformal_warmup_and_chronology():
    rows = pd.DataFrame({
        'target_date_local': pd.date_range('2024-01-01', periods=8, freq='D').date,
        'valid_time_utc': pd.date_range('2024-01-01 18:00:00+00:00', periods=8, freq='D'),
        'y_tmax': [30, 31, 32, 33, 34, 35, 36, 37],
    })
    pred = pd.DataFrame({
        'q_0.010': [25]*8, 'q_0.025': [26]*8, 'q_0.050': [27]*8, 'q_0.100': [28]*8,
        'q_0.200': [29]*8, 'q_0.300': [30]*8, 'q_0.400': [31]*8, 'q_0.500': [32]*8,
        'q_0.600': [33]*8, 'q_0.700': [34]*8, 'q_0.800': [35]*8, 'q_0.900': [36]*8,
        'q_0.950': [37]*8, 'q_0.975': [38]*8, 'q_0.990': [39]*8,
    })
    qs = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]

    st = init_conformal_state(window=365, min_warmup=3)
    corr, diag = apply_rolling_conformal(rows, pred, qs, st, update_state=True)
    assert len(corr) == 8
    assert diag['hist_len'].is_monotonic_increasing
    assert bool(diag['conformal_warmup'].iloc[0])
