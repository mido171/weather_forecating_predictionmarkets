import pandas as pd

from pipelines.quantile_knn_conformal.leakage_audit import run_leakage_audit


def test_leakage_audit_detects_future_neighbor():
    all_rows = pd.DataFrame({
        'valid_time_utc': pd.to_datetime(['2024-01-01T00:00:00Z']),
        'valid_time_ny': pd.to_datetime(['2023-12-31T19:00:00-05:00']),
        'target_date_local': ['2023-12-31'],
        'KJFK_source_valid_time_utc': pd.to_datetime(['2024-01-01T00:30:00Z']),
    })
    dev = pd.DataFrame({'target_date_local': ['2022-01-01']})
    test = pd.DataFrame({'target_date_local': ['2024-01-01']})
    tune = pd.DataFrame({'target_date_local': ['2021-01-01']})
    knn_diag = pd.DataFrame({'query_date': ['2024-01-01'], 'neighbor_date': ['2024-01-01']})
    gate = pd.DataFrame({'target_date_local': ['2022-06-01']})
    conf = pd.DataFrame({'valid_time_utc': ['2024-01-01T00:00:00Z'], 'hist_len': [1]})
    folds = [{'fold': 1, 'train_max_date': '2023-12-31', 'pred_min_date': '2024-01-01'}]

    out = run_leakage_audit(all_rows, dev, test, tune, knn_diag, gate, conf, folds)
    assert not out.passed
