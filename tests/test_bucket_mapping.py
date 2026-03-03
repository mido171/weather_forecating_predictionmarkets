import numpy as np
import pandas as pd

from pipelines.quantile_knn_conformal.cdf_bucket_mapper import Bucket, integer_pmf_to_bucket_probs, quantile_rows_to_integer_pmf


def test_bucket_pmf_sums_to_one():
    pred = pd.DataFrame({
        'q_0.010': [10.0],
        'q_0.025': [12.0],
        'q_0.050': [14.0],
        'q_0.100': [16.0],
        'q_0.200': [20.0],
        'q_0.300': [24.0],
        'q_0.400': [27.0],
        'q_0.500': [30.0],
        'q_0.600': [33.0],
        'q_0.700': [36.0],
        'q_0.800': [40.0],
        'q_0.900': [45.0],
        'q_0.950': [49.0],
        'q_0.975': [52.0],
        'q_0.990': [55.0],
    })
    qs = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]
    pmf = quantile_rows_to_integer_pmf(pred, qs)
    pcols = [c for c in pmf.columns if c.startswith('p_int_')]
    assert np.isclose(float(pmf[pcols].sum(axis=1).iloc[0]), 1.0, atol=1e-8)

    buckets = [Bucket('L', None, 30), Bucket('M', 31, 40), Bucket('H', 41, None)]
    b = integer_pmf_to_bucket_probs(pmf, buckets)
    yes_cols = [c for c in b.columns if c.startswith('bucket_yes::')]
    assert np.isclose(float(b[yes_cols].sum(axis=1).iloc[0]), 1.0, atol=1e-8)
