# V5+8 Calibration Comparison

- best_method: quantile_residual

## conformal_residual
### val
- crps: 0.863115
- PIT mean=0.5000 std=0.2887 chi2=0.08
- p50: coverage=0.500 avg_width=0.606
- p80: coverage=0.799 avg_width=2.382
- p90: coverage=0.898 avg_width=3.957
- pinball: 0.05:0.1694, 0.1:0.2501, 0.25:0.3445, 0.5:0.3477, 0.75:0.3172, 0.9:0.2082, 0.95:0.1305

### test
- crps: 0.861204
- PIT mean=0.4897 std=0.2911 chi2=8.84
- p50: coverage=0.484 avg_width=0.607
- p80: coverage=0.786 avg_width=2.397
- p90: coverage=0.907 avg_width=4.014
- pinball: 0.05:0.1582, 0.1:0.2439, 0.25:0.3496, 0.5:0.3452, 0.75:0.3109, 0.9:0.2060, 0.95:0.1349

## quantile_residual
### val
- crps: 0.558737
- PIT mean=0.4995 std=0.2722 chi2=342.39
- p50: coverage=0.548 avg_width=0.804
- p80: coverage=0.795 avg_width=2.394
- p90: coverage=0.877 avg_width=3.478
- pinball: 0.05:0.1647, 0.1:0.2437, 0.25:0.3462, 0.5:0.3479, 0.75:0.3195, 0.9:0.2039, 0.95:0.1254

### test
- crps: 0.548084
- PIT mean=0.4923 std=0.2733 chi2=311.04
- p50: coverage=0.541 avg_width=0.806
- p80: coverage=0.791 avg_width=2.426
- p90: coverage=0.880 avg_width=3.492
- pinball: 0.05:0.1479, 0.1:0.2301, 0.25:0.3454, 0.5:0.3450, 0.75:0.3082, 0.9:0.2026, 0.95:0.1276
