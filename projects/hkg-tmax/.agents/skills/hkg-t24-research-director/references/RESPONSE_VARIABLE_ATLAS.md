# Response Variable Atlas

Every experiment must name one primary response before scoring. Secondary responses are allowed for predeclared diagnostics. A result on one response cannot be presented as proof for another.

## 1. Raw target Tmax

Name: `target_tmax_c`

Definition: canonical HKO daily maximum temperature for target date T.

Use:

- direct-model diagnostics;
- climatology and target-memory research;
- station-level physical relationship mapping.

Caution:

- strong correlation may duplicate the official forecast;
- target-derived features must exclude T;
- direct-model performance is not directly comparable to anchor residual performance.

## 2. Official signed residual

Name: `official_residual_c`

Preferred sign convention:

```text
actual_hko_tmax_c - official_tmax_c
```

Positive means the official forecast was too cool; negative means it was too warm.

Use:

- residual correction;
- source bias;
- station or target-state blind spots;
- signed specialists.

Always state sign convention in every artifact.

## 3. Absolute official residual

Name: `official_abs_error_c`

Definition:

```text
abs(actual_hko_tmax_c - official_tmax_c)
```

Use:

- uncertainty;
- forecast trust;
- abstention;
- high-error prediction.

A feature can predict magnitude without direction.

## 4. Overforecast response

Names:

- `official_overforecast_c = max(official_tmax_c - actual_tmax_c, 0)`
- `official_overforecast_flag = 1[official_tmax_c > actual_tmax_c + threshold]`

Use:

- marine/humidity/cloud suppression;
- source overforecast streaks;
- cold or suppressed regimes.

Threshold must be predeclared or fitted on prior data.

## 5. Underforecast response

Names:

- `official_underforecast_c = max(actual_tmax_c - official_tmax_c, 0)`
- `official_underforecast_flag = 1[actual_tmax_c > official_tmax_c + threshold]`

Use:

- weak-wind heat buildup;
- subsidence;
- cool-surge breakdown;
- rapid warm transitions.

## 6. High-error flag

Name: `official_high_error_flag`

Definition based on a predeclared absolute-error threshold, such as 1.5 °C, or a fold-training quantile.

Use:

- tail detection;
- uncertainty;
- specialist routing.

Report prevalence and precision/recall. Do not optimize the threshold on outer-fold outcomes.

## 7. Hot-day underforecast flag

Definition combines:

- actual target above a prior-defined hot threshold or causal seasonal percentile;
- official residual positive beyond a material threshold.

Use:

- capture costly hot misses;
- heat-potential specialists.

Thresholds must be prior-only.

## 8. Cold-day overforecast flag

Definition combines:

- actual target below a prior-defined cool threshold or causal seasonal percentile;
- official residual negative beyond a material threshold.

Use:

- cool surge, cloud/rain, or marine-suppression specialists.

## 9. MAM high-error flag

Definition:

- target month in March, April, or May;
- official absolute error above a predeclared threshold.

Use:

- transition specialist discovery.

Also examine signed MAM residual because spring errors may be asymmetric by regime.

## 10. Station-only model residual

Definition:

```text
actual_tmax_c - station_only_oof_prediction_c
```

Use:

- discover target-memory or anchor information missing from the station array;
- avoid repeatedly adding station information already captured.

The station prediction must be out of fold.

## 11. Online-memory residual

Definition:

```text
actual_tmax_c - (official_tmax_c + online_memory_correction_c)
```

Use:

- test whether station/target/regime signals add beyond source-aware bias memory.

This is a critical promotion response.

## 12. Current champion residual

Definition:

```text
actual_tmax_c - current_champion_oof_prediction_c
```

Use:

- residual-of-residual anatomy;
- select the next incremental lane.

The champion predictions must be frozen and out of fold on the same canonical frame.

## 13. Forecast trust state

Possible continuous response:

- expected absolute error;
- probability official raw beats the correction model;
- prior rolling relative loss of official versus alternative.

Use:

- router;
- abstention;
- blend weight.

A trust label derived from realized errors is training-only and must not appear as a live feature.

## 14. Expert selection response

For experts A and B:

```text
expert_a_wins = 1[abs(error_a) < abs(error_b)]
```

Use:

- routing diagnostics.

Caution: hard winner labels are noisy. Compare with continuous loss difference and fixed blends.

## 15. Threshold event responses

Examples:

- `1[target_tmax_c >= 30.5]`
- market bucket membership;
- rounded official-resolution outcomes.

Use later for calibrated probabilities. The threshold must match the intended operational or settlement definition. Probability evaluation requires Brier score, log loss, calibration, and discrimination—not MAE alone.

## 16. Information-gain-only diagnostic responses

Examples:

- blocked upper-air regime;
- retrospective marine suppression state;
- physical cluster labels.

These may be used to train or evaluate safe proxies, but cannot become production targets whose value is unavailable live unless the proxy output is based entirely on safe inputs and evaluated against the final forecast response.

## Required reporting for every response

- exact formula;
- sign;
- units;
- target availability;
- row count;
- date range;
- prevalence for flags;
- relationship to baseline;
- whether it is primary or secondary;
- whether it can be computed only after target resolution;
- how it is used during training versus live inference.
