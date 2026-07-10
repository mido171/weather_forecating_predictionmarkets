# Protocol

1. Load DB source-of-truth target labels, lead-1 HKO historical forecasts, lead-0 diagnostics, and HKO daily climate.
2. Build one row per `(target_date, cutoff)` for all four cutoffs.
3. Engineer GPT-Pro requested feature families: latest official state, revision sequence, fold-safe residual history, T-2 target history, T-2 climate state, seasonal climatology, coastal/weather regimes, analog residuals, and interactions.
4. Run expanding yearly walk-forward validation for `2011` through `2023`. Each validation year trains on all prior years only.
5. Score baselines and candidate residual models on official rows, all rows, and identical cutoff intersections.
6. Select a production point-forecast strategy by the strict cutoff/model rule, with 23:59 HKT as the default cutoff unless an earlier cutoff clearly improves MAE without RMSE damage.
