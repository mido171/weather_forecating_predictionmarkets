# Negative And Null Results

This file records constraints and rejected shortcuts discovered during Phase A/B.

| Finding | Status | Reason |
|---|---|---|
| Same-day daily climate labels as predictors | rejected | They are observed after/during target day and leak target-day information at T-1 15:00. |
| Reanalysis fields as operational features | rejected for now | Final products have release lag and are retrospective unless product-specific availability is proven. |
| Current-only NWP snapshots for backtest | rejected for now | They are prospective from acquisition start only and cannot support retrospective performance claims. |
| Official daily Tmax as feature | rejected | It is the target. |
| Public high-frequency HKO archive before 2020/2021 | unavailable in acquired archive | Existing public DATA.GOV historical ZIPs begin in 2020/2021 for the station feeds parsed here. |
