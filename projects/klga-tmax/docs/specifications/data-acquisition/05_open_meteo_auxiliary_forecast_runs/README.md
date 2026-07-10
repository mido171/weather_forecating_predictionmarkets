# 05 Open-Meteo Auxiliary Forecast Runs

Source spec:

```text
06_open_meteo_auxiliary_forecast_runs.md
```

Execution role:

This task is placed before GribStream bulk work because it can provide auxiliary exact-run forecast coverage while the larger GribStream request allowance is pending.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Fetch single-run families and model update metadata only under the source-family distinctions in the acquisition spec. Keep historical-forecast and previous-run products separate from exact as-of runs unless the spec proves equivalence.
