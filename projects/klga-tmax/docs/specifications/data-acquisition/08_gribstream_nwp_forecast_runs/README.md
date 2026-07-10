# 08 GribStream NWP Forecast Runs

Source spec:

```text
03_gribstream_nwp_forecast_runs.md
```

Execution role:

This task covers high-value deterministic, ensemble, quantile, and audition NWP forecast runs from GribStream. It is ordered after smaller public feeds because the user noted that a temporary larger-request allowance is needed for bulk GribStream pulls.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Do not launch large historical GribStream pulls until the request allowance is resolved. When implementing requests, use the live GribStream catalog/API for exact dataset selectors and never invent selectors from memory.
