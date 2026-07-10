# 02 Wunderground Settlement Actuals

Source spec:

```text
01_wunderground_settlement_actuals.md
```

Execution role:

This task is the first external data source because Wunderground is the settlement-source actuals backbone for target labels and historical verification.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Fetch KLGA first, then the surrounding station list from the station registry. Preserve raw provider responses and store revision-aware daily actuals according to the main strategy and supplemental patch.
