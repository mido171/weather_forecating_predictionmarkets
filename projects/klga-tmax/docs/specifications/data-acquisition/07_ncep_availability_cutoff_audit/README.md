# 07 NCEP Availability Cutoff Audit

Source spec:

```text
09_ncep_availability_cutoff_audit.md
```

Execution role:

This task records production-status timing evidence for cutoff auditing. It is not a predictor source; it supports leakage-safe availability validation.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Poll and store NCEP/NCO production status evidence on the configured cadence. Keep these records out of model predictors unless a later spec explicitly promotes derived availability diagnostics.
