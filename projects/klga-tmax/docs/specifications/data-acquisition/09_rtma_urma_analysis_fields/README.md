# 09 RTMA URMA Analysis Fields

Source spec:

```text
05_rtma_urma_analysis_fields.md
```

Execution role:

This task captures analysis fields for current-state estimation, retrospective diagnostics, and research comparisons. It is placed near GribStream because the practical retrieval path may share the same bulk-access constraints.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Keep RTMA live/current-state usage separate from URMA retrospective/research usage. Apply the availability contract before any analysis field is allowed into cutoff-sensitive features.
