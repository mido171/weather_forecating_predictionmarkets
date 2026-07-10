# 10 NOAA Raw Archives Optional Bulk Fallback

Source spec:

```text
07_noaa_raw_archives_optional_bulk_fallback.md
```

Execution role:

This task is last because it is an optional fallback path for bulk historical archives when primary provider paths are insufficient, too expensive, or unavailable.

Persistence target:

```text
postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Use only when the primary data-provider path cannot satisfy the required history, coverage, or auditability. Keep raw archive provenance distinct from GribStream and other provider-normalized sources.
