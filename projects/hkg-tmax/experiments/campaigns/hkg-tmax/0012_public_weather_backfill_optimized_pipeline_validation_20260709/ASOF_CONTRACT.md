# As-Of Contract

Model issues use `issued_at_utc + 6h` as a conservative availability proxy unless a stronger
provider timestamp is captured. `valid_at_utc` is the model forecast valid time and must not be
used as the information-availability time.

Himawari issues use the later of native HSD file creation time and `observed_at_utc + 30m`.
Source-side 404 rows retain this conservative expected availability and have no feature rows.

Every `station_feature` and `area_feature` row carries the same `available_at_utc` as its
`source_issue`. Leakage-safe consumers must enforce:

```sql
available_at_utc <= cutoff_utc
```

The accepted experiment 0012 robustness scope excludes radar.
