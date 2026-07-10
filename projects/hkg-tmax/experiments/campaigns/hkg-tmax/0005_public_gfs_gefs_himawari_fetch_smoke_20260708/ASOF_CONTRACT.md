# As-Of Contract

This smoke does not score a model. It verifies provider access and timestamp fields.

For GFS/GEFS:

```text
issuedAt = model cycle / run initialization time
validAt = issuedAt + forecast lead
availableAt for strict H24N backtests = issuedAt + configured publication buffer
default buffer = 6 hours unless a provider-specific release audit proves a tighter value
eligible if availableAt <= target_date T-1 15:00 HKT
```

For Himawari:

```text
observedAt = image timestamp parsed from object key
availableAt proxy in this smoke = S3 LastModified
eligible if observedAt + latency buffer <= target_date T-1 15:00 HKT
```
