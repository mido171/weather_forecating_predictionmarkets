# Operational Runbook

## Daily research/shadow sequence

### Before cutoff

1. verify clock synchronization;
2. fetch/validate current market rules and rules hash;
3. confirm target date/timezone;
4. verify all expected NWP cycles and forecast vintages;
5. inspect source freshness/quality;
6. freeze the input manifest.

### At cutoff

1. record exact cutoff;
2. select records with `available_at <= cutoff`;
3. generate forecast with effective model version;
4. save distribution, quantiles, buckets, inputs, hashes;
5. timestamp prediction;
6. do not regenerate under the same prediction ID.

### After forecast

1. archive market book/price;
2. paper-trade decision through cost/risk layer;
3. continue source archive;
4. record operational incidents.

### After target publication

1. archive first Daily Extract publication;
2. compute/verify winning bucket;
3. store target and source hash;
4. score forecast;
5. update live dashboard;
6. investigate anomalies without rewriting prediction.

## Incident categories

- rules changed;
- target publication missing/revised;
- source stale;
- model cycle incomplete;
- station anomaly;
- forecast runner failed;
- market feed desynchronized;
- disk/clock/network failure.

Each incident gets an immutable record with start/end, impact, action, and prevention.

## Recovery

- raw archive is append-only;
- derived data can be rebuilt from raw + code + config;
- predictions and first-published targets are backed up;
- test restore periodically;
- never resolve an incident by silently editing history.
