# Threat Model and Failure Modes

## Research threats

### Wrong target

**Failure:** model predicts CLMMAXT latest but contract uses first Daily Extract publication.

**Controls:** G1 parity, first-publication archive, rules hash, resolved-event fixtures.

### Leakage

**Failure:** future cycle, corrected observation, final best track, or reanalysis enters features.

**Controls:** five timestamps, row-level availability validation, independent auditor, negative controls.

### Overfitting through experimentation

**Failure:** thousands of hypotheses produce a lucky validation gain.

**Controls:** experiment registry, family declaration, locked test, FDR awareness, live shadow.

### Nonstationarity

**Failure:** urbanization, climate trend, instrumentation, source/model upgrades.

**Controls:** metadata timelines, rolling windows, trend analysis, drift monitoring, version indicators.

### Sample-selection bias

**Failure:** hard days are absent because a source was missing.

**Controls:** coverage report, common-sample metrics, missingness diagnostics.

### Metric gaming

**Failure:** MAE improves while calibration/log loss deteriorates.

**Controls:** distributional primary metrics and guardrails.

## Data threats

- provider endpoint/schema changes;
- HTML error page with status 200;
- partial model files;
- clock skew;
- duplicate or out-of-order WebSocket events;
- station code/name mismatch;
- unit changes;
- source outage;
- corrupted archive;
- licensing restriction.

Controls include hashes, schema/range checks, completeness manifests, UTC clocks, backups, and fail-closed adapters.

## Market threats

- rules/source changes;
- ambiguous/disputed resolution;
- fee changes;
- thin books;
- apparent mid-price edge with no fills;
- adverse selection;
- stale quotes;
- token mapping error;
- market cancellation;
- correlated inventory;
- other participants adapt.

## Model threats

- regime collapse;
- overconfident tails;
- ensemble underdispersion;
- official forecast methodology change;
- NWP upgrade;
- feature source latency drift;
- model calibration decay;
- high leverage from a few dates.

## Operational threats

- credentials leak;
- code version mismatch;
- wrong timezone/date;
- forecast generated after cutoff;
- stale source used silently;
- alert failure;
- scheduler failure;
- disk full;
- network partition.

## Sensor-integrity anomalies

A single implausible observation should trigger monitoring, not opportunistic assumptions. Compare network consistency, provider quality flags, and subsequent official publication. Never interact with or attempt to influence physical sensors.

## Kill-switch hierarchy

1. unknown rules/source/target;
2. time or archive integrity failure;
3. stale/missing critical sources;
4. station anomaly;
5. forecast/model validation failure;
6. book synchronization failure;
7. risk limit breach.

Any higher-priority failure forces no-trade regardless of apparent edge.
