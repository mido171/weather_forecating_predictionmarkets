# HKG T-24 campaign

This campaign contains two historical records. Their cutoffs and conclusions
must not override the current binding HKG T-24 contract.

| Experiment | Status | Main conclusion |
|---|---|---|
| [0214 tactical H24N GribStream backfill](0214_tactical_h24n_gribstream_backfill/README.md) | Audited with warnings | Large tactical archive exists, but source-specific quality filters are mandatory |
| [0215 GPT-Pro point forecast strategy](0215_gpt_pro_point_forecast_strategy/README.md) | Historical candidate; no promotion | B3 improved raw official MAE by only 0.00588 C and failed promotion gates |

## Authority warning

Experiment 0215 evaluated late T-1 cutoffs through 23:59 HKT. The current
operational contract is governed elsewhere and uses a 15:00 HKT decision
cutoff with an earlier operational freeze. Treat 0215 as historical evidence,
not as the deployable H24N champion.

Machine artifacts remain inside each experiment. Retired prose is recoverable
through [the campaign provenance ledger](../DOCUMENT_PROVENANCE.csv).
