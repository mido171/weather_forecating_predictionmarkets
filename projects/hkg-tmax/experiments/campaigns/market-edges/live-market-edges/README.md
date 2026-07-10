# HKG Tmax market-edge snapshot: 2026-07-06

Status: `historical_analysis_only`. No orders were placed.

## Event and forecast

- Event: “Highest temperature in Hong Kong on July 6?”
- Target date: 2026-07-06.
- HKO forecast update: 2026-07-06 01:05 HKT.
- HKO forecast range: 27.0-31.0 C.
- Model: `B4_hierarchical_residual_pmf` trained on 9,644 rows from
  2000-01-02 through 2026-05-31.

## Bucket probabilities

| Bucket | Probability |
|---|---:|
| 27 | 0.80% |
| 28 | 4.85% |
| 29 | 10.71% |
| 30 | 24.67% |
| 31 | 28.12% |
| 32 | 22.83% |
| 33 | 6.40% |
| 34 or higher | 1.63% |

Buckets at or below 26 were each below 0.01%.

## Best reported edge

Buy No on bucket 32 at 58.00c; model fair 77.17c; edge +19.17 percentage
points; classification `ELITE`. This is a frozen model-versus-market
calculation, not current advice or execution authority.

## Rounding contract

31.9 C remains bucket 31; 32.0 C begins bucket 32; 34.0 C and above belongs to
`34_or_higher`.

## Evidence

`latest_edge_report.json` preserves the event URL, complete PMF, prices,
forecast timestamp, alpha-selection evidence, edges, and no-order statement.
The retired Markdown snapshot is recoverable through
[`DOCUMENT_PROVENANCE.csv`](../../DOCUMENT_PROVENANCE.csv).
