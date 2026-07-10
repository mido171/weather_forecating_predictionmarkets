# HKG Tmax Polymarket Edge Report

Event: `Highest temperature in Hong Kong on July 6?`
URL: https://polymarket.com/event/highest-temperature-in-hong-kong-on-july-6-2026
Target date: `2026-07-06`

## Latest HKO Forecast

- Source: `HKO fnd 9-day forecast`
- Update time HKT: `2026-07-06T01:05:00+08:00`
- Forecast min: `27.0` C
- Forecast max: `31.0` C

## Champion Model

- Method: `B4_hierarchical_residual_pmf`
- Training rows: `9644`
- Training span: `2000-01-02` to `2026-05-31`
- Month alpha: `20.0`
- Cell alpha: `10.0`

## Bucket Probabilities

| Bucket | Model probability |
|---|---:|
| `24_or_below` | <0.01% |
| `25` | <0.01% |
| `26` | <0.01% |
| `27` | 0.80% |
| `28` | 4.85% |
| `29` | 10.71% |
| `30` | 24.67% |
| `31` | 28.12% |
| `32` | 22.83% |
| `33` | 6.40% |
| `34_or_higher` | 1.63% |

## Positive Edges

| Rank | Side | Bucket | Market price | Model fair | Edge | ROI on cost | Class |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Buy No | `32` | 58.00c | 77.17c | +19.17pp | 33.1% | ELITE |
| 2 | Buy Yes | `30` | 14.00c | 24.67c | +10.67pp | 76.2% | very good |
| 3 | Buy Yes | `29` | 1.90c | 10.71c | +8.81pp | 463.7% | very good |
| 4 | Buy No | `31` | 66.00c | 71.88c | +5.88pp | 8.9% | good |
| 5 | Buy No | `33` | 88.80c | 93.60c | +4.80pp | 5.4% | normal |
| 6 | Buy Yes | `28` | 0.30c | 4.85c | +4.55pp | 1515.1% | normal |
| 7 | Buy Yes | `27` | 0.20c | 0.80c | +0.60pp | 299.0% | normal |
| 8 | Buy No | `34_or_higher` | 98.30c | 98.37c | +0.07pp | 0.1% | normal |

## Best Edge

Buy No `32` at 58.00c; model fair 77.17c; edge +19.17pp; classification `ELITE`.

## Rounding Contract

`31.9C` stays bucket `31`; `32.0C` starts bucket `32`; `34.0C+` is `34_or_higher`.

No orders were placed. This is model-vs-market analysis only.
