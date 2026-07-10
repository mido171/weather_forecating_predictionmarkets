# HKO Station-Network Research Plan

## Target distinction

The target is one authoritative daily maximum, not the territory-wide maximum. The network is used to understand and forecast the target station.

## Metadata first

For every station:

- official code and name;
- coordinates/elevation;
- environment classification;
- elements/cadence;
- start/end;
- relocations/instrument changes;
- missingness;
- distance/bearing/elevation relative to target.

Do not merge records across station changes without a versioned bridge.

## Spatial group concepts

- target-near urban;
- harbour/coastal;
- inland/northern New Territories;
- eastern marine-exposed;
- western Pearl River influence;
- elevated;
- airport/outlying island.

Groups are hypotheses and may be refined through metadata and covariance analysis.

## Classical analyses

1. pairwise and conditional correlations;
2. HKO-minus-station residual distributions;
3. wind-sector-conditioned relationships;
4. month/regime conditioned relationships;
5. lead-lag analysis using only pre-cutoff history;
6. principal spatial modes fit rolling training-only;
7. graph smoothing/kriging with physical covariates;
8. analogue matching of network state;
9. break detection;
10. source/station reliability ranking.

## Potential pre-event features

At H24N, target date’s live network is not yet available. Useful information includes:

- T-1 daytime network pattern at cutoff;
- recent overnight/daytime anomalies;
- persistence of regional gradients;
- prior-day urban heat retention;
- initial state for model bias correction;
- model forecast of next-day network gradients.

## Later intraday features

Only after pre-event system:

- HKO heating slope;
- neighboring station heating lead;
- arrival of marine cooling;
- localized rain-cooled gradients;
- spatial max-so-far pattern;
- network-conditioned remaining-heating distribution.

## Guardrails

- no target-day future station readings;
- no interpolation using later values;
- station missingness not silently filled;
- transformations fit on prior data only;
- station code verified against official table;
- network gain evaluated after data availability costs.
