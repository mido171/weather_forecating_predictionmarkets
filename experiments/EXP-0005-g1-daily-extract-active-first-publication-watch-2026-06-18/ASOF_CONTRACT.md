# As-Of Contract

EXP-0005 does not create forecast rows. It creates target-publication evidence
rows for G1.

## Candidate Rule

`PROVIDER_FIRST_PUBLICATION_CANDIDATE` is allowed only when all are true:

- `local_date` is explicitly listed in `--watch-candidate-date`;
- `--active-polling-start-at` is timezone-aware;
- a raw monthly payload at or after active start and before first presence
  proves the watched local date was absent;
- a later raw monthly payload proves first presence;
- the first-present value has not been revised in later archived payloads.

## Timestamp Semantics

| Field | Meaning | Source |
|---|---|---|
| valid_at | Hong Kong local calendar date represented by the Daily Extract row | parsed row |
| published_at | not provider-declared; bounded only by absent/present archive pair | inferred from immutable snapshots |
| available_at | successful archive retrieval time for each raw payload | raw sidecar `retrieved_at` |
| retrieved_at | same as archive retrieval time | raw sidecar `retrieved_at` |

## Eligibility Rule

Only raw snapshots retrieved by this repository may support candidate status.
Latest-only historical payloads can support equality checks but cannot prove
first publication.

## Forbidden Data

- finalized CLMMAXT as a substitute for first-published Daily Extract;
- a first-present snapshot without a prior active absent snapshot;
- market prices, books, trades, or execution data;
- predictive features or model outputs.
