# As-Of Contract

EXP-0006 creates target-publication evidence rows only. It does not create
forecast rows or features.

## Active Watch Window

- inherited active polling start: `2026-06-18T17:48:59.956593Z`
- watched local date: `2026-06-18`
- evidence bound: provider publication can only be bounded between the last
  active absent snapshot and first active present snapshot.

## Candidate Rule

`PROVIDER_FIRST_PUBLICATION_CANDIDATE` is allowed only when:

- the local date is explicitly watched;
- the inherited active polling start is timezone-aware;
- at least one raw monthly payload after active start and before first presence
  proves the date was absent;
- a later raw payload proves first presence;
- no later value revision is observed.

## Timestamp Semantics

| Field | Meaning | Source |
|---|---|---|
| valid_at | Hong Kong local Daily Extract date | parsed row |
| published_at | not provider-declared; bounded by absent/present archive pair | inference from raw snapshots |
| available_at | successful archive retrieval time | raw sidecar `retrieved_at` |
| retrieved_at | successful archive retrieval time | raw sidecar `retrieved_at` |

## Forbidden Data

- CLMMAXT as a substitute for first-published Daily Extract.
- Any first-publication claim without active absence evidence.
- Predictive model features or target labels.
- Market prices, books, trades, liquidity, or execution data.
