# Hypothesis

## Claim

If HKO publishes the `2026-06-18` Daily Extract row during this continuation
window, the ledger can identify it as a provider-first-publication candidate
because active absent snapshots have already been archived since
`2026-06-18T17:48:59.956593Z`.

## Mechanism

The monthly Daily Extract backing payload is append-like while the current
month is active. Continued immutable polling should either extend the absent
evidence or capture the first-present payload.

## Expected Direction

Either `2026-06-18` remains missing with zero candidates, or it appears as a
single provider-first candidate with active absence evidence and no revision.

## Falsification

The experiment fails if the poller cannot complete, raw snapshots are missing,
candidate status appears without active absence evidence, or the report implies
G1 is passed without first-present evidence.

## Leakage Risks

- Treating current latest payloads as historical first-publication proof.
- Inferring exact provider publication time beyond the absent/present bounds.
- Allowing target-publication evidence to enter a model feature table.
