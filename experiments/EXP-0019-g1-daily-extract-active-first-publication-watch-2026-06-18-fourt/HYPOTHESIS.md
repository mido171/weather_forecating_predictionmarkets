# Hypothesis

## Mechanism

G1 requires point-in-time evidence for when HKO first publishes the Daily
Extract value that can act as settlement truth for HKG Tmax. Repeated immutable
poll snapshots can prove an absent-to-present transition only if the watched
row is absent in earlier snapshots and appears in a later public provider
payload.

## Exact prediction

The fourteenth continuation will either:

- keep `2026-06-18` absent from the June 2026 Daily Extract monthly payload; or
- capture the first provider-public candidate for `2026-06-18`, including the
  raw payload, metadata sidecar, HTTP metadata, and derived ledger update.

No forecast distribution, feature set, model, or market rule is evaluated.

## Null hypothesis

The watched row remains absent, or any observed row cannot be tied to a clean
absent-before-present provider-public transition.

## Falsification

The experiment cannot support G1 if:

- immutable raw snapshots or metadata sidecars are missing;
- snapshot hashes do not match sidecars;
- HTTP metadata is incomplete;
- the watched row remains absent;
- the row appears without prior absence evidence from this active watch family;
- the provider payload is ambiguous, revised without trace, or date parsing is
  uncertain.

## Novelty and prior evidence

EXP-0003 through EXP-0018 established polling, first-observed ledger mechanics,
retry-backed fetching, and active absence evidence through
`2026-06-18T19:42:55.501790Z`. EXP-0019 extends that same live watch.

## Leakage risks anticipated

This experiment inspects target-publication evidence only. The evidence is not
eligible as a predictive feature, cannot be used for horizon selection, and
does not open G2, modelling, machine learning, or market backtesting.
