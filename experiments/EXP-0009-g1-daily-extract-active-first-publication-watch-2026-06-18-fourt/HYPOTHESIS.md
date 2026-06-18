# Hypothesis

## Mechanism

HKO Daily Extract monthly payloads are mutable during the current month. Repeated
immutable raw snapshots can establish whether a watched local date was absent
before later first appearance. The first snapshot that contains the watched row,
combined with at least one post-active-start absent snapshot, is the required
evidence pattern for a provider-first-publication candidate.

## Exact prediction

If HKO publishes the `2026-06-18` Daily Extract row during this watch, the
publication ledger will emit one `PROVIDER_FIRST_PUBLICATION_CANDIDATE` for
`2026-06-18`, with the current snapshot hash and the latest prior absent
snapshot hash.

If HKO has not yet published that row, the poll will complete with
`2026-06-18` still listed in `watched_candidate_dates_missing` and zero provider
first-publication candidates.

## Null hypothesis

The watch only extends absence evidence: the June 2026 Daily Extract payload
remains unchanged and `2026-06-18` is not present.

## Falsification

The infrastructure hypothesis fails if the poll cannot archive immutable raw
snapshots, if sidecar metadata/hashes are missing, if the watched date appears
without a documented active absent-before-present snapshot, or if validation
finds leakage/as-of violations.

## Novelty and prior evidence

EXP-0005 established the active watch and stricter candidate gating. EXP-0006
added retry-backed fetching and per-iteration poll snapshot metrics. EXP-0007
and EXP-0008 extended the same watch through `2026-06-18T18:20:46.160262Z`
with no row present.

## Leakage risks anticipated

The main risk is treating a latest mutable target payload as first-published
truth. This experiment only records source publication evidence and does not
create forecast features, predictive labels, market outcomes, or model metrics.
