# Hypothesis

## Claim

If HKO publishes the `2026-06-18` Daily Extract row during an active polling
window, the archive should contain:

1. at least one post-active-start monthly payload where `2026-06-18` is absent;
2. a later immutable monthly payload where `2026-06-18` is present; and
3. no later value revision for that date.

Only that sequence may be labelled
`PROVIDER_FIRST_PUBLICATION_CANDIDATE`.

## Mechanism

The Daily Extract monthly backing payload appears to grow as new local dates are
published. A pair of archived absent/present snapshots bounds the publication
window without relying on provider undocumented timing assumptions.

## Expected Direction

The experiment may produce either zero candidates if the row is still absent or
already present without active absence evidence, or one candidate for
`2026-06-18` if the absent-to-present transition is captured.

## Falsification

The claim is weakened if the row can be labelled candidate without a prior
active absent snapshot, if row absence is not traceable to raw hashes, if a
revision overrides the first value, or if the poller cannot complete a bounded
run reproducibly.

## Leakage Risks

- Treating latest Daily Extract payloads as historical first publication.
- Treating a first fetch after `now` as proof the row was not already public.
- Ignoring later revisions to the same local date.
- Using this target-publication evidence as a forecast feature.
