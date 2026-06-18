# Hypothesis

## Mechanism

The HKO Daily Extract backing payload for June 2026 is mutable until newly
completed daily rows are published. Continued active polling can establish a
direct absence-before-presence window for the `2026-06-18` row, which is the
minimum evidence needed to treat an observed row as a provider first-publication
candidate rather than merely an archive first-observed historical row.

## Exact prediction

If the provider publishes the `2026-06-18` row during this bounded window, the
polling ledger will record the row as a provider first-publication candidate
because EXP-0005 through EXP-0029 already observed it absent after active
polling began at `2026-06-18T17:48:59.956593Z`.

If the provider does not publish it during this window, the ledger will extend
the absence evidence beyond EXP-0029 without changing G1 status.

## Null hypothesis

The watched row remains absent for all iterations, adding only a later absence
checkpoint.

## Falsification

The hypothesis is falsified or blocked if the source cannot be fetched after the
declared retry budget, if raw snapshots or sidecars are missing or mutable, if
the parser cannot identify the target field/date, or if the metrics artifact
cannot distinguish absent rows from provider first-publication candidates.

## Novelty and prior evidence

Prior evidence is the accepted active watch chain EXP-0005 through EXP-0029.
Those experiments prove this date was absent after active polling began and
through `2026-06-18T21:21:19.382593Z`; they do not prove provider first
publication yet.

## Leakage risks anticipated

- Treating latest Daily Extract or CLMMAXT as first-published truth.
- Overwriting the first observed raw payload with a later revision.
- Using target evidence as a predictive feature.
- Promoting G2 or modelling before G1 target parity passes.
