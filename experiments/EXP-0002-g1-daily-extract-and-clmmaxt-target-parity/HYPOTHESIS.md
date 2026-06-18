# Hypothesis

## Mechanism

The target system names a specific HKO product, field, and station. The
machine-readable `CLMMAXT` file is a convenient historical climate series. If
Daily Extract and CLMMAXT are semantically and numerically identical for
verifiable target-station dates, then the project can use `CLMMAXT station=HKO`
as a historical proxy while still treating first-published Daily Extract as
official target truth. If they differ, downstream modelling on `CLMMAXT` would
be target-leaked or mislabelled.

## Exact prediction

- HKO Daily Extract exposes `Absolute Daily Max (deg. C)` with one-decimal
  precision for the intended local calendar date and HKO target station.
- `CLMMAXT station=HKO` equals the corresponding Daily Extract value on
  ordinary verifiable dates. Any mismatch is rare, explainable, and quarantined.
- No Polymarket backtesting, price-history processing, or forecast distribution
  change is claimed in this G1 checkpoint.

## Null hypothesis

The null is that target truth cannot be proven: the HKO field/date/station or
precision is not stable, first-publication evidence is insufficient, or
`CLMMAXT` materially differs from Daily Extract values.

## Falsification

- HKO Daily Extract does not expose the expected product/field/station/date, or
  contains revision/fallback/date language that cannot be parsed safely.
- Daily Extract and `CLMMAXT station=HKO` differ on dates not explainable by
  documented revision/missingness/date handling.
- First-publication timing cannot be established sufficiently to support the
  no-later-revision contract wording.

## Novelty and prior evidence

Prior state:

- EXP-0001 proved archive mechanics and fetched current HKO target/proxy
  sources without parsing target parity.
- `docs/02_TARGET_AND_SETTLEMENT.md` states current target-source evidence but
  explicitly keeps `CLMMAXT` candidate-only until G1.
- `docs/13_SOURCES_AND_VERIFICATION.md` cites the HKO URLs as
  starting points, not proof.

This experiment must archive exact current external evidence and distinguish
provider facts from inference.

## Leakage risks anticipated

- Treating latest revised CLMMAXT as first-published Daily Extract without
  measuring revisions.
- Using current latest Daily Extract pages as if they were historical first
  publication payloads.
- Silently allowing unsupported precision, missing station identity, or missing
  source fields into target generation.
