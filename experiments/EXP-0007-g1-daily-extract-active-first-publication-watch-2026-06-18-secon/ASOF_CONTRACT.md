# As-Of Contract

EXP-0007 creates target-publication evidence only. It does not create forecast
rows, features, labels for training, or market signals.

## Active Watch Window

- inherited active polling start: `2026-06-18T17:48:59.956593Z`
- watched local date: `2026-06-18`
- publication timing evidence is bounded only by raw absent/present snapshots.

## Candidate Rule

`PROVIDER_FIRST_PUBLICATION_CANDIDATE` requires:

- explicit watched local date;
- timezone-aware active start;
- at least one post-active-start raw monthly payload proving absence before
  first presence;
- later raw payload proving first presence;
- no observed later revision for that local date.

## Forbidden Data

- CLMMAXT as a substitute for first-published Daily Extract.
- Latest-only payloads as historical first-publication proof.
- Predictive modelling, machine learning, or market execution data.
