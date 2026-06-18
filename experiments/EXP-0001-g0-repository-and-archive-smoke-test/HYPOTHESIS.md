# Hypothesis

## Mechanism

G0 is an infrastructure validation rather than a meteorological signal test.
If the repository can run its checks and archive raw source bytes immutably,
then later target-parity and forecast experiments can depend on reproducible
source provenance instead of mutable live pages.

## Exact prediction

- No forecast distribution change is claimed.
- Every `bootstrap_now` source can be fetched twice with distinct retrieval
  events.
- Every accepted payload has non-empty raw bytes, SHA-256, retrieval timestamp,
  URL/request metadata, HTTP status, response headers, and sidecar JSON.
- HTTP-error, empty, or malformed payload paths fail loudly under tests.
- No raw snapshot is overwritten in place.

## Null hypothesis

The repository cannot be considered ready for G1 if doctor/test/validation
fail, if any bootstrap source cannot be archived, if hashes or sidecars are
missing, or if failure paths can pass silently.

## Falsification

- `doctor`, `pytest`, or `validate all` exits nonzero after proper dependency
  setup.
- Any `bootstrap_now` source fails to fetch without an explicit documented
  provider-side reason.
- Any archived raw payload has a mismatched hash, missing sidecar, missing HTTP
  metadata, empty body, or non-2xx status.
- A repeated fetch overwrites an existing raw snapshot in place.
- Fetch/storage/parser failure paths do not raise explicit exceptions.

## Novelty and prior evidence

No prior experiment exists. The test follows G0 in `FIRST_GOALS.md`, the raw
archive requirements in `AGENTS.md`, and the timestamp/availability discipline
in `docs/03_ASOF_AND_LEAKAGE.md`.

## Leakage risks anticipated

No features or labels are created in G0. Leakage risk is limited to accidentally
treating latest-only or revised sources as operational training data. This
experiment stores raw snapshots only and does not derive predictive features.
