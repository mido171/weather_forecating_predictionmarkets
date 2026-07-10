# Target and Settlement Specification

## Current evidence hierarchy

For each market, use evidence in this order:

1. exact market rules and event metadata archived before resolution;
2. the rules-named HKO product and field as first published;
3. actual resolved winner;
4. official HKO machine-readable climate history;
5. all other sites only as secondary diagnostics.

The current Hong Kong market template names the Hong Kong Observatory Daily Extract and its `Absolute Daily Max (deg. C)` field, states one-decimal source precision, and says later revisions do not affect resolution. This must be reverified for every new event.

## Canonical target

For local date T:

```text
first public value of
HKO Daily Extract
field = Absolute Daily Max (deg. C)
date = T in Asia/Hong_Kong
```

The target is not:

- highest value among all HKO network stations;
- Hong Kong International Airport;
- a city forecast;
- a third-party weather history;
- a nearest-integer rounded value;
- a later corrected value, when current rules exclude revisions.

## Crucial unresolved parity question

HKO’s official API documents `CLMMAXT` as Daily Maximum Temperature and supports station `HKO` (Hong Kong Observatory). That series is an excellent candidate historical label. However:

- the contract names a displayed Daily Extract field;
- the climate file may be updated or quality-controlled later;
- first-publication timing may differ;
- rare missing/revised values may exist.

Therefore:

```text
Daily Extract first publication = canonical candidate
CLMMAXT station HKO = historical proxy pending G1 parity
```

No model may silently collapse these into one target before G1.

## Bucket semantics

Buckets must be represented as explicit intervals:

```yaml
label: "31°C"
lower_inclusive: 31.0
upper_exclusive: 32.0
```

For a one-decimal source under the integer-range template:

- 30.9°C belongs to `[30.0, 31.0)`;
- 31.0°C belongs to `[31.0, 32.0)`.

This is range containment, not ordinary nearest-integer rounding.

Tail examples:

```yaml
label: "25°C or below"
lower_inclusive: null
upper_exclusive: 26.0

label: "35°C or higher"
lower_inclusive: 35.0
upper_exclusive: null
```

These examples are valid only if the actual event outcomes/rules match. Production must parse each event.

## Required event record

```text
event_id
event_slug
condition_id
market_ids
token_ids
title
description
rules_original
rules_normalized
rules_sha256
source_url
target_local_date
timezone
source_precision
outcome_labels
explicit_bucket_boundaries
event_created_at
market_open_at
market_close_at
resolution_at
resolved_winner
raw_event_payload_sha256
```

## Rules normalization

Normalization may remove nonsemantic whitespace for change detection, but the original text must remain archived. Store both:

- exact-byte hash;
- normalized-text hash.

A changed hash is not automatically dangerous, but it must trigger semantic comparison and manual review.

## First-publication archive

For each target date:

1. poll the Daily Extract product around expected publication;
2. retain every distinct payload;
3. record provider timestamp when present;
4. record first successful retrieval;
5. extract the target field only after saving raw;
6. never replace the first payload with a revision;
7. record later differences separately.

## Parity table

Minimum columns:

```text
local_date
event_slug
rules_hash
daily_extract_first_value
daily_extract_first_available_at
daily_extract_latest_value
clmmaxt_value
clmmaxt_retrieved_at
actual_winner
computed_winner
first_vs_clmmaxt_delta
computed_winner_matches
quality_state
notes
```

## Mismatch taxonomy

- `RULES_CHANGED`
- `DATE/TIMEZONE`
- `FIELD_PARSE`
- `MISSING_FIRST_PUBLICATION`
- `LATER_REVISION`
- `CLMMAXT_DIFFERENCE`
- `BUCKET_BOUNDARY`
- `RESOLUTION_DISPUTE`
- `SOURCE_OUTAGE`
- `UNKNOWN`

Every mismatch is quarantined until explained.

## Settlement adapter invariants

- exactly one bucket contains every permitted value;
- buckets do not overlap;
- no interior gaps;
- tails cover all values if tails exist;
- decimal parsing uses `Decimal`, not binary floating point, at boundaries;
- date is explicitly Hong Kong local;
- unknown rules halt computation;
- computed winners are regression-tested on resolved markets.
