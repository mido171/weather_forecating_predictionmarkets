# Point-in-Time and Leakage Specification

## Why this is the central risk

Weather archives often contain corrected observations, final analyses, reanalysis, final best tracks, and the newest version of forecasts. A retrospective model can appear exceptional while using information that did not exist at the forecast cutoff.

The system therefore treats availability as data, not metadata.

## Five timestamp model

Every time-varying record should carry:

| Field | Meaning |
|---|---|
| `valid_at` | Physical time or forecast-valid time |
| `issued_at` | Provider’s initialization/issue time |
| `published_at` | Time provider made it public, when known |
| `available_at` | Earliest defensible time our system could consume it |
| `retrieved_at` | Time our archive actually received it |

For observations, `valid_at` may precede publication. For NWP, `issued_at` is not the same as file availability. For revised climate files, retrieval today does not prove historical availability.

## Eligibility rule

For forecast cutoff \(c\):

```text
feature is eligible only if available_at <= c
```

If `available_at` is missing or ambiguous, the default is exclusion, not a guessed zero latency.

## Source availability contracts

Each source adapter must define:

- provider timestamp fields;
- expected issue cycle;
- typical and conservative latency;
- timeout/incomplete-file behavior;
- revision behavior;
- how actual availability is observed;
- whether historical vintages exist;
- eligibility class.

## Common leakage modes

### Forecast-vintage leakage

Using a 12Z run that was delivered after the cutoff, or downloading only the latest forecast and assigning it to earlier dates.

### Final-observation leakage

Using day-T rainfall, radiation, minimum, or max-so-far in a T-1 forecast.

### Revised-target leakage

Training on corrected CLMMAXT while evaluating against first-published settlement without measuring revisions.

### Reanalysis leakage

ERA5 assimilates observations and is released after valid time. It is excellent for mechanism discovery but not automatically an operational predictor.

### Best-track leakage

Final tropical-cyclone tracks and intensity are retrospective. Use contemporaneous advisories/forecast tracks for operational features.

### Cross-validation leakage

- random K-fold across dates;
- scaling/imputation before temporal split;
- feature selection on all years;
- calibration on test;
- repeated test-set inspection;
- overlapping target windows without embargo.

### Missingness leakage

A source may be missing because of an event only knowable later, or a repaired archive may reveal which days were problematic.

### Label-proxy leakage

Feature names, file paths, revision flags, or source selection can encode the final target.

## Dataset construction

Build forecasts by iterating forecast issuance records, not by taking a daily table and shifting columns casually.

Pseudocode:

```python
for target_date in dates:
    cutoff = cutoff_for(target_date, horizon)
    target = target_store.first_published(target_date)
    eligible = source_store.query(available_at_lte=cutoff)
    row = feature_builder.transform(eligible, cutoff=cutoff)
```

Every output row stores `max_source_available_at` and source-vintage IDs.

## Preprocessing

All learned transformations must fit inside each rolling training window:

- means/stds;
- imputation;
- quantile maps;
- regime clusters;
- PCA/spatial modes;
- feature selection;
- calibration;
- blend weights;
- hyperparameters.

Static physical transformations are allowed when they do not use future outcomes.

## Split discipline

1. Development data is reusable for exploration.
2. Validation confirms candidates.
3. Locked test is opened only for a frozen shortlist and documented protocol.
4. After test inspection, it is no longer pristine.
5. Live shadow becomes the strongest future evidence.

Maintain a `TEST_ACCESS_LOG.md` whenever the locked test is opened.

## Leakage audit checklist

- [ ] exact cutoff stored per row;
- [ ] all timestamps timezone-aware;
- [ ] all features have source lineage;
- [ ] max `available_at` <= cutoff;
- [ ] model cycles delivered after cutoff excluded;
- [ ] no day-T observation in pre-day forecast;
- [ ] no finalized reanalysis/best track operationally;
- [ ] preprocessing fit training-only;
- [ ] no target-derived imputation;
- [ ] no random CV;
- [ ] no duplicate target dates across folds;
- [ ] calibration training-only;
- [ ] locked-test access logged;
- [ ] negative controls fail as expected.

## Required automated sentinels

- reject naive datetimes;
- reject missing availability;
- reject future source timestamp;
- reject target-date observations for pre-event horizon unless explicitly allowed;
- reject unknown source role;
- reject target columns in feature matrix;
- reject overlap between train/validation/test target dates;
- reject predictions generated after target publication;
- test that a deliberately future-shifted feature trips the guardrail.
