# EXP-0043 / HKG-T24-R11 Long-Form Experiment Report

## Purpose

R11 was specified as the dynamic upwind station-selection and flow-relative advection experiment. Its purpose is to test whether the station or regional source area that matters for tomorrow's official HKO Headquarters Tmax changes with the observed flow at the T-1 15:00 cutoff. This is a different question from R08 and R09. R08 asked whether wind state by itself carries signal. R09 asked whether fixed station-temperature contrasts carry signal. R11 asks whether the wind vector can choose which surrounding thermal and moisture observations are upstream, downstream, or dynamically irrelevant.

## Required Design

The uploaded goal requires upstream cones of 30, 60, 90, and 120 degrees; distance-decayed upwind averages using multiple length scales; downstream-minus-upstream contrasts; advection proxies from spatial gradients; surface plus 925/850 hPa flow; fallback behavior when winds are weak; and randomized station-coordinate negative controls. Every one of those operations depends on verified station geometry. A dynamic upwind cone cannot be computed from station names, file order, or manually guessed neighborhoods. The geometry must be canonical and point-in-time documented for the same station identities used by the high-frequency HKO feeds.

## Inputs Audited

The available surface wind table from R08 has `897394` sampled rows, `30` stations, and observed timestamps from `2021-12-29 11:40:00+08:00` through `2026-06-18 14:50:00+08:00`. The available station-temperature table from R09 has `1240613` sampled rows, `39` stations, and observed timestamps from `2020-06-30 11:40:00+08:00` through `2026-06-18 14:50:00+08:00`. These are legitimate cutoff-safe source families under the conservative 20-minute availability rule. They are necessary for R11, but they are not sufficient.

The current canonical experiment registry at `config/hkg_t24/station_registry.parquet` contains HKO station identities, aliases, feed membership, target flags, and unresolved official-code status. It does not contain latitude, longitude, or elevation columns for the named HKO high-frequency stations. The separate static-context registry under the data root contains one HKO target point and NOAA ISD station metadata, but it does not resolve the full HKO named station network used in R08/R09. The static distance/bearing table is therefore not a valid substitute for dynamic upwind station geometry across the public HKO feed stations.

## Leakage Decision

R11 is blocked rather than approximated. That is an intentional leakage and scientific-validity decision. If I computed cones using station-name ordering, hand-assigned compass groups, or the NOAA-only static table, the resulting features would look mathematically precise but would not correspond to the station field actually used by the modern HKO high-frequency archive. Worse, randomized-coordinate negative controls would become meaningless because there would be no trusted coordinate baseline to randomize from.

## What Was Completed

R11 completed an input-readiness and no-go gate. The experiment folder contains a machine-readable readiness table, metrics JSON, empty OOF prediction table with a blocked reason, explicit feature specification, ablation plan, negative controls, date ranges, data manifest, and conclusion. The repository report records that the surface wind and temperature ingredients exist, but the dynamic geometry layer and eligible upper-air wind parser do not.

## Why No OOF Model Was Scored

Scoring a model would require at least one valid dynamic feature family. The available fixed-group fallback features were already tested in R08 and R09. Repackaging them as R11 would double-count earlier work and would not answer the dynamic upwind hypothesis. The R11 model ladder is therefore intentionally empty until station coordinates/elevations and the IGRA wind parser are available. This is not a missing effort item; it is the correct blocked outcome under the non-negotiable no-forward-looking and no-fake-geometry rules.

## Readiness Table

| requirement | status | evidence | disposition |
| --- | --- | --- | --- |
| surface wind network before cutoff | available | 897394 sampled rows; 30 stations | usable as raw flow signal from R08 |
| surrounding station temperature field before cutoff | available | 1240613 sampled rows; 39 stations | usable as fixed station-field signal from R09 |
| canonical HKO station latitude/longitude/elevation for all dynamic candidates | blocked | config/hkg_t24/station_registry.parquet has no latitude/longitude/elevation_m columns for HKO high-frequency stations | do not compute cones, distances, gradients, or random-coordinate controls |
| distance/bearing matrix for HKO high-frequency station names | blocked | static distance table rows=1369; HKO named station ids present=False | NOAA/HKO target static table is not a substitute for 39 HKO public feed stations |
| eligible IGRA 925/850 hPa wind by cutoff | blocked | IGRA raw period-of-record zip is downloaded but no parsed eligible sounding table exists | upper-air flow cannot enter R11 scoring |
| flow-relative upwind cones and length-scale selection | blocked | depends on verified station coordinates and elevations | document blocker; do not fake geometry from station names |
| fixed-group fallback comparison | available | R08 and R09 fixed wind/spatial group diagnostics exist | already benchmarked; insufficient to answer R11 dynamic-geometry question |

## Data Ranges

The audited wind observations span `2021-12-29 11:40:00+08:00` through `2026-06-18 14:50:00+08:00`. The audited temperature observations span `2020-06-30 11:40:00+08:00` through `2026-06-18 14:50:00+08:00`. The existing modern pre-validation feature matrices still end at 2023-12-31, and no 2024 validation outcomes or 2025+ locked-test rows were accessed. Because no dynamic R11 feature matrix exists, there is no R11 development OOF date range and no candidate can be promoted.

## Blockers

The exact blockers are: verified latitude/longitude for all HKO high-frequency station names, verified elevation for those same station identities, a station-distance/bearing matrix keyed to the canonical HKO station ids rather than NOAA-only ids, an eligible 00 UTC T-1 IGRA parser for 925/850 hPa wind, and a fold-safe hyperparameter procedure for cone angle and length scale once the geometry exists. All of these are engineering inputs, not model-tuning choices.

## Next Action

The next lawful task for this research branch is to enrich the station registry from official HKO station metadata or another citable source, preserve aliases such as Wong Chuk Han versus Wong Chuk Hang without blind merging, derive the HKO-named station distance/bearing/elevation matrix, and then rerun R11 with the predeclared cone and length-scale grid. Until then, R12 can proceed because it uses already parsed King's Park solar observations and does not depend on station geometry.

## Decision Record

R11 status is `BLOCKED_INPUTS_MISSING`. Surface wind and temperature inputs are available, but the required dynamic upwind geometry is not. No validation data was read. No locked-test data was read. No model was trained. No feature was promoted. The null/blocker result is retained and indexed so later work does not accidentally treat dynamic upwind information as tested.

This result should be treated as a hard engineering prerequisite, not as evidence that flow-relative advection lacks meteorological value. The experiment says only that the current repository cannot test the idea honestly yet.

## Guardrail Detail

The most tempting shortcut would be to use a hand-built list of "north", "south", "coastal", or "inland" station groups as a pseudo-upwind geometry. R11 rejects that shortcut. Those groups were already represented in R08/R09-style fixed diagnostics, and they do not satisfy the dynamic upwind specification. A flow-relative cone needs bearings from HKO to each candidate station, and a distance-decay calculation needs verified distances in kilometers. Without those fields, every downstream number would be arbitrary even if the code looked precise.

Another tempting shortcut would be to use the NOAA ISD static registry as a proxy for the HKO public feed stations. That is also rejected. The NOAA stations are useful for future long-history regional work, but their identifiers and station histories are not the same as the named HKO high-frequency stations in the current temperature, humidity, pressure, and wind feeds. Mixing those registries would create a false sense of spatial precision and could silently merge unresolved station aliases.

The R11 folder is therefore designed to make the blocker operationally actionable. It tells the next worker exactly what to add: official HKO station coordinates, elevations, validity periods, alias resolution, a distance/bearing matrix keyed to the canonical HKO feed station ids, and an eligible upper-air wind parser. Once those are present, the same experiment id can be rerun as a scored dynamic-upwind test rather than a no-go gate.
