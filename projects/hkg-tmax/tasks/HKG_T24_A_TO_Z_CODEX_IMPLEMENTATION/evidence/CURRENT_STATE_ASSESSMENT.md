# Current State Assessment After Complete Experiment-Corpus Audit

## Executive judgment

The corpus contains genuine information gain and a credible system architecture, but it does **not** currently demonstrate a path from roughly 1.04–1.06 C expanded-frame MAE to 0.45 C through incremental residual tweaks alone. The repository has established that the official forecast is a strong anchor, causal residual memory is useful, regional station state adds complementary signal, and tail/regime targeting can improve a parent. It has also established that many increasingly elaborate micro-gates, analog blends and stacked residual layers deliver only thousandths of a degree or fail transfer.

The current evidence supports building a disciplined multi-anchor post-processing system. It does not support claiming that the 0.45 C benchmark is already close.

## Canonical score interpretation

| Evidence | MAE | Interpretation |
|---|---:|---|
| Raw official expanded frame | 1.079107 | Primary baseline on 5,265 rows |
| 0185 strict T-7 online residual memory | about 1.063927 | Best clearly strict-core expanded evidence |
| 0196 station-tail research proxy | 1.038829 | Best promoted expanded research-proxy candidate; conditional on ISD repair/availability |
| 0200 analogue child | 1.038296 | Slightly lower raw MAE, but did not pass promotion robustness |
| 0080/0081 narrow common frame | 0.943755 | Useful narrow-frame result, not comparable to 5,265-row frame |
| 0113/0172–0178 | 0.799044 | Invalid predictive score due same-row error/target-derived inputs |
| Aspirational target | 0.450000 | Unproven external benchmark |

The 0196 result reduces expanded raw-official MAE by only about 3.73%. Reaching 0.45 from the raw-official baseline requires about a 58.30% reduction and remains 0.58883 C below 0196. That gap is too large to bridge honestly by continuing to add tiny station buckets to the current anchor.

## What is genuinely promising

### 1. Official forecast as anchor

The strongest repeated finding is that direct weather-only or station-only models do not replace the official forecast. The official product embeds human and numerical-model information not reproduced by the current historical feature store. The system should model `actual - official`, trust, uncertainty and specialist corrections rather than discard the anchor.

### 2. Causal residual memory

Experiments 0074–0075 and 0185 show that forecast errors have adaptive bias structure. The strict expanded replay at T-7 maturity improves the official baseline without relying on current outcomes. Multi-resolution, hierarchical and change-point-aware residual state remains a high-value lane, but its correction must stay small and parent-fallback-safe.

### 3. Regional station network as a regime sensor

Experiments 0050–0052 and 0194–0196 support temperature anomalies, dewpoint changes, pressure state, station disagreement and cross-station structure. The station network is most useful for identifying marine suppression, inland heat realization, cool-surge evolution, humidity/cloud regimes and official-forecast blind spots. It is not proven as a stand-alone replacement forecast.

### 4. Tail-conditioned correction

0196 improves the 0194 parent, especially in historical parent-tail cases. This supports a two-stage design: estimate whether the anchor is likely wrong and in which direction, then permit a bounded correction only when prior evidence is strong.

### 5. MAM and transition-day focus

MAM remains the hardest seasonal slice, with official MAE around 1.35 C in the expanded frame. Spring transition mechanisms are a legitimate specialist target. Earlier hard-cell approaches were too sparse or unstable; the next version should use smooth regime probabilities and multivariate physics rather than dozens of brittle thresholds.

### 6. Diagnostic upper-air/marine mechanisms

IGRA heights/thermal profile and HKO daily/marine variables repeatedly rank highly as explanatory signals. Their current operational timing is blocked and IGRA values are contaminated by sentinels/scaling defects. Their best use now is teacher-state definition and safe-proxy distillation, not direct production input.

## What is weak, exhausted or negative

- Station-only forecasts remain near 1.3 C MAE and do not replace official guidance.
- Pressure-only or single-feature experts generally yield small or unstable lift.
- Repeated hard bucket/cell specialists produce tiny development gains and poor transfer.
- Broad feature soup and stacked routers often dilute the best simple expert.
- Mature T-7 synoptic trajectories in 0205 did not improve 0196.
- The climatological analogue lane in 0200 gained only 0.00053 C versus 0196 and failed promotion robustness.
- Further micro no-harm gates in 0195, 0197, 0198 and 0203 left predictions unchanged.
- Source-era routing and additional feature-group stacking did not reliably improve the champion.
- Short-history high-frequency experiments 0179–0183 are same-day/prospective diagnostics, not evidence for the strict T-24 horizon.

## Critical invalid-result block

The following experiment IDs have predictive scores that must not enter any champion registry: 0111, 0112, 0113, 0114, 0115, 0116, 0117, 0118, 0119, 0120, 0121, 0122, 0123, 0124, 0125, 0126, 0127, 0128, 0129, 0130, 0131, 0132, 0133, 0134, 0135, 0136, 0137, 0138, 0140, 0141, 0142, 0143, 0144, 0145, 0146, 0147, 0148, 0149, 0150, 0151, 0152, 0153, 0154, 0155, 0156, 0157, 0158, 0159, 0164, 0165, 0166, 0167, 0168, 0171, 0172, 0173, 0174, 0175, 0176, 0177, 0178.

Their `FEATURE_SPEC.yaml` files admit current-row outcome-derived fields including `official_underforecast_c`, `official_overforecast_c`, `hot_day_underforecast_flag` and `cold_day_overforecast_flag`. In 0111, changed-row candidate correction correlates about 0.825 with the same-row realized official residual and usually behaves like that residual clipped at ±0.35 C. In 0113, the corresponding cap is ±0.50 C. Many nominally different folders share identical scoreboards and identical candidate predictions. The hypotheses may be retained, but each lane requires clean reimplementation from source data.

## Reproducibility concern

Experiments 0105–0183 use wrappers importing `scripts/run_hkg_t24_0105_0183_beastmode_roadmap.py`; that shared generator is absent from the supplied archive. Even apparently causal results from 0161–0170 should be rebuilt self-contained before promotion.

## Data-quality constraint on the apparent expanded champion

0194/0196 use ISD-derived station features and descendants of 0187. The supplied dataset profile shows `wind_direction_deg` is exactly 20 degrees across 4,029,291 ISD observations and the daily cutoff summary is also constant at 20 degrees. It also marks ISD as retrospective archive/proxy rather than proven exact operational vintage. Therefore 0196 is a **promising research-proxy champion**, not yet a deployable champion. Experiment 0207 must repair/rebuild the parser and produce strict-core and proxy scoreboards separately.

## Do we have an edge?

There is evidence of a small edge over raw official guidance in historical development data. There is not yet evidence of a 0.45 C system. The best chance of a step-change is not another tiny residual cell; it is an exact-vintage independent NWP/ensemble anchor combined with local station-aware MOS, causal online calibration, regime routing and a distributional median. HKO itself states that computer-model products form the basis of local forecasts, and its automatic regional forecasts provide model-generated location guidance. The current repository's ARWF and NCEP holdings are too short or undecoded to test that path robustly.

## Objective priority order

1. Repair data and establish strict/proxy canonical benchmarks.
2. Build hierarchical residual memory and a feature-family residual superlearner.
3. Rebuild the 36-station array as a geospatial graph with repaired wind and time proof.
4. Convert blocked physics into safe student-state probabilities.
5. Acquire/decode exact-vintage GFS/GEFS and continuously archive ARWF.
6. Add text/revision contradiction features.
7. Route a small set of independently credible experts.
8. Fit a calibrated residual distribution and use its conditional median as the final point forecast.
9. Run the championship tournament and freeze the first robust system.
