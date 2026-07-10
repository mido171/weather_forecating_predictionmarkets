# Current Evidence Seed

This file initializes the Director. It must be refreshed from actual experiment artifacts before decisions.

## Strongest currently documented anchor-correction result

Experiment/folder:

`0075_online_residual_memory_refinement`

Candidate:

`causal_onmem_refine_all_h20_n10_cap0p2_lift_weighted`

Reported:

- MAE: `0.9456033267531754`
- RMSE: `1.201776422355206`
- rows: `2670`
- dates: `2000-01-02` through `2023-12-31`
- no 2024+ rows.

Interpretation:

- real evidence that small source-aware causal online residual correction can help;
- not directly comparable with the expanded 5265-row frame;
- replay on harmonized frames is a priority.

## Expanded current frame

Folder:

`0103_current_rss_safe_continuation`

Reported pre-2024 frame:

- rows: `5265`
- dates: `2000-01-02` through `2023-12-31`
- 0101 candidate MAE: `1.0490737910402705`
- RMSE: `1.3514232511866526`
- improvement versus raw official: approximately `-0.030033521400375296` MAE.

Current scoreable RSS slice:

- rows: `992`
- dates: `2021-04-14` through `2023-12-31`
- candidate MAE: `0.989647316624874`
- improvement versus official: approximately `-0.026179296278351805`.

Largest current forecast archive gap:

- `2011-09-14` through `2021-04-13`
- about 3500 missing days.

## Timestamp audit

Folder:

`0102_timestamp_proof_unlock_queue`

Reported:

- 258 blocked features audited;
- 207 upper-air;
- 39 HKO daily climate;
- 12 marine proxy;
- zero unlocked.

Interpretation:

- diagnostic mechanisms may be real;
- production-style scoring remains blocked without available-at proof.

## Feature atlas

Folder:

`0100_stable_mam_cell_feature_atlas`

Reported:

- strong diagnostic families: upper-air, target memory, HKO daily climate, marine proxy, calendar climatology, ISD network;
- top raw diagnostic: `igra_hgt_1000hpa_m`, timestamp blocked;
- top safe feature: `target_roll14_std_lag7_c`;
- approximately 204 safe future-allowed features, mainly target memory, calendar climatology, and ISD network.

Safe examples:

- `target_roll14_std_lag7_c`
- `target_roll7_std_lag7_c`
- `target_lag365_tmax_c`
- `trajectory_7_30_slope_c_per_day`
- `target_roll3650_std_lag7_c`
- `target_lag7_tmax_c`
- `target_lag60_tmax_c`
- `target_lag30_tmax_c`
- `volatility_mad_14_lag7_c`
- `volatility_iqr_14_lag7_c`
- `target_lag7_minus_roll365_c`
- `clim_lag7_anomaly_vs_hl20_c`
- `isd_dew_point_mean_c_change_1d`
- `clim_harmonic_sin5_year_drift`
- `trajectory_range_lag7_60_c`

## Station evidence

Prior work indicates potential value from:

- regional dew-point change;
- station temperature anomaly versus recent baseline;
- station pressure tendency;
- temperature-dew-point spread;
- wind fields;
- station pair/group interactions;
- one station's regime state gating another station's thermal level.

A documented interaction involved station IDs `450110-99999` and `592870-99999`. Map them to verified metadata before physical interpretation.

## Long-history robust screens

Documented scoreboards:

- R14 `r14_upper_air_core`: MAE about `1.489763`, N 21,313;
- R15 `r15_coupling_terms`: MAE about `1.243243`, N 18,624;
- R16 `r16_isd_regional_aggregate`: MAE about `1.258535`, N 18,627;
- R17 `r17_era_transfer_terms`: MAE about `1.232364`, N 18,624.

Interpretation:

- these are signal screens, not production champions;
- upper-air, regional surface, and era/coupling information matter;
- inherited timestamp eligibility must be audited.

## Research posture

Current likely priorities:

1. canonical frame and scoreboard harmonization;
2. replay 0075 logic on compatible frames;
3. deeper multi-station interaction/regime mining;
4. target-memory state specialists;
5. source/season/regime residual memory;
6. MAM transition and high-error specialists;
7. station disagreement and uncertainty;
8. diagnostic-to-deployable proxy conversion;
9. forecast archive backfill preparation;
10. timestamp proof acquisition.

Do not assume these remain optimal after the latest experiment batch. Recompute from the corpus.
