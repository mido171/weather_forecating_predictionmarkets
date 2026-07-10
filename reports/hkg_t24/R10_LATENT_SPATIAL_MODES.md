# EXP-0042 / HKG-T24-R10 Long-Form Experiment Report

## Purpose

R10 tests whether fold-fit latent modes of the all-station temperature field capture mesoscale structure that hand-designed R09 contrasts miss. The key leakage constraint is that PCA/EOF preprocessing must be fit inside each chronological training fold only. R10 therefore does not create a single global PCA feature table. It builds fold-local imputation, scaling, PCA loadings, mode scores, and reconstruction-error features during OOF prediction.

## Data Used

The input is the R09 station-temperature-gradient feature matrix, including the HKO-minus-station offset columns generated from immutable high-frequency station-temperature snapshots. The feature target-date period is `2020-07-02` through `2023-12-31`, and the OOF prediction period is `2021-07-01` through `2023-12-31`. Validation 2024 and locked-test rows are not used.

## Methods

R10 runs baseline, PCA-3, PCA-5, PCA-8 station-offset Ridge models, a PCA-5 plus reconstruction-error Ridge model, and a shallow boosting model using fold-local PCA scores. PCA operates on station-offset columns, not target values. The mode catalog records fold, model id, principal component, station-offset feature loading, and explained variance ratio. Because station coordinates are not yet in the registry, graph-Laplacian modes are blocked rather than fabricated.

## Leakage Controls

For each chronological fold, imputation medians, scaling parameters, PCA components, reconstruction-error calculations, and regression/boosting models are fit using training rows only. Test rows are transformed by that fold's fitted preprocessing objects. No full-sample mode score is written. The deliberately global-fit negative control requested by the specification is not run as a model because it is a known leakage design; it is documented as forbidden and left for a leakage-test fixture rather than scored.

## OOF Gate

The strict four-year OOF check is `BLOCKED`: R10 modern latent-spatial-mode pre-validation feature period: 3.50 years available, requires at least 4.0 years. R10 is therefore a completed latent-mode diagnostic but not promotable under the hard four-year OOF rule.

## Main Result

The best non-control model by OOF MAE is `r10_pca8_station_offsets_ridge` with MAE `1.4543` C, RMSE `1.8236` C, bias `-0.1178` C, and CRPS `1.0238` over `911` rows. The mode catalog and fold-delta tables determine whether latent spatial structure is stable enough to carry forward.

## Interpretation

If a small PCA model beats baseline and R09 spatial summaries, it suggests the station field has coherent latent structure. If PCA loadings rotate unpredictably or only boosting improves, the modes are not yet operationally robust. If reconstruction error helps, field coherence or station-disagreement may be a useful missingness/transition signal. If all mode models lose, transparent R09 features or later graph/coordinate-aware modes are better next steps.

## Blockers

Graph-Laplacian modes, geography-aware loading maps, elevation-aware interpretation, and terrain adjacency are blocked by missing station coordinates/elevation in the current registry. Sparse PCA and probabilistic PCA are not added as dependencies in this pass; the first diagnostic uses standard fold-local PCA with explicit loadings.

## Decision Record

R10 is complete as a fold-local latent spatial mode diagnostic once artifacts and tests pass. It does not authorize validation access. The next planned experiment is R11 dynamic upwind station selection, which can use R08 vector winds and R09/R10 station-field representations but must preserve fold-local preprocessing.

## Operational Details and What Was Deliberately Not Done

The experiment uses the station-offset family because it is the highest-dimensional station-field representation already constructed under the current as-of contract. Each offset compares the HKO target station cutoff state with one neighboring station's cutoff-safe sampled temperature. This choice keeps the mode extraction tied to physical surface thermal contrasts rather than to target-day labels. It also avoids accidental use of since-midnight maximum/minimum values whose label semantics were shown in R03 to include carryover behavior that is not equivalent to an ordinary minute-level trace.

The PCA features are intentionally refit in every chronological fold. That means the loadings file is not a single global map of Hong Kong station modes. Instead, it is a fold-local audit table showing which stations were active, how each component loaded, and how much variance was explained inside that fold. This is less tidy than one global chart, but it is the only acceptable leakage-free design for an OOF experiment. If a future report wants a nice visual map, it must either be descriptive only or be produced from training-window-only objects for the exact fold being evaluated.

The shallow boosting model is included as a weak nonlinearity probe, not as a production candidate. Its tree depth and iteration count are deliberately small. The goal is to learn whether latent station modes contain interactions with seasonal and local cutoff-temperature state. It is not allowed to search a large hyperparameter space, because the modern OOF sample is short and already below the user's four-year reliability requirement.

R10 also refuses to turn the missing station-coordinate problem into a fake solution. A true graph-mode experiment needs coordinates, elevation, terrain/coastline context, and a defensible adjacency kernel. Those inputs must come from station metadata and static geospatial tables, not from station-name ordering or hand-waved groups. Until those fields are canonicalized, the graph portion of the R10 title remains a documented blocker while the fold-local PCA portion is complete.

The month/season relationship is handled by the baseline feature set inherited from R04, while the station field contributes only residual spatial shape. If a PCA model beats the baseline by a small amount but the loadings are dominated by one station or rotate dramatically across folds, the correct decision is not promotion. The correct decision is to treat the result as an unstable diagnostic and use it to design robustness work in R27 or catastrophic-error specialist work in R22.

## Date-Range Discipline

The effective feature matrix remains bounded by the modern HKO high-frequency archive and by the pre-validation cutoff used across R04-R10. R10 does not extend into 2024 to make the statistics look better. The source feature target-date period is `2020-07-02` through `2023-12-31`, and the OOF predictions cover `2021-07-01` through `2023-12-31`. Under the strict user requirement, this is not long enough for promotion. The short span is not hidden inside the result; it is the central reason for the experiment status.

## Reproducibility Notes

All primary outputs are written both to the immutable-style data root output directory and to the repository experiment folder. The repository folder contains the narrative, run config, date ranges, metrics JSON, scoreboard, fold deltas, mode catalog, prediction copy, as-of contract, data manifest with SHA256 values, and reproduction command. That makes this experiment handoff-safe: a later GPT-Pro or Codex conversation can inspect the folder without needing to reconstruct the rationale from terminal history.

The practical takeaway is deliberately narrow: fold-local latent station modes are promising enough to revisit, but the experiment cannot override the four-year OOF gate or the missing graph metadata blocker.

# R10 Machine-Readable Summary Tables

Generated: `2026-06-20T10:07:48.854222Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r10_pca8_station_offsets_ridge | 911 | 2021-07-01 | 2023-12-31 | 1.454342 | 1.823584 | 1.246843 | -0.117758 | 1.023781 | 0.813392 | 0.902305 |
| r10_pca3_station_offsets_ridge | 911 | 2021-07-01 | 2023-12-31 | 1.459778 | 1.856261 | 1.250081 | 0.010914 | 1.039851 | 0.809001 | 0.899012 |
| r10_pca5_station_offsets_ridge | 911 | 2021-07-01 | 2023-12-31 | 1.466581 | 1.859331 | 1.276385 | 0.012196 | 1.042159 | 0.811196 | 0.901207 |
| r10_pca5_with_reconstruction_error_ridge | 911 | 2021-07-01 | 2023-12-31 | 1.469119 | 1.859745 | 1.245975 | -0.069669 | 1.041766 | 0.806806 | 0.899012 |
| r10_baseline_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.472338 | 1.886078 | 1.216189 | 0.029801 | 1.051187 | 0.829857 | 0.908891 |
| r10_shallow_boosting_pca5_modes | 911 | 2021-07-01 | 2023-12-31 | 1.664589 | 2.092089 | 1.457854 | 0.015314 | 1.174800 | 0.740944 | 0.850714 |

## Fold Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2023_h2 | r10_pca5_station_offsets_ridge | 184 | 1.257117 | 1.326541 | 0.069423 | 0.040270 |
| fold_2022_h2 | r10_baseline_temp_calendar | 182 | 1.273130 | 1.273130 | 0.000000 | 0.000000 |
| fold_2023_h2 | r10_pca3_station_offsets_ridge | 184 | 1.277956 | 1.326541 | 0.048585 | 0.026769 |
| fold_2023_h2 | r10_pca5_with_reconstruction_error_ridge | 184 | 1.289803 | 1.326541 | 0.036737 | 0.027935 |
| fold_2022_h2 | r10_pca3_station_offsets_ridge | 182 | 1.297962 | 1.273130 | -0.024832 | -0.000342 |
| fold_2023_h2 | r10_pca8_station_offsets_ridge | 184 | 1.298290 | 1.326541 | 0.028251 | 0.034773 |
| fold_2022_h2 | r10_pca5_station_offsets_ridge | 182 | 1.309223 | 1.273130 | -0.036093 | -0.006453 |
| fold_2022_h2 | r10_pca5_with_reconstruction_error_ridge | 182 | 1.325209 | 1.273130 | -0.052080 | -0.007939 |
| fold_2023_h2 | r10_baseline_temp_calendar | 184 | 1.326541 | 1.326541 | 0.000000 | 0.000000 |
| fold_2022_h2 | r10_pca8_station_offsets_ridge | 182 | 1.332706 | 1.273130 | -0.059576 | 0.003332 |
| fold_2021_h2 | r10_pca8_station_offsets_ridge | 183 | 1.428746 | 1.588463 | 0.159716 | 0.079843 |
| fold_2023_h2 | r10_shallow_boosting_pca5_modes | 184 | 1.473514 | 1.326541 | -0.146973 | -0.067803 |
| fold_2023_h1 | r10_baseline_temp_calendar | 181 | 1.482125 | 1.482125 | 0.000000 | 0.000000 |
| fold_2023_h1 | r10_pca3_station_offsets_ridge | 181 | 1.485321 | 1.482125 | -0.003196 | 0.013099 |
| fold_2023_h1 | r10_pca5_station_offsets_ridge | 181 | 1.487436 | 1.482125 | -0.005311 | 0.018345 |
| fold_2023_h1 | r10_pca8_station_offsets_ridge | 181 | 1.493842 | 1.482125 | -0.011717 | 0.019488 |
| fold_2021_h2 | r10_pca5_with_reconstruction_error_ridge | 183 | 1.497975 | 1.588463 | 0.090488 | 0.042647 |
| fold_2023_h1 | r10_pca5_with_reconstruction_error_ridge | 181 | 1.517130 | 1.482125 | -0.035005 | -0.000672 |
| fold_2021_h2 | r10_pca3_station_offsets_ridge | 183 | 1.532052 | 1.588463 | 0.056411 | 0.027526 |
| fold_2021_h2 | r10_pca5_station_offsets_ridge | 183 | 1.544383 | 1.588463 | 0.044080 | 0.019325 |
| fold_2023_h1 | r10_shallow_boosting_pca5_modes | 181 | 1.562299 | 1.482125 | -0.080174 | -0.067733 |
| fold_2021_h2 | r10_baseline_temp_calendar | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2021_h2 | r10_shallow_boosting_pca5_modes | 183 | 1.625825 | 1.588463 | -0.037362 | -0.029067 |
| fold_2022_h2 | r10_shallow_boosting_pca5_modes | 182 | 1.664172 | 1.273130 | -0.391042 | -0.210301 |
| fold_2022_h1 | r10_baseline_temp_calendar | 181 | 1.693665 | 1.693665 | 0.000000 | 0.000000 |
| fold_2022_h1 | r10_pca3_station_offsets_ridge | 181 | 1.708706 | 1.693665 | -0.015041 | -0.010739 |
| fold_2022_h1 | r10_pca5_with_reconstruction_error_ridge | 181 | 1.718924 | 1.693665 | -0.025259 | -0.015446 |
| fold_2022_h1 | r10_pca8_station_offsets_ridge | 181 | 1.721667 | 1.693665 | -0.028002 | -0.000975 |
| fold_2022_h1 | r10_pca5_station_offsets_ridge | 181 | 1.738225 | 1.693665 | -0.044560 | -0.026892 |
| fold_2022_h1 | r10_shallow_boosting_pca5_modes | 181 | 2.000734 | 1.693665 | -0.307069 | -0.244650 |

## Mode Catalog

| model_id | pc | mean_explained_variance_ratio | top_features | top_abs_loading_sum |
| --- | --- | --- | --- | --- |
| r10_pca3_station_offsets_ridge | 1 | 0.374557 | station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_kowloon_city_c; station_offset_hko_minus_wong_tai_sin_c; station_offset_hko_minus_ta_kwu_ling_c | 1.682662 |
| r10_pca3_station_offsets_ridge | 2 | 0.167036 | station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c | 2.376129 |
| r10_pca3_station_offsets_ridge | 3 | 0.085338 | station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c | 2.653879 |
| r10_pca5_station_offsets_ridge | 1 | 0.374557 | station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_kowloon_city_c; station_offset_hko_minus_wong_tai_sin_c; station_offset_hko_minus_ta_kwu_ling_c | 1.682662 |
| r10_pca5_station_offsets_ridge | 2 | 0.167036 | station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c | 2.376129 |
| r10_pca5_station_offsets_ridge | 3 | 0.085338 | station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c | 2.653879 |
| r10_pca5_station_offsets_ridge | 4 | 0.071683 | station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_lau_fau_shan_c | 2.910054 |
| r10_pca5_station_offsets_ridge | 5 | 0.033603 | station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_ngong_ping_c | 3.368806 |
| r10_pca5_with_reconstruction_error_ridge | 1 | 0.374557 | station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_kowloon_city_c; station_offset_hko_minus_wong_tai_sin_c; station_offset_hko_minus_ta_kwu_ling_c | 1.682662 |
| r10_pca5_with_reconstruction_error_ridge | 2 | 0.167036 | station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c | 2.376129 |
| r10_pca5_with_reconstruction_error_ridge | 3 | 0.085338 | station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c | 2.653879 |
| r10_pca5_with_reconstruction_error_ridge | 4 | 0.071683 | station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_lau_fau_shan_c | 2.910054 |
| r10_pca5_with_reconstruction_error_ridge | 5 | 0.033603 | station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_ngong_ping_c | 3.368806 |
| r10_pca8_station_offsets_ridge | 1 | 0.374557 | station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_kowloon_city_c; station_offset_hko_minus_wong_tai_sin_c; station_offset_hko_minus_ta_kwu_ling_c | 1.682662 |
| r10_pca8_station_offsets_ridge | 2 | 0.167036 | station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c | 2.376129 |
| r10_pca8_station_offsets_ridge | 3 | 0.085338 | station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c | 2.653879 |
| r10_pca8_station_offsets_ridge | 4 | 0.071683 | station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_lau_fau_shan_c | 2.910054 |
| r10_pca8_station_offsets_ridge | 5 | 0.033603 | station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_ngong_ping_c | 3.368806 |
| r10_pca8_station_offsets_ridge | 6 | 0.024114 | station_offset_hko_minus_ngong_ping_c; station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_ngong_ping_c; station_offset_hko_minus_ngong_ping_c; station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_peng_chau_c | 3.491668 |
| r10_pca8_station_offsets_ridge | 7 | 0.022031 | station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_cheung_chau_c; station_offset_hko_minus_stanley_c; station_offset_hko_minus_tai_po_c; station_offset_hko_minus_stanley_c; station_offset_hko_minus_tai_mei_tuk_c; station_offset_hko_minus_king_s_park_c | 2.977192 |
| r10_pca8_station_offsets_ridge | 8 | 0.019818 | station_offset_hko_minus_kai_tak_runway_park_c; station_offset_hko_minus_kai_tak_runway_park_c; station_offset_hko_minus_stanley_c; station_offset_hko_minus_stanley_c; station_offset_hko_minus_kai_tak_runway_park_c; station_offset_hko_minus_sham_shui_po_c; station_offset_hko_minus_sham_shui_po_c; station_offset_hko_minus_tai_mei_tuk_c | 2.397104 |
| r10_shallow_boosting_pca5_modes | 1 | 0.374557 | station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_sha_tin_c; station_offset_hko_minus_kowloon_city_c; station_offset_hko_minus_wong_tai_sin_c; station_offset_hko_minus_ta_kwu_ling_c | 1.682662 |
| r10_shallow_boosting_pca5_modes | 2 | 0.167036 | station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_clear_water_bay_c; station_offset_hko_minus_shau_kei_wan_c; station_offset_hko_minus_shau_kei_wan_c | 2.376129 |
| r10_shallow_boosting_pca5_modes | 3 | 0.085338 | station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_happy_valley_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c | 2.653879 |
| r10_shallow_boosting_pca5_modes | 4 | 0.071683 | station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_wong_chuk_hang_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_lau_fau_shan_c; station_offset_hko_minus_chek_lap_kok_c; station_offset_hko_minus_lau_fau_shan_c | 2.910054 |
| r10_shallow_boosting_pca5_modes | 5 | 0.033603 | station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_the_peak_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_tai_mo_shan_c; station_offset_hko_minus_ngong_ping_c | 3.368806 |
