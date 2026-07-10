# HKG T+24 Tmax Beastmode Information-Gain Research Program

**Target:** Hong Kong Observatory daily maximum temperature at T+24.  
**Development cutoff:** all target dates before 2024 only.  
**Purpose:** extract the maximum credible predictive information from the acquired target, station-network, forecast, upper-air, daily-climate, marine, static-context, and short-history feeds before committing to a final ML architecture.  
**Deliverable type:** repository-wide evidence synthesis plus an executable queue of 80 new experiments.

## 1. Executive recommendation

### Single best next Codex task

Build `experiments/0104_canonical_frame_evidence_registry` before running another score-seeking model. It must reproduce the 2,670-row historical common frame, the 5,265-row expanded official frame, the 992-row exact RSS frame, the long-history station/target frame, and the diagnostic blocked-physics frame from immutable inputs. It must then publish one canonical response library and a cross-frame scoreboard. Without this, a 0.9438 score on the narrow frame and a 1.0491 score on the expanded frame will continue to be quoted as though they were comparable.

### Top three research routes

1. **Causal spatial-sensor-array mining:** station anomalies, rank reversals, pressure/dewpoint wave propagation, wind-conditioned gradients, sea-breeze penetration, and physically grouped station modes. Use these primarily for residual correction, regime detection, uncertainty, and routing—not as a wholesale replacement for the official anchor.
2. **Target-memory × station-regime specialists:** recent HKO trajectory, volatility, spell state, and seasonal-transition position should be crossed with station-network confirmation to determine when persistence holds and when synoptic/spatial evidence breaks it.
3. **Expanded-frame source-aware online correction:** replay and generalize the strongest 0075/0081 residual-memory logic on the 5,265-row frame, with source/era/season hierarchy, bounded corrections, no-regret abstention, and explicit MAM/high-error-tail objectives.

### Top three blockers

1. The scoreable official archive has a roughly 3,500-day gap from 2011-09-14 through 2021-04-13. This blocks continuous anchor-history studies but does not block target/station information-gain work.
2. The timestamp audit left 207 upper-air, 39 HKO daily-climate, and 12 marine features blocked; zero were unlocked. They can explain mechanisms and train proxy students, but they cannot enter production-style walk-forward scoring yet.
3. Station metadata and temporal QA are incomplete enough to matter: at least one high-profile coordinate is implausible (`450090-99999` longitude `-114.283`), station eras differ sharply, and historical timestamp offsets must be reconciled before spatial physics is trusted.

### Current benchmark treatment

- **Canonical expanded-frame research benchmark:** 0103/0101 candidate MAE `1.0490737910402705`, RMSE `1.3514232511866526`, N=`5,265`, versus official MAE `1.0791073124406458`.
- **Strong narrow-frame evidence, not directly comparable:** 0075 MAE `0.9456033267531754` on N=`2,670`; 0081 later reached roughly `0.943755` on the same narrow/common geometry but relied on a small modern RSS gate and remains fragile.
- **Long-history weather-only reference:** `long_history_core_v1` MAE `1.2365984893663038` on the corrected 2020–2023 common-row benchmark. It is useful for mechanism discovery, not a replacement for the official forecast anchor.

## 2. What was actually reviewed

The supplied archive contains **165 top-level folders and 1,639 files**: 763 Markdown files, 405 CSV files, 251 JSON files, and 220 YAML files. Every file was opened and parsed successfully. The review covered numbered research/analysis records from `0000` through `0103`, formal experiments from `EXP-0046` through `EXP-0099`, their scoreboards, subgroup metrics, manifests, hypotheses, conclusions, leakage notes, and negative results.

The supplied dataset document was also read in full. It describes the normalized dataset families, the 48,577-row × 566-column expanded feature matrix, R14–R17 matrices and OOF artifacts, official forecast archives, and acquired high-frequency HKO feeds.

**Important scope statement:** the attachment contained the complete experiment archive and the dataset explanation, not the raw Parquet payloads themselves. Therefore this document performs a complete evidence audit of the experiment outputs and designs the next numerical analyses; it does not pretend to have recomputed every raw correlation from unattached source rows. Each proposed experiment below is written so Codex can perform that raw-row computation in the repository.

## 3. Evidence map: what is already known

### 3.1 Top evidence already found

| Evidence | What it proves | What it does not prove |
|---|---|---|
| 0075 online residual-memory MAE 0.945603 on 2,670 rows | Prior-only, source-aware online residual correction can materially improve an official anchor on the old common frame. | That the same gain survives the expanded 5,265-row frame or future continuous archive. |
| 0081 narrow-frame MAE about 0.943755 | A small, hardened RSS-era gate can add incremental lift after 0075. | Robustness outside the modern RSS slice; the active gate touched only a limited set of rows. |
| 0103 expanded candidate MAE 1.049074 on 5,265 rows | The current safe MAM/trajectory specialist improves official by about 0.030°C on the expanded pre-2024 frame. | Production readiness; improvement is uneven by era and correction activates sparsely. |
| 0102 zero timestamp unlocks | Meteorological valid time is not equivalent to operational availability; blocked physics must stay diagnostic. | That upper-air/daily/marine mechanisms are unimportant. |
| 0100 top safe feature `target_roll14_std_lag7_c` and 204-feature safe backlog | Target-memory shape and volatility contain deployable residual information. | That a broad model using all safe features will improve MAE. |
| 0050 station trajectory atlas | Station 450070 temperature anomaly vs its 14-day baseline had strong target-anomaly relation and nontrivial official-residual relation. | That the individual station is stable enough to use without metadata, coverage, and regime checks. |
| 0051 station interaction atlas | A 450110 anomaly/trend gate crossed with 592870 temperature level created large target and residual spreads. | That hard 3×3 cells generalize; sparse cells need smooth shrinkage and fold-local thresholds. |
| Formal EXP-0051–0099 mostly rejected | Broad direct models, Euclidean/DTW analogs, threshold soups, and unconstrained expert additions usually dilute the core. | That the underlying physical signals are false; many should be repurposed as conditional residual or uncertainty features. |

### 3.2 Performance comparison across non-equivalent frames

| Record | Candidate/frame | N | Date range | MAE | RMSE | Delta vs official/baseline | Interpretation |
|---|---|---:|---|---:|---:|---:|---|
| 0075 | `causal_onmem_refine_all_h20_n10_cap0p2_lift_weighted`; old common official frame | 2,670 | 2000-01-02 to 2023-12-31, non-contiguous | 0.945603 | 1.201776 | favorable; exact baseline differs by record | Real online-memory evidence; requires expanded-frame replay. |
| 0081 | hardened narrow-frame research champion | 2,670 | same endpoint geometry | 0.943755 | 1.200926 | incremental over 0075 | Best narrow-frame research score; modern RSS gate is small and fragile. |
| 0103 | 0101 MAM trajectory specialist; expanded official frame | 5,265 | 2000-01-02 to 2023-12-31, non-contiguous | 1.049074 | 1.351423 | −0.030034 vs official | Current canonical expanded-frame benchmark. |
| 0103 RSS slice | same candidate; exact RSS only | 992 | 2021-04-14 to 2023-12-31 | 0.989647 | not separately promoted | −0.026179 vs official | Useful modern exact-vintage slice, but only 2.7 years. |
| EXP-0050 core | `long_history_core_v1`; common-row 2020–2023 | 1,460 | 2020-01-01 to 2023-12-31 | 1.236598 | 1.571377 | core reference | Weather-only mechanism benchmark, not official-anchor replacement. |
| R17 scoreboard | era-transfer terms | 18,624 | OOF 1965–2023 | 1.232364 | 1.561883 | different long-history frame | Best R14–R17 headline; cannot be compared directly to official frames. |

### 3.3 Numbered official-anchor progression

The numbered sequence demonstrates genuine but diminishing returns. Approximate best-MAE milestones on the old 2,670-row frame were 0.9959 (0016), 0.9921 (0017), 0.9898 (0037), 0.9880 (0039), 0.9846 (0040), 0.9831 (0041), 0.9803 (0049), 0.9521 (0067), 0.9504 (0068), 0.9493 (0069), 0.9456 (0075), 0.9449 (0078), 0.9439 (0079), and 0.9438 (0081). On the expanded 5,265-row frame, the path was roughly 1.0599 (0083/0084), 1.0540 (0086), 1.0517 (0087), 1.0507 (0092), 1.05066 (0093), 1.04984 (0094), 1.04952 (0095), 1.04947 (0099), and 1.04907 (0101/0103).

The lesson is not “more buckets always win.” The lesson is that source-aware memory created the largest step, while subsequent specialist gains are small and vulnerable to frame changes. The next work must improve identification of *when* a correction is deserved rather than merely generating more corrections.

## 4. Dataset and source eligibility matrix

| Source family | Approximate documented coverage | Main attributes | Point-in-time status | Use now | Blocker or caveat | Highest-value role |
|---|---|---|---|---|---|---|
| HKO target Tmax labels | 1884–2023 development | daily target Tmax | label only | yes as outcome; lagged self-history only | never use T value as predictor | target memory, climatology, residual labels |
| Expanded robust feature matrix | 48,577 rows, 566 columns, 1884–2023 | target memory, calendar, IGRA, ISD, climate, engineered terms | mixed eligibility by column | yes only through whitelist | feature-level timestamp status required | unified atlas and proxy discovery |
| R14 upper-air matrix | 26,632 rows, 120 columns | target/calendar + upper-air/stability + ISD | upper-air blocked for deployable scoring | diagnostic | provider release/available-at proof missing | mechanism teacher |
| R15/R17 matrices | 23,943 rows, 120 columns | coupling, upper-air, surface, era transfer | mixed/blocked | diagnostic plus safe subsets | same timestamp issue | interaction priors and proxy targets |
| R16 matrix | 25,202 rows, 120 columns | station/network and ISD | station cutoff summaries treated as currently usable in numbered analyses; still require explicit archive-vintage contract | safe research with audit | ISD is quality-controlled retrospective archive | long-history station signal discovery |
| HKO official press forecast | scoreable 2000–2011 era | forecast max/min/text/revisions | eligible only when issue time and parser row valid | yes | incomplete raw-detail continuity, null fields, era changes | anchor and residual labels |
| HKO RSS forecast | 992 pre-2024 scored rows, 2021-04-14–2023-12-31 | exact-vintage numerical ranges/text | deployable on exact vintage | yes | short modern history | anchor, revision, trust routing |
| NOAA ISD regional surface | many stations, mainly 1940s/1950s onward | temperature, dewpoint, pressure, wind, report/QC timing | research-safe only under conservative T−1 pre-15:00 construction and explicit eligibility record | yes for current safe queue | retrospective QC and heterogeneous station eras | spatial state, advection, uncertainty |
| NOAA IGRA upper air | 1949–2026 archive | heights, temperatures, moisture, winds, stability | diagnostic-only after 0102 | no production-style scoring | no issue/available-at or all-row latency proof | physics teacher and safe-proxy target |
| HKO daily climate elements | 1884–2026 archive | rain, RH, cloud, wet bulb, sunshine, radiation, sea temperature, wind, visibility | diagnostic-only unless safely lagged with publication proof | mechanism research only | first-publication timestamps missing | teacher labels and proxy design |
| HKO marine/daily sea temperature | long values in climate table; live feeds short | marine moderation/tide/coastal state | long history diagnostic; live prospective | diagnostic/prospective | publication timestamp missing | marine-suppression teacher |
| TC best track | long retrospective track | position, intensity, motion | diagnostic-only | regime labels only | post-analysis revisions | teacher for TC mechanism proxies |
| Radar/satellite/lightning/nowcast | short current/live | cloud/rain/convection | prospective after retrieval | later | insufficient long history | live regime layer and teacher |
| HKO high-frequency temperature | 2020-06–2026-06; 16.86M rows, 39 stations | 1-minute temperature | prospective/exact retrieval; 2024+ sealed | pre-2024 diagnostic | only ~3.5 development years | heating curves and propagation |
| Since-midnight max/min | 2020-06–2026-06; 16.88M rows, 39 stations | running extrema | prospective/exact retrieval | pre-2024 diagnostic | short history | ceiling trajectory and live calibration |
| High-frequency humidity | 2020-06–2026-06; 10.67M rows, 28 stations | 1-minute RH | prospective | pre-2024 diagnostic | short history | moisture/cloud suppression teacher |
| High-frequency pressure | 2021-06–2026-06; 4.64M rows, 12 stations | 1-minute pressure | prospective | pre-2024 diagnostic | shorter still | front/pressure-wave timing |
| High-frequency solar/UV | 2021/2020–2026 | solar, UV | prospective | pre-2024 diagnostic | only two solar sites; daylight/missingness | cloud-break/radiative efficiency |
| High-frequency wind | 2021-06–2026-06; 11.61M rows, 31 stations | speed/direction | prospective | pre-2024 diagnostic | short history, exposure heterogeneity | sea breeze, convergence, advection |
| NCEP GRIB inventory | inventory only | not decoded | not eligible | no | variables/cycles/leads/bytes not defined | later only for a named signal gap |
| Static geospatial packages | static inventory | elevation, coast, terrain, urban context | eligible after deterministic mapping | engineering now | not yet converted to station-level features | physics-informed grouping and graph priors |

## 5. Station-network evidence and immediate interpretation

The station network is the most underexploited safe source family, but it should be treated as a spatial sensor array rather than a bag of columns.

- The 0047 atlas evaluated 36 stations and 540 station attributes. The strongest stations differed by response and era; recurring IDs include `450070-99999`, `450110-99999`, `592870-99999`, `592930-99999`, `594930-99999`, `595010-99999`, `596730-99999`, `590960-99999`, and `590870-99999`.
- `450070-99999` temperature relative to its own 14-day mean had a 2000–2023 target-anomaly correlation near 0.581 and official-error correlation near −0.108. That is strong level/regime evidence, but residual value is far smaller than target-level value.
- The 0051 interaction screen found that a `450110-99999` temperature-anomaly/trend gate crossed with `592870-99999` temperature level produced a target-anomaly spread of about 4.78°C and an official-error spread around 0.61°C. Hard cells are not automatically deployable; the next step is smooth conditional response estimation with temporal shrinkage.
- Long-history aggregate evidence includes pressure-plane latitude slope, station wind, morning-to-midday temperature rise, and regional dewpoint change. Pressure gradients and dewpoint changes repeatedly appear more useful than static pressure or humidity levels.
- Station identity remains partly anonymous. Physical interpretation must wait for a station dossier with names, coordinates, elevation, bearing, distance, coastal/inland/urban/island role, and coverage/QC history. The implausible `450090-99999` longitude is a mandatory quarantine case.

### Physics-first station hypotheses

1. **Inland heating vs marine suppression:** compare inland/continental stations with coastal, airport, island, and urban groups; condition on wind sector and dewpoint.
2. **Advection arrival:** use pressure/dewpoint/temperature tendency timing across ordered stations to infer whether a regime boundary is approaching HKO.
3. **Sea-breeze penetration:** detect rank reversals and growing coastal–inland spreads, not merely daily means.
4. **Cool-surge modification:** northern stations should lead HKO temperature/pressure/wind changes, with magnitude decaying toward the coast.
5. **Network disagreement:** dispersion may not fix point MAE directly, but it can predict absolute error and tell the router to abstain.

## 6. What the formal EXP-0051–0099 failures teach us

The formal long-history sequence is extremely valuable precisely because it preserves failures. Nearly every broad candidate was worse than `long_history_core_v1` on the 1,460-row 2020–2023 common row set. The closest candidates were still materially worse: conditional champion gate roughly +0.082°C MAE, soft upper-air mixture roughly +0.104°C, synoptic archetype roughly +0.107°C, Waglan wind roughly +0.146°C, cloud/sunshine memory roughly +0.150°C, recency ensemble roughly +0.165°C, and nested residual stacking roughly +0.178°C. DTW trajectories and spell-hazard models were among the weakest.

This rules out several naive routes:

- Do not train a giant weather-only model on every diagnostic feature and expect it to beat the official forecast.
- Do not repeat broad Euclidean or DTW analog matching. Restrict analog distance to a physically defined state and predict residuals, not raw Tmax.
- Do not reuse sparse hard threshold cells without shrinkage and fold-local support checks.
- Do not add an expert merely because its standalone target correlation is high. It must add orthogonal residual information or improve uncertainty/trust decisions.
- Do not interpret upper-air or daily-climate diagnostic rank as permission to use those rows operationally.

The failures redirect the program toward **conditional information gain, safe proxy conversion, bounded residual specialists, and abstaining routers**.

## 7. Response-variable library for information-gain mining

Every feature family must be tested against more than raw Tmax. Experiment 0108 will materialize these definitions once and all later experiments must import them.

| Response | Definition | Strategic use |
|---|---|---|
| `target_tmax_c` | observed HKO Tmax on T | direct level modelling and physical interpretation |
| `target_anomaly_c` | T Tmax minus past-only day-of-year climatology | removes dominant seasonal cycle |
| `official_residual_c` | target minus exact eligible official max forecast | correction direction and magnitude |
| `official_abs_error_c` | absolute official residual | uncertainty and trust routing |
| `official_overforecast_c` | max(official−target, 0) | cold/cloud/marine failure specialist |
| `official_underforecast_c` | max(target−official, 0) | hot/subsidence/cloud-break failure specialist |
| `high_error_flag` | prior-defined absolute error threshold, e.g. >1.5°C and >2.0°C | tail enrichment and specialist activation |
| `hot_day_underforecast_flag` | hot climatology state plus positive target−official tail | high-value hot miss detection |
| `cold_day_overforecast_flag` | cool state plus negative target−official tail | surge/cloud/rain miss detection |
| `mam_high_error_flag` | MAM row plus high error | transition-specific research |
| `station_core_residual_c` | target minus station-only OOF prediction | information beyond regional surface core |
| `online_memory_residual_c` | target minus 0075-style OOF corrected forecast | remaining error after strongest simple correction |
| `correction_uplift_c` | absolute baseline error minus absolute candidate error | causal/no-harm gate target |
| `forecast_trust_state` | prior-only probability raw official beats each specialist | routing target, never fit on future outcomes |

## 8. Information-gain measurement standard

A feature is not “useful” because one correlation is large. Every atlas must provide:

1. Pearson and Spearman correlation after seasonal residualization.
2. Quantile response curves using thresholds estimated in training history only.
3. Monotonicity and sign-reversal diagnostics.
4. Fold-local mutual-information-style binning with bias correction and permutation nulls.
5. Conditional response curves after controlling for calendar, target memory, and official anchor.
6. Signed residual asymmetry: overforecast and underforecast effects separately.
7. High-tail enrichment: odds ratios for >1.5°C, >2°C, and >3°C errors.
8. Year, season, month, source, source-era, and late-window stability.
9. Block-bootstrap confidence intervals preserving serial dependence.
10. False-discovery-rate control across the full tested family, not only the reported top rows.
11. Redundancy/orthogonality against existing champion features.
12. Coverage, missingness, observation age, and whether missingness itself predicts regime or simply archive quality.

### Signal promotion ladder

1. Diagnostic relation exists with adequate support.
2. Timestamp eligibility passes.
3. Feature construction is causal and reproducible.
4. Fold-local addition improves a relevant baseline.
5. Sign and magnitude are stable across years/seasons/sources.
6. P90/P95 and signed tails do not worsen.
7. Simpler residual-memory and fixed-bias baselines are beaten.
8. Row-level lineage and leakage audit are complete.
9. 2024+ confirmation remains sealed.
10. Candidate can be replayed unchanged after forecast backfill.

## 9. Canonical frame design

| Frame ID | Purpose | Rows/date span | Permitted families | Primary comparisons |
|---|---|---|---|---|
| `F-O2670` | reproduce old official-anchor research | 2,670, 2000-01-02–2023-12-31 non-contiguous | exact eligible forecast plus safe features | 0075/0081 lineage only |
| `F-O5265` | canonical expanded official-anchor development | 5,265, same endpoints with expanded press rows | exact eligible forecast plus safe features | 0103 baseline and all new anchor work |
| `F-RSS992` | exact modern RSS robustness | 992, 2021-04-14–2023-12-31 | exact RSS vintage, safe station/target features | modern source behavior, leave-year-out |
| `F-LONG` | long-history target/station discovery | source-dependent, mainly 1947/1949–2023 | target memory, calendar, eligible station features | five-year rolling-origin information gain |
| `F-DIAG` | blocked physical mechanism research | long history through 2023 | IGRA, HKO daily/marine, best track as teachers | diagnostic association/proxy alignment only |
| `F-HF-DEV` | high-frequency short-history discovery | 2020/2021–2023 only | pre-cutoff high-frequency values | leave-year/event-block diagnostics; never final confirmation |
| `F-CONFIRM` | final sealed confirmation | 2024+ | unopened until owner instruction | one-time final audit only |

## 10. Universal validation, sample, and no-harm gates

- Long-history feature screening: at least 1,000 usable rows globally; at least 300 per broad quantile; at least 150 per two-feature cell after fold-local binning; and evidence in at least three non-overlapping temporal eras.
- Official-frame specialists: at least 200 historical trigger rows globally, 100 within a season/source analysis, and 60 only for a heavily shrunk exploratory specialist. No hard unshrunk correction may activate on fewer than 100 prior rows.
- High-frequency diagnostics: at least 300 independent target days and leave-one-year-out or event-block validation. Minute rows do not count as independent sample size.
- Promotion-oriented point correction: improve canonical F-O5265 MAE by at least 0.003°C, or improve a predeclared severe slice by at least 0.01°C without global harm. Require nonpositive MAE delta in at least 70% of temporal folds and no material (>0.01°C) degradation in any major season/source unless a documented tradeoff is explicitly accepted.
- Tail gate: P90/P95 absolute error, >2°C count, and >3°C count may not worsen materially. A specialist aimed at one tail must not create the opposite tail.
- Complexity gate: beat global bias correction, source bias, 0075-style residual memory, and a fixed small blend. Otherwise preserve the finding as diagnostic only.
- Correction caps: start with ±0.15, ±0.25, and ±0.40°C; larger corrections require strong, repeated tail evidence.
- All hyperparameters, thresholds, source weights, station normals, PCA/graph transforms, calibration maps, and text vocabularies are fold-local.

## 11. Ranked immediate experiment sequence

1. 0104 canonical frame/evidence registry.
2. 0105 station dossier and coordinate/QC correction map.
3. 0106 feature availability and lineage graph.
4. 0108 canonical response library.
5. 0109 temporal multiplicity/stability harness.
6. 0161 exact 0075/0081 replay on F-O5265.
7. 0111 cross-fitted conditional information-gain atlas on safe features.
8. 0123 target-memory × station-network coherence atlas.
9. 0127 station rank/reversal motif atlas.
10. 0131 pressure/dewpoint/temperature propagation atlas.
11. 0171 MAM latent-transition state atlas.
12. 0117 correction-uplift/no-harm atlas.
13. 0162 hierarchical source/era residual memory.
14. 0177 extreme-error precursor model.
15. 0170 abstaining trust router.

The remaining experiments are ordered below so that foundation, signal mining, mechanism conversion, and promotion are separated rather than mixed.

# Part II — 80 new experiments


## 0104 — Canonical Frame, Evidence, and Scoreboard Registry

**Priority:** P0  
**Research mode:** Foundation / audit  
**Eligibility:** Deployable now  
**Dependencies:** None; this is the organizing prerequisite

### Decision question and hypothesis

The largest avoidable research risk is frame confusion rather than lack of candidate features. A single immutable registry that recreates every major score frame will prevent false progress, expose row-selection differences, and let all later experiments compare like with like.

### Why this is new rather than a relabelled prior experiment

Existing folders document individual frames, but no one artifact reconciles F-O2670, F-O5265, F-RSS992, R14–R17, the 1,460-row common-row benchmark, and diagnostic-only rows under one row-level identity map. This is not another model; it is the missing research control plane.

### Response variables

No new predictive response is required. The experiment verifies `target_tmax_c`, official prediction, official residual, current candidate prediction, and frame-membership flags row by row.

### Exact inputs

Canonical repository inputs (resolve from repository root, never from ad-hoc copies):
- `experiments/0000_research_state_and_data_contract/README.md` through `experiments/0103_current_rss_safe_continuation/README.md` and all formal `experiments/EXP-*` records.
- `data/datasets/01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet`.
- `data/datasets/04_noaa_isd_regional_surface/noaa_isd_station_day_cutoff_summary.parquet` and, when sub-daily timing is required, `noaa_isd_core_observations.parquet`.
- `data/datasets/05_hko_historical_rss_forecasts/hko_official_t15_scored_pre2024.parquet` and `hko_press_archive_temperature_forecast_days.parquet` for exact-vintage official-anchor work.
- `data/datasets/12_hkg_t24_robust_experiment_outputs/hkg_t24_exp0050_0099_feature_matrix.parquet` plus the R14–R17 feature, OOF-prediction, diagnostic, fold-delta, and scoreboard Parquets.

Also read every `_manifest.json`, `summary.json`, `scoreboard.csv`, `predictions.csv/parquet`, and data-range declaration in all experiment folders.

### Feature constructions and calculations

Create deterministic keys `target_date`, `forecast_source_family`, `issued_at_hkt`, `available_at_hkt`, `selected_vintage_id`, `frame_id`, and `row_hash`. Add Boolean memberships for every known frame and columns explaining each exclusion: missing target, no eligible vintage, parser null, source gap, blocked feature, fold geometry, or confirmation seal.

### Procedure

Rebuild each frame from source files rather than copying prior row lists. Join historical predictions by immutable row key; reproduce published MAE/RMSE within numerical tolerance; produce pairwise overlap/difference tables and date-gap calendars. For every score difference, decompose it into row-selection effect and prediction-change effect. Hash all inputs and publish a machine-readable frame contract.

### Walk-forward validation and minimum evidence

Reproduction audit, not model validation. Require exact row counts and target-date boundaries for all canonical frames; compare each rebuilt metric with its folder-reported value to 1e-9 where predictions are available and document any unavoidable serialization rounding.

**Minimum sample rules:** 100% of rows in every declared frame must have a disposition. Zero unexplained row drops and zero duplicate row keys.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0104_canonical_frame_evidence_registry/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`frame_registry.csv`, `frame_membership.parquet`, `frame_pairwise_overlap.csv`, `metric_reproduction.csv`, `gap_calendar.csv`, `row_exclusion_reasons.csv`, `source_era_map.csv`, and `canonical_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Accept only if F-O2670=2,670 rows, F-O5265=5,265 rows, F-RSS992=992 rows, published date endpoints match, the 3,500-day major gap is reproduced, and all benchmark scores reconcile within tolerance. Any discrepancy blocks downstream score claims until resolved.

### Expected failure modes and interpretation

Likely failures are inconsistent issue-time selection, old research/data_analysis path aliases, duplicated forecast vintages, and predictions joined only on date when source/vintage is also required. A failure is useful: it identifies why historical scores were not comparable.

### Expected information gain

No immediate MAE gain. Extremely high information value because it converts every later delta into an auditable apples-to-apples result and is likely to eliminate the largest class of false discoveries.

---

## 0105 — Station Dossier, Identity, Geography, and Coverage Forensics

**Priority:** P0  
**Research mode:** Foundation / data quality  
**Eligibility:** Deployable now after deterministic metadata derivation  
**Dependencies:** 0104 frame registry

### Decision question and hypothesis

Anonymous station IDs hide physical roles and allow metadata defects to masquerade as signal. Mapping every station to verified identity, geometry, era, exposure, and coverage will improve grouping, prevent bad spatial calculations, and explain why station importance changes by regime.

### Why this is new rather than a relabelled prior experiment

0047 and 0050 ranked station IDs and reported coordinates, but did not deliver a complete, adjudicated station dossier with coast/terrain/urban roles, metadata conflicts, era changes, and a quarantine policy. Formal EXP-0082 addressed homogenization broadly, not this station-by-station scientific registry.

### Response variables

Metadata/QC responses: coordinate sanity, coverage continuity, observation cadence, variable availability, missingness, station offset against peers, and station contribution by target anomaly, official residual, absolute error, and signed tail.

### Exact inputs

`noaa_isd_station_day_cutoff_summary.parquet`, `noaa_isd_core_observations.parquet`, static geospatial inventory, station metadata embedded in 0047/0050/0051, and all station IDs appearing in top features or interactions.

### Feature constructions and calculations

For each station compute name, aliases, latitude, longitude, elevation, start/end, per-year counts, cadence, report types, distance/bearing to HKO, pair distances, elevation difference, coastline distance/orientation when derivable, urban/coastal/inland/island/airport/open-exposure labels with confidence, topography sector, and likely meteorological role. Create metadata-break and relocation flags. Explicitly quarantine `450090-99999` until its longitude is verified.

### Procedure

Cross-reconcile all available metadata sources; never silently choose one. Plot station trajectories and coverage heatmaps. Test coordinate swaps/sign errors by geographic plausibility and pair-distance discontinuity. Generate physical station groups and a confidence score; allow unknown rather than forced labels. Recalculate prior top station interactions with and without suspect stations.

### Walk-forward validation and minimum evidence

Metadata reconciliation plus sensitivity analysis. Station groups must be defined without target outcomes. Re-run top 0047/0050/0051 statistics after quarantines to quantify whether conclusions survive.

**Minimum sample rules:** Every station used by a proposed feature must have a dossier row. Group-level analyses require at least three stations when possible and at least 70% group coverage on a target date.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0105_station_dossier_geography_coverage/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`station_dossier.csv`, `station_aliases.csv`, `metadata_conflicts.csv`, `station_coverage_by_year.csv`, `station_pair_geometry.csv`, `station_groups.csv`, `quarantine_list.csv`, `top_signal_sensitivity.csv`, and map-ready GeoJSON.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Accept if 100% of stations in the normalized ISD files and all top-ranked experiment features are mapped or explicitly unresolved, every coordinate receives a sanity status, and top conclusions are recalculated after exclusion of bad metadata.

### Expected failure modes and interpretation

Geospatial packages may lack coast distance or urban classification. Do not fabricate labels; preserve confidence and use only geometry that can be reproduced. A station with strong signal but unstable identity is diagnostic-only.

### Expected information gain

High information gain and indirect model value. It enables physically coherent groups, prevents nonsensical gradients, and should reduce fragile station-specific feature proliferation.

---

## 0106 — Per-Feature Availability, Eligibility, and Lineage Graph

**Priority:** P0  
**Research mode:** Foundation / leakage audit  
**Eligibility:** Deployable now  
**Dependencies:** 0104 and source manifests

### Decision question and hypothesis

A column-level whitelist is safer and more useful than family-level assumptions. Encoding source valid time, provider issue time, retrieval time, release-latency proof, and operational cutoff logic for every feature will separate safe research from attractive but blocked retrospective data.

### Why this is new rather than a relabelled prior experiment

0053 and 0102 audited selected candidates/families, but the expanded 566-column matrix and derived experiment features still lack one complete lineage graph. This experiment makes eligibility executable rather than narrative.

### Response variables

Eligibility classes: `label_only`, `deployable_exact`, `deployable_assumed_with_proof`, `lagged_safe`, `diagnostic_only`, `prospective_only`, `blocked_unknown`, plus reason codes.

### Exact inputs

All source manifests; 0102 source-timing evidence; 0100 feature atlas; 0053 candidate audit; schemas and feature-definition files for the expanded matrix, R14–R17, station summaries, and official forecasts.

### Feature constructions and calculations

For every raw and derived column store source ID, raw columns, formula, units, valid-time rule, issue/available-at fields, retrieval timestamp, required lag, operational cutoff, first eligible date, revision risk, transformation-fit scope, missingness rule, and downstream dependents. Build a directed acyclic lineage graph from raw source to prediction.

### Procedure

Parse definitions where available, infer nothing silently, and mark unknowns. Execute synthetic boundary tests around T−1 15:00 HKT and target-day midnight. Generate an allowlist for each frame and fail closed when a feature lacks proof. Trace the 0101 champion and all 204 safe-backlog features to raw inputs.

### Walk-forward validation and minimum evidence

Unit-test at least 100 sampled rows per source, all boundary timestamps, and every champion feature. Recompute derived values from raw inputs where source rows are present. Compare statuses with 0102 and explain every difference.

**Minimum sample rules:** All 566 expanded-matrix columns, every R14–R17 column, and every feature used by a promoted or near-promoted numbered experiment must have one lineage record.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0106_feature_availability_lineage_graph/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`feature_registry.parquet`, `lineage_edges.csv`, `eligibility_by_frame.csv`, `cutoff_unit_tests.csv`, `blocked_unlock_queue.csv`, `unknown_lineage.csv`, and `deployable_whitelist.yaml`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Zero unclassified promoted features; zero champion dependencies on blocked columns; all unknowns fail closed; 0102 blocked counts reconcile or a documented proof artifact justifies the difference.

### Expected failure modes and interpretation

Old experiments may lack machine-readable formulas. Mark them unreconstructable and do not promote them. Do not upgrade eligibility from mere historical valid time.

### Expected information gain

Very high audit value; moderate model value by preventing wasted work and enabling safe automated feature sweeps.

---

## 0107 — Timezone, Daily Boundary, Unit, Duplicate, and Observation-Age Audit

**Priority:** P0  
**Research mode:** Foundation / data quality  
**Eligibility:** Deployable now  
**Dependencies:** 0104–0106

### Decision question and hypothesis

A one-hour timezone shift, full-day aggregate crossing the cutoff, unit inconsistency, or duplicate report can create more apparent skill than any real feature. A destructive red-team audit should precede deeper spatial timing work.

### Why this is new rather than a relabelled prior experiment

Several experiments state conservative timing, and EXP-0084 tested report/QC weighting, but no repository-wide adversarial audit aligns historical offsets, HKT boundaries, station report ages, target dates, and forecast issue times across every source.

### Response variables

Audit outcomes: duplicate rate, timestamp-offset distribution, latest-pre-cutoff age, post-cutoff contamination flag, unit-range violation, impossible rate-of-change, daily-label mismatch, and sensitivity of top signals to ±1/±8-hour shifts.

### Exact inputs

Raw/normalized target labels, ISD core observations and station-day summaries, official forecast issue/vintage rows, IGRA valid times, high-frequency manifests, and experiment-derived feature rows where available.

### Feature constructions and calculations

Construct `observation_age_minutes`, `seconds_from_cutoff`, `source_timezone_offset`, historical offset regime, same-day/full-day inclusion flags, duplicate signature, report-type priority, unit plausibility checks, and shifted-control versions of top station features.

### Procedure

Rebuild station cutoff summaries from core observations under explicit timezone rules. Compare to normalized summaries. Intentionally shift timestamps and daily boundaries to measure how much reported signal changes; a suspicious skill spike under the wrong boundary is a red flag. Check Celsius/Fahrenheit, hPa/Pa, knots/m/s, direction wrapping, and missing-code decoding.

### Walk-forward validation and minimum evidence

Use deterministic row-level reconciliation and negative-control shifts. Top features must retain sign and most effect after removing stale, duplicate, suspect-unit, and boundary-adjacent rows.

**Minimum sample rules:** Audit all rows. For sensitivity claims require at least 1,000 rows or 80% of the feature’s normal support.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0107_temporal_unit_qc_audit/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`timestamp_reconciliation.csv`, `cutoff_rebuild_comparison.csv`, `duplicate_report_audit.csv`, `unit_range_audit.csv`, `observation_age_profiles.csv`, `shift_negative_controls.csv`, and `quarantined_rows.parquet`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

No unexplained difference between rebuilt and normalized cutoff summaries above a predeclared tolerance; no promoted feature dominated by boundary-adjacent or stale reports; all unit anomalies dispositioned.

### Expected failure modes and interpretation

Historical daylight-saving/offset metadata may be incomplete. Preserve raw UTC and documented local offset, and flag ambiguous eras rather than force modern HKT. If top signal collapses after QC, treat prior finding as artifact.

### Expected information gain

High downside protection. It may reduce apparent information, which is a success if that information was leakage or timestamp error.

---

## 0108 — Canonical Multi-Response Label and Residual Library

**Priority:** P0  
**Research mode:** Foundation / target engineering  
**Eligibility:** Deployable now  
**Dependencies:** 0104

### Decision question and hypothesis

Mining only raw Tmax conflates seasonal level with forecast blind spots. A shared, causal response library will let every source be evaluated for level, residual, uncertainty, sign, high-error tails, and trust without redefining labels in each experiment.

### Why this is new rather than a relabelled prior experiment

Prior atlases used several targets independently, but definitions and thresholds are dispersed. This experiment freezes one row-level set of responses and prior-only threshold policies for all later work.

### Response variables

Materialize the full response table listed in Section 7: target, past-only anomaly, official residual/absolute/signed components, high-error flags at fixed and prior-quantile thresholds, MAM tails, station-core residual, 0075/0103 residual, correction uplift, and forecast-trust outcomes.

### Exact inputs

Target labels, canonical official frames, long-history OOF predictions, 0075/0081/0103 predictions where reconstructable, and canonical season/source/era mappings.

### Feature constructions and calculations

Create past-only day-of-year climatology with circular kernel and minimum history; prior-only hot/cold thresholds; residual sign with zero tolerance policy; severity thresholds; source-aware and source-agnostic variants; event IDs grouping consecutive high-error days; and censoring flags where an anchor is unavailable.

### Procedure

Build responses once from immutable predictions. Do not recompute a baseline inside downstream experiments. Store provenance for every response. Validate algebraic identities such as residual = target−forecast and absolute error = |residual|. Produce distribution and overlap tables.

### Walk-forward validation and minimum evidence

No model fitting except fold-safe climatology/threshold estimation. Use expanding history and compare threshold stability across eras. Verify that response construction never uses current/future target values beyond the label itself.

**Minimum sample rules:** All rows in F-LONG receive target/anomaly when possible; all F-O5265 rows receive official responses; derived baseline residuals require exact OOF prediction availability.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0108_canonical_response_library/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`canonical_responses.parquet`, `response_definitions.csv`, `threshold_history.parquet`, `event_labels.csv`, `response_coverage.csv`, and `algebra_unit_tests.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

One authoritative response definition per name, no duplicate conflicting labels, no threshold fitted using its evaluation row, and exact reconciliation to official benchmark metrics.

### Expected failure modes and interpretation

Some historical predictions may not be recoverable row by row. Mark those response variants unavailable rather than substituting in-sample estimates.

### Expected information gain

Foundational information value; it changes what “signal” means and is essential for finding residual and uncertainty features that raw-target correlations miss.

---

## 0109 — Temporal Multiple-Testing, Stability, and Negative-Control Harness

**Priority:** P0  
**Research mode:** Foundation / statistical validity  
**Eligibility:** Deployable now  
**Dependencies:** 0104, 0108

### Decision question and hypothesis

Testing thousands of features, lags, bins, and pairs will inevitably produce impressive false positives unless serial dependence, multiplicity, and era stability are handled centrally.

### Why this is new rather than a relabelled prior experiment

Existing atlases report train/eval sign checks, but there is no reusable framework for block permutations, false-discovery-rate control, effect shrinkage, temporal replication, and negative controls across all new searches.

### Response variables

Any canonical response, with special support for continuous residual, absolute error, binary tail flags, and correction uplift.

### Exact inputs

Canonical frames/responses and a feature matrix supplied by each downstream experiment.

### Feature constructions and calculations

No meteorological features. Statistical artifacts include block-bootstrap confidence intervals, year-block permutation nulls, Benjamini–Hochberg q-values, empirical-Bayes shrinkage, sign-stability score, era heterogeneity, late-window decay, and placebo lags/leads.

### Procedure

Provide a library and CLI that takes a feature table and response specification. Use contiguous time blocks rather than row shuffling. Include negative controls: future-shifted features (which must fail eligibility), randomized station IDs, seasonal phase rotations, and synthetic noise with matched autocorrelation. Rank by replicated effect, not raw p-value.

### Walk-forward validation and minimum evidence

Self-test on synthetic known-signal and null datasets, then reproduce selected 0046/0047 atlas effects. Calibrate empirical false-positive rate under null permutations.

**Minimum sample rules:** At least 20 temporal blocks for formal q-values when possible; otherwise label inference exploratory. No interaction is “stable” unless it appears with consistent direction in at least three separated eras or folds.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0109_temporal_multiplicity_stability_harness/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

Reusable `stability.py`, `permutation_nulls.parquet`, `effect_confidence.csv`, `fdr_results.csv`, `negative_control_results.csv`, and `stability_rank.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Empirical false-positive rate near declared alpha on null tests; known synthetic signals recovered; all future experiment scoreboards include q-value, block CI, and temporal stability fields.

### Expected failure modes and interpretation

With only 992 RSS rows, formal power will be low. Report wide intervals and use long-history station/target evidence for mechanism replication rather than relaxing thresholds.

### Expected information gain

Very high research quality gain; likely to demote many fragile signals and concentrate compute on effects that can survive production.

---

## 0110 — One-Command Walk-Forward Replay and Artifact Harness

**Priority:** P0  
**Research mode:** Foundation / implementation  
**Eligibility:** Deployable now  
**Dependencies:** 0104, 0106, 0108, 0109

### Decision question and hypothesis

Research throughput and trust improve when every candidate is run through the same immutable folds, baseline definitions, metrics, lineage checks, and artifact schema.

### Why this is new rather than a relabelled prior experiment

Formal experiments have a template and many numbered folders are self-contained, but implementations and output shapes vary. This creates hidden degrees of freedom and makes reruns after archive backfill expensive.

### Response variables

All canonical responses, chosen by config. Default promotion response is official residual/point MAE on F-O5265.

### Exact inputs

Canonical frame registry, response library, deployable whitelist, baseline predictions, and any candidate feature table.

### Feature constructions and calculations

Configurable preprocessors restricted to fold-local transforms; online state API with score-then-update semantics; specialist gate API; source hierarchy; metric suite; and artifact writers.

### Procedure

Build a CLI such as `python -m hkg_t24.run_experiment --config experiments/XXXX/config.yaml`. Enforce frame ID, feature whitelist, fold geometry, random seed, model/correction cap, min history, and confirmation seal. Fail before fitting on any blocked feature or row >=2024.

### Walk-forward validation and minimum evidence

Replay at least 0074, 0075, 0101, and the raw official baseline. Compare predictions/metrics with historical artifacts and explain any differences.

**Minimum sample rules:** Harness must support all rows, sparse specialists, and online state without silently dropping rows. Every dropped row gets a reason code.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0110_walkforward_replay_harness/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

Reusable package, `config_schema.json`, baseline configs, replay comparison reports, deterministic environment lock, and example generated experiment folder.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A clean checkout can recreate canonical baseline scores with one command; repeated runs are byte-identical except timestamps; forbidden features and confirmation rows trigger hard failures.

### Expected failure modes and interpretation

Historical scripts may encode undocumented behavior. Preserve both “exact legacy replay” and “corrected canonical implementation,” and never blend their scores.

### Expected information gain

No direct lift; very high compounding value because it turns 70 subsequent ideas into comparable, backfill-replayable experiments.

---

## 0111 — Cross-Fitted Conditional Information-Gain Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now for safe features; blocked families diagnostic-only  
**Dependencies:** 0104–0110

### Decision question and hypothesis

Many features correlate with Tmax only because of season or recent temperature. The valuable features are those retaining information after conditioning on calendar, target memory, and the official anchor. Cross-fitted conditional residualization should reveal orthogonal station and trajectory signals hidden by raw correlation rankings.

### Why this is new rather than a relabelled prior experiment

0002/0046/0100 rank raw or partially adjusted relationships. This experiment computes out-of-fold conditional information against multiple baselines and separates target-level, anchor-residual, uncertainty, and tail value.

### Response variables

`target_anomaly_c`, `official_residual_c`, `official_abs_error_c`, signed tail flags, `station_core_residual_c`, and `online_memory_residual_c`.

### Exact inputs

Canonical safe feature whitelist from target-memory, calendar, and ISD station/network families; optional blocked features run in a physically separate diagnostic table. Include all 204 safe backlog features and station trajectory variants from 0050.

### Feature constructions and calculations

For each candidate create raw, robust-z, within-month rank, anomaly-to-own-14/30/60-day baseline, 1/3/7-day change, 7/30 slope contrast, missingness/age, and interactions only after main-effect screening. Residualize feature and response using fold-local calendar plus baseline predictions; calculate conditional Pearson/Spearman, incremental deviance, conditional MI, and block-permutation importance.

### Procedure

Run nested cross-fitting: outer temporal fold for evaluation; inner training folds fit nuisance models for response and feature. Measure incremental information over (a) calendar, (b) target memory, (c) official raw, and (d) 0075-style memory. Rank separately by response and season/source. Preserve all negative rows.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Require effect replication in at least three temporal folds and compare against matched-autocorrelation noise features.

**Minimum sample rules:** 1,000 global rows per feature for long-history ranking; 500 official-overlap rows; 200 per quantile; 100 per source-season slice for descriptive reporting only.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0111_conditional_information_gain_atlas/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`conditional_ig_atlas.parquet`, `nuisance_model_scores.csv`, `response_specific_leaderboards.csv`, `incremental_vs_baseline.csv`, `fold_effects.csv`, and `blocked_diagnostic_atlas.parquet`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Promote a signal to the interaction queue only if conditional effect direction is stable, block-permutation q<0.05 or replicated effect is compelling, coverage is adequate, and it adds information beyond at least one relevant simple baseline.

### Expected failure modes and interpretation

Flexible nuisance models can remove true nonlinear signal or leak through tuning. Use simple predeclared nuisance classes and report sensitivity. Strong raw correlation that vanishes conditionally is a level proxy, not residual value.

### Expected information gain

Very high discovery value; likely to shrink the feature universe dramatically and identify a small orthogonal station/target set for modelling.

---

## 0112 — Nonlinear Monotonicity and Conditional Response-Curve Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now for safe features  
**Dependencies:** 0111

### Decision question and hypothesis

A feature can have near-zero global correlation yet strongly alter residuals near physical thresholds, saturation points, or sign reversals. Smooth response curves are more stable than hard tertile cells.

### Why this is new rather than a relabelled prior experiment

0009 examined threshold lifts and later experiments used discrete cells. This experiment replaces threshold chasing with cross-fitted isotonic/spline response curves, uncertainty bands, derivative maps, and explicit monotonicity tests.

### Response variables

Official residual, absolute error, hot-underforecast, cold-overforecast, target anomaly, and correction uplift.

### Exact inputs

Top 100 response-specific safe candidates from 0111 plus predeclared physical variables: dewpoint change, T−Td spread, pressure tendency, wind components, coastal-inland spread, target slope, target volatility, and forecast-minus-memory disagreement.

### Feature constructions and calculations

Use robust percentiles, circular wind transforms, log transforms where physically appropriate, and fold-local seasonal standardization. Estimate partial response curves after controlling for baseline state. Compute local slope, turning points, saturation threshold, hysteresis by prior state, and sign asymmetry.

### Procedure

Fit low-degree monotone splines, unconstrained penalized splines, isotonic regressions, and quantile curves inside each outer fold. Compare out-of-fold curve shape across eras and bootstrap blocks. Convert only stable curve regions into candidate smooth correction functions with shrinkage to zero outside support.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Evaluate curve-shape stability with integrated absolute derivative difference and sign agreement across folds.

**Minimum sample rules:** At least 500 rows per global curve, 250 per seasonal curve, 100 effective rows on either side of any claimed turning point.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0112_nonlinear_response_curve_atlas/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`response_curves.parquet`, `curve_derivatives.parquet`, `turning_points.csv`, `monotonicity_tests.csv`, `curve_stability.csv`, and `candidate_smooth_functions.yaml`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A curve is actionable only if its material region repeats in three folds, has sufficient support, improves a predeclared baseline when converted to a bounded cross-fitted correction, and does not rely on one extreme bin.

### Expected failure modes and interpretation

Spline wiggles and sparse tails can create false structure. Penalize complexity, compare to monotone/no-effect nulls, and label unstable thresholds as negative results.

### Expected information gain

High insight, moderate model-lift potential. Most value will come from replacing brittle hard buckets with small smooth corrections.

---

## 0113 — Asymmetric Tail Dependence and Copula-Style Error Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain / tail risk  
**Eligibility:** Deployable now for safe features  
**Dependencies:** 0108, 0109, 0111

### Decision question and hypothesis

Features useful for catastrophic hot underforecasts may have weak mean effects. Tail co-occurrence, quantile dependence, and residual-sign asymmetry should identify precursors that average-MAE screens miss.

### Why this is new rather than a relabelled prior experiment

Prior high-error autopsies and directional MAM cells used thresholds. This experiment measures continuous upper/lower tail dependence across all safe families and tests whether tail enrichment survives temporal blocks.

### Response variables

Top/bottom 10% official residual, >1.5/2/3°C absolute error, hot-underforecast flag, cold-overforecast flag, and P90 conditional absolute error.

### Exact inputs

Safe target-memory, station-network, calendar, forecast-state, and missingness features; blocked upper-air/marine values in a separate diagnostic teacher atlas.

### Feature constructions and calculations

Convert each feature to fold-local ranks. Estimate lower/upper quantile dependence, tail odds ratios, quantile mutual information, conditional exceedance curves, and pairwise copula cells for physically selected pairs. Include residual sign and season/source conditioning.

### Procedure

Use rolling-origin folds and year-block bootstrap. Compare observed tail dependence with autocorrelation-preserving permutations. Test whether a feature enriches only one tail or merely high variance. Build a candidate precursor score from no more than five replicated features with logistic shrinkage.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Primary objective is Brier/log loss and recall at fixed 10% alert rate, followed by tail-count reduction under a bounded specialist.

**Minimum sample rules:** At least 100 tail events globally, 50 per directional tail for exploratory reporting, and 30 events per fold before claiming stability.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0113_asymmetric_tail_dependence_atlas/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`tail_dependence_atlas.parquet`, `tail_odds_ratios.csv`, `quantile_dependence_curves.csv`, `tail_event_catalog.csv`, and `precursor_candidate_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Require replicated tail enrichment, calibrated out-of-fold event probability, no use of future quantiles, and either meaningful alert precision lift or a reduction in P95/>2°C errors without average-MAE damage.

### Expected failure modes and interpretation

Rare-event estimates are noisy and thresholds drift. Use fixed physical thresholds alongside prior-quantile thresholds, shrink probabilities, and preserve failed alerts.

### Expected information gain

Moderate-to-high information gain; potentially more valuable for trading risk and uncertainty than for mean point MAE.

---

## 0114 — Feature Redundancy Graph and Minimal Orthogonal Signal Set

**Priority:** P1  
**Research mode:** Exploratory information-gain / dimensionality discipline  
**Eligibility:** Deployable now for safe features  
**Dependencies:** 0111–0113

### Decision question and hypothesis

The 566-column matrix contains many transformations of the same seasonal or thermal state. A graph of conditional redundancy can identify a minimal physically diverse feature set and prevent feature-soup dilution.

### Why this is new rather than a relabelled prior experiment

Formal stacking and broad feature additions failed, but no experiment explicitly built a response-specific redundancy graph with temporal conditional dependence and chose one representative per physical mechanism.

### Response variables

Separate graphs for target anomaly, official residual, absolute error, hot tail, cold tail, and online-memory residual.

### Exact inputs

Features surviving 0111/0112/0113 plus all existing champion components and baseline predictions.

### Feature constructions and calculations

Compute pairwise rank correlation, distance correlation, conditional MI, shared permutation importance, residualized correlation, and prediction substitution loss. Annotate each node with source family, station, transform, coverage, eligibility, and physical mechanism.

### Procedure

Build graph edges when two features are highly redundant after seasonal residualization. Cluster with stability consensus across folds. Select representatives by coverage, timestamp certainty, physical interpretability, and incremental response information. Run leave-cluster-out ablations around simple residual models.

### Walk-forward validation and minimum evidence

Construct graph in each training era and compare cluster adjusted Rand index and representative stability. Never use evaluation outcomes to select the representative for that fold.

**Minimum sample rules:** Features need 1,000 overlapping long-history rows or 500 F-O5265 rows; pairwise overlap must exceed 70% of the smaller feature’s support.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0114_feature_redundancy_orthogonal_set/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`redundancy_edges.parquet`, `feature_clusters.csv`, `representative_set_by_response.csv`, `cluster_stability.csv`, `leave_cluster_out_ablation.csv`, and graph visualization files.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Produce a compact set that retains at least 95% of cross-fitted diagnostic information or candidate lift while materially reducing feature count. No blocked feature may represent a deployable cluster.

### Expected failure modes and interpretation

Correlation can label complementary nonlinear features as redundant. Use response-conditional substitution tests, not correlation alone. If representatives change wildly by fold, retain the mechanism as unstable.

### Expected information gain

High modelling discipline; likely improves generalization and interpretability more than headline score immediately.

---

## 0115 — Hierarchical Two- and Three-Feature Interaction ANOVA

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now for safe inputs  
**Dependencies:** 0111, 0112, 0114

### Decision question and hypothesis

The strongest weather information is conditional, but exhaustive hard cells overfit. Hierarchical functional ANOVA with shrinkage can quantify whether a pair or triplet adds genuine interaction beyond smooth main effects.

### Why this is new rather than a relabelled prior experiment

0010/0046/0051/0087/0094 mined interactions through discrete cells or candidate models. This experiment decomposes main and interaction effects cross-fitted, ranks incremental interaction variance, and applies heredity constraints.

### Response variables

Target anomaly, official residual, absolute error, and directional tails.

### Exact inputs

At most 40 representative safe features from 0114, grouped into target memory, thermal level, moisture, pressure, wind, station spatial state, forecast state, and missingness/quality.

### Feature constructions and calculations

Predeclare physically meaningful pairs and a limited triplet queue: target slope×dewpoint change; pressure tendency×wind sector; coastal-inland spread×humidity; forecast disagreement×station heat anomaly; residual memory×network disagreement; MAM phase×moisture×pressure. Use smooth tensor products or hierarchical group lasso, never an unrestricted 40³ search.

### Procedure

Fit fold-local additive main-effect model, then add one interaction block at a time. Estimate incremental out-of-fold deviance/MAE, interaction variance, sign maps, support density, and fold stability. Shrink sparse surface regions to main effects.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Compare every interaction to its exact main-effect-only parent and to a permuted interaction preserving marginals.

**Minimum sample rules:** At least 1,000 rows for pair surfaces, 2,000 for triplets; no reported local region below 150 rows globally or 60 prior rows at activation.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0115_hierarchical_interaction_anova/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`interaction_queue.csv`, `functional_anova_scores.csv`, `interaction_surfaces.parquet`, `surface_support.parquet`, `main_vs_interaction_ablation.csv`, and `stable_interactions.yaml`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

An interaction advances only if it improves its parent in most folds, has stable broad topology, adds information after correction for multiple tests, and can be expressed as a bounded interpretable surface.

### Expected failure modes and interpretation

Interactions can proxy source era or missingness. Include these as controls and test within source/era. If only one tiny surface patch helps, do not promote it as a global feature.

### Expected information gain

High information gain; modest point-lift potential from a small number of physically coherent surfaces.

---

## 0116 — Information-Gain Drift, Era Transfer, and Effect-Lifetime Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain / robustness  
**Eligibility:** Deployable now  
**Dependencies:** 0111–0115

### Decision question and hypothesis

A feature may be physically real but operationally obsolete because station networks, urbanization, forecast sources, or climate state changed. Measuring effect lifetime and transfer decay will determine appropriate training windows and shrinkage.

### Why this is new rather than a relabelled prior experiment

0007 and EXP-0052/0057 studied stability and eras broadly. This experiment focuses on response-specific conditional effect drift for the newly narrowed safe signal set and estimates when each feature stops transferring.

### Response variables

Conditional target anomaly effect, official residual effect, tail odds, and candidate correction uplift.

### Exact inputs

Stable candidates from 0111–0115, station metadata eras from 0105, source eras from 0104, and baseline predictions.

### Feature constructions and calculations

Rolling 5/10-year effect estimates, expanding estimates, CUSUM/Page-Hinkley diagnostics, coefficient half-life, sign-persistence duration, pre/post station-break contrast, and source-era interaction. Use physical season standardization.

### Procedure

Estimate feature response curves in successive historical windows and score transfer to the next block. Fit a hierarchical state-space effect model diagnostically, then compare static, trailing-window, and exponentially decayed estimates. Attribute shifts to source/station changes where possible.

### Walk-forward validation and minimum evidence

No change point may be selected using future evaluation rows. Report forward transfer error and whether a drift alarm would have fired before degradation.

**Minimum sample rules:** Each rolling window needs at least 500 rows overall or 150 seasonal rows; era comparisons require 200 rows per side.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0116_information_gain_drift_lifetime/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`effect_time_series.parquet`, `effect_half_life.csv`, `transfer_matrix.csv`, `drift_alarms.csv`, `station_break_attribution.csv`, and `recommended_memory_by_feature.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Assign each feature a stable-long, decaying, source-specific, season-specific, or unusable class. Promotion requires an update rule supported by forward transfer, not hindsight.

### Expected failure modes and interpretation

Apparent drift may reflect coverage composition. Reweight to common stations/features and report composition-adjusted estimates. If effect lifetime is shorter than available history supports, keep diagnostic.

### Expected information gain

High robustness value; directly informs online halflives and prevents old-history dilution.

---

## 0117 — Correction Uplift and No-Harm Opportunity Atlas

**Priority:** P1  
**Research mode:** Promotion-oriented information gain  
**Eligibility:** Deployable now  
**Dependencies:** 0108, 0111–0116

### Decision question and hypothesis

Predicting the residual is insufficient: a correction should activate only when it is likely to reduce absolute error versus leaving the official forecast untouched. Modelling correction uplift directly can discover safer gates and abstention regions.

### Why this is new rather than a relabelled prior experiment

Prior routers compare expert performance retrospectively and use prior-lift gates. This experiment constructs row-level potential uplift for a library of fixed bounded corrections and learns which pre-cutoff states justify each action.

### Response variables

For actions δ∈{−0.40,−0.25,−0.15,0,+0.15,+0.25,+0.40}, response is `|official−target| − |official+δ−target|`; also best-action label and no-harm probability.

### Exact inputs

F-O5265, exact official forecast, safe representative features, online residual state, target-memory state, station-network state, and source/era identifiers.

### Feature constructions and calculations

Action-specific uplift curves; forecast-vs-memory disagreement; residual-memory magnitude/confidence; network disagreement; station heat/moisture/pressure regimes; MAM phase; source age; and missingness confidence. Use only pre-cutoff inputs.

### Procedure

Use nested temporal folds. Estimate action uplift with conservative doubly robust or direct outcome models, heavily regularized. Compare to simple sign-of-prior-bias and fixed correction. Activate only when lower confidence bound on expected uplift is positive; otherwise choose zero correction.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Primary metrics are realized MAE delta, harmful-activation rate, regret versus zero action, and calibration of predicted uplift.

**Minimum sample rules:** At least 200 prior examples for any action-state region; at least 50 realized activations per evaluation fold; shrink to zero below support.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0117_correction_uplift_no_harm_atlas/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`action_uplift_oof.parquet`, `uplift_calibration.csv`, `action_support.csv`, `harmful_activation.csv`, `policy_scoreboard.csv`, and `abstention_map.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Policy must beat zero action and prior-bias baselines on F-O5265, reduce or preserve tails, and show a harmful-activation rate materially below an ungated residual model. Improvement must survive source/season ablation.

### Expected failure modes and interpretation

Uplift is hard to estimate because actions are counterfactual but outcomes for fixed deterministic shifts are calculable; still, flexible policy selection can overfit. Keep action grid tiny and thresholds nested.

### Expected information gain

One of the highest likely routes to safe incremental lift because it optimizes the actual correction decision instead of residual fit alone.

---

## 0118 — Value of Information Under Missingness, Age, and Source Dropout

**Priority:** P2  
**Research mode:** Exploratory information-gain / operational robustness  
**Eligibility:** Deployable now  
**Dependencies:** 0105–0111

### Decision question and hypothesis

A station or feature may be useful when fresh and widely observed but harmful when stale, sparse, or supported by a changing subset. Quantifying conditional value of information will guide fallbacks and prevent silent distribution shift.

### Why this is new rather than a relabelled prior experiment

EXP-0083 tested missingness-robust learning and numbered work notes dropout, but no response-specific atlas measures the incremental value of each source conditional on freshness, group coverage, and fallback composition.

### Response variables

Target anomaly, official residual, absolute error, high-error flags, and candidate uplift.

### Exact inputs

Safe station features with observation times/QC, source coverage from 0105/0107, canonical responses, and representative signal set.

### Feature constructions and calculations

Observation age, stations-present count, group coverage, nearest-station availability, pattern hash, missingness burst length, source dropout transition, imputation distance, and leave-one-source-out prediction delta.

### Procedure

For each source/group, compare cross-fitted baseline performance with and without the source across freshness and coverage states. Use missingness-pattern clustering fitted on prior rows. Test whether missingness is meteorological by controlling for year/source era and report type. Simulate operational dropout on high-value days.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Include leave-station-out and synthetic dropout stress tests; all imputers fit fold-locally.

**Minimum sample rules:** 200 rows per freshness/coverage band, 100 dropout events for a claimed signal, and at least three stations for group fallback.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0118_value_of_information_missingness/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`source_value_atlas.csv`, `freshness_response_curves.csv`, `dropout_pattern_scores.csv`, `leave_source_out.csv`, `fallback_policy.yaml`, and `operational_stress_results.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Define explicit source-use and fallback rules that preserve score under realistic dropout. Treat missingness as a feature only if its meteorological effect survives era/QC controls.

### Expected failure modes and interpretation

Archive gaps often mimic weather regimes. If missingness signal vanishes after era control, retain it solely for confidence/fallback, not prediction.

### Expected information gain

Moderate information gain; high operational value and useful for uncertainty scaling.

---

## 0119 — Causal Target Level–Slope–Curvature State Space

**Priority:** P1  
**Research mode:** Exploratory then promotion-oriented  
**Eligibility:** Deployable now  
**Dependencies:** 0108–0111

### Decision question and hypothesis

Tomorrow’s Tmax constraint depends on the joint state of recent level, trend, acceleration, and seasonal anomaly. A robust causal state representation should outperform isolated lag and slope columns and reveal when persistence versus mean reversion dominates.

### Why this is new rather than a relabelled prior experiment

EXP-0053 trajectory analogs failed and 0101 found `trajectory_7_30_slope_c_per_day` useful in a narrow MAM cell. This experiment does not repeat broad trajectory matching; it decomposes recent HKO history into interpretable local derivatives and tests residual value conditionally.

### Response variables

Target anomaly for direct constraint; official residual and 0075-memory residual for correction; absolute error for forecastability.

### Exact inputs

HKO target labels through T−1; existing safe target-memory fields including lags, rolling means, slope contrasts, volatility, and decayed climatology.

### Feature constructions and calculations

Fit fold-local robust local-linear and local-quadratic regressions over 7/14/21/30/45/60-day windows ending T−1 and, as a conservative sensitivity, ending T−7. Derive level, slope, curvature, slope disagreement across windows, acceleration sign, distance from past-only seasonal normal, percentile state, and state uncertainty. Add Kalman-filter-like causal latent level/slope estimates with fixed predeclared process-noise grid.

### Procedure

Create a target-state atlas first; compare each state component to raw lag features. Then fit tiny bounded residual corrections conditioned on state regions, using shrinkage and no-harm gating. Test whether station information in 0123 adds lift beyond this state.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. All smoothing is one-sided and reinitialized without future data in each fold. Compare ending T−1 versus T−7 to isolate operational-lag assumptions.

**Minimum sample rules:** 1,000 long-history rows per state variant; 200 rows per state quintile; 100 official rows per promoted correction region.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0119_target_level_slope_curvature_state/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`target_state.parquet`, `state_feature_atlas.csv`, `window_sensitivity.csv`, `persistence_mean_reversion_map.csv`, `bounded_correction_scoreboard.csv`, and `state_examples.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A state representation must add conditional information beyond `target_lag7_tmax_c`, `target_roll14_std_lag7_c`, and `trajectory_7_30_slope_c_per_day`; any promoted correction must improve F-O5265 and not worsen tails.

### Expected failure modes and interpretation

Smooth states can merely restate seasonality. Require conditional residual tests and compare to simple rolling mean/slope. Overly adaptive filters can chase noise; select process noise only inside training folds.

### Expected information gain

High information gain, moderate lift potential, and foundational value for later regime interaction work.

---

## 0120 — Robust Local Derivatives, Reversal Hazard, and Trend Exhaustion

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0119

### Decision question and hypothesis

A warming or cooling run does not have constant continuation probability. Reversal risk may depend on derivative consistency, accumulated anomaly, volatility, and station-network disagreement rather than run length alone.

### Why this is new rather than a relabelled prior experiment

EXP-0055’s broad spell-hazard model failed badly. This experiment narrows the question to derivative exhaustion and requires external station confirmation; it avoids direct raw-Tmax hazard replacement.

### Response variables

Next-day target change, sign reversal, official residual, hot underforecast, cold overforecast, and correction uplift.

### Exact inputs

Causal target state from 0119, target lags, volatility, and later station confirmation fields from 0123; first phase can run target-only.

### Feature constructions and calculations

Theil–Sen slopes; median pairwise derivative; slope sign agreement across 7/14/30 days; cumulative signed move; distance traveled in rolling MAD units; curvature; number of consecutive same-sign changes; plateau length; overshoot beyond seasonal quantile; and reversal-pressure score. Interact only with prior-known volatility and station thermal/dewpoint tendency.

### Procedure

Estimate fold-local reversal probability and residual sign curves. Compare target-only, target+volatility, and target+station-confirmation specifications. Use a small hazard/logistic model diagnostically; promotion is a bounded gate between persistence and zero correction, not a full direct forecast.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Evaluate Brier score, calibration, continuation/reversal precision, and realized correction uplift by season.

**Minimum sample rules:** At least 150 reversal events for a global specification and 50 per season for descriptive curves; no state-specific action below 100 prior rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0120_target_derivative_reversal_exhaustion/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`derivative_features.parquet`, `reversal_event_catalog.csv`, `hazard_curves.csv`, `calibration.csv`, `station_confirmation_ablation.csv`, and `persistence_gate_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Advance only if reversal probability is calibrated in multiple eras and a gate improves a simple persistence/official baseline without recreating EXP-0055’s broad degradation.

### Expected failure modes and interpretation

Autocorrelated target changes can make run features look predictive in-sample. Use forward folds and compare to empirical season-only reversal rates. If station confirmation adds no lift, preserve target-only curves as diagnostic.

### Expected information gain

Moderate information gain; likely useful in MAM and cool-surge termination slices rather than globally.

---

## 0121 — Volatility Compression–Expansion Transition Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain / uncertainty  
**Eligibility:** Deployable now  
**Dependencies:** 0108, 0119

### Decision question and hypothesis

The change in volatility regime—not volatility level alone—may mark transitions where official forecasts lose trust. Compression followed by directional station movement can precede breakouts; expansion followed by coherence can mark stabilization.

### Why this is new rather than a relabelled prior experiment

EXP-0056 tested broad volatility/entropy modelling and was rejected. This experiment isolates transition dynamics and uses volatility mainly for uncertainty and gating, not as a direct weather-only predictor.

### Response variables

Official absolute error, high-error flag, target breakout magnitude, residual sign, and correction uplift.

### Exact inputs

Target history, safe station temperatures/dewpoints/pressure, canonical official residuals, and source/season labels.

### Feature constructions and calculations

Rolling MAD/IQR/std over 7/14/30/60 days; log ratio short/long volatility; volatility slope; realized range; sign-change rate; compression percentile; expansion shock; station-network dispersion ratio; target vs network volatility divergence; and post-compression directional coherence.

### Procedure

Map conditional error distributions across volatility transition states. Test whether network coherence disambiguates harmless volatility from synoptic transition. Train a calibrated high-error probability model with at most six predeclared features, then use it only to scale confidence or gate specialists.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Compare against target_roll14_std_lag7_c alone and report calibration/P90 error by volatility state.

**Minimum sample rules:** 200 rows per broad compression/neutral/expansion state; 100 high-error events per fitted uncertainty model.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0121_volatility_compression_expansion/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`volatility_transition_features.parquet`, `transition_error_atlas.csv`, `breakout_curves.csv`, `uncertainty_calibration.csv`, and `gating_ablation.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Require stable ranking of error risk, improved absolute-error calibration versus simple volatility, and no point-MAE damage when used as a gate.

### Expected failure modes and interpretation

Volatility can be a season proxy. Residualize within month/season and test cross-season transfer. Entropy-like features are demoted unless they beat simple MAD ratios.

### Expected information gain

High uncertainty information; modest direct MAE potential but important for abstention and tail control.

---

## 0122 — Causal Seasonal Phase-Boundary and Monsoon-Transition Index

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now with safe inputs  
**Dependencies:** 0119, 0121

### Decision question and hypothesis

Calendar month is too coarse for Hong Kong transitions. A causal phase index based on recent thermal, moisture, pressure, and wind evolution may locate the actual onset, pause, reversal, or maturity of seasonal regimes each year.

### Why this is new rather than a relabelled prior experiment

Calendar harmonics and MAM specialists exist, but no fold-safe year-specific phase coordinate has been built from safe pre-cutoff observations and tested against residuals.

### Response variables

Target anomaly, official residual, MAM high-error, cold-overforecast, hot-underforecast, and source trust.

### Exact inputs

Safe target state, station-network dewpoint/pressure/wind/temperature changes, past-only calendar harmonics, and station coverage controls. Blocked daily-climate values can be used only as diagnostic labels for phase interpretation.

### Feature constructions and calculations

Cumulative thermal anomaly, dewpoint rise, pressure trend, northerly/easterly wind persistence, station gradient change, volatility transition, and circular day-of-year. Fit a causal hidden-state or changepoint index on prior years only; output continuous phase, phase speed, confidence, and transition type.

### Procedure

Define simple physics-informed states first: cool monsoon, humid transition, unstable oscillation, warm maritime, hot mature summer, autumn retreat. Fit fold-local soft state probabilities and compare to calendar-only states. Test residual/error behavior by phase and phase speed.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. State labels/parameters must be trained on prior years; assess year-to-year onset-date stability and forward classification without using future season outcomes.

**Minimum sample rules:** At least 10 prior years before estimating year-specific phase; 200 rows per broad state and 80 per transition edge.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0122_seasonal_phase_boundary_index/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`seasonal_phase.parquet`, `phase_definitions.csv`, `onset_dates.csv`, `phase_response_atlas.csv`, `calendar_ablation.csv`, and `transition_examples.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Phase must explain residual or high-error variation beyond day-of-year/month in multiple years and improve a MAM or seasonal gate without damaging other seasons.

### Expected failure modes and interpretation

Latent states may be arbitrary or swap labels. Anchor them with explicit physical constraints and publish raw component scores. If phase adds no conditional value, keep calendar harmonics.

### Expected information gain

High information gain for MAM/DJF/SON specialists; moderate global lift potential.

---

## 0123 — Target-Memory × Station-Network Coherence Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain then promotion  
**Eligibility:** Deployable now  
**Dependencies:** 0105, 0119, 0111

### Decision question and hypothesis

Recent HKO history constrains tomorrow only when the regional network confirms the same air-mass trajectory. Divergence between target memory and surrounding station state should identify persistence failures and official-forecast blind spots.

### Why this is new rather than a relabelled prior experiment

0051 found station pair interactions and 0101 used a target trajectory cell, but no experiment systematically measures target-vs-network coherence across level, slope, moisture, pressure, and spatial gradient with smooth shrinkage.

### Response variables

Official residual, online-memory residual, target change, high-error flag, and correction uplift.

### Exact inputs

Causal target state, top safe station features from 0050/0111, station groups from 0105/0140, and official frames.

### Feature constructions and calculations

Target slope minus regional thermal slope; target anomaly minus nearby/urban/coastal group anomaly; sign agreement; rank agreement; target volatility vs network dispersion; target warming with dewpoint/pressure opposition; forecast max minus target envelope crossed with network heat anomaly; and coherence confidence based on coverage.

### Procedure

Create one-dimensional coherence scores and a limited set of pair surfaces. Rank by incremental information over target state and station state separately. Test tiny corrections only in persistent disagreement regions, with prior-only support and source/season hierarchy.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Include leave-station-group-out tests and compare against the existing 450110×592870 hard-cell finding.

**Minimum sample rules:** 1,000 long-history rows for atlas; 200 official rows per disagreement regime; 100 prior rows before correction activation.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0123_target_station_coherence_atlas/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`coherence_features.parquet`, `coherence_atlas.csv`, `disagreement_surfaces.parquet`, `group_ablation.csv`, `bounded_specialist_scoreboard.csv`, and `case_studies.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Require signal beyond both component main effects, stable disagreement direction, and a bounded specialist that improves F-O5265 or a predeclared MAM/tail slice with no global harm.

### Expected failure modes and interpretation

Differences may reflect station calibration or coverage. Use station anomalies, metadata eras, and common-station sensitivity. Hard pair dependence without group replication is not promoted.

### Expected information gain

One of the most promising safe routes because it directly asks when target persistence should be trusted.

---

## 0124 — Spell Termination with Spatial Confirmation

**Priority:** P2  
**Research mode:** Exploratory specialist  
**Eligibility:** Deployable now  
**Dependencies:** 0120, 0123

### Decision question and hypothesis

Hot/cold spell age alone failed, but spell termination may become predictable when regional pressure, dewpoint, wind, and station rank changes confirm an approaching transition.

### Why this is new rather than a relabelled prior experiment

This is a disciplined rescue of EXP-0055: it does not fit a broad spell-duration forecast. It tests a small set of physically confirmed termination triggers and applies no correction when confirmation is absent.

### Response variables

Termination next day, official residual on termination days, hot-spell underforecast/overforecast, cold-spell overforecast/underforecast, and uplift from bounded reversal corrections.

### Exact inputs

Past-only target spell definitions, target derivative exhaustion, station propagation features, wind/pressure/dewpoint changes, seasonal phase, and official forecast.

### Feature constructions and calculations

Spell age and cumulative anomaly crossed with northern pressure rise, dewpoint surge/drop, wind-sector change, thermal-front approach, station rank reversal, and forecast disagreement. Separate hot and cold spells and season-specific mechanisms.

### Procedure

Define spells using prior climatology percentiles. Predeclare confirmation rules, then estimate smooth termination probability. Compare target-only, station-only, and combined confirmation. Apply a correction only when prior lower confidence bound of uplift is positive.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Event-level folds prevent consecutive spell days from splitting across train/test; score onset/termination events, not inflated daily rows.

**Minimum sample rules:** At least 100 completed hot spells and 100 cold spells for global analysis; 40 confirmed termination events per specialist class.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0124_spell_termination_spatial_confirmation/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`spell_events.csv`, `termination_confirmations.csv`, `event_level_scores.csv`, `termination_probability_oof.parquet`, and `reversal_correction_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Combined confirmation must beat duration-only and target-derivative-only baselines, and any correction must reduce termination-day errors without harming continuation days.

### Expected failure modes and interpretation

Few extreme spells and overlapping definitions reduce power. Use broad anomaly spells first; preserve null results rather than tuning thresholds repeatedly.

### Expected information gain

Moderate targeted information, likely strongest in DJF cold-surge breakdown and MAM warm transitions.

---

## 0125 — Phase-Aligned Year-over-Year and Submonth Analog Residuals

**Priority:** P2  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0122, 0114

### Decision question and hypothesis

Same calendar day in prior years is too rigid, while unrestricted DTW analogs diluted signal. Restricting analogs to causal seasonal phase and a minimal safe state may recover recurring transition patterns and residual bias.

### Why this is new rather than a relabelled prior experiment

EXP-0053 and EXP-0092 rejected broad trajectory analogues. This experiment uses a small physically normalized state, prior dates only, explicit phase alignment, and predicts official residual rather than raw Tmax.

### Response variables

Official residual, online-memory residual, residual sign, and analog uncertainty.

### Exact inputs

Seasonal phase index, minimal orthogonal safe feature set, target state, station group state, and official frames.

### Feature constructions and calculations

Distance on phase, target anomaly/slope, regional temperature anomaly, dewpoint change, pressure tendency, and wind sector. Normalize each by training-history variability; enforce same broad season and exclude nearby dates/episodes. Derive kNN residual mean, median, dispersion, sign agreement, and effective sample size.

### Procedure

Compare same-day-of-year climatological residual, phase-only analog, target-only analog, station-only analog, and combined physical analog. Use k and distance weights chosen inside training folds. Add analog correction only when effective sample and sign agreement exceed predeclared thresholds.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Analog pool contains only prior dates; use a 60-day temporal exclusion and event exclusion. Compare to online residual memory directly.

**Minimum sample rules:** At least 10 acceptable analogs per prediction and 200 predictions per policy; no specialist if median effective sample <8.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0125_phase_aligned_year_analog_residuals/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`analog_neighbors.parquet`, `analog_feature_scaling.csv`, `analog_oof_predictions.parquet`, `distance_ablation.csv`, `memory_comparison.csv`, and `analog_casebook.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Must beat simple source/season residual memory on at least one stable slice or provide materially better uncertainty calibration. Otherwise preserve nearest-neighbor diagnostics only.

### Expected failure modes and interpretation

Regime recurrence may be too weak and distances may collapse to target level. Feature ablations and phase-only controls reveal this. Do not add dimensions after seeing test results.

### Expected information gain

Moderate insight, low-to-moderate lift probability; worthwhile because the failure criterion is explicit and avoids repeating unrestricted analog work.

---

## 0126 — Causal Thermal Constraint Envelope and Breakout Detector

**Priority:** P1  
**Research mode:** Promotion-oriented diagnostic  
**Eligibility:** Deployable now  
**Dependencies:** 0119–0123

### Decision question and hypothesis

Recent target history and season define a plausible next-day Tmax envelope. Official forecasts outside that envelope may be correctly anticipating a regime break—or may be overreacting. Station confirmation can distinguish the two.

### Why this is new rather than a relabelled prior experiment

Existing features include forecast-minus-roll means and target ranges, but no calibrated causal envelope with explicit breakout probability and station-confirmation logic.

### Response variables

Target relative to envelope, official forecast relative to envelope, official residual, breakout flag, and correction uplift.

### Exact inputs

Past-only target state/climatology, target volatility, seasonal phase, official forecast, and station coherence/propagation features.

### Feature constructions and calculations

Fold-local conditional q10/q50/q90 target envelope based on day-of-year, recent level/slope/volatility and spell state; `official_minus_envelope_mid`, distance beyond upper/lower bound, envelope width, station-confirmed breakout score, and false-breakout score.

### Procedure

Fit simple quantile models on prior history. Classify official forecasts as inside, mild outside, or extreme outside. Determine when station evidence supports the breakout. Test policies: trust official when confirmed; shrink toward envelope when unconfirmed; abstain under wide envelope.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Evaluate quantile coverage, breakout precision, and point correction separately. Quantile models and calibration are fold-local.

**Minimum sample rules:** At least 1,000 training rows for envelope; 100 upper and 100 lower out-of-envelope events; 50 confirmed events per direction for policy scoring.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0126_thermal_constraint_envelope_breakout/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`thermal_envelope_oof.parquet`, `coverage_calibration.csv`, `breakout_events.csv`, `station_confirmation_scores.csv`, `shrinkage_policy_scoreboard.csv`, and `tail_effects.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Envelope must be calibrated across seasons; confirmation must improve breakout classification; any shrink policy must beat fixed shrinkage and not suppress genuine extreme heat/cold days.

### Expected failure modes and interpretation

A narrow overfit envelope creates false breakouts. Penalize complexity and prioritize coverage. Official forecasts may contain NWP information that safely violates memory; unconfirmed does not automatically mean wrong.

### Expected information gain

High decision value and plausible small MAE lift from selective shrinkage, especially around overreactive transitions.

---

## 0127 — Station Rank, Rank-Reversal, and Permutation Motif Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0105, 0107, 0111

### Decision question and hypothesis

Absolute station temperatures carry calibration and seasonal level. Relative rank and rank reversals can expose movement of marine fronts, inland heating, and surge boundaries more robustly across station eras.

### Why this is new rather than a relabelled prior experiment

Prior station atlases used network-relative values and pair spreads, but did not treat the full ordering of stations as a dynamic object or mine physically grouped permutation motifs.

### Response variables

Target anomaly, official residual, hot/cold tails, high-error flag, and seasonal phase transition.

### Exact inputs

T−1 pre-cutoff station temperature, dewpoint, pressure, and wind-derived features; station groups/dossier; target and official responses.

### Feature constructions and calculations

Within-network and within-group ranks; rank of HKO-near proxies; Spearman footrule distance from previous day; number and identity of pairwise inversions; coastal-vs-inland rank reversal; north-south ordering; hot-station centroid; rank persistence; and coverage-normalized permutation signatures.

### Procedure

Compute ranks only among present, quality-passing stations and create comparable group-level motifs. Mine frequent motifs using training folds, then evaluate response spreads in future folds. Replace exact permutations with low-dimensional inversion counts and group orderings to avoid sparsity.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Use common-station sensitivity and leave-one-station-out checks. Motifs are defined from training history only.

**Minimum sample rules:** At least five stations present per network motif, three per group, 200 rows per coarse motif, and 100 official-overlap rows for residual claims.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0127_station_rank_reversal_motifs/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`station_ranks.parquet`, `rank_reversal_events.csv`, `motif_catalog.csv`, `motif_response_atlas.csv`, `common_station_sensitivity.csv`, and `candidate_motif_features.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A motif must replicate across eras and be interpretable as a group ordering, not one station artifact. Promotion requires conditional residual value beyond temperature level/spread.

### Expected failure modes and interpretation

Changing station availability alters ranks. Use within-group normalized ranks and pattern confidence; if motif effect disappears on common station sets, classify as coverage artifact.

### Expected information gain

High novel information potential, particularly for sea-breeze and frontal transition detection.

---

## 0128 — Robust Spatial Interpolation and HKO Counterfactual Field Estimate

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now after 0105 geometry  
**Dependencies:** 0105, 0107, 0144 later optional

### Decision question and hypothesis

The regional network can estimate the large-scale field expected at HKO; the difference between this counterfactual field and recent HKO state or official forecast may isolate local urban/marine effects and anchor errors.

### Why this is new rather than a relabelled prior experiment

R16 aggregates, EXP-0075 planes, and graph modes used network fields broadly. This experiment focuses on a robust leave-target-location interpolation with uncertainty, causal station availability, and response-specific residual diagnostics.

### Response variables

Target anomaly, official residual, HKO-minus-field residual, absolute error, and local-effect state.

### Exact inputs

Station dossier/geometry, T−1 cutoff temperature/dewpoint/pressure, station anomalies and trends, target history, and official forecast.

### Feature constructions and calculations

Inverse-distance/elevation-adjusted mean; robust thin-plate or kriging-like interpolation with fixed covariance learned in training; anisotropic distance by wind sector; leave-one-station-out error; interpolation uncertainty; local field gradient; and `official_minus_interpolated_hko_state`.

### Procedure

Compare simple IDW, robust plane, anisotropic kernel, and graph smoother. Fit geometry/hyperparameters inside training folds or from static priors. Use station anomalies rather than raw level where appropriate. Test whether interpolation residual predicts HKO local behavior or official residual.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Spatial cross-validation leaves stations out; temporal OOF evaluates HKO response. Never tune geometry on target test rows.

**Minimum sample rules:** At least five valid stations per date; 1,000 temporal rows; each spatial method must be evaluated on at least 10 stations with adequate coverage.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0128_robust_spatial_interpolation_hko/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`hko_field_estimates.parquet`, `spatial_cv.csv`, `method_comparison.csv`, `interpolation_uncertainty.csv`, `local_effect_atlas.csv`, and `residual_specialist_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Method must beat simple network median as a field estimator and its disagreement must add conditional residual or uncertainty information. A correction requires F-O5265 no-harm evidence.

### Expected failure modes and interpretation

Sparse/heterogeneous station geometry and metadata errors can dominate. Report support radius and uncertainty; abstain when geometry is poor.

### Expected information gain

High information gain and moderate lift potential through a physically meaningful network summary.

---

## 0129 — Thermal-Front Geometry, Orientation, and Distance-to-HKO

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now after station geometry audit  
**Dependencies:** 0105, 0127, 0128

### Decision question and hypothesis

A sharp spatial temperature transition and its orientation can signal an approaching maritime boundary, surge, or inland heat edge. Distance and motion toward HKO should be more informative than a single north-south/east-west gradient.

### Why this is new rather than a relabelled prior experiment

EXP-0075/0077/0078 tested planes and fixed directional gradients. This experiment detects robust fronts, curvature, and front movement in station-anomaly space rather than assuming a fixed axis.

### Response variables

Next-day target change, target anomaly, official residual, high-error flag, and sea-breeze/cool-surge event labels.

### Exact inputs

Station temperature anomalies, geometry, target history, wind state, and pressure/dewpoint companion fields.

### Feature constructions and calculations

Robust spatial gradient vector; piecewise two-region front fit; front normal and tangent; signed HKO distance; gradient magnitude; curvature; thermal centroid; front confidence; 1/2/3-day movement vector; and alignment with wind/pressure gradient.

### Procedure

Fit simple robust plane and change-surface candidates using only present stations. Choose complexity by training-only spatial CV. Track front parameters across prior days. Estimate conditional response curves and event enrichment; compare to fixed N-S/E-W gradients.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Require spatial-fit confidence and common-station sensitivity; movement at T uses only fields through T−1.

**Minimum sample rules:** At least eight stations per front fit and 500 high-confidence days; 100 official rows per front-distance band.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0129_thermal_front_geometry/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`thermal_fronts.parquet`, `front_fit_quality.csv`, `front_tracks.csv`, `fixed_gradient_ablation.csv`, `front_response_atlas.csv`, and `case_maps/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Front geometry must outperform gradient magnitude/axis alone for at least one response and repeat across multiple years. Promotion requires stable bounded correction or uncertainty lift.

### Expected failure modes and interpretation

With sparse stations, a two-region fit can hallucinate a front. Use bootstrap station subsamples and confidence thresholds; low-confidence days produce missing features rather than forced estimates.

### Expected information gain

High exploratory value; likely useful as a regime trigger more than a standalone predictor.

---

## 0130 — Dewpoint-Front, Moisture-Boundary, and Dry-Air Intrusion Geometry

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now with safe station dewpoint  
**Dependencies:** 0105, 0129

### Decision question and hypothesis

Moisture boundaries can alter cloud, haze, sensible heating, and marine influence. Dewpoint change and spatial gradient may explain residuals even when temperature fields look similar.

### Why this is new rather than a relabelled prior experiment

Dewpoint change repeatedly ranks well, but prior work did not map a moving dewpoint boundary, its alignment with thermal fronts, or dry/moist intrusion into HKO.

### Response variables

Target anomaly, official residual, hot underforecast, cold overforecast, high-error probability, and cloud/rain suppression proxy states.

### Exact inputs

Station dewpoint and T−Td spread through cutoff, geometry, thermal-front features, wind vectors, pressure tendency, and target/official responses.

### Feature constructions and calculations

Dewpoint gradient/front orientation, HKO distance, moisture centroid, dry-air intrusion score, moist-surge score, dewpoint-front speed, thermal–moisture front alignment/misalignment, T−Td spatial gradient, and upwind moisture contrast.

### Procedure

Replicate 0129 front methodology for dewpoint, then study joint four-state regimes: warm/moist, warm/dry, cool/moist, cool/dry intrusions. Estimate response curves by season and wind sector. Build a diagnostic cloud/heating suppression score.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Compare dewpoint level, one-day change, simple gradient, and front geometry; require added value over temperature fronts.

**Minimum sample rules:** Eight stations for geometry, 500 confident days, 100 official rows per broad intrusion state.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0130_dewpoint_front_geometry/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`moisture_fronts.parquet`, `front_alignment.csv`, `intrusion_states.csv`, `response_atlas.csv`, `temperature_front_ablation.csv`, and `bounded_candidate_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Require stable conditional residual or tail information beyond temperature, season, and wind. Any correction must have physically consistent sign and no-harm evidence.

### Expected failure modes and interpretation

Dewpoint coverage is lower and temperature-dependent humidity can confuse interpretation. Use dewpoint/T−Td, not RH alone, and expose support/coverage.

### Expected information gain

High information potential because regional dewpoint change is one of the strongest recurring station signals.

---

## 0131 — Pressure-Tendency Wave Propagation and Arrival-Time Map

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0105, 0107

### Decision question and hypothesis

Synoptic transitions arrive across the regional station network with structured pressure-tendency timing. Estimated propagation direction, speed, and HKO arrival time may anticipate cool surges, fronts, ridges, and TC-related pressure changes better than pressure level or static gradient.

### Why this is new rather than a relabelled prior experiment

0026–0028 pressure experts and EXP-0074 pressure-gradient fields were weak. This experiment tests propagation timing and crosses pressure change with moisture/temperature/wind instead of repeating pressure-only buckets.

### Response variables

Next-day target change, official residual, high-error flag, cool-surge breakdown/onset, hot-underforecast under ridge building, and uncertainty.

### Exact inputs

Station pressure latest-before-cutoff and 1/3/6/12/24-hour or daily tendencies where available; station geometry; temperature/dewpoint/wind changes; target/official responses.

### Feature constructions and calculations

Station pressure tendency anomaly; onset time of threshold changes; cross-correlation lead estimates fitted only on training data; robust propagation plane in time; direction/speed; predicted HKO arrival; pressure-wave coherence; rise/fall asymmetry; and pressure×dewpoint×wind confirmation.

### Procedure

Use sub-daily observations where eligibility is proven; otherwise T−1 day-cutoff daily tendencies. Detect coherent waves, estimate arrival from stations excluding HKO proxy, and evaluate next-day response. Compare pressure-only to confirmed multi-field propagation.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Event-level blocked validation; lead estimates and thresholds learned from prior events only. Include shuffled-station and reversed-time negative controls.

**Minimum sample rules:** At least five pressure stations per event; 100 coherent rise and 100 fall events globally; 30 per season for descriptive analysis.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0131_pressure_wave_propagation/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`pressure_wave_events.csv`, `propagation_parameters.parquet`, `arrival_predictions.csv`, `confirmation_ablation.csv`, `negative_controls.csv`, and `response_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Arrival features must beat static pressure tendency/gradient for event or residual prediction in multiple folds. No promotion if signal depends on post-cutoff observations.

### Expected failure modes and interpretation

Daily summaries may be too coarse and station pressure elevations differ. Use tendencies/anomalies and explicit observation-age controls. If only high-frequency recent data resolves waves, move result to F-HF-DEV.

### Expected information gain

High novel information gain and strong physical interpretability; direct lift uncertain but regime value is substantial.

---

## 0132 — Wind-Field Divergence, Vorticity, Deformation, and Directional Persistence

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now where station wind is cutoff-safe  
**Dependencies:** 0105, 0107

### Decision question and hypothesis

Wind speed alone is weak because thermal impact depends on direction, convergence, shear, and station exposure. Low-order wind-field kinematics may identify sea-breeze convergence, monsoon surges, stagnation, and changing ventilation.

### Why this is new rather than a relabelled prior experiment

Wind sectors, Waglan persistence, and upwind weighting were tested, but no systematic station-network divergence/vorticity/deformation atlas with geometry, exposure confidence, and residual responses exists.

### Response variables

Target anomaly, official residual, high-error flag, marine suppression, weak-wind heat buildup, and MAM transition error.

### Exact inputs

Station wind speed/direction through cutoff, station geometry/roles, temperature/dewpoint gradients, pressure tendency, and target/official responses.

### Feature constructions and calculations

Convert to u/v. Fit robust affine vector field to derive divergence, vorticity, stretching and shearing deformation; compute directional coherence, calm fraction, sector persistence, wind-shift angle, circular station disagreement, onshore/offshore components relative to local coastline, and exposure-weighted confidence.

### Procedure

Fit kinematics on training-defined station groups and per-date observations. Compare raw speed/sector to field terms. Cross with thermal/moisture gradients to distinguish advection from local sea breeze. Use smooth response curves and event clusters.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Bootstrap stations and exclude exposed outliers. Circular variables use proper angular metrics; transforms are fold-local where learned.

**Minimum sample rules:** At least six wind stations per field fit, 500 high-confidence days, and 100 rows per broad kinematic state.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0132_wind_field_kinematics/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`wind_kinematics.parquet`, `field_fit_quality.csv`, `station_bootstrap.csv`, `thermal_moisture_interactions.csv`, `response_atlas.csv`, and `candidate_regime_flags.yaml`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A kinematic term must add information beyond mean wind u/v, speed, and sector; promoted flags require stable physical sign and no-harm correction/uncertainty evidence.

### Expected failure modes and interpretation

Station exposure differences can mimic divergence. Use role/exposure weights and common-station sensitivity. If field fit is unstable, retain simpler group wind contrasts.

### Expected information gain

High regime-discovery potential, especially for sea breeze and monsoon transitions.

---

## 0133 — Surface Moisture-Flux Convergence and Ventilation Potential

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now with safe station data  
**Dependencies:** 0130, 0132

### Decision question and hypothesis

The product of moisture and wind can distinguish humid advection/convergence from locally high dewpoint. Moisture-flux convergence may proxy cloud/rain suppression, while dry ventilation may permit stronger heating.

### Why this is new rather than a relabelled prior experiment

Prior coupling terms and dewpoint/wind features were largely additive. This experiment constructs a physically meaningful surface moisture transport field and tests its incremental value.

### Response variables

Official residual, target anomaly, hot underforecast, cold overforecast, high-error flag, and blocked cloud/rain teacher labels.

### Exact inputs

Station dewpoint or specific-humidity proxy, pressure/temperature for conversion, u/v winds, station geometry, and thermal-front/phase features.

### Feature constructions and calculations

Approximate vapor pressure/specific humidity; q·u and q·v flux; robust spatial divergence/convergence; upwind moisture flux into HKO; ventilation index wind×T−Td; convergence persistence; moisture-flux change; and alignment with pressure/thermal fronts.

### Procedure

Compute humidity thermodynamics from pre-cutoff values using documented formulas. Fit flux gradients only when geometry/coverage is adequate. Evaluate residual and tail response; use blocked daily cloud/rain only to interpret mechanism and train proxy alignment, never as deployable predictor.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Compare against dewpoint, wind, and their simple product; propagate input QC/coverage uncertainty.

**Minimum sample rules:** Six stations for flux field, 500 confident days, 100 official rows per convergence quintile.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0133_moisture_flux_convergence/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`moisture_flux_fields.parquet`, `thermodynamic_definitions.csv`, `field_uncertainty.csv`, `incremental_ablation.csv`, `teacher_alignment.csv`, and `response_atlas.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Require stable incremental information beyond component features and physically coherent seasonal behavior. Promotion only through a bounded safe specialist.

### Expected failure modes and interpretation

Approximate humidity and sparse wind can amplify noise. Report uncertainty, cap extreme conversions, and do not use RH without temperature correction.

### Expected information gain

Moderate-to-high information gain; likely strongest for cloud-prone MAM/JJA error and uncertainty.

---

## 0134 — Flow-Conditioned Transport and Station-to-HKO Arrival-Time Atlas

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0105, 0132

### Decision question and hypothesis

A station’s relevance should depend on whether it is upwind and on travel time, not merely geographic distance. Flow-conditioned lag selection may recover lead signals diluted in fixed one-day aggregates.

### Why this is new rather than a relabelled prior experiment

EXP-0073 upwind weighting and EXP-0081 lead-lag propagation were rejected as broad models. This experiment narrows the goal to response-specific arrival diagnostics, uses station roles/geometry, and compares physically implied travel time with empirical prior-only lag.

### Response variables

Next-day target change, official residual, target anomaly, and regime-event onset.

### Exact inputs

Station temperature/dewpoint/pressure anomalies and changes, wind-field direction/speed, station geometry, target/official responses.

### Feature constructions and calculations

Projected upwind distance; travel time = projected distance / capped representative wind speed; crosswind distance; age-adjusted feature sampled at nearest prior lag; upwind station set; flow persistence; empirical lead-lag prior from training history; and transport-confidence score.

### Procedure

For each target date, select candidate stations based on pre-cutoff flow. Build travel-time-aligned anomalies using only historical observations. Compare fixed T−1 features, static distance weights, flow-conditioned weights, and arrival-aligned values. Evaluate by wind sector/season.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Travel-time formulas and empirical lag priors fit on training folds. Include wrong-wind and randomized-bearing controls.

**Minimum sample rules:** At least three upwind stations and 500 high-confidence dates; 100 rows per major wind sector for claims.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0134_flow_conditioned_transport_arrival/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`flow_conditioned_features.parquet`, `arrival_time_estimates.csv`, `station_selection_frequency.csv`, `lag_ablation.csv`, `negative_controls.csv`, and `response_atlas.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Flow conditioning must add information over static station anomalies in repeated folds and not depend on one station. A model lift claim requires bounded residual testing.

### Expected failure modes and interpretation

Surface wind may not represent transport aloft and long travel times exceed feature cadence. Cap/flag implausible travel times; treat flow selection as regime weighting, not literal parcel tracking.

### Expected information gain

High scientific value and plausible residual lift for monsoon/advection regimes.

---

## 0135 — Sea-Breeze Penetration Phase and Marine-Suppression Index

**Priority:** P1  
**Research mode:** Exploratory information-gain then specialist  
**Eligibility:** Deployable now from station proxies; marine daily source remains diagnostic  
**Dependencies:** 0105, 0127, 0129, 0132

### Decision question and hypothesis

HKO Tmax can be capped when marine air penetrates inland, but a static coastal–inland spread misses onset, depth, and timing. A phase index using coastal cooling, inland heating, wind shift, rank reversal, and moisture convergence may predict marine suppression and official overforecast.

### Why this is new rather than a relabelled prior experiment

EXP-0076 coastal–inland contrast and EXP-0089/0090 marine variables were weak as broad models. This experiment builds a dynamic multi-sensor phase and explicitly targets residual direction.

### Response variables

Official overforecast, target anomaly, target heating shortfall, high-error flag, and marine-suppression teacher state.

### Exact inputs

Station groups from 0105/0140, temperature/dewpoint/wind/pressure through cutoff, coastal geometry, target state, official forecast, blocked sea-temperature/daily marine values only as diagnostic teachers.

### Feature constructions and calculations

Coastal-inland temperature/dewpoint spread and change; coastal wind onshore component; inland heating anomaly; rank reversal; sea-breeze front distance; convergence; penetration depth proxy; onset/persistence/retreat phase; marine teacher alignment; and confidence/coverage.

### Procedure

Construct a rule-based physical phase, then a small fold-local soft-state model. Test effects by JJA/MAM/SON, humidity, and synoptic wind strength. Build a bounded negative correction only for stable official-overforecast states and compare to simple coastal-inland spread.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Station groups fixed from metadata, not outcome. Evaluate leave-one-coastal and leave-one-inland-station robustness.

**Minimum sample rules:** At least three stations in each coastal/inland side when possible; 200 high-confidence phase events; 100 official trigger rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0135_sea_breeze_penetration/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`sea_breeze_phase.parquet`, `group_spreads.csv`, `phase_event_catalog.csv`, `marine_teacher_alignment.csv`, `station_ablation.csv`, and `suppression_specialist_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Phase must add residual/tail information beyond mean spread and wind sector; specialist must reduce overforecast without increasing hot underforecasts or JJA tail risk.

### Expected failure modes and interpretation

Synoptic maritime flow can resemble a sea breeze. Separate weak-wind diurnal penetration from strong background flow using pressure/wind coherence. If groups are uncertain, keep diagnostic.

### Expected information gain

High targeted potential for marine-suppressed days and a key route to operationalize blocked sea-temperature mechanisms.

---

## 0136 — Cool-Surge Propagation, Coastal Modification, and Breakdown Index

**Priority:** P1  
**Research mode:** Exploratory information-gain then specialist  
**Eligibility:** Deployable now  
**Dependencies:** 0122, 0131, 0132, 0134

### Decision question and hypothesis

Northern pressure rise, northerly wind shift, dewpoint/temperature fall, and southward propagation define cool-surge onset; coastal modification and weakening define breakdown. These phases should explain DJF/MAM residual direction better than pressure-only experts.

### Why this is new rather than a relabelled prior experiment

EXP-0072 front/surge detector and pressure specialists were broad and weak. This experiment uses ordered propagation, multi-field confirmation, phase progression, and separate onset versus breakdown corrections.

### Response variables

Next-day target drop/rise, official over/underforecast, DJF/MAM high-error, and surge event phase.

### Exact inputs

Northern/nearby station groups, pressure-wave features, wind kinematics, temperature/dewpoint fronts, seasonal phase, target derivative state, and official forecast.

### Feature constructions and calculations

North-to-south pressure-rise timing; northerly u/v persistence; temperature/dewpoint fall; gradient steepness; arrival confidence; surge age; coastal moderation ratio; weakening pressure tendency; wind veer; target slope reversal; and onset/mature/breakdown probabilities.

### Procedure

Identify candidate events from training-only thresholds and cluster consecutive days. Fit a simple phase state machine; score event transitions. Test separate bounded corrections: negative on underpredicted onset cooling, positive on overextended official cooling during breakdown.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Event blocks remain intact across folds; compare to calendar+pressure-only and target-reversal baselines.

**Minimum sample rules:** At least 100 surge events across long history; 40 onset and 40 breakdown official-overlap events for promotion-oriented tests.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0136_cool_surge_propagation_breakdown/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`surge_events.csv`, `surge_phase.parquet`, `propagation_diagnostics.csv`, `phase_response_scores.csv`, `onset_breakdown_specialists.csv`, and `event_casebook.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Multi-field phase must beat pressure-only detection and show consistent signed residual behavior. Corrections must improve target phase without harming non-event DJF/MAM rows.

### Expected failure modes and interpretation

Event labels may depend on arbitrary thresholds. Use sensitivity grid chosen before evaluation and report consensus events. Sparse official overlap may limit promotion.

### Expected information gain

High physical information and moderate specialist potential, especially for cold overforecast/underforecast tails.

---

## 0137 — Spatial-Field Topology, Hotspot Connectivity, and Boundary Morphology

**Priority:** P2  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now after geometry/QC  
**Dependencies:** 0105, 0128–0130

### Decision question and hypothesis

Mean, plane, and variance may miss whether heat/moisture anomalies form one coherent region, isolated hotspots, or a boundary wrapping around HKO. Low-complexity topology/morphology descriptors may capture regime geometry robustly.

### Why this is new rather than a relabelled prior experiment

EXP-0080 tested distribution shape and hotspots, but not spatial connectivity, HKO component membership, or persistence of anomaly regions under a fixed neighbor graph.

### Response variables

Target anomaly, official residual, high-error, marine suppression, and heat-buildup regimes.

### Exact inputs

Station anomaly fields for temperature/dewpoint/pressure, station graph from geometry, target/official responses.

### Feature constructions and calculations

Threshold fields at training-only quantiles; number/size of connected hot/cool/moist/dry components; whether HKO-neighbor stations belong to largest component; boundary edge count; component centroid/distance; persistence over prior days; multi-field overlap; and graph total variation.

### Procedure

Build a static neighbor graph from geometry. Calculate descriptors at several predeclared anomaly thresholds, with missing-node robustness. Compare topology to network mean/dispersion/plane. Cross only top descriptors with wind/season.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Thresholds are fold-local; graph is static from metadata. Use station-dropout bootstrap and common-station graph sensitivity.

**Minimum sample rules:** At least eight present graph nodes and 500 valid days; 150 rows per coarse topology state.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0137_spatial_field_topology/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`spatial_topology.parquet`, `component_events.csv`, `graph_sensitivity.csv`, `mean_plane_ablation.csv`, `response_atlas.csv`, and `topology_visual_examples/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A descriptor must add stable information beyond simple spatial moments and survive node dropout. Promotion requires an interpretable bounded use.

### Expected failure modes and interpretation

Topology can be unstable to thresholds and network coverage. Require consensus across adjacent thresholds and confidence flags.

### Expected information gain

Moderate novel information; more likely a regime/uncertainty feature than direct point correction.

---

## 0138 — Response-Specific Station Contribution and Group-Shapley Leaderboard

**Priority:** P1  
**Research mode:** Exploratory information-gain / simplification  
**Eligibility:** Deployable now  
**Dependencies:** 0105, 0111, 0127–0137

### Decision question and hypothesis

A station can matter for one response, season, wind sector, or residual sign but not globally. A response- and regime-specific contribution leaderboard can identify robust station groups and eliminate dozens of fragile one-off station features.

### Why this is new rather than a relabelled prior experiment

0047 produced station rankings and 0051 interactions, but not conditional group contribution after controlling for target memory/anchor, nor a systematic station-group simplification test.

### Response variables

Target anomaly, official residual, absolute error, hot underforecast, cold overforecast, MAM high error, station-core residual, and online-memory residual.

### Exact inputs

All safe station feature families and newly derived spatial features, minimal nonstation controls, station groups, canonical responses.

### Feature constructions and calculations

Group and station leave-out loss; conditional permutation importance with temporal blocks; approximate Shapley contribution over physically grouped station sets; contribution by month/season/source/wind/moisture/pressure/target state; and stability/coverage penalty.

### Procedure

Fit the same simple cross-fitted response model with station groups as blocks. Calculate marginal contribution of each group and representative station, then test compressed models: all stations, top one per group, group aggregates only, and hybrid. Avoid interpreting correlated individual Shapley values without group results.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Contributions computed only on held-out temporal folds; run station dropout and era-common subsets.

**Minimum sample rules:** 1,000 long-history rows; 500 official rows; 100 per reported regime; a station must contribute in at least three folds to be called robust.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0138_station_contribution_shapley/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`station_contribution_by_response.csv`, `group_shapley.csv`, `regime_leaderboards.parquet`, `compressed_model_ablation.csv`, `station_stability.csv`, and `recommended_station_core.yaml`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Produce a compact group-based station core that retains or improves information/OOF score relative to the full station set and reduces dependence on any single station.

### Expected failure modes and interpretation

Correlated stations make individual attributions unstable. Treat group-level contribution as primary; if groups are wrong, revisit 0105/0140 rather than overinterpreting IDs.

### Expected information gain

Very high design value; likely improves robustness and computational efficiency, with possible small lift through noise reduction.

---

## 0139 — Deterministic Static Station Context Feature Store

**Priority:** P0  
**Research mode:** Foundation / feature engineering  
**Eligibility:** Deployable now after reproducible derivation  
**Dependencies:** 0105

### Decision question and hypothesis

Static geography cannot predict day-to-day weather alone, but it can define which stations should matter under specific flow, marine, elevation, and urban regimes. Converting the static inventory into deterministic station context is prerequisite to physics-informed interactions.

### Why this is new rather than a relabelled prior experiment

Static packages are documented as an inventory, and earlier experiments used latitude/longitude/elevation. No complete versioned station-level context store with coast, terrain, land/sea, urban, and pair-geometry attributes exists.

### Response variables

No target-derived response in feature construction. Downstream validation uses target anomaly, official residual, and regime labels.

### Exact inputs

Station dossier, static geospatial packages already acquired, HKO target coordinates, and station-pair geometry.

### Feature constructions and calculations

Station elevation; elevation relative to HKO; distance/bearing to HKO; distance and bearing to nearest coast; coastline normal; fraction land/sea/urban/vegetation within predeclared radii; terrain height/roughness and exposure by sector; valley/ridge/island flag with confidence; pair distance/bearing/elevation/coastal contrast; and data-source/version hashes.

### Procedure

Create deterministic geospatial joins using fixed coordinate reference systems and documented raster/vector resolutions. Preserve missing/unavailable attributes. Validate stations visually and numerically. Freeze context features independently of target outcomes, then rerun top station pair/group relationships with these annotations.

### Walk-forward validation and minimum evidence

Geospatial unit tests: coordinate bounds, CRS transformations, known coast/land sanity points, pair-distance symmetry, and reproducibility across runs. Predictive tests occur only downstream.

**Minimum sample rules:** 100% of non-quarantined stations receive core geometry; advanced terrain/urban fields may be missing but must have coverage flags.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0139_static_station_context_store/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`station_static_context.parquet`, `station_pair_context.parquet`, `geospatial_sources.json`, `crs_audit.md`, `context_missingness.csv`, `sanity_maps/`, and `context_feature_catalog.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Core geometry is complete and reproducible; every feature has units/source/resolution; no outcome data enters static labels; suspect stations remain quarantined.

### Expected failure modes and interpretation

Packages may not support coast/urban/terrain derivations at required resolution. Do not estimate unsupported values from target correlations; retain simpler geometry and confidence.

### Expected information gain

High enabling value for station grouping, flow-relative features, and robust graph priors; no standalone MAE claim.

---

## 0140 — Physics-Informed Station-Role Clustering and Group Compression

**Priority:** P1  
**Research mode:** Exploratory information-gain / simplification  
**Eligibility:** Deployable now  
**Dependencies:** 0105, 0139, 0138

### Decision question and hypothesis

Stable physical station groups—near-HKO urban, airport/open exposure, marine/island, coastal, inland PRD, northern continental, elevated—should transfer better than arbitrary individual station IDs and support robust group contrasts.

### Why this is new rather than a relabelled prior experiment

Previous analyses named potential group roles but did not infer/freeze a reproducible role taxonomy from metadata plus climate behavior, nor test whether group aggregates preserve station information.

### Response variables

Target anomaly, official residual, signed tails, high-error probability, and station-core residual.

### Exact inputs

Station static context, long-history station climatology/anomaly distributions, wind exposure statistics, coverage/QC, and response-specific contribution results.

### Feature constructions and calculations

Metadata-only cluster features plus outcome-free station climatology: seasonal mean/variance, diurnal range proxy, dewpoint/pressure/wind distributions, marine response, elevation and coast context. Create hard role labels and soft memberships; compute group mean/median/trimmed mean, spread, slope, rank, and coverage.

### Procedure

Compare expert-defined groups, hierarchical clustering, Gaussian mixture, and constrained clustering with physical must-link/cannot-link rules. Fit clustering on training-era station characteristics or static data; freeze before response evaluation. Test group aggregates versus top individual stations and all-station models.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Cluster stability under bootstrap years/stations and transfer across eras; no target outcome used to define clusters.

**Minimum sample rules:** At least three stations per promoted group where possible; groups with one station are explicit roles, not clusters. Aggregate requires 60% member coverage.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0140_station_role_clustering/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`station_roles.csv`, `soft_memberships.csv`, `group_feature_store.parquet`, `cluster_stability.csv`, `individual_vs_group_ablation.csv`, and `role_dossiers.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Groups must be physically interpretable, stable, and retain most station response information while reducing station-specific fragility. Any outcome-tuned grouping is rejected.

### Expected failure modes and interpretation

Sparse or era-limited stations can distort clusters. Separate coverage/QC from meteorological characteristics and allow soft/unknown roles.

### Expected information gain

High robustness and modelling value; likely the best path to simplify dozens of station columns.

---

## 0141 — Flow-Relative Land–Sea–Urban Fetch Index

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now after static context  
**Dependencies:** 0132, 0139

### Decision question and hypothesis

The same wind speed/direction has different thermal consequences depending on upwind fetch over sea, urban land, vegetation, valley, or elevated terrain. A flow-relative fetch index should explain marine suppression, urban heating, and downslope warming.

### Why this is new rather than a relabelled prior experiment

Prior wind and coastal experiments used station sectors/spreads but lacked a reproducible path-integrated upwind surface context for each station and HKO.

### Response variables

Target anomaly, official residual, marine overforecast, hot underforecast, and weak-wind heat-buildup state.

### Exact inputs

Station/HKO static land-sea-urban/terrain layers, pre-cutoff wind vectors/field, station thermal/moisture anomalies, official responses.

### Feature constructions and calculations

For each location and wind direction trace predeclared 10/25/50/100 km upwind rays/sectors; calculate sea fraction, urban fraction, roughness, elevation change, terrain blocking, coast crossings, and fetch uncertainty. Aggregate HKO fetch and station-group fetch contrasts; use calm-wind fallback.

### Procedure

Precompute directional lookup tables independent of outcomes. At each date map observed wind to fetch descriptors. Compare raw wind sector, coastline-normal component, and fetch-aware interactions. Test physical responses by season/humidity/pressure.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Directional lookup is static; wind used must be pre-cutoff. Test sensitivity to ray length/sector width using nested selection.

**Minimum sample rules:** 200 rows per broad fetch class; 100 official rows for a promoted residual regime.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0141_flow_relative_fetch_index/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`directional_fetch_lookup.parquet`, `daily_fetch_features.parquet`, `fetch_response_atlas.csv`, `wind_only_ablation.csv`, `length_sensitivity.csv`, and `candidate_fetch_flags.yaml`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Fetch index must add conditional information beyond wind u/v and station coastal labels, with stable physical sign across years. Corrections remain bounded.

### Expected failure modes and interpretation

Coarse land-cover or uncertain wind representativeness can weaken results. Preserve uncertainty and avoid fine-grained interpretations unsupported by resolution.

### Expected information gain

High novel physics value, moderate specialist potential for marine/urban/downslope regimes.

---

## 0142 — Elevation, Lapse-Rate, and Downslope Realization Index

**Priority:** P2  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0139, 0141, 0132

### Decision question and hypothesis

Elevated and inland stations can reveal air-mass warmth, while flow over terrain may produce downslope drying/warming at HKO. Dynamic lapse and elevation-adjusted anomalies should separate synoptic temperature from local exposure.

### Why this is new rather than a relabelled prior experiment

Station elevations were present and upper-air parcel descent was tested, but no safe surface-network experiment estimates dynamic lapse rates and flow-conditioned downslope realization at HKO.

### Response variables

Target anomaly, official residual, hot underforecast, T−Td spread response, and dry-subsidence proxy.

### Exact inputs

Station elevation/context, temperature/dewpoint/wind/pressure fields, target/official responses.

### Feature constructions and calculations

Robust temperature/dewpoint vs elevation slope within spatial groups; elevation-adjusted station anomaly; upwind terrain descent toward HKO; along-flow elevation change; lee-side exposure; dry-air confirmation; lapse-rate deviation from seasonal normal; and realization uncertainty.

### Procedure

Fit per-date robust lapse only with adequate elevation spread and remove horizontal gradient confounding. Compare static seasonal lapse, dynamic lapse, and flow-conditioned descent. Evaluate interactions with pressure rise/subsidence proxy and humidity.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Bootstrap stations and require minimum elevation range; geometry is static and all atmospheric values pre-cutoff.

**Minimum sample rules:** At least five stations and 100 m elevation range for dynamic lapse; 500 valid days and 100 candidate downslope official rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0142_elevation_downslope_realization/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`dynamic_lapse.parquet`, `downslope_index.parquet`, `fit_quality.csv`, `horizontal_gradient_ablation.csv`, `response_atlas.csv`, and `specialist_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Added residual/tail information beyond regional temperature and wind, with stable warm/dry sign. No promotion if driven by one elevated station.

### Expected failure modes and interpretation

Regional horizontal gradients can masquerade as lapse rate. Jointly fit location/elevation or use matched station pairs; low-confidence estimates are missing.

### Expected information gain

Moderate information gain, potentially useful for hot underforecast/subsidence days.

---

## 0143 — Coastline Orientation and Onshore-Penetration Exposure Index

**Priority:** P2  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0139, 0141

### Decision question and hypothesis

Onshore/offshore is location-specific because coastline orientation varies. Local coast-normal wind components and cross-coast station chains should improve marine-flow interpretation over fixed compass sectors.

### Why this is new rather than a relabelled prior experiment

Existing coastal-inland and wind-sector work does not account for local coastline normals, multiple coast crossings, or station-specific exposure.

### Response variables

Marine suppression, target anomaly, official overforecast, sea-breeze phase, and dewpoint-front state.

### Exact inputs

Coastline geometry, station/HKO locations, wind vectors, station thermal/moisture features, target/official responses.

### Feature constructions and calculations

Local coastline normal/tangent; onshore/offshore and alongshore components; distance to coast along wind path; cross-coast thermal/dewpoint gradients; coast-crossing count; sheltered/exposed sector; persistence and shift of onshore component.

### Procedure

Precompute local orientation at multiple coastline smoothing scales. Evaluate which scale transfers in nested folds. Construct station chains from sea-facing to inland. Compare to raw wind sector and generic coastal-inland spread.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Static orientation must not be outcome-tuned; scale chosen within training folds.

**Minimum sample rules:** 200 onshore and 200 offshore rows; 100 official rows in each promoted exposure state.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0143_coastline_orientation_exposure/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`coast_orientation.csv`, `daily_coast_exposure.parquet`, `cross_coast_chains.csv`, `scale_sensitivity.csv`, `response_atlas.csv`, and `marine_ablation.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Must add stable information beyond u/v wind and coast distance; physical sign should vary sensibly by season/background flow.

### Expected failure modes and interpretation

Complex Hong Kong coastline creates unstable normals. Use multi-scale confidence and group exposure rather than a single exact normal when uncertain.

### Expected information gain

Moderate-to-high marine-regime information and better interpretability of wind effects.

---

## 0144 — Physics-Informed Graph Kernels and Causal Graph Modes

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0139–0143, 0105

### Decision question and hypothesis

Graph modes failed as a broad direct model partly because a distance-only graph ignores coast, elevation, and flow. Physics-informed static and flow-conditioned kernels may produce more meaningful low-dimensional station modes.

### Why this is new rather than a relabelled prior experiment

EXP-0079 used graph-Laplacian modes. This experiment explicitly tests graph construction—not merely more graph components—and evaluates response-specific residual/uncertainty value with fold-local transforms.

### Response variables

Target anomaly, official residual, absolute error, sea-breeze/cool-surge regimes, and station-core residual.

### Exact inputs

Station geometry/context, daily wind/pressure/thermal state, safe station anomalies, canonical responses.

### Feature constructions and calculations

Static kernels based on distance/elevation/coast/role; directed flow kernels using upwind projection; variable-specific graphs; graph smoothness/total variation; first 3–5 eigenmodes fitted on training-eligible station covariance; HKO-local graph interpolation; and graph disagreement uncertainty.

### Procedure

Predeclare a small graph family. Fit normalization and covariance only in training folds; static geometry graph may be fixed. Compare distance graph, role graph, coast/elevation graph, and flow-directed graph. Use modes as atlas features and tiny residual models, not unrestricted feature soup.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Leave-station-out and graph-perturbation stability; compare directly to simple group aggregates and robust plane.

**Minimum sample rules:** At least eight stations present for modes; 1,000 long-history rows; 500 official rows; graph components capped at five.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0144_physics_graph_kernels/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`graph_definitions.json`, `graph_modes.parquet`, `graph_stability.csv`, `simple_group_ablation.csv`, `leave_station_out.csv`, and `response_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A graph must beat distance-only and simple groups on a predeclared response, remain stable under node dropout, and add orthogonal information. Otherwise archive as negative evidence.

### Expected failure modes and interpretation

Dynamic graphs can overfit and eigenvector signs/order can switch. Align modes deterministically and favor simple groups when performance is tied.

### Expected information gain

Moderate information gain with upside from better graph priors; complexity gate is strict.

---

## 0145 — Surface Wet-Bulb, Enthalpy, and Moist-Heat Network State

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now from station temperature/dewpoint/pressure  
**Dependencies:** 0107, 0130

### Decision question and hypothesis

Dewpoint level alone mixes warm maritime air and cloud-limited heating. Surface wet-bulb/enthalpy, T−Td, and their spatial/tendency structure may better represent moisture load, nighttime memory, and sensible-heating potential.

### Why this is new rather than a relabelled prior experiment

Upper-air theta-e/MSE and coupling experiments were broad/blocked. This is a safe surface-network thermodynamic reconstruction aimed at residual and uncertainty responses.

### Response variables

Target anomaly, official residual, hot underforecast, cold overforecast, absolute error, and humid-heat state.

### Exact inputs

Station temperature, dewpoint, pressure through cutoff; station geometry/groups; target memory and official forecast.

### Feature constructions and calculations

Vapor pressure, mixing ratio/specific humidity approximation, wet-bulb approximation, moist enthalpy, virtual temperature, T−Td, station anomalies and tendencies, coastal-inland enthalpy spread, network percentile/rank, and moist-heat convergence with wind.

### Procedure

Use documented thermodynamic formulas with unit/QC gates. Rank raw level, one-day change, anomaly-to-own-baseline, group spreads, and target-memory interactions. Separate summer warm-moist level from spring moisture-surge transitions.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Compare against temperature and dewpoint components; formulas are deterministic and use only pre-cutoff values.

**Minimum sample rules:** 1,000 long-history rows per core feature, 200 per quantile, 100 official rows per specialist state.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0145_surface_enthalpy_network/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`thermodynamic_features.parquet`, `formula_audit.md`, `component_ablation.csv`, `moist_heat_atlas.csv`, `seasonal_response_curves.csv`, and `candidate_specialists.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Derived thermodynamics must add conditional information beyond components and exhibit stable physical season interactions. No promotion solely from raw target correlation.

### Expected failure modes and interpretation

Pressure missingness and approximation errors may add noise. Provide fallback formulas and uncertainty flags; do not infer impossible humidity states.

### Expected information gain

High information potential for MAM/JJA moisture regimes and uncertainty scaling.

---

## 0146 — Temperature–Dewpoint Spread and Dry-Heating Potential Atlas

**Priority:** P1  
**Research mode:** Exploratory then promotion-oriented  
**Eligibility:** Deployable now  
**Dependencies:** 0145, 0112

### Decision question and hypothesis

Large T−Td can indicate dry air and greater sensible-heating potential, while a rapidly closing spread can signal moisture/cloud suppression. Level, tendency, and spatial distribution should have asymmetric effects by season and wind.

### Why this is new rather than a relabelled prior experiment

T−Td has appeared among station features but has not received a dedicated multi-scale, network, response-specific study with nonlinear curves and bounded residual actions.

### Response variables

Official residual, hot underforecast, cold overforecast, target anomaly, high-error, and radiative-suppression teacher.

### Exact inputs

Station T−Td features, dewpoint/temperature changes, wind/pressure/seasonal phase, target/official responses.

### Feature constructions and calculations

Station and group T−Td level; 1/3-day change; anomaly vs 14/30-day baseline; network mean/min/max/IQR; coastal-inland contrast; upwind spread; dry-wedge front; interaction with weak wind, pressure rise, target slope, and forecast-vs-memory disagreement.

### Procedure

Estimate smooth conditional curves by season and wind/pressure state. Distinguish dry heating from cool dry surge using temperature tendency and pressure/wind. Test positive corrections only in stable hot-underforecast regions and negative corrections for moisture-surge overforecasts.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Compare to dewpoint-only and temperature-only; use no-harm uplift policy for actions.

**Minimum sample rules:** 500 rows per curve, 150 per major interaction regime, 100 official trigger rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0146_dry_heating_potential/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`spread_features.parquet`, `conditional_curves.csv`, `dry_vs_cool_classification.csv`, `component_ablation.csv`, `bounded_policy_scores.csv`, and `tail_diagnostics.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Must show conditional residual asymmetry beyond components and improve a predeclared bounded policy with no opposite-tail harm.

### Expected failure modes and interpretation

T−Td mechanically rises with temperature. Residualize against temperature and season; treat residual spread and change as primary.

### Expected information gain

High practical promise because it is safe, physically interpretable, and directly tied to heating efficiency.

---

## 0147 — Dewpoint Tendency × Wind-Advection Interaction Lab

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable now  
**Dependencies:** 0132, 0134, 0145

### Decision question and hypothesis

Regional dewpoint change is repeatedly informative, but its meaning depends on whether moist/dry air is advecting toward HKO, recirculating locally, or arriving with a pressure transition.

### Why this is new rather than a relabelled prior experiment

This focuses on the strongest station clue—dewpoint change—and asks why/when it works, rather than adding another generic multi-signal model.

### Response variables

Official residual, high-error, MAM transition error, hot underforecast, cold overforecast, and target change.

### Exact inputs

Station/group dewpoint tendencies, wind field/transport features, pressure waves, temperature changes, seasonal phase, official response.

### Feature constructions and calculations

Upwind-weighted dewpoint change; along-flow dewpoint gradient; moisture arrival time; moisture tendency sign agreement; dewpoint surge with pressure fall/rise; dry intrusion with offshore/downslope flow; persistence; and target-memory coherence.

### Procedure

Predeclare interaction surfaces by season. Compare global mean dewpoint change, best individual station, group change, and flow-conditioned change. Use functional ANOVA and bounded policies for stable signed regions.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Require incremental interaction over both main effects and test station-group transfer.

**Minimum sample rules:** 1,000 global rows, 200 per broad advection state, 100 official trigger rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0147_dewpoint_tendency_wind_advection/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`dew_advection_features.parquet`, `interaction_surfaces.parquet`, `main_effect_ablation.csv`, `station_group_transfer.csv`, `policy_scores.csv`, and `physical_cases.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Flow conditioning must improve response information over simple dewpoint change and preserve sign across multiple years/sources.

### Expected failure modes and interpretation

Wind field may not represent moisture transport; poor-confidence transport rows must abstain. If simple group dewpoint change wins, prefer it.

### Expected information gain

Very high information-gain priority because it deepens a repeatedly observed safe signal.

---

## 0148 — Antecedent Wetness, Drying Recovery, and Nonlinear Surface-Memory Atlas

**Priority:** P2  
**Research mode:** Diagnostic-to-safe-proxy research  
**Eligibility:** Safe proxy inputs deployable; finalized rainfall/daily climate diagnostic-only without publication proof  
**Dependencies:** 0106, 0145

### Decision question and hypothesis

Wet surfaces and recent rain can suppress Tmax through evaporation/cloud persistence, but effects decay nonlinearly with sun, wind, season, and urban context. Safe station moisture proxies may reconstruct this memory.

### Why this is new rather than a relabelled prior experiment

EXP-0085 broad antecedent rainfall/dry-spell model failed. This experiment separates diagnostic rainfall teacher from safe proxy students and focuses on decay/recovery curves rather than direct feature addition.

### Response variables

Target anomaly, official overforecast, heating shortfall, high-error, and diagnostic rain/cloud teacher states.

### Exact inputs

Blocked lagged daily rainfall/wetness only in F-DIAG; safe station dewpoint/T−Td, pressure/wind, target volatility, visibility where eligible, and high-frequency pre-2024 humidity/rain proxies if available.

### Feature constructions and calculations

Teacher wetness index with multiple decay half-lives; safe moisture persistence, dewpoint surge, low T−Td, pressure/wind/cloud-suppression proxies; drying degree-days; wind ventilation; urban/coastal group differences; and nonlinear recovery time.

### Procedure

First quantify teacher effect and optimal decay by season. Then train simple safe proxy scores to predict teacher state using prior-only safe features. Test whether the proxy predicts residual/heating shortfall; never insert teacher values into deployable scoring.

### Walk-forward validation and minimum evidence

Separate F-DIAG mechanism validation from F-LONG/F-O5265 proxy evaluation. Proxy models and decay selection are fold-local; 2024+ sealed.

**Minimum sample rules:** 1,000 diagnostic rows, 500 proxy-evaluable rows, 100 wet-state official rows for candidate action.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0148_antecedent_wetness_recovery/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`wetness_teacher.parquet`, `decay_curves.csv`, `safe_proxy_oof.parquet`, `teacher_student_alignment.csv`, `residual_atlas.csv`, and `eligibility_separation.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Safe proxy must transfer across years and add residual/tail information beyond moisture level. Teacher association alone is not promotion.

### Expected failure modes and interpretation

Rainfall publication timing and target-day rain make leakage easy. Use only diagnostic teacher rows or proven prior lags; do not infer availability from archive presence.

### Expected information gain

Moderate information gain and a disciplined way to rescue a physically plausible but previously weak family.

---

## 0149 — Radiative-Suppression Safe Proxy Distillation

**Priority:** P1  
**Research mode:** Diagnostic-to-deployable proxy conversion  
**Eligibility:** Teacher diagnostic-only; student deployable if safe  
**Dependencies:** 0106, 0145–0148

### Decision question and hypothesis

Cloud, rain, fog, haze, and moisture suppress daytime heating. Blocked daily sunshine/cloud/rain fields can define a teacher state; station moisture, pressure, wind, visibility, thermal trajectory, and forecast text may safely proxy it.

### Why this is new rather than a relabelled prior experiment

EXP-0086/0094 tested cloud/sunshine memory and teacher/student broadly but degraded the core. This experiment targets a narrow latent “radiative suppression” response and evaluates proxy alignment before any residual correction.

### Response variables

Diagnostic suppression teacher, observed target heating shortfall, official overforecast, absolute error, and safe proxy probability.

### Exact inputs

F-DIAG HKO cloud/sunshine/rain/visibility/wet-bulb values; safe station dewpoint/T−Td, pressure/wind, temperature tendencies, target memory, official text/range where exact; optional pre-2024 UV/solar high-frequency teacher.

### Feature constructions and calculations

Teacher as fold-local residual of Tmax/heating versus seasonal thermal state; proxy features include moisture surge, low spread, pressure transition, wind convergence, station warming shortfall, visibility/haze, text tokens, range width, and network disagreement. Use sparse monotone model.

### Procedure

Define multiple teacher variants and assess consistency. Train safe students on prior rows only, then evaluate teacher AUC/calibration and official-overforecast enrichment. Apply a small negative correction only when proxy confidence and prior uplift are high.

### Walk-forward validation and minimum evidence

Strictly separate blocked teacher from deployable student at inference. Use temporal folds and source-era robustness; compare to simple dewpoint-change baseline.

**Minimum sample rules:** 1,000 teacher rows, 500 safe overlap rows, 100 high-confidence proxy events, 60 official activations per evaluation fold where feasible.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0149_radiative_suppression_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`suppression_teacher.csv`, `proxy_feature_catalog.csv`, `student_oof.parquet`, `alignment_metrics.csv`, `overforecast_enrichment.csv`, and `bounded_specialist_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Student must predict teacher and residual behavior out of time; correction must improve F-O5265 or a stable slice without hot-underforecast harm. Otherwise preserve proxy as uncertainty only.

### Expected failure modes and interpretation

Teacher may encode target-day outcome and should never be treated as an input. Text coverage differs by era; report source-specific student variants.

### Expected information gain

High potential because it converts strong but blocked physical information into a safe operational state.

---

## 0150 — State-Dependent Solar-to-Tmax Conversion Efficiency

**Priority:** P2  
**Research mode:** Diagnostic mechanism and safe-proxy research  
**Eligibility:** Long daily solar diagnostic-only; short high-frequency pre-2024 diagnostic/prospective  
**Dependencies:** 0149, 0179 later

### Decision question and hypothesis

The temperature response to available solar energy depends on moisture, wind, antecedent wetness, season, and thermal state. Conversion efficiency—not solar level alone—may reveal heat buildup and model blind spots.

### Why this is new rather than a relabelled prior experiment

EXP-0087 tested solar-radiation-to-Tmax efficiency as a broad candidate and failed. This redesign treats efficiency as a teacher/mechanism and searches for safe state proxies and residual-specific regimes.

### Response variables

Diagnostic realized heating efficiency, target anomaly, official residual, hot underforecast, and radiative-suppression state.

### Exact inputs

Diagnostic daily sunshine/radiation; pre-2024 high-frequency solar/UV; safe moisture/wind/pressure/target/station features; official responses.

### Feature constructions and calculations

Solar input normalized by day length; target heating per unit solar; station warming slope; moisture and wetness state; ventilation; recent heat storage; cloud-break timing in high-frequency data; and safe predicted-efficiency score.

### Procedure

Estimate teacher efficiency after controlling for season and starting thermal state. Stratify by moisture/wind/wetness. Train safe proxies without solar if long-history promotion is desired. Compare to EXP-0087 broad feature and preserve mechanism even if point lift fails.

### Walk-forward validation and minimum evidence

Teacher remains diagnostic; high-frequency analysis uses only pre-2024 days and leave-year-out folds; student evaluated on long safe frames.

**Minimum sample rules:** 500 diagnostic days per broad regime, 300 pre-2024 high-frequency days, 100 high-efficiency/low-efficiency official rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0150_solar_tmax_efficiency/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`efficiency_teacher.parquet`, `state_response_curves.csv`, `high_frequency_alignment.csv`, `safe_efficiency_student.parquet`, `residual_scores.csv`, and `negative_results.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Require stable state-dependent efficiency and a student that adds information beyond moisture/wind components. No promotion from teacher-only performance.

### Expected failure modes and interpretation

Daily Tmax and solar timing are jointly determined; causal interpretation is limited. Treat as predictive mechanism, not causal proof.

### Expected information gain

Moderate information gain; likely better as regime/uncertainty signal than standalone correction.

---

## 0151 — Visibility, Haze, and Aerosol-Cloud Suppression Proxy Atlas

**Priority:** P2  
**Research mode:** Diagnostic-to-safe research  
**Eligibility:** Eligibility must be proven per source/lag  
**Dependencies:** 0106, 0149

### Decision question and hypothesis

Low visibility may proxy fog, haze, cloud, rain, or aerosols, each with different Tmax effects. Combining visibility change with dewpoint spread, wind, and pressure can distinguish humid suppression from dry haze and identify forecast bias.

### Why this is new rather than a relabelled prior experiment

Visibility exists in daily climate context but has not been separated into meteorological mechanisms or converted into a safe station/forecast proxy.

### Response variables

Target anomaly, official overforecast, high-error, radiative-suppression teacher, and humid-vs-dry low-visibility state.

### Exact inputs

Visibility fields with eligibility status, station dewpoint/T−Td/wind/pressure, rainfall/cloud teacher diagnostics, forecast text where exact.

### Feature constructions and calculations

Visibility level/change/anomaly; low-visibility persistence; humidity-adjusted visibility; dry-haze vs fog score; wind ventilation; pressure regime; text consistency; and network spatial contrast if station visibility exists.

### Procedure

First diagnose mechanisms using blocked climate labels. Then restrict deployable tests to proven pre-cutoff visibility or safe proxy features. Estimate nonlinear response and residual sign by season/source.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Eligibility status is explicit. Compare to moisture-only proxy and exclude target-day finalized visibility from deployable scoring.

**Minimum sample rules:** 500 rows per mechanism class, 100 official rows per low-visibility regime.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0151_visibility_haze_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`visibility_mechanism_atlas.csv`, `humidity_adjusted_features.parquet`, `eligibility_split.csv`, `proxy_scores.csv`, `residual_curves.csv`, and `specialist_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Only safe features can promote; they must add conditional information beyond moisture and wind. Teacher-only insight remains diagnostic.

### Expected failure modes and interpretation

Visibility is heterogeneous and quality codes matter. Audit units/report types and avoid conflating missing with unlimited visibility.

### Expected information gain

Moderate insight; potential niche MAM/haze/cloud specialist.

---

## 0152 — Urban Heat-Storage Hysteresis and Nocturnal Thermal Memory

**Priority:** P2  
**Research mode:** Exploratory information-gain  
**Eligibility:** Long target/station proxies deployable; detailed high-frequency later  
**Dependencies:** 0105, 0140, 0119

### Decision question and hypothesis

Urban heat storage and warm-night persistence can alter next-day starting state and heating efficiency. The relationship may show hysteresis: the same morning/previous Tmax implies different next-day outcomes after sustained warm, calm, humid nights versus ventilated cooling.

### Why this is new rather than a relabelled prior experiment

EXP-0088 broad grass-minimum/evaporation/urban memory failed. This experiment uses safe target and station contrasts, separates nocturnal retention from daytime level, and tests response-specific hysteresis loops.

### Response variables

Target anomaly, official residual, hot underforecast, high-minimum proxy state, and next-day heating increment.

### Exact inputs

Target Tmax history, safe near-urban vs open/coastal station temperature anomalies, dewpoint, wind, target volatility/spell state; high-frequency pre-2024 temperature later as teacher.

### Feature constructions and calculations

Multi-day warm anomaly integral; cooling deficit proxy from T−1 station temperature versus recent Tmax/seasonal norm; urban-open contrast; calm/humid persistence; heat-storage decay half-life; loop state based on warming vs cooling branch; and network nocturnal-retention proxy where timing permits.

### Procedure

Map hysteresis curves between accumulated warmth, ventilation/moisture, and next-day residual. Compare simple lag/roll features to urban-open contrasts. Use high-frequency nights only to validate proxy meaning, not long-history promotion.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Branch/state definitions are prior-only; station roles fixed from metadata. High-frequency validation uses pre-2024 leave-year-out.

**Minimum sample rules:** 1,000 long-history rows; 200 per hysteresis branch; 100 official hot-state rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0152_urban_heat_storage_hysteresis/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`heat_storage_features.parquet`, `hysteresis_curves.csv`, `urban_open_ablation.csv`, `high_frequency_teacher_alignment.csv`, `residual_atlas.csv`, and `bounded_hot_specialist.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Must add residual or tail information beyond target rolling means and station temperature level; correction may not amplify already-hot overforecasts.

### Expected failure modes and interpretation

Without true nocturnal minima, proxies may simply reflect recent Tmax. Require incremental tests and use high-frequency data to validate physical interpretation.

### Expected information gain

Moderate information gain; targeted upside during persistent weak-wind heat episodes.

---

## 0153 — 1000-hPa Geopotential Height Diagnostic-to-Safe Proxy Conversion

**Priority:** P1  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** IGRA teacher diagnostic-only; safe student deployable  
**Dependencies:** 0102, 0111, 0131–0134

### Decision question and hypothesis

`igra_hgt_1000hpa_m` was the strongest raw diagnostic feature in 0100, likely representing low-level pressure/thermal structure, ridge strength, or air-mass state. Safe surface pressure gradients, tendencies, temperature fields, and wind may proxy the same mechanism.

### Why this is new rather than a relabelled prior experiment

Upper-air height was ranked and broad ridge/height models failed, but no feature-by-feature teacher–student conversion has targeted the exact top blocked signal with safe surface mechanisms and residual responses.

### Response variables

Teacher value/anomaly/tendency, target anomaly, official residual, MAM high-error, hot underforecast, and safe student error.

### Exact inputs

F-DIAG IGRA 1000-hPa height and related levels; safe station pressure level/tendency/plane, temperature/dewpoint gradients, wind field, target memory, calendar; official frames.

### Feature constructions and calculations

Teacher anomalies relative to causal seasonal norm; surface MSLP plane/intercept and gradients; pressure tendency/wave; elevation-adjusted station pressure; regional temperature anomaly; wind divergence; target slope; source/season interactions. Train sparse monotone/linear students and report residual teacher information left unexplained.

### Procedure

Quantify teacher response conditionally. Cross-fit students using safe inputs only. Evaluate teacher reconstruction and whether the student preserves the teacher’s target/residual ordering. Test student as uncertainty/regime feature and bounded residual specialist; teacher never enters deployable prediction.

### Walk-forward validation and minimum evidence

Two separate scoreboards: diagnostic teacher value and safe student value. Temporal folds span multiple eras; all student preprocessing is fold-local.

**Minimum sample rules:** 2,000 teacher-overlap rows for mechanism, 1,000 safe training rows, 200 official rows per student regime.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0153_hgt1000_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`teacher_atlas.csv`, `student_oof.parquet`, `proxy_feature_coefficients.csv`, `teacher_residual_atlas.csv`, `response_transfer.csv`, `eligibility_firewall.md`, and `specialist_scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Student must reconstruct a meaningful fraction of teacher state out of time and add safe conditional response information beyond component pressure features. No promotion based on teacher-only score.

### Expected failure modes and interpretation

1000-hPa level may be below ground or inconsistently reported, and the diagnostic signal may be seasonal. Audit level validity and use anomalies/availability flags. If student cannot transfer, preserve mechanism only.

### Expected information gain

High information-gain priority because it targets the strongest blocked feature with a specific safe substitute.

---

## 0154 — Low-Level Inversion and Mixing-Cap Safe Proxy Conversion

**Priority:** P1  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** Upper-air teacher diagnostic-only; station student deployable  
**Dependencies:** 0102, 0145–0147, 0132

### Decision question and hypothesis

Low-level inversion/mixing-cap structure influences cloud, fog, sea-breeze depth, and surface heating. Surface station temperature/dewpoint gradients, pressure/wind, and heating-rate contrasts may proxy inversion presence and erosion.

### Why this is new rather than a relabelled prior experiment

EXP-0060 full inversion geometry and upper-air mixtures failed as direct models. This experiment uses inversion only as a latent teacher, separates inversion presence from surface realization, and tests a safe proxy state.

### Response variables

Diagnostic inversion strength/depth/erosion, target heating increment, official residual, radiative suppression, and marine suppression.

### Exact inputs

IGRA profile features in F-DIAG; safe station thermal/moisture fields, morning-to-midday rise, wind kinematics, pressure tendency, target memory; high-frequency pre-2024 heating curves as optional recent teacher.

### Feature constructions and calculations

Teacher: inversion base/top/depth/strength and 24/48-hour tendency. Student: coastal-inland thermal stratification, elevated-vs-low station anomaly, morning warming shortfall, dewpoint spread, calm/onshore flow, pressure rise, spatial variance, and target volatility.

### Procedure

Define several inversion teacher labels robust to missing levels. Train separate safe students by season. Test whether proxy predicts heating shortfall and official overforecast; compare to simple moisture/wind features. Use student probability as gate/uncertainty, not direct replacement forecast.

### Walk-forward validation and minimum evidence

Blocked teacher and safe student strictly separated. Profile quality included. Temporal folds and leave-season transfer reported.

**Minimum sample rules:** 1,000 valid soundings; 500 student-overlap dates; 100 high-confidence proxy official rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0154_inversion_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`inversion_teacher.parquet`, `profile_quality.csv`, `student_oof.parquet`, `seasonal_alignment.csv`, `heating_response.csv`, and `safe_proxy_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Proxy must retain stable relation to teacher and residual/heating response in multiple folds; correction cannot rely on profile values at inference.

### Expected failure modes and interpretation

Sounding launch/valid times and profile coverage are blocked and irregular. Teacher is diagnostic only; missingness must not be mistaken for no inversion.

### Expected information gain

High mechanism value, moderate safe-regime potential in fog/cloud/sea-breeze seasons.

---

## 0155 — Lower-Tropospheric Heat-Content and Thickness Safe Proxy Conversion

**Priority:** P1  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** Upper-air teacher diagnostic-only  
**Dependencies:** 0153, 0154, 0142

### Decision question and hypothesis

Layer thickness and lower-tropospheric heat content describe air-mass thermal potential that surface stations may partially observe through regional temperature/elevation patterns and pressure tendencies.

### Why this is new rather than a relabelled prior experiment

EXP-0061/0065 heat-content/ridge models were direct and weak. This redesign targets teacher state and asks whether a compact safe surface proxy predicts residuals when official forecast misses air-mass warmth.

### Response variables

Diagnostic thickness/heat anomaly and tendency, target anomaly, official residual, hot/cold directional tails.

### Exact inputs

IGRA 1000–925–850–700 hPa temperature/height features; safe station temperature anomalies, dynamic lapse, pressure field, wind/transport, target state, official forecast.

### Feature constructions and calculations

Teacher layer mean potential temperature/thickness and 24/48-hour change. Student regional warm anomaly, elevation-adjusted temperature, pressure-height proxy, upwind thermal advection, target-vs-network coherence, dry-heating potential, and seasonal phase.

### Procedure

Cross-fit teacher students with strong regularization. Quantify whether teacher residual (unexplained by surface) still relates to target—indicating a genuinely missing source. Test student as residual feature and envelope-breakout confirmation.

### Walk-forward validation and minimum evidence

Teacher/students scored in long temporal blocks; student compared to regional temperature mean and target memory baselines.

**Minimum sample rules:** 2,000 teacher rows, 1,000 safe overlap rows, 200 official overlap rows per broad state.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0155_heat_content_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`heat_content_teacher.parquet`, `surface_student_oof.parquet`, `unexplained_teacher_effect.csv`, `baseline_ablation.csv`, `breakout_alignment.csv`, and `specialist_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Promote only if student adds response information beyond surface temperature and target memory; otherwise document the unresolved source gap for future operational NWP.

### Expected failure modes and interpretation

Surface proxy may saturate and miss elevated warm layers. This negative result would justify a precisely scoped operational NWP acquisition rather than broad gridded downloads.

### Expected information gain

High diagnostic value and clear go/no-go evidence for whether upper-air thermal potential can be replaced safely.

---

## 0156 — Moisture-Profile and Stability Safe Proxy Conversion

**Priority:** P1  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** Upper-air teacher diagnostic-only  
**Dependencies:** 0145–0149, 0154

### Decision question and hypothesis

Column moisture and dry layers affect cloud/rain/heating differently from surface dewpoint. Spatial dewpoint, T−Td, wind convergence, pressure and heating-rate patterns may proxy low-level moisture depth or dry-air entrainment.

### Why this is new rather than a relabelled prior experiment

EXP-0062/0063 profile moisture features were broad direct candidates. This experiment creates explicit moisture-profile teacher states and tests whether safe surface networks can distinguish shallow humid, deep humid, and dry-over-moist regimes.

### Response variables

Teacher profile class, radiative-suppression teacher, target heating shortfall, official residual, high-error and directional tails.

### Exact inputs

IGRA humidity/temperature profiles; station thermodynamics, moisture fronts/flux, wind/pressure, target state; diagnostic daily cloud/rain.

### Feature constructions and calculations

Teacher precipitable/moist-layer proxies, RH/dewpoint-depression by layer, dry-layer strength, stability and tendency. Student surface enthalpy, T−Td distribution, dewpoint convergence, visibility proxy, pressure/wind transition, warming-rate shortfall, and seasonal phase.

### Procedure

Cluster/label teacher profiles using training-only physically anchored thresholds. Train calibrated safe students and test whether profile-state probability adds to surface moisture. Evaluate official residual and uncertainty, not only teacher accuracy.

### Walk-forward validation and minimum evidence

Temporal folds, profile-quality gating, teacher/student separation, and comparison to dewpoint-only baseline.

**Minimum sample rules:** 1,000 valid profiles, 500 overlap rows, 100 official high-confidence proxy rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0156_moisture_profile_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`moisture_profile_teacher.parquet`, `teacher_classes.csv`, `student_oof.parquet`, `surface_baseline_ablation.csv`, `residual_response.csv`, and `uncertainty_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Student must improve teacher classification/calibration and residual/uncertainty metrics beyond surface dewpoint alone. Teacher-only association is diagnostic.

### Expected failure modes and interpretation

Historical humidity profile sparsity/quality may limit stable labels. Reduce to broad classes and report missing-level sensitivity.

### Expected information gain

High MAM/JJA mechanism value and possible uncertainty lift.

---

## 0157 — Vertical Shear, Veering, and Advection Safe Proxy Conversion

**Priority:** P2  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** Upper-air teacher diagnostic-only  
**Dependencies:** 0132, 0134, 0102

### Decision question and hypothesis

Vertical shear/veering can distinguish monsoon depth, frontal structure, and elevated advection that surface wind misses. Spatial surface wind/pressure/temperature behavior may provide a safe proxy for key shear regimes.

### Why this is new rather than a relabelled prior experiment

EXP-0064 tested upper-air shear directly and failed. This experiment seeks specific surface signatures and evaluates whether any residual effect is recoverable or truly requires operational upper-air/NWP.

### Response variables

Teacher shear/veering class, target change, official residual, cool-surge/sea-breeze phase, and high-error.

### Exact inputs

IGRA winds by level; safe surface wind field, pressure waves, temperature/dewpoint fronts, transport features, official responses.

### Feature constructions and calculations

Teacher bulk shear magnitude/direction, veering/backing, low-level jet proxy. Student surface directional disagreement, pressure-gradient alignment, station wind persistence, propagation speed, front alignment, and exposure-weighted group winds.

### Procedure

Build broad teacher classes, train safe students, and test conditional response. Compare student to surface wind-only; assess unexplained teacher residual. Use no point correction unless a stable safe state appears.

### Walk-forward validation and minimum evidence

Temporal folds; circular metrics; profile quality; source/season stability.

**Minimum sample rules:** 1,000 teacher profiles, 500 overlap rows, 100 official proxy-state rows.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0157_shear_veering_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`shear_teacher.csv`, `surface_student_oof.parquet`, `circular_alignment.csv`, `unexplained_effect.csv`, and `response_atlas.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Safe student must add information beyond surface wind; otherwise use result to specify a future issue-time-safe NWP variable request.

### Expected failure modes and interpretation

Surface winds are exposure-dependent and may poorly represent vertical flow. Negative result is expected/valuable.

### Expected information gain

Moderate diagnostic value; lower immediate lift probability than thermal/moisture proxies.

---

## 0158 — Sea-Temperature and Marine-Moderation Safe Proxy Conversion

**Priority:** P1  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** Daily marine teacher diagnostic-only; live feeds prospective  
**Dependencies:** 0135, 0141, 0143, 0102

### Decision question and hypothesis

North Point sea temperature and marine daily values likely rank well because they control air–sea contrast and marine suppression. Coastal/inland station spreads, onshore fetch, dewpoint, and recent seasonal state may safely proxy marine thermal influence.

### Why this is new rather than a relabelled prior experiment

EXP-0089 used sea–air contrast directly and failed; 0100 still found marine diagnostics valuable. This experiment isolates the teacher mechanism and creates a deployable station/geospatial student.

### Response variables

Diagnostic sea-air contrast, target anomaly, official overforecast, marine-suppression state, and JJA/MAM high-error.

### Exact inputs

Blocked marine/daily sea-temperature values; safe station groups, coastline/fetch, onshore wind, target/climatology, dewpoint and pressure; official responses.

### Feature constructions and calculations

Teacher sea-surface/air contrast and tendency. Student seasonal sea-temperature climatology based only on prior years, coastal station thermal inertia, coastal-inland spread, onshore fetch, dewpoint, wind persistence, target seasonal phase, and lagged marine state only if publication proof exists.

### Procedure

Measure teacher effect and transfer across decades. Train safe students excluding blocked values. Compare student to simple day-of-year and coastal spread. Test a bounded marine-overforecast correction and uncertainty flag.

### Walk-forward validation and minimum evidence

Teacher/student firewall; causal climatology uses prior years only; leave-year-out and season/source stability.

**Minimum sample rules:** 1,000 teacher rows, 500 safe overlap rows, 100 marine-suppression official events.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0158_marine_temperature_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`marine_teacher.parquet`, `safe_marine_student.parquet`, `climatology_ablation.csv`, `teacher_alignment.csv`, `suppression_response.csv`, and `specialist_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Student must add residual information beyond calendar and coastal spread and reduce overforecast without suppressing genuine hot days.

### Expected failure modes and interpretation

Sea temperature evolves slowly and may be mostly calendar. If student adds no conditional value, retain calendar/coastal proxy and do not chase blocked data.

### Expected information gain

High targeted information potential and direct bridge from blocked marine signal to safe station features.

---

## 0159 — Tropical-Cyclone Subsidence, Cloud-Shield, and Flow-Regime Safe Proxy

**Priority:** P1  
**Research mode:** Diagnostic teacher → deployable student  
**Eligibility:** Best-track teacher diagnostic-only; safe student deployable  
**Dependencies:** 0132–0135, 0149

### Decision question and hypothesis

TC proximity is not monotonic: outer subsidence/offshore flow can produce exceptional heat, while cloud shield/rain bands suppress Tmax. Safe pressure, wind, moisture, cloud-proxy, and station-gradient states may distinguish these opposing effects without retrospective best track.

### Why this is new rather than a relabelled prior experiment

EXP-0093 TC teacher/student was mechanism-only and did not beat the core. This experiment splits the teacher into physically opposite quadrants/states, uses response-specific tails, and demands operationally observable student triggers.

### Response variables

Teacher TC regime (subsidence-hot, cloudy/wet, transition, none), hot underforecast, cold/overforecast, target anomaly, and high-error.

### Exact inputs

Retrospective best-track geometry in F-DIAG; safe pressure tendency/gradient, wind field/fetch, dewpoint/front, radiative proxy, target state, official forecast/text.

### Feature constructions and calculations

Teacher distance/bearing/intensity/motion/quadrant and lag. Student cyclonic pressure/wind signature, directional persistence, offshore/onshore fetch, pressure fall/rise, moisture/radiative proxy, regional asymmetry, forecast text tokens, and transition confidence.

### Procedure

Create teacher states before looking at residuals using physical rules; quantify signed effects. Train safe multiclass student and evaluate transfer. Test hot and suppression specialists separately with bounded actions and abstention.

### Walk-forward validation and minimum evidence

Best track never enters deployable scoring. Event-level folds keep TC episodes intact; compare against generic pressure/wind/radiative states.

**Minimum sample rules:** At least 50 TC episodes and 100 teacher-state days for broad mechanism; 50 high-confidence student official events per action for exploratory promotion.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0159_tc_regime_safe_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`tc_teacher_events.csv`, `tc_regime_labels.csv`, `safe_student_oof.parquet`, `episode_level_scores.csv`, `generic_state_ablation.csv`, and `directional_specialists.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Student must distinguish opposing TC effects and add tail information beyond generic states. A specialist must improve its direction without global/tail harm.

### Expected failure modes and interpretation

TC events are rare and best-track revisions are retrospective. Use event-level uncertainty and do not promote if student confidence is low.

### Expected information gain

High tail-risk information; likely limited average-MAE lift but important for extreme hot/cloudy misses.

---

## 0160 — Timestamp-Proof Acquisition and Prospective Shadow-Latency Experiment

**Priority:** P0  
**Research mode:** Data unblock / prospective audit  
**Eligibility:** Prospective only; does not retroactively unlock history without proof  
**Dependencies:** 0102, 0106

### Decision question and hypothesis

The only defensible way to promote blocked families is to capture issue/publication/availability timestamps prospectively and, where possible, obtain provider documentation or archived first-publication evidence for historical latency.

### Why this is new rather than a relabelled prior experiment

0102 produced a zero-unlock queue. This experiment is the operational plan to collect the missing proof, measure real release latency, and decide whether conservative historical lag assumptions can ever be justified.

### Response variables

Release-latency distribution, missing publication events, revision frequency, earliest safe cutoff, and feature availability status.

### Exact inputs

Live IGRA/provider feeds, HKO daily climate/marine publications, retrieval logs, HTTP headers where relevant, source documentation, and normalized source rows.

### Feature constructions and calculations

`valid_at`, `issued_at`, `first_seen_at`, `retrieved_at`, content hash, revision hash, latency minutes/hours, cutoff eligibility, provider outage, and first-publication confidence.

### Procedure

Poll/capture sources on a fixed schedule, store immutable raw snapshots and hashes, measure first appearance and revisions. Search existing archive metadata for first-seen evidence. Define conservative latency contracts by source/product only after sufficient observations; never infer historical exact availability from current behavior alone.

### Walk-forward validation and minimum evidence

Clock synchronization, duplicate retrieval handling, snapshot integrity, and cross-check against provider timestamps. Produce a red-team assessment of whether proof supports historical promotion or prospective-only use.

**Minimum sample rules:** At least 90 days and 50 publication events for provisional latency characterization; a full seasonal year preferred before generalization. Historical unlock requires direct documentary/archival proof, not sample extrapolation.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0160_timestamp_proof_shadow_latency/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`availability_events.parquet`, `latency_distributions.csv`, `revision_history.parquet`, `snapshot_manifest.jsonl`, `provider_evidence.md`, `eligibility_decisions.csv`, and `historical_unlock_assessment.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Every captured row has immutable first-seen evidence; conservative cutoff rule has quantified violation rate. Zero historical rows are unlocked without explicit proof.

### Expected failure modes and interpretation

Provider pages may overwrite values or lack issue timestamps. First-seen retrieval still enables future deployment but not retrospective scoring.

### Expected information gain

Critical long-term value; no immediate historical MAE gain, but it can unlock the strongest physical families safely for future/live use.

---

## 0161 — Exact 0075/0081 Online-Memory Replay on the 5,265-Row Expanded Frame

**Priority:** P0  
**Research mode:** Promotion-oriented benchmark  
**Eligibility:** Deployable now  
**Dependencies:** 0104, 0110

### Decision question and hypothesis

The largest proven narrow-frame lift came from online residual memory. Replaying the exact logic on F-O5265 will determine whether it is robust or an artifact of the old frame and modern RSS composition.

### Why this is new rather than a relabelled prior experiment

0103 reports the 0101 candidate on both frames but does not provide an exact expanded replay of every 0075/0081 state key, halflife, cap, min-history, and prior-lift rule with row-by-row attribution.

### Response variables

Official residual and realized correction uplift; secondary absolute-error/tail responses.

### Exact inputs

F-O5265/F-O2670/F-RSS992, exact official predictions, source/era/season labels, legacy 0075/0081 configs and predictions.

### Feature constructions and calculations

Exact legacy all-context/source-context residual states; halflife 20 and legacy variants; min-history 10 and hardened alternatives; correction cap 0.2 and sensitivity; prior-lift weighting; RSS gate; state confidence and active-row reason.

### Procedure

First reproduce legacy predictions on F-O2670. Then run identical code and parameters on F-O5265 without retuning. Decompose performance by old rows versus newly added 2,595 rows, source era, season, year, correction sign, and active-state support. Only after frozen replay run a nested sensitivity grid.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Online state must score T before updating with T residual. Legacy frozen replay is distinct from tuned canonical variant.

**Minimum sample rules:** All 5,265 rows scored; any state requires its declared prior history; sensitivity promotion requires at least 100 active rows and multiple temporal blocks.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0161_expanded_online_memory_replay/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`legacy_reproduction.csv`, `expanded_oof_predictions.parquet`, `old_vs_new_rows.csv`, `state_trace.parquet`, `frozen_vs_retuned.csv`, `slice_scoreboard.csv`, and `robustness_decision.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Legacy replay must reproduce old score. A promoted expanded candidate must beat 0103 on F-O5265, improve most eras/seasons or have explicit no-harm gates, and preserve P90/P95/tails.

### Expected failure modes and interpretation

New press-era rows may reveal that the narrow-frame lift was composition-specific. That result is decisive; retain online memory only in sources/eras with prior evidence.

### Expected information gain

Highest immediate model-lift priority because it tests the strongest existing mechanism on the proper canonical frame.

---

## 0162 — Hierarchical Source–Era–Season Online Residual Memory

**Priority:** P1  
**Research mode:** Promotion-oriented  
**Eligibility:** Deployable now  
**Dependencies:** 0161, 0116

### Decision question and hypothesis

Residual bias evolves at multiple resolutions. Partial pooling across global, source, source-era, season, and source-season states should adapt without the sparsity of independent cells.

### Why this is new rather than a relabelled prior experiment

0074/0075 and later source/era shrinkage used several keys and gates. This experiment formalizes a hierarchical state with uncertainty-weighted shrinkage and feature-specific effect lifetimes derived from 0116.

### Response variables

Official residual, absolute error, directional tails, and correction uplift.

### Exact inputs

F-O5265, source/era/season/submonth, online residual history, exact official forecast metadata.

### Feature constructions and calculations

Exponentially weighted mean/median residual at global, source, era, season, source-season, and submonth levels; effective sample size; residual variance; sign streak; time since last observation; learned half-life from training only; Bayesian/empirical shrinkage weights; and cap/confidence.

### Procedure

At each date compute states from prior scored dates only. Compare fixed hierarchy, empirical-Bayes pooling, and conservative rule-based shrinkage. Predeclare half-life grid; nested folds choose at most one per hierarchy level. Ablate each level.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. State trace must prove score-then-update. Report performance on sparse era transitions and after long archive gaps.

**Minimum sample rules:** Global state always; lower-level state begins after 20 prior rows, reaches full influence after 100; active corrections at least 200 rows for promotion.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0162_hierarchical_online_memory/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`hierarchical_state_trace.parquet`, `shrinkage_weights.parquet`, `level_ablation.csv`, `gap_reset_sensitivity.csv`, `oof_predictions.parquet`, and `scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Beat frozen 0161 and simple global/source memory on F-O5265 with stable slices and no tail degradation. Complexity must yield a meaningful, reproducible delta.

### Expected failure modes and interpretation

Hierarchy may overreact to source gaps or double-count correlated contexts. Reset/decay states across long gaps and prefer simpler memory if tied.

### Expected information gain

High likely incremental lift and robustness; one of the core production candidates.

---

## 0163 — Signed Residual Streak, Bias Momentum, and Multi-Halflife State

**Priority:** P1  
**Research mode:** Promotion-oriented  
**Eligibility:** Deployable now  
**Dependencies:** 0161, 0162

### Decision question and hypothesis

A sequence of same-sign errors or acceleration in bias can signal source/model drift faster than a single exponentially weighted mean. Short and long memory disagreement may indicate a turning point where correction should be reduced or reversed.

### Why this is new rather than a relabelled prior experiment

Residual streaks and half-lives were mentioned in prior work, but no dedicated expanded-frame study decomposes sign persistence, magnitude momentum, and multi-timescale disagreement with no-regret action policies.

### Response variables

Next official residual sign/magnitude, correction uplift, and harmful-activation flag.

### Exact inputs

Prior official residuals by source, hierarchical memory states, source gaps/eras, canonical official frame.

### Feature constructions and calculations

Consecutive over/underforecast length; signed run magnitude; EW means/medians at halflives 5/10/20/40/80; short-minus-long bias; residual slope/curvature; sign entropy; last error magnitude; time-gap adjusted decay; and reversal probability.

### Procedure

Estimate conditional residual curves and uplift for small actions. Compare mean-only memory, sign streak, multi-halflife, and combined state. Use a conservative state machine: reinforce, hold, shrink, or abstain.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. All residual state uses outcomes only after they became known and only prior target dates. Event/gap sensitivity explicit.

**Minimum sample rules:** 20 prior source rows before use, 100 occurrences per streak/momentum state, 50 activations per fold.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0163_residual_streak_momentum/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`residual_momentum_state.parquet`, `state_response_curves.csv`, `memory_ablation.csv`, `action_policy_oof.parquet`, `harmful_activation.csv`, and `scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Combined state must beat the best single halflife and reduce harmful corrections around reversals; no material source/season regressions.

### Expected failure modes and interpretation

Streaks can arise by chance and sparse states overfit. Use shrinkage and broad states; do not tune run thresholds repeatedly.

### Expected information gain

Moderate-to-high lift potential as a cheap enhancement to online memory.

---

## 0164 — Full Forecast-Vintage Revision Path and Shape Features

**Priority:** P1  
**Research mode:** Exploratory then promotion-oriented  
**Eligibility:** Deployable only where all revisions are timestamp-eligible  
**Dependencies:** 0104, 0106, official archive completeness

### Decision question and hypothesis

The sequence of official forecast revisions contains human/model information: persistent upward revisions, late reversals, widening ranges, or stale unchanged forecasts may signal confidence and residual direction better than the final selected value alone.

### Why this is new rather than a relabelled prior experiment

0035/0036 examined revision momentum/centroids, but this experiment standardizes the entire pre-cutoff revision path across press/RSS eras, handles missing revisions, and separates path shape from source-era artifacts.

### Response variables

Official residual of the selected operational vintage, absolute error, trust state, and correction uplift.

### Exact inputs

All parsed forecast vintages with issue/available-at times, exact selected vintage, numeric max/min/range and text fields, source-era parser metadata.

### Feature constructions and calculations

Revision count; time since first/last issue; first-to-final change; monotone revision run; max excursion; path slope/curvature; late reversal; revision volatility; range midpoint/width changes; text-numeric direction consistency; stale/no-change duration; and path completeness confidence.

### Procedure

Construct paths using only revisions issued before cutoff; never select after cutoff. Normalize product-era cadence. Compare final value alone, last change, legacy momentum features, and full path shape. Fit tiny smooth residual/trust models.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Exact issue-time audit; source-era stratification; missing revision path is a confidence state, not zero change.

**Minimum sample rules:** At least two eligible vintages for path features; 300 multi-vintage rows globally; 100 per source era for claims.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0164_forecast_revision_path/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`forecast_revision_paths.parquet`, `path_feature_catalog.csv`, `era_cadence_map.csv`, `path_completeness.csv`, `legacy_ablation.csv`, and `oof_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Path shape must add information beyond final forecast and last revision, survive source-era controls, and improve trust/correction without leakage.

### Expected failure modes and interpretation

Press archive may not retain all revisions and product cadence changes. Limit claims to verified complete paths; do not infer absent revisions.

### Expected information gain

High information potential where vintage density is adequate, especially after backfill.

---

## 0165 — Causal Forecast-Text Ontology and Meteorological State Extraction

**Priority:** P1  
**Research mode:** Exploratory information-gain  
**Eligibility:** Deployable where exact pre-cutoff text vintage exists  
**Dependencies:** 0104, 0106, 0164

### Decision question and hypothesis

Official text encodes cloud, rain, wind, temperature trend, uncertainty, and TC context not fully captured by the numeric max. A causal, era-robust ontology may identify when numeric forecasts are biased or uncertain.

### Why this is new rather than a relabelled prior experiment

0029 and prior text buckets examined source/text dynamics, but no complete ontology with exact vintage, product-era vocabulary drift, multi-label state, and out-of-era validation has been built.

### Response variables

Official residual, absolute error, signed tails, forecast trust, and diagnostic physical regimes.

### Exact inputs

Exact selected press/RSS forecast text, issue time, numeric ranges, source era, target/official responses.

### Feature constructions and calculations

Normalized bilingual/English phrase ontology if present: rain intensity/probability, cloud/fog/mist, sunny periods, wind direction/strength/change, temperature trend, uncertainty qualifiers, TC mentions, thunderstorms, visibility. Add negation, temporal qualifier, phrase co-occurrence, text completeness, and era vocabulary mapping. Avoid unrestricted embeddings initially.

### Procedure

Build lexicon from training text and provider terminology; manually document mappings. Fit sparse multi-label indicators and low-dimensional topic/state models fold-locally. Test residual/error curves and interactions with numeric range/station state. Compare lexicon to bag-of-words and simple keyword buckets.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Vocabulary and token statistics fit inside folds; test transfer from press to RSS and by era. Exact pre-cutoff vintage mandatory.

**Minimum sample rules:** 100 occurrences per promoted phrase/state; 500 rows for a text model; rare phrases descriptive only.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0165_forecast_text_ontology/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`text_ontology.yaml`, `text_state.parquet`, `phrase_counts_by_era.csv`, `transfer_scores.csv`, `numeric_interactions.csv`, `oof_predictions.parquet`, and `error_cases.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Ontology must add residual/uncertainty information beyond numeric forecast/source and transfer across at least two eras or be explicitly source-specific with support.

### Expected failure modes and interpretation

Vocabulary drift, parser loss, and correlated source eras can create fake effects. Require within-era tests and preserve unknown text states.

### Expected information gain

High potential because human forecaster context is otherwise unavailable in station-only data.

---

## 0166 — Numeric–Text–Station Consistency and Contradiction Index

**Priority:** P1  
**Research mode:** Promotion-oriented trust research  
**Eligibility:** Deployable on exact text/numeric vintages  
**Dependencies:** 0165, 0123, 0149

### Decision question and hypothesis

Official forecasts may be least trustworthy when the numeric max, text-implied weather, recent target envelope, and station-network state contradict one another. Agreement should favor abstention; contradiction may indicate a correction direction or high uncertainty.

### Why this is new rather than a relabelled prior experiment

Existing stacks combine signals but do not explicitly encode semantic consistency among the official product’s own components and independent observations.

### Response variables

Official absolute error, residual sign, trust state, correction uplift, and high-error flag.

### Exact inputs

Numeric forecast/range, text ontology, target envelope/state, station heat/moisture/pressure/radiative proxies, source era.

### Feature constructions and calculations

Text-implied heating/suppression score; numeric forecast anomaly versus target envelope; station-implied thermal direction; pairwise sign agreement; contradiction count; weighted consistency confidence; range width; source/text completeness; and contradiction type (hot numeric+rain text, cool numeric+dry-hot station, etc.).

### Procedure

Define semantic direction mappings before outcome analysis. Estimate error/tail enrichment by contradiction type. Train a calibrated trust model with few features; test correction actions only where contradiction has stable signed residual evidence.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Ontology and thresholds fold-local; compare to numeric-only, text-only, and station-only trust models.

**Minimum sample rules:** 100 rows per broad contradiction type, 500 total text-complete rows, 60 correction activations per fold for promotion.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0166_forecast_consistency_contradiction/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`consistency_features.parquet`, `contradiction_catalog.csv`, `error_enrichment.csv`, `component_ablation.csv`, `trust_oof.parquet`, and `correction_policy_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Contradiction index must improve high-error calibration/trust and any action must reduce error with abstention/no-harm gates.

### Expected failure modes and interpretation

Text direction can be ambiguous and product templates era-specific. Use coarse, documented states and confidence; do not force sentiment-style scores.

### Expected information gain

High decision value and plausible small point-lift plus uncertainty improvement.

---

## 0167 — Forecast Range, Predictive-Scale, and Trust Calibration

**Priority:** P1  
**Research mode:** Uncertainty and routing  
**Eligibility:** Deployable where exact range fields exist  
**Dependencies:** 0108, 0164–0166

### Decision question and hypothesis

Forecast min/max range width, revisions, station disagreement, and recent source error may predict absolute error. Calibrated uncertainty can determine correction strength and abstention even when point prediction does not improve directly.

### Why this is new rather than a relabelled prior experiment

Forecast range has been used as a feature, and EXP-0098 attempted Student-t uncertainty broadly. This experiment calibrates empirical absolute-error distributions around the official anchor with simple source-aware scales and explicit utility for gating.

### Response variables

Absolute official error, >1.5/2/3°C flags, signed residual quantiles, and realized correction uplift.

### Exact inputs

Forecast range/midpoint/max, revision/text consistency, station disagreement, target volatility, source residual variance, seasonal phase.

### Feature constructions and calculations

Raw and normalized range width; recent range change; source/season empirical scale; station dispersion; target volatility transition; contradiction; online residual variance; and missing/null confidence. Fit conformal-style or quantile calibration using prior residuals only.

### Procedure

Compare range-only, online-scale-only, station uncertainty, and combined scales. Produce calibrated P50/P80/P90/P95 absolute-error forecasts and signed intervals. Use predicted uncertainty to modulate correction cap or abstain.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Evaluate interval coverage, width, CRPS/pinball where applicable, high-error Brier/AUC, and point-policy effect.

**Minimum sample rules:** 200 prior residuals per source for source calibration; shrink to global below support; at least 100 tail events.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0167_forecast_range_trust_calibration/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`uncertainty_oof.parquet`, `coverage_tables.csv`, `scale_ablation.csv`, `tail_calibration.csv`, `uncertainty_gated_policy.csv`, and `reliability_plots/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Intervals calibrated across source/season and uncertainty-gated policy improves or preserves point MAE while reducing harmful activations/tails. Complexity must beat simple rolling residual MAD.

### Expected failure modes and interpretation

Forecast range may reflect temperature range, not uncertainty. Test semantics by product era and do not assume width meaning is constant.

### Expected information gain

High operational/trading value; moderate contribution to point corrections through gating.

---

## 0168 — Forecast Staleness, Null Fields, Parser Confidence, and Product-Quality State

**Priority:** P1  
**Research mode:** Data quality + trust routing  
**Eligibility:** Deployable now  
**Dependencies:** 0104, 0106, 0164

### Decision question and hypothesis

Missing max/min fields, stale unchanged vintages, parser ambiguity, truncated text, and product-era changes can predict data-quality risk and should alter trust or trigger fallback—not silently become zeros or imputed forecasts.

### Why this is new rather than a relabelled prior experiment

Archive gap and null concerns are documented, but no row-level quality state is integrated into score/tail analysis and fallback policy.

### Response variables

Parser/quality failure, official absolute error, residual sign, high-error, and fallback uplift.

### Exact inputs

Raw/parsed forecast records, selected vintage metadata, parser logs, null fields, source-era templates, retrieval/issue times, official responses.

### Feature constructions and calculations

Field completeness; parse confidence; text length/template match; numeric range consistency; issue age; unchanged-vintage duration; duplicate/conflicting records; source fallback used; row provenance count; and anomaly flags.

### Procedure

Create deterministic quality rules independent of outcome. Audit error by quality state. Compare dropping, fallback to prior official, source-memory correction, and safe station/target fallback. Never impute from future or choose a better later record.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Raw-to-parsed spot checks by era; fallback evaluated OOF on rows where simulated quality failure occurs.

**Minimum sample rules:** All forecast rows quality-scored; 100 rows per major quality state for error claims; rare parser failures documented individually.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0168_forecast_quality_state/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`forecast_quality.parquet`, `parser_audit.csv`, `quality_error_atlas.csv`, `fallback_simulation.csv`, `null_field_policy.yaml`, and `raw_examples/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

No silent null/imputation paths; quality state explains trust or supports a fallback that reduces damage; parser confidence reproducible.

### Expected failure modes and interpretation

Quality flags may correlate with source era. Report within-era effects and keep quality controls even if predictive value is low.

### Expected information gain

High robustness value and possible tail reduction; essential before continuous backfill replay.

---

## 0169 — Invariant Press–RSS Cross-Era Calibration and Source Bridge

**Priority:** P1  
**Research mode:** Promotion-oriented robustness  
**Eligibility:** Deployable now on current disjoint eras; stronger after backfill  
**Dependencies:** 0104, 0161–0168

### Decision question and hypothesis

Press and RSS products differ in format, era, cadence, and possibly semantics. A source-invariant feature set plus hierarchical source offsets can transfer correction logic without mistaking climate era for source effect.

### Why this is new rather than a relabelled prior experiment

Source/era-aware fusion exists, but the archive gap makes source and time nearly confounded. This experiment explicitly seeks invariant features, runs negative controls, and quantifies what cannot be identified until overlap/backfill.

### Response variables

Official residual, absolute error, correction uplift, and trust state.

### Exact inputs

Press and RSS canonical rows, common numeric/text/quality features, safe station/target states, online residual memory.

### Feature constructions and calculations

Common semantic numeric features; standardized forecast anomaly/range; source-specific intercept and scale; invariant station/target interactions; domain classifier probability; residualized source effect; and overlap-simulation via temporal holdouts.

### Procedure

Train correction on press, test on RSS; train RSS, test late press only as exploratory due chronology; leave-era-out within press; adversarially identify source-predictive features and remove/stratify them. Compare common-rule, separate-rule, and hierarchical pooled models.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Make confounding explicit; no claim of causal source effect without temporal overlap. Report domain-classifier accuracy and transfer gap.

**Minimum sample rules:** At least 500 rows per source for pooled models; source-specific specialists require 200 activations.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0169_press_rss_invariant_calibration/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`common_feature_schema.csv`, `domain_shift_report.csv`, `cross_source_transfer.csv`, `hierarchical_calibration_oof.parquet`, `source_ablation.csv`, and `identifiability_limits.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A bridge is useful only if common rules transfer better than source-blind baseline and hierarchical pooling beats fully separate/single-source approaches without hidden era leakage.

### Expected failure modes and interpretation

Source and era confounding may be irresolvable until archive continuity improves. In that case freeze source-specific rules and prepare unchanged replay after backfill.

### Expected information gain

High robustness information; moderate immediate lift, larger future value after backfill.

---

## 0170 — Uplift-Based Abstaining Trust Router Across Credible Experts

**Priority:** P1  
**Research mode:** Promotion-oriented integration  
**Eligibility:** Deployable now after expert qualification  
**Dependencies:** 0117, 0161–0169, selected station/target specialists

### Decision question and hypothesis

The best system should choose among a small set of credible actions—raw official, online memory, target-memory correction, station-network correction, MAM specialist, and tail specialist—and abstain when expected uplift is uncertain.

### Why this is new rather than a relabelled prior experiment

Prior trust routers and expert stacks often diluted gains. This router uses action-specific uplift, lower confidence bounds, no-regret online performance, and a hard complexity budget instead of feature-soup prediction.

### Response variables

Action uplift versus raw official, harmful activation, regret, point/tail metrics, and expert trust state.

### Exact inputs

OOF predictions from raw official, 0162/0163 memory, one qualified target expert, one station expert, one MAM expert, and one tail expert; safe router features only.

### Feature constructions and calculations

Prior expert MAE/bias by source/season/state; predicted uncertainty; expert disagreement; uplift lower bounds; activation support; source quality; contradiction; network confidence; and recent no-regret weights. Maximum six experts.

### Procedure

Compare fixed blend, inverse-prior-MAE blend, positive-lift gate, contextual bandit/no-regret weights with delayed outcomes, and abstaining uplift router. All expert predictions must be genuinely OOF and generated without router test outcomes. Penalize switching/complexity.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Nested OOF prevents meta-leakage. Report oracle gap only as diagnostic, not attainable score.

**Minimum sample rules:** Each expert requires its own promotion evidence; router context needs 200 prior rows or shrinks to global; at least 100 non-raw actions overall.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0170_uplift_abstaining_router/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`expert_oof_matrix.parquet`, `router_state_trace.parquet`, `router_predictions.parquet`, `action_counts.csv`, `regret.csv`, `expert_ablation.csv`, and `scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Router must beat the best single expert and simple fixed blend on F-O5265, preserve all major tails, avoid one-era concentration, and justify complexity by a reproducible delta.

### Expected failure modes and interpretation

Meta-model can exploit noisy OOF artifacts or select sparse winners. Use minimal features, nested folds, abstention, and action support thresholds. If no router beats 0162, deploy 0162.

### Expected information gain

High final integration value, but only after individual experts pass. Expected gains are small; robustness matters more than headline decimals.

---

## 0171 — MAM Latent Transition-Phase Classifier and Error Map

**Priority:** P1  
**Research mode:** Exploratory then promotion-oriented  
**Eligibility:** Deployable now with safe features  
**Dependencies:** 0122, 0123, 0130–0136, 0162

### Decision question and hypothesis

MAM’s high MAE is a mixture of cool-monsoon persistence, humid maritime onset, fog/low-cloud suppression, frontal passages, dry breaks, and rapid warming. A single MAM correction fails because these states require opposite actions.

### Why this is new rather than a relabelled prior experiment

0059/0060 and 0094–0101 developed spring/MAM specialists and stable cells. This experiment replaces narrow historical cells with a physically defined soft phase model using safe target/station states and evaluates each phase’s residual anatomy.

### Response variables

MAM official residual, absolute error, overforecast, underforecast, >1.5/2°C flags, online-memory residual, and action uplift.

### Exact inputs

MAM rows in F-LONG/F-O5265; target phase/slope/volatility; station thermal/dewpoint fronts; pressure waves; wind kinematics; sea-breeze/cool-surge states; radiative safe proxy; official range/text/quality where available.

### Feature constructions and calculations

Soft probabilities for cool-monsoon, cool-humid/fog, frontal transition, post-front dry warming, warm-humid maritime, cloud-break warming, and ambiguous/disagreement. Add phase speed, confidence, duration, source/era and forecast-vs-target envelope disagreement.

### Procedure

Define physical anchors without target residual, then fit a fold-local soft classifier/state model. Produce residual/tail maps by phase and compare to calendar submonth and 0101 cell. Qualify no more than three phase-specific bounded actions through uplift/no-harm rules.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. MAM-specific leave-year-out and early/late-era folds; non-MAM rows receive zero specialist correction and are checked for implementation leakage.

**Minimum sample rules:** At least 200 rows per broad phase for formal residual claims; 100 official rows and 50 high-error events per promoted phase action.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0171_mam_transition_phase/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`mam_phase_oof.parquet`, `phase_definitions.csv`, `phase_residual_map.csv`, `0101_cell_comparison.csv`, `phase_specialist_scores.csv`, `non_mam_zero_check.csv`, and `casebook.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Phase model must improve MAM residual/tail understanding beyond month/submonth and 0101 trajectory cell. Promoted actions must reduce MAM MAE/tails without global/non-MAM harm and remain stable across years/sources.

### Expected failure modes and interpretation

Soft states can be unstable or simply encode date. Calendar-only ablation and physical feature importance are mandatory. If only one era supports a phase, keep exploratory.

### Expected information gain

Among the highest likely specialist lifts because MAM is the largest current seasonal error slice and existing evidence already shows small safe gains.

---

## 0172 — Cool-Surge Breakdown and Rebound-Warming Specialist

**Priority:** P1  
**Research mode:** Promotion-oriented specialist  
**Eligibility:** Deployable now  
**Dependencies:** 0136, 0120, 0171

### Decision question and hypothesis

Official forecasts may extend cool conditions too long when a surge weakens, pressure falls, winds veer, dewpoint/temperature recover, and regional warming reaches HKO. A positive bounded correction should help only on confirmed breakdown days.

### Why this is new rather than a relabelled prior experiment

Spring pressure/dew specialists and surge detectors exist, but this is a narrowly defined event-phase action with propagation confirmation, target derivative exhaustion, and explicit opposite-tail protection.

### Response variables

Official overforecast magnitude on breakdown/rebound days, action uplift for +0.15/+0.25/+0.40°C, and cold-to-warm transition tails.

### Exact inputs

Cool-surge phase, pressure-wave weakening, wind veer, north-south thermal recovery, dewpoint change, target reversal state, forecast envelope, source memory.

### Feature constructions and calculations

Surge age; pressure tendency reversal; northerly component decay; warm-front/thermal centroid approach; target slope turning positive; station rank reversal; dewpoint recovery; official below target envelope; and confidence/support.

### Procedure

Predeclare trigger logic from 0136/0171. Estimate uplift curves for small positive actions; fit a shrunk probability of beneficial correction. Compare event-only residual memory, simple pressure reversal, and full confirmation. Correction is zero outside trigger.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Event-level folds and leave-year-out MAM/DJF; test hot-overforecast harm after false rebounds.

**Minimum sample rules:** At least 100 breakdown candidates and 60 official trigger rows; 20 events in each of at least three temporal folds.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0172_cool_surge_breakdown_specialist/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`breakdown_events.csv`, `trigger_oof.parquet`, `uplift_curves.csv`, `confirmation_ablation.csv`, `specialist_predictions.parquet`, and `tail_safety.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Positive action must improve breakdown-day and overall F-O5265 MAE, not increase hot underforecast/overforecast tails, and activate across multiple years/source eras.

### Expected failure modes and interpretation

Sparse event support and uncertain phase timing. Use abstention and shrinkage; do not tune trigger thresholds after seeing test years.

### Expected information gain

Moderate targeted lift with strong interpretability; likely contributes more to MAM/DJF tail reduction than global MAE.

---

## 0173 — Humid Maritime Suppression versus Cloud-Break Warming Specialist

**Priority:** P1  
**Research mode:** Promotion-oriented specialist  
**Eligibility:** Safe proxy deployable; blocked teachers diagnostic-only  
**Dependencies:** 0149, 0150, 0171, 0135

### Decision question and hypothesis

High moisture can either cap Tmax through cloud/fog/rain or support a warm maritime air mass that heats rapidly after a cloud break. Station warming rate, radiative proxy, pressure/wind, and spatial coherence should distinguish opposite residual directions.

### Why this is new rather than a relabelled prior experiment

Prior moisture/cloud features were often treated monotonically. This experiment explicitly models two competing humid states and only acts when the direction is identified.

### Response variables

Official overforecast in suppression state, underforecast in cloud-break state, high-error tails, and action uplift for ±0.15/0.25°C.

### Exact inputs

Safe surface enthalpy/T−Td, dewpoint tendency/advection, radiative-suppression proxy, morning warming proxy, sea-breeze phase, pressure/wind, target envelope, exact forecast text/range where available.

### Feature constructions and calculations

Humid-level score; moisture-surge score; station warming coherence/shortfall; cloud-break acceleration; pressure stabilization; wind ventilation; text sunny/rain/cloud states; suppression probability; cloud-break probability; and ambiguity confidence.

### Procedure

Train two calibrated state probabilities, not one linear moisture feature. Estimate signed residual and uplift in each state. Compare moisture-only, proxy-only, warming-only, and combined. Apply negative correction for suppression and positive for cloud break with strict abstention.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Separate MAM/JJA/SON curves and source-era text availability; blocked cloud/sun labels never enter deployable inference.

**Minimum sample rules:** 100 official events per broad state, 50 high-confidence actions, and at least three years represented.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0173_humid_maritime_cloudbreak_specialist/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`humid_state_oof.parquet`, `state_confusion_diagnostics.csv`, `component_ablation.csv`, `signed_uplift.csv`, `specialist_predictions.parquet`, and `opposite_tail_safety.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Combined state must resolve opposite signs better than moisture alone and improve relevant tails without cross-state harm.

### Expected failure modes and interpretation

Cloud-break timing may require intraday data unavailable in long history. If safe proxies are weak, keep as uncertainty/short-history layer rather than force correction.

### Expected information gain

High MAM/JJA information and meaningful tail upside; average-MAE gain uncertain but plausible.

---

## 0174 — Weak-Wind Heat-Buildup and Dry-Subsidence Specialist

**Priority:** P1  
**Research mode:** Promotion-oriented specialist  
**Eligibility:** Deployable now from safe proxies  
**Dependencies:** 0141, 0142, 0146, 0153, 0155

### Decision question and hypothesis

Weak ventilation, dry-heating potential, warm regional air, pressure/ridge proxy, and persistent target warmth can create heat buildup that official forecasts understate. Weak wind alone is insufficient; the multi-signal physical conjunction should be required.

### Why this is new rather than a relabelled prior experiment

Waglan wind, ridge, and heat-content direct models failed. This specialist uses them only as safe proxy components, targets hot underforecast, and applies a small positive action under replicated conjunctions.

### Response variables

Hot-day underforecast magnitude, >1.5/2°C hot miss, target breakout, and +0.15/+0.25/+0.40°C uplift.

### Exact inputs

Safe wind stagnation/ventilation, T−Td/dry-air, target heat-storage/state, regional thermal/elevation fields, pressure/height students, forecast envelope and source memory.

### Feature constructions and calculations

Calm fraction and persistence; low ventilation index; dry-heating potential; regional warm anomaly; pressure/ridge proxy; urban heat-storage state; official below upper envelope; absence of radiative-suppression/marine state; and specialist confidence.

### Procedure

Predeclare conjunctive score and smooth probability. Compare each component, additive score, and interaction. Use uplift policy to select small positive correction only when lower-bound gain is positive; abstain under moisture/cloud/marine ambiguity.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Hot-event and leave-summer/year folds; report false-hot corrections and P95.

**Minimum sample rules:** At least 100 high-confidence heat-buildup rows and 50 hot-underforecast events; actions represented in multiple years.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0174_weak_wind_heat_buildup/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`heat_buildup_state.parquet`, `component_ablation.csv`, `hot_event_uplift.csv`, `specialist_predictions.parquet`, `false_activation.csv`, and `tail_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Must reduce hot underforecast/tail errors and improve global or hot-slice MAE without increasing hot overforecast; no blocked teacher dependency.

### Expected failure modes and interpretation

Weak wind can accompany cloud/rain or marine stagnation. Negative gates for suppression/marine states are mandatory.

### Expected information gain

Moderate-to-high targeted potential for extreme hot misses.

---

## 0175 — Marine Suppression versus Inland Heat-Potential Duel Specialist

**Priority:** P1  
**Research mode:** Promotion-oriented specialist  
**Eligibility:** Deployable now from station/geospatial proxies  
**Dependencies:** 0135, 0141, 0143, 0158

### Decision question and hypothesis

Regional inland stations may signal strong heat potential while coastal/onshore state caps HKO. The residual depends on which regime wins. A duel score between inland heat and marine penetration can decide whether to trust high official forecasts or shrink them.

### Why this is new rather than a relabelled prior experiment

Prior coastal-inland models used spreads or broad experts. This experiment explicitly models competing latent forces, includes dynamic penetration confidence, and targets forecast trust rather than raw Tmax.

### Response variables

Official overforecast/underforecast, marine-suppression tail, hot breakout, action uplift, and forecast trust.

### Exact inputs

Inland/northern/urban/coastal group anomalies, sea-breeze phase, onshore fetch, dewpoint/moisture front, wind field, target envelope, official forecast.

### Feature constructions and calculations

Inland heat-potential score; marine suppression score; difference/ratio; dominance confidence; boundary distance/motion; forecast position relative to both implied levels; target-memory coherence; and source uncertainty.

### Procedure

Estimate each force independently using safe features and no residual labels where possible; then fit a small duel-to-residual curve. Compare group spread, separate scores, and duel. Actions: shrink high forecast under marine dominance, retain/raise under inland dominance, abstain near balance.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Station-group and coast-geometry ablations; JJA/MAM/SON stability; no blocked marine input.

**Minimum sample rules:** 200 rows per broad dominance state, 100 official rows per action, 50 severe events.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0175_marine_vs_inland_duel/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`duel_scores.parquet`, `dominance_response.csv`, `group_ablation.csv`, `action_uplift.csv`, `specialist_predictions.parquet`, and `tail_safety.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Duel must add information beyond coastal-inland spread and improve trust/actions in multiple seasons/years without suppressing real hot extremes.

### Expected failure modes and interpretation

Station roles/coverage or synoptic wind can blur dominance. Use confidence and abstention; do not force a winner.

### Expected information gain

High physical and modelling promise; one of the strongest multi-station specialist concepts.

---

## 0176 — TC Quadrant Subsidence–Cloud Transition Specialist

**Priority:** P2  
**Research mode:** Exploratory tail specialist  
**Eligibility:** Safe student only; best track diagnostic  
**Dependencies:** 0159, 0173, 0174

### Decision question and hypothesis

The largest TC-related errors occur during transitions between subsidence/offshore heating and cloud-shield/rain suppression. A safe student that detects transition direction may reduce extreme misses.

### Why this is new rather than a relabelled prior experiment

0159 establishes broad TC student states; this experiment focuses on transitions and signed specialist actions rather than static TC proximity.

### Response variables

Hot underforecast, cool/cloud overforecast, >2/3°C errors, and action uplift.

### Exact inputs

Safe TC student probabilities, pressure/wind trajectory, radiative proxy, moisture front, station heat state, official text/revisions/range.

### Feature constructions and calculations

State probability change; pressure fall/rise acceleration; wind rotation; offshore-to-onshore fetch switch; radiative proxy surge/clearance; moisture-front arrival; forecast revision response; and transition confidence.

### Procedure

Build event blocks from diagnostic best track for research, but generate deployable transitions from safe student only. Evaluate signed tail enrichment and ±0.25/0.40 corrections with strict confidence and episode-level validation.

### Walk-forward validation and minimum evidence

Episode-level leave-event-out; no best-track inputs at inference; report average MAE and rare-tail effects separately.

**Minimum sample rules:** At least 30 diagnostic episodes and 50 high-confidence transition days; promotion requires multiple independent episodes.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0176_tc_quadrant_transition_specialist/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`tc_transition_events.csv`, `safe_transition_oof.parquet`, `episode_scores.csv`, `tail_uplift.csv`, `specialist_predictions.parquet`, and `eligibility_firewall.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Student transition must reduce severe signed errors across multiple episodes without material non-event harm. Otherwise retain as alert/uncertainty state.

### Expected failure modes and interpretation

Very low event count and source-era imbalance. Do not chase average-MAE decimals or tune by individual famous storms.

### Expected information gain

High tail relevance, low global activation; valuable for risk control more than average score.

---

## 0177 — Extreme-Error Precursor and Catastrophic-Miss Prevention Model

**Priority:** P1  
**Research mode:** Tail-risk promotion  
**Eligibility:** Deployable now with safe features  
**Dependencies:** 0113, 0121, 0167, 0171–0176

### Decision question and hypothesis

Severe forecast errors share precursors such as station disagreement, phase transitions, forecast contradictions, pressure/moisture fronts, unusual target-envelope departures, and source residual instability. Predicting high-error probability can gate specialist strength or force abstention.

### Why this is new rather than a relabelled prior experiment

Prior error autopsies and EXP-0098/0099 uncertainty/tail models were broad. This experiment uses the narrowed causal states, event clustering, class-imbalance discipline, and an explicit operational action: reduce harmful corrections and flag tail days.

### Response variables

Absolute error >1.5, >2, >3°C; top-decile error; signed hot/cold catastrophic misses; and correction harmfulness.

### Exact inputs

Safe representative feature set, uncertainty scale, online memory, network disagreement, transition states, text/numeric contradiction, quality/staleness, target envelope, specialist confidences.

### Feature constructions and calculations

No more than 20 predeclared features selected by response-specific stability, not unrestricted ML. Include interactions only for known mechanisms. Output calibrated probabilities by severity/sign and a reason-code decomposition.

### Procedure

Compare logistic/GAM, monotone boosted shallow trees, and a simple additive score under nested temporal folds. Use event-grouped sampling and class weights fitted in training only. Evaluate alert precision at fixed alert budgets and use probability to scale/abstain corrections.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Primary metrics: Brier, log loss, PR-AUC, calibration, recall at 5/10/20% alerts, P95 and >2/3°C counts after gating.

**Minimum sample rules:** At least 100 events for >2°C model; >3°C may remain exploratory if fewer. At least 20 events per temporal fold for formal stability.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0177_extreme_error_precursor/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`extreme_event_catalog.csv`, `precursor_oof.parquet`, `calibration.csv`, `alert_budget_scores.csv`, `reason_codes.csv`, `gated_system_scores.csv`, and `model_card.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Model must be calibrated, beat simple residual-MAD/network-disagreement baselines, and reduce severe-error counts or harmful specialist activations without average-MAE damage.

### Expected failure modes and interpretation

Rare-event overfitting and threshold instability. Keep models simple, calibrate fold-locally, and report confidence intervals. A good alert model need not directly change point forecast.

### Expected information gain

Very high risk-management value and plausible indirect MAE/tail gains.

---

## 0178 — Signed Hot-Underforecast and Cold-Overforecast Action Specialists

**Priority:** P1  
**Research mode:** Promotion-oriented tail specialists  
**Eligibility:** Deployable now  
**Dependencies:** 0113, 0117, 0171–0177

### Decision question and hypothesis

Hot underforecasts and cold/cloud overforecasts arise from different physics and should not share one residual model. Separate action policies can target each tail with asymmetric features, caps, and no-harm constraints.

### Why this is new rather than a relabelled prior experiment

0095–0099 split MAM direction, but this experiment generalizes signed tails across seasons using the newly built physical states and action-specific uplift rather than fixed cells.

### Response variables

Positive residual tail, negative residual tail, hot-day underforecast, cold-day overforecast, action uplift, and opposite-tail harm.

### Exact inputs

Heat-buildup, marine/radiative suppression, surge/MAM/TC states; target envelope; station/network and forecast trust; online memory.

### Feature constructions and calculations

Hot policy: dry heat, weak ventilation, inland dominance, warm-air proxy, official below envelope, positive bias memory. Cold/overforecast policy: radiative suppression, marine dominance, moisture surge, surge onset, official above envelope, negative bias memory. Each has calibrated confidence/support.

### Procedure

Fit two separate conservative action policies on ±0.15/0.25/0.40 grid. Optimize lower confidence bound of uplift and penalize opposite-tail creation. Compare to symmetric residual model and MAM-only direction split. Combine only by mutual exclusion/abstention.

### Walk-forward validation and minimum evidence

Use expanding or rolling walk-forward evaluation with thresholds, normalizers, station normals, encoders, feature selectors, and model parameters fitted inside each training fold. Use at least four years of OOF evidence wherever the source permits; for long-history work use multiple five-year test blocks across early, middle, and modern eras. Report MAE, RMSE, bias, median absolute error, P80/P90/P95 absolute error, errors above 2°C and 3°C, hot-underforecast rate, cold-overforecast rate, and correction activation count. Report all metrics globally and by year, season, month, forecast source, source era, residual sign, and error-severity bucket. Report each tail, opposite tail, season/source and global metrics; nested policy selection.

**Minimum sample rules:** 100 directional events and 60 actions per policy; at least three temporal folds with activations.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0178_signed_tail_action_specialists/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`signed_policy_features.parquet`, `hot_policy_oof.parquet`, `cold_policy_oof.parquet`, `symmetric_ablation.csv`, `opposite_tail_penalty.csv`, `combined_predictions.parquet`, and `scoreboard.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Each policy must improve its tail, not worsen the opposite tail, and combined system must preserve/improve global MAE and P90/P95. Sparse one-era success is rejected.

### Expected failure modes and interpretation

Directional labels can overlap with season/temperature level. Use past-only hot/cold definitions and within-season validation.

### Expected information gain

High tail-reduction potential and a disciplined final specialist layer.

---

## 0179 — High-Frequency Morning-Heating Curve Shape Atlas

**Priority:** P1 later-track  
**Research mode:** Short-history diagnostic / prospective layer  
**Eligibility:** Pre-2024 development only; 2024+ sealed  
**Dependencies:** 0107, high-frequency path manifest, 0108

### Decision question and hypothesis

Minute-scale warming rate, acceleration, plateaus, and cross-station coherence before cutoff can reveal cloud breaks, marine penetration, and heating potential that daily summaries miss.

### Why this is new rather than a relabelled prior experiment

EXP-0070 used raw ISD intraday trajectory on a long-history setup. This experiment exploits the much denser 1-minute HKO archive, explicitly treats days—not minutes—as samples, and builds shape features for a future live layer.

### Response variables

Same-day pre-cutoff heating state as diagnostic, next-day T+24 Tmax/official residual where timing aligns, high-error probability, and radiative/sea-breeze proxy states.

### Exact inputs

1-minute temperature archive (39 stations) through 2023 only for development; humidity/wind/pressure where overlapping; target/official responses and station dossier.

### Feature constructions and calculations

Temperature at fixed local times; slopes over 15/30/60/120 minutes; acceleration; first/second warming onset; plateau duration; cloud-break jump; station-curve PCA/shapelets fitted fold-locally; cross-station lead/lag; urban/coastal/inland heating ratios; and data-quality/coverage.

### Procedure

Resolve operational timing first. Aggregate minute data into one feature row per target date and cutoff; never treat minute rows as independent. Run leave-one-year-out and event-block analysis on 2020-2023. Compare simple slopes to complex shapes and distill robust daily proxies.

### Walk-forward validation and minimum evidence

F-HF-DEV only, 2024+ untouched. Days/events are units. All curve transforms fit on training years. Report exact cutoff variants.

**Minimum sample rules:** At least 300 independent days, 100 per major season where possible, and 50 high-error days for exploratory tail claims.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0179_hf_morning_heating_curves/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`daily_heating_shapes.parquet`, `curve_feature_catalog.csv`, `leave_year_out_scores.csv`, `station_group_curves.csv`, `timing_eligibility.md`, `proxy_distillation.csv`, and `case_plots/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A curve feature must transfer across at least two held-out years and beat fixed-time/slope baselines. No long-history production promotion from only 3.5 years.

### Expected failure modes and interpretation

Short history, outages, and target/cutoff alignment may dominate. Label findings prospective and use as teacher for 0183.

### Expected information gain

Very high live-system information potential, but intentionally not a 39-year production claim.

---

## 0180 — High-Frequency Spatial Propagation, Convergence, and Boundary Timing

**Priority:** P1 later-track  
**Research mode:** Short-history diagnostic / prospective layer  
**Eligibility:** Pre-2024 development only  
**Dependencies:** 0179, high-frequency wind/pressure/humidity feeds

### Decision question and hypothesis

Minute/10-minute observations can identify the actual timing and speed of sea-breeze fronts, pressure waves, moisture surges, and station heating disruptions, validating the coarser long-history proxies.

### Why this is new rather than a relabelled prior experiment

Long-history propagation experiments rely on sparse cadence. This uses dense recent data as a mechanism microscope and teacher, not as evidence of long-history robustness.

### Response variables

Boundary arrival at HKO/nearby stations, next-day Tmax residual, same-day heating disruption, and long-history proxy alignment.

### Exact inputs

1-minute temperature/humidity/pressure, 10-minute wind through 2023; station geometry; long-history front/propagation proxy outputs.

### Feature constructions and calculations

Change-point time per station; wavefront direction/speed; cross-correlation lag; convergence onset; wind shift; pressure jump; dewpoint surge; curve synchronization; boundary confidence; and mapped coarse-proxy score.

### Procedure

Detect events using predeclared robust thresholds, group related station changes into fronts, estimate propagation, and compare with daily/T−1 proxy features from 0129–0136. Use leave-event-out validation and produce teacher labels for distillation.

### Walk-forward validation and minimum evidence

Events/days, not observations, are samples. Leave one year and one event cluster out. 2024+ remains sealed.

**Minimum sample rules:** At least 100 coherent events overall and 20 per major event type for descriptive conclusions.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0180_hf_spatial_propagation/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`hf_boundary_events.parquet`, `propagation_tracks.parquet`, `event_quality.csv`, `coarse_proxy_alignment.csv`, `leave_event_out.csv`, and `event_visualizations/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Dense-data event labels must be reproducible and at least one long-history safe proxy must show meaningful alignment. Otherwise use results to redesign proxies, not promote minute features historically.

### Expected failure modes and interpretation

Station clock/cadence differences and missingness can create false lags. Apply 0107 timing audit and uncertainty.

### Expected information gain

High mechanism validation value and important bridge between long-history proxies and live deployment.

---

## 0181 — Since-Midnight Tmax Trajectory, Ceiling, and Remaining-Upside Model

**Priority:** P1 later-track  
**Research mode:** Short-history live/prospective research  
**Eligibility:** Pre-2024 development only; operational timing must match market/forecast cutoff  
**Dependencies:** 0179, since-midnight archive

### Decision question and hypothesis

Running maximum and current temperature trajectory reveal whether the daily ceiling is already approached, still rising, or likely capped. While this may be more relevant to same-day markets, it can also validate thermal-ceiling states and next-day memory.

### Why this is new rather than a relabelled prior experiment

No prior experiment systematically uses the 16.88M-row running max/min feed, its timing, and cross-station ceiling progression with strict market-horizon separation.

### Response variables

Same-day final Tmax minus running max at cutoff; next-day Tmax/official residual only where operationally legitimate; ceiling-hit time and remaining upside.

### Exact inputs

Since-midnight max/min and 1-minute temperature through 2023, solar/UV/wind/humidity, station groups, target labels.

### Feature constructions and calculations

Running max level; time since new max; increments over 15/30/60/120 min; current-minus-running-max; cross-station fraction making new highs; urban/inland/coastal ceiling divergence; solar/wind/moisture state; and probabilistic remaining upside.

### Procedure

First write a horizon contract separating T+24 from same-day use. For any legitimate T+24 feature, only use prior-day trajectory. For same-day diagnostic, model final remaining upside with leave-year-out validation. Distill prior-day curve states for next-day use.

### Walk-forward validation and minimum evidence

Strict target-date/horizon audit; one row per day; leave-year-out; 2024+ sealed.

**Minimum sample rules:** 300 independent days and 50 extreme remaining-upside events.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0181_hf_since_midnight_ceiling/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`horizon_contract.md`, `daily_ceiling_features.parquet`, `remaining_upside_oof.parquet`, `cross_station_ceiling.csv`, `next_day_distillation.csv`, and `leakage_audit.md`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

No T+24 result may use target-day trajectory. Same-day findings are clearly separated. A next-day distilled feature must beat simple prior Tmax memory across held-out years.

### Expected failure modes and interpretation

High leakage risk and horizon mismatch. Any ambiguity defaults to diagnostic/same-day only.

### Expected information gain

High live-market value in the correct horizon; moderate mechanism value for T+24, with strict separation required.

---

## 0182 — UV–Solar–Humidity–Wind Cloud-Suppression and Cloud-Break Proxy

**Priority:** P1 later-track  
**Research mode:** Short-history diagnostic / prospective layer  
**Eligibility:** Pre-2024 development only  
**Dependencies:** 0149, 0150, 0179

### Decision question and hypothesis

UV/solar deviations relative to clear-sky seasonal expectation, combined with humidity and wind, can identify cloud suppression and sudden cloud breaks in real time and validate long-history safe radiative proxies.

### Why this is new rather than a relabelled prior experiment

Daily cloud/sunshine models and broad solar efficiency failed. This experiment uses high-frequency radiation timing, clear-sky normalization, and multi-station thermal response to construct a prospective proxy and teacher for distillation.

### Response variables

Radiative suppression/cloud-break event, heating-rate change, final/next-day Tmax residual where eligible, and long-history proxy alignment.

### Exact inputs

15-minute UV, 1-minute solar (2 sites), temperature/humidity, 10-minute wind, target/official responses through 2023.

### Feature constructions and calculations

Clear-sky-normalized UV/solar; rolling deficit; rapid recovery; cloud intermittency; station heating response lag; humidity/wind state; daylight geometry; missingness/sensor confidence; and event labels.

### Procedure

Build deterministic solar-geometry baseline, derive suppression/break events, and validate against station warming curves. Train a safe low-frequency proxy using station moisture/pressure/wind/text. Use leave-year-out/event validation.

### Walk-forward validation and minimum evidence

One day/event is sample unit; daylight and sensor QC explicit; 2024+ sealed.

**Minimum sample rules:** 300 valid days, 100 suppression events, 50 cloud-break events.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0182_hf_radiative_proxy/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`radiative_events.parquet`, `clear_sky_baseline.csv`, `sensor_qc.csv`, `heating_response.csv`, `safe_proxy_distillation.csv`, and `leave_year_out_scores.csv`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

Event detection transfers across years and improves heating/radiative state classification over humidity alone; long-history proxy must be evaluated separately before promotion.

### Expected failure modes and interpretation

Only two solar sites and short history limit generality. Use as teacher/prospective feature, not long-history proof.

### Expected information gain

High live-layer value and strong validation for radiative-suppression mechanism.

---

## 0183 — Short-History Teacher to Long-History Deployable Feature Distillation

**Priority:** P1 synthesis  
**Research mode:** Diagnostic-to-deployable bridge  
**Eligibility:** Students deployable only if built from long-history safe inputs  
**Dependencies:** 0179–0182 plus long-history proxy experiments

### Decision question and hypothesis

Dense recent data can reveal hidden physical states—heating curve, boundary arrival, sea-breeze penetration, cloud suppression, nocturnal retention—that are absent historically. Safe long-history station/target features can be trained to approximate those states and then tested over decades.

### Why this is new rather than a relabelled prior experiment

This is the explicit bridge preventing short-history feeds from becoming overfit production features. It uses recent data only to define teachers, then requires long-history students and independent temporal evidence.

### Response variables

Teacher states from 0179–0182; long-history target anomaly, official residual, absolute error, and tails for student evaluation.

### Exact inputs

Pre-2024 high-frequency teacher labels; overlapping safe daily station/target/forecast features; F-LONG/F-O5265 for long-history student deployment.

### Feature constructions and calculations

For each teacher, a minimal safe student using daily station anomalies/slopes, pressure/dewpoint/wind fields, target memory, geospatial groups, and forecast text/range. Record teacher confidence, student probability, out-of-domain score, and feature availability.

### Procedure

Train students only on 2020–2023 overlap with leave-year-out and severe regularization. Freeze student formulas, then apply them to long history and test whether their response relations remain stable by era. Compare teacher-driven feature to its raw safe components and to simpler rule-based proxy.

### Walk-forward validation and minimum evidence

Two-stage evaluation: teacher reconstruction on held-out recent years, then long-history response stability without retraining on future outcomes. 2024+ sealed throughout.

**Minimum sample rules:** At least 300 overlap days per teacher, 50 positive events, and 1,000 long-history student rows. Low-support teachers remain prospective only.

### Leakage and point-in-time contract

Hard leakage rules: reject `target_date >= 2024-01-01`; target date T may use only information proven available by the operational cutoff; never use target-day finalized observations; construct every lag/rolling feature from T−1 or earlier; fit bins and transformations on prior rows only; update online residual state only after scoring T; keep IGRA, finalized HKO daily climate, marine daily values, and retrospective TC best track diagnostic-only until availability proof passes; never select a later forecast revision because it happens to verify better.

### First concrete implementation

Create `experiments/0183_hf_teacher_long_history_distillation/run.py` with a YAML config and deterministic CLI entry point.

### Required artifacts

`teacher_registry.csv`, `student_models/`, `student_oof_recent.parquet`, `long_history_student_scores.parquet`, `component_ablation.csv`, `domain_shift.csv`, `promotion_decisions.csv`, and `model_cards/`.

In addition, apply the repository-wide artifact contract: Every experiment folder must be self-contained and contain, at minimum: `README.md`, `summary.json`, `data_range.csv`, `input_hashes.json`, `feature_definitions.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `scoreboard.csv`, `fold_metrics.csv`, `year_stability.csv`, `season_stability.csv`, `source_stability.csv`, `high_error_tail.csv`, `negative_results.md`, and `next_recommendation.md`. Model-producing experiments must additionally write row-level OOF `predictions.parquet`; atlas experiments must write the complete scored atlas rather than only a top-N extract. All tables must include row counts, first/last target dates, and explicit frame IDs.

### Acceptance criteria

A student promotes only if it predicts teacher state out of year, adds long-history response information beyond components, passes eligibility, and improves a bounded official-anchor or uncertainty use case. Otherwise retain as live-only feature.

### Expected failure modes and interpretation

2020–2023 may not span enough regimes; student can learn era-specific artifacts. Domain-shift diagnostics and simple rule comparisons are mandatory.

### Expected information gain

Very high strategic value: it extracts maximum insight from millions of recent observations without sacrificing long-history robustness or leakage discipline.

---

# Part III — Program control, sequencing, and audit

## 12. Most likely next lift, ranked without overpromising

The ranges below are **research targets, not promised gains**. They indicate plausible order of magnitude relative to the current 0103 expanded-frame candidate if the hypothesis survives strict OOF testing.

| Rank | Route | Why it is credible now | Plausible outcome | Primary risk |
|---:|---|---|---|---|
| 1 | 0161 exact 0075/0081 replay on F-O5265 | Largest proven prior step; no need for new data | decisive robustness answer; perhaps 0.005–0.03°C MAE if transferable | narrow-frame composition may explain most gain |
| 2 | 0162 hierarchical source/era/season memory | Extends the strongest mechanism with partial pooling | small stable incremental lift, roughly 0.002–0.01°C | sparse contexts and gap resets |
| 3 | 0123 target-memory × station coherence | Directly tests when persistence should fail | orthogonal residual signal and bounded lift | station era/coverage confounding |
| 4 | 0171 MAM latent phase | MAM is current largest seasonal error and prior cells already helped | meaningful MAM/tail reduction; small global lift | phase states may collapse to calendar |
| 5 | 0117 correction uplift/no-harm atlas | Optimizes action value instead of residual fit | fewer harmful activations and safer small gains | flexible policy overfit |
| 6 | 0147 dewpoint tendency × advection | Dewpoint change repeatedly appears in station evidence | residual/tail information in MAM/JJA | surface wind may not track moisture transport |
| 7 | 0131/0136 pressure-wave and surge phase | Converts weak pressure level into transition timing | DJF/MAM event skill and tail reduction | cadence/pressure-station coverage |
| 8 | 0175 marine-vs-inland duel | Combines first-class station roles and opposing physics | overforecast reduction without suppressing hot extremes | group definitions and sea-breeze ambiguity |
| 9 | 0177 extreme-error precursor | Explicitly targets severe misses and unsafe corrections | calibrated high-error alerts and P95 improvement | rare-event power |
| 10 | 0170 abstaining router | Integrates only qualified experts | small cumulative lift if expert signals are orthogonal | meta-overfit/dilution |
| 11 | 0153/0158 blocked-signal proxy conversion | Top diagnostic signals identify real missing mechanisms | safe proxy or clear evidence that operational NWP/live source is needed | teacher does not transfer to surface proxy |
| 12 | 0183 high-frequency teacher distillation | Millions of recent rows can reveal hidden states | future live-layer value and better long-history proxies | only 2020–2023 unsealed development history |

## 13. Parallel execution tracks

### Track A — Safe long-history signal discovery now

Run 0104–0118, 0119–0147, and the safe portions of 0171–0178. This track is not blocked by forecast archive continuity. Its objective is to build response-specific station/target information maps and a compact deployable feature whitelist.

### Track B — Official-anchor replay and correction now

Run 0161–0170 on the current canonical F-O5265 frame. Every script must be able to replay unchanged when backfill adds rows. Do not tune a separate rule after backfill; append rows and rerun the same protocol.

### Track C — Diagnostic physics and proxy conversion

Run 0148–0160 with strict teacher/student separation. Upper-air, finalized daily climate, marine values, and best track may explain mechanisms but never appear in deployable OOF predictions unless timestamp proof is attached.

### Track D — Short-history prospective/live layer

Run 0179–0183 only on pre-2024 development rows. Minute records are not millions of independent samples; the statistical unit is the target day or meteorological event. Treat outcomes as prospective architecture evidence, not final 39-year proof.

### Track E — Final integration and confirmation later

After robust gains survive F-O5265, source/season/year ablation, high-error tails, and backfill replay, freeze the system. Open 2024+ only when explicitly authorized and only once for confirmation. Market/Polymarket backtesting follows weather-system confirmation, not before.

## 14. Recommended execution waves

| Wave | Experiments | Objective | Exit condition |
|---|---|---|---|
| 0 | 0104–0110 | Frames, station identity, lineage, QC, responses, statistical/replay harness | all benchmarks reproduce and no champion feature is unclassified |
| 1 | 0111–0118 | Conditional IG, nonlinear curves, tails, redundancy, interactions, drift, uplift, missingness | compact stable response-specific queue with FDR/stability evidence |
| 2 | 0119–0138 | Target dynamics and deep spatial sensor-array mining | physically annotated target/station states and group core |
| 3 | 0139–0152 | Geospatial context and moisture/energy/radiative mechanisms | safe physical feature store and qualified proxy candidates |
| 4 | 0153–0160 | Blocked physics proxy conversion and timestamp proof | promote safe students or document irreducible source gaps |
| 5 | 0161–0170 | Official-anchor memory, revisions, text, uncertainty, quality, routing | one expanded-frame candidate beating 0103 with no-harm proof |
| 6 | 0171–0178 | MAM, surge, cloud-break, heat, marine, TC and signed-tail specialists | only stable specialists survive; others archived as negative evidence |
| 7 | 0179–0183 | High-frequency mechanism microscope and distillation | live/prospective features separated from long-history promotion |

## 15. Data cleanup and blockers

| Blocker | What it blocks | What can continue now | Exact unlocking artifact |
|---|---|---|---|
| Official archive gap 2011-09-14–2021-04-13 | continuous source-memory/revision/text validation and era overlap | F-O5265 replay, station/target atlases, backfill-ready scripts | parsed exact-vintage forecast rows with issue/available-at, raw detail hash, parser status, and canonical row key |
| Zero timestamp proof for 207 upper-air features | deployable use of IGRA values | diagnostic physics, safe proxy students | provider issue/availability field or documented conservative release-latency proof covering all promoted rows |
| Zero timestamp proof for 39 daily-climate and 12 marine features | deployable cloud/rain/radiation/sea-temp features | diagnostic teachers and safe station proxies | first-publication timestamp or immutable first-seen archive with valid conservative cutoff rule |
| Station `450090-99999` coordinate anomaly and other metadata conflicts | trustworthy spatial gradients/graphs | nonspatial station anomaly work excluding suspect stations | verified source metadata, correction provenance, and sensitivity rerun |
| ISD retrospective quality-control/vintage ambiguity | strongest production claim for historical station inputs | conservative T−1 pre-15:00 research and prospective validation | source latency/vintage contract or explicit policy accepting archive proxy status |
| Static geospatial inventory not converted | coast/terrain/urban physics | ID/coordinate-based features | deterministic station context store with source hashes/CRS audit |
| High-frequency archives not in clean normalized folders | reproducible recent/live experiments | long-history work | resolved-path manifest, normalized per-feed schema, eligibility/cutoff metadata, daily/event feature store |
| Parsed forecast null min/max/text fields | trustworthy revision/text/quality work | rows with valid exact fields and explicit null states | raw-to-parsed audit, null policy, parser confidence and era template tests |
| NCEP is inventory only | issue-time-safe gridded/NWP features | all current station/target/forecast work | named variable/cycle/lead/domain/level extraction, byte/legal policy, issue/valid-time contract |
| Missing row-level predictions for some old experiments | exact legacy comparison | metric-level evidence synthesis | deterministic replay or recovered OOF prediction file with row keys |

## 16. Negative results preservation: do not repeat these mistakes

| Preserved negative evidence | Interpretation | Better next move |
|---|---|---|
| Broad dynamic climatology/change-point/recency models underperformed core | long-term drift is not free lift on modern common rows | use drift to set feature memory/era shrinkage, not replace anchor |
| DTW and multivariate climate trajectory analogs were among worst candidates | broad distance mixes meteorologically incompatible days | phase-restricted, minimal-state residual analogs; keep only if they beat memory |
| Spell-duration hazard was poor | duration alone cannot locate synoptic termination | require pressure/wind/dewpoint/station propagation confirmation |
| Upper-air heat/inversion/moisture/shear direct models failed and are timestamp-blocked | strong explanatory physics is not deployable point skill | teacher→safe student conversion; seek NWP only for irreducible gap |
| Pressure-only experts and hard pressure cells were weak | level/threshold misses transition timing and moisture/wind interaction | pressure-wave arrival plus multi-field confirmation |
| Broad upwind/gradient/graph/hotspot models underperformed | geometry alone or unrestricted modes dilute source signal | station dossier, physical groups, dynamic fronts, simple modes, ablations |
| Station-only systems plateaued around materially worse MAE than official anchor | station network cannot replace human/NWP forecast | use station state for residual correction, trust, uncertainty and tails |
| Sparse hard interaction cells gave tiny/unstable gains | cell winners are vulnerable to support and frame changes | smooth curves, hierarchical shrinkage, FDR, uplift lower bounds |
| Generic expert stacks diluted gains | adding correlated weak experts hurts | at most six qualified experts, nested OOF, abstention, no-regret router |
| Broad Student-t/tail-distribution candidates failed | distributional flexibility does not create signal | calibrate simple source residual scale and high-error precursor features |
| TC/cloud teacher students were mechanism-only | teacher signal did not transfer sufficiently in broad form | split opposing physical states and target signed tails |
| Source/era scores changed materially when frame expanded | narrow-frame leaderboard can mislead | F-O5265 is canonical; frozen replay before tuning |

## 17. Most dangerous leakage traps

1. Using target-day daily climate, rain, cloud, sea temperature, station extrema, or full-day summaries for target T.
2. Treating meteorological valid time as proof that the operational system had the row before cutoff.
3. Using retrospective IGRA release timing, quality-controlled ISD revisions, or finalized TC best track as exact live vintage.
4. Computing a rolling feature whose window accidentally includes target T or whose shift occurs after aggregation.
5. Choosing the best forecast revision after cutoff or selecting a revision based on verification quality.
6. Confusing archive retrieval timestamp in 2026 with the forecast’s historical issue/availability timestamp.
7. Building a station-day summary from observations after T−1 15:00 HKT or mishandling historical timezone offsets.
8. Computing climatology, standardization, PCA, graph modes, station normals, bins, text vocabulary, or imputation on full history.
9. Joining predictions by target date alone when multiple source/vintage rows exist.
10. Updating online residual state with the current target outcome before emitting the current prediction.
11. Treating missing forecast fields or station reports as zero and letting the model exploit parser/source era.
12. Training high-frequency models on millions of minute rows as though each minute were an independent sample.
13. Using 2024–2026 rows to choose features, thresholds, hyperparameters, or teacher definitions.
14. Reporting the 0.9438 narrow-frame result beside the 1.0491 expanded-frame result without frame warnings.
15. Distilling a blocked teacher but accidentally retaining teacher-derived values or full-history normalization in the student inference graph.

## 18. Do not do yet

- Do not run Polymarket pricing/backtesting or optimize market decisions before the weather signal and calibration system is frozen.
- Do not open final 2024+ confirmation rows.
- Do not train unrestricted XGBoost/LightGBM/neural models on all 566 columns.
- Do not use timestamp-blocked upper-air, HKO daily-climate, marine daily, or retrospective best-track rows as deployable predictors.
- Do not use target-day observations, finalized target-day extrema, or post-cutoff station summaries.
- Do not download huge gridded/NWP archives without a named unresolved mechanism, exact variables, cycle/lead/levels/domain, issue-time eligibility, byte budget, and legal policy.
- Do not keep creating hard 3×3 cells or tiny source/submonth buckets without smooth shrinkage, support thresholds, and multiple-testing control.
- Do not promote a feature because it predicts raw Tmax if it adds nothing to the official residual or routing problem.
- Do not claim 0.45°C is reachable from current evidence. Treat it as an external aspiration and demand auditable incremental progress.

## 19. ML modelling architecture after the analysis phase

Only after the promotion ladder is satisfied should the project move to flexible modelling. The recommended architecture is deliberately modular:

1. **Anchor:** exact pre-cutoff official maximum forecast when available; otherwise a separately validated weather-only fallback.
2. **Online bias state:** hierarchical source/era/season residual memory with conservative caps.
3. **Small interpretable state encoders:** target level/slope/volatility/phase; station thermal/moisture/pressure/wind state; forecast quality/text/range state.
4. **At most a few specialists:** MAM transition, cool-surge breakdown, humid suppression/cloud break, weak-wind heat, marine-vs-inland, signed tail.
5. **Uncertainty:** calibrated source residual scale plus station disagreement/transition risk.
6. **Router:** nested-OOF uplift/no-regret abstaining policy. Raw official is the default action.
7. **Distribution and market layer later:** translate the frozen, calibrated weather distribution into market probabilities only after confirmation.

A future flexible model must have a feature whitelist, exact timestamp contract, rolling refit cadence, fold-local preprocessing, fixed hyperparameter budget, feature-block ablations, calibration, uncertainty, source fallback, and direct comparison with simple online memory. A black box that cannot beat 0162/0170 across canonical frames is rejected.

## 20. Mental red-team of this roadmap

- **Is any route merely a renamed prior experiment?** Each rescue route changes the response or decision: pressure level becomes wave arrival; spell duration becomes spatially confirmed termination; DTW raw-target analogs become phase-restricted residual analogs; graph modes become graph-construction ablation; blocked upper air becomes teacher/student conversion; broad moisture becomes opposing suppression/cloud-break states.
- **Are expected gains plausible?** The roadmap expects most single experiments to improve understanding, not MAE. Promotion targets are millidegree-to-hundredth-degree scale and require no-harm evidence. The largest immediate question is whether 0075 transfers to F-O5265.
- **Does any deployable route depend on a blocked source?** No. Blocked data appears only in F-DIAG teacher/mechanism stages and is separated by an eligibility firewall.
- **Is history sufficient?** Long-history station/target atlases use multiple eras. Official specialists acknowledge 5,265 non-contiguous rows and 992 RSS rows. High-frequency work is explicitly recent/prospective and counts days/events, not minutes.
- **Will work remain useful after archive backfill?** Every official experiment reads canonical frame manifests and is required to replay unchanged when new rows arrive.
- **Will Codex know what to build?** Each experiment names its folder, inputs, features, validation, minimum support, artifacts, acceptance, and failure interpretation.
- **Could the experiment queue itself overfit by breadth?** Yes. Therefore waves 0–1 must narrow the queue through FDR, temporal replication, redundancy, and conditional information before models are fit.

## 21. Minimum viable next Codex command

> **Create `experiments/0104_canonical_frame_evidence_registry`. Read every folder under `experiments`, especially `0000_research_state_and_data_contract`, `0075_online_residual_memory_refinement`, `0081_rss_gate_stability_stress`, `0101_stable_mam_cell_feature_specialists`, `0102_timestamp_proof_unlock_queue`, `0103_current_rss_safe_continuation`, and `EXP-0050` through `EXP-0099`. Rebuild from repository source files the F-O2670, F-O5265, F-RSS992, F-LONG, and F-DIAG row sets with immutable row keys and hashes. Reproduce all quoted benchmark metrics, write `canonical_scoreboard.csv`, `frame_registry.csv`, `frame_membership.parquet`, `metric_reproduction.csv`, `row_exclusion_reasons.csv`, `gap_calendar.csv`, `source_era_map.csv`, `feature_eligibility.csv`, `leakage_audit.md`, `summary.json`, and `README.md`. Reject all target dates on or after 2024-01-01. Do not use upper-air, HKO daily-climate, marine, or best-track values in deployable scores. Acceptance requires exact frame row counts, exact date endpoints, the 3,500-day major gap, zero unexplained row drops/duplicates, and score reconciliation within numeric tolerance. Stop and document discrepancies instead of starting a model if any frame cannot be reproduced.**

# Part IV — Repository evidence appendices

## Appendix A. Complete top-level experiment-folder inventory

The table below is included to demonstrate the review scope and prevent accidental duplication. Empty alias/control folders are preserved rather than silently omitted.

| folder | title | files | md | csv | json | yaml | readme_chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0000_research_state_and_data_contract | Research State And Data Contract | 1 | 1 | 0 | 0 | 0 | 6777 |
| 0001_official_forecast_pre_cutoff_bias | Official Forecast Pre-Cutoff Bias | 2 | 2 | 0 | 0 | 0 | 6738 |
| 0002_feature_correlation_atlas | Feature Correlation Atlas | 1 | 1 | 0 | 0 | 0 | 8938 |
| 0003_station_network_information_gain | Station Network Information Gain | 1 | 1 | 0 | 0 | 0 | 19175 |
| 0004_slope_trend_regime_signals | Slope, Trend, And Regime Signals | 1 | 1 | 0 | 0 | 0 | 16250 |
| 0005_champion_error_autopsy | Champion Error Autopsy | 1 | 1 | 0 | 0 | 0 | 17393 |
| 0006_press_archive_offline_export | Press Archive Offline Export | 1 | 1 | 0 | 0 | 0 | 6973 |
| 0007_long_history_signal_stability | Long-History Signal Stability | 1 | 1 | 0 | 0 | 0 | 6342 |
| 0008_station_network_forensics | Station Network Forensics | 1 | 1 | 0 | 0 | 0 | 15101 |
| 0009_nonlinear_threshold_lifts | Nonlinear Threshold Lifts | 1 | 1 | 0 | 0 | 0 | 4838 |
| 0010_cross_feature_interaction_screens | Cross-Feature Interaction Screens | 1 | 1 | 0 | 0 | 0 | 6286 |
| 0011_past_only_residual_correction_screen | Past-Only Residual Correction Screen | 1 | 1 | 0 | 0 | 0 | 8623 |
| 0012_official_forecast_residual_anatomy | Official Forecast Residual Anatomy | 1 | 1 | 0 | 0 | 0 | 4030 |
| 0013_official_forecast_feature_conditioning | Official Forecast Feature Conditioning | 1 | 1 | 0 | 0 | 0 | 8508 |
| 0014_official_anchor_blend_sensitivity | Official Anchor Blend Sensitivity | 1 | 1 | 0 | 0 | 0 | 6957 |
| 0015_official_forecast_archive_coverage_gap | Official Forecast Archive Coverage Gap | 1 | 1 | 0 | 0 | 0 | 2280 |
| 0016_past_only_official_anchor_correction_ladder | Past-Only Official Anchor Correction Ladder | 1 | 1 | 0 | 0 | 0 | 12208 |
| 0017_past_only_official_residual_analog_screen | Past-Only Official Residual Analog Screen | 1 | 1 | 0 | 0 | 0 | 16451 |
| 0018_past_only_official_expert_blend_screen | Past-Only Official Expert Blend Screen | 1 | 1 | 0 | 0 | 0 | 7630 |
| 0019_multistation_attribute_information_gain_matrix |  | 0 | 0 | 0 | 0 | 0 | 0 |
| 0019_multistation_info_gain | Multistation Attribute Information-Gain Matrix | 1 | 1 | 0 | 0 | 0 | 34219 |
| 0020_regime_experts | Regime Expert Factory | 1 | 1 | 0 | 0 | 0 | 16209 |
| 0021_failure_modes | Official Forecast Failure-Mode Segmentation | 1 | 1 | 0 | 0 | 0 | 28775 |
| 0022_failure_specialists | Fold-Local Failure-Mode Specialists | 1 | 1 | 0 | 0 | 0 | 24225 |
| 0023_composite_expert_stack | Composite Official-Anchor Expert Stack | 1 | 1 | 0 | 0 | 0 | 11243 |
| 0024_press_archive_raw_detail_gap_audit |  | 0 | 0 | 0 | 0 | 0 | 0 |
| 0024_press_gap_audit | Press Archive Raw-Detail Gap Audit | 1 | 1 | 0 | 0 | 0 | 11473 |
| 0025_longhist_signal_atlas | Long-History Calendar-Adjusted Signal Atlas | 1 | 1 | 0 | 0 | 0 | 25470 |
| 0026_pressure_gradient_experts | Pressure-Gradient Official Residual Experts | 1 | 1 | 0 | 0 | 0 | 37665 |
| 0027_pressure_regime_interaction_atlas | Pressure-Regime Interaction Atlas | 1 | 1 | 0 | 0 | 0 | 19009 |
| 0028_smooth_gated_pressure_experts | Smooth Gated Pressure Experts | 1 | 1 | 0 | 0 | 0 | 18452 |
| 0029_official_residual_source_text_range_dynamics | Official Residual Source/Text/Range Dynamics | 1 | 1 | 0 | 0 | 0 | 40709 |
| 0030_multi_signal_local_residual_lab | Multi-Signal Local Residual Lab | 1 | 1 | 0 | 0 | 0 | 50982 |
| 0031_regime_gated_specialist_selector | Regime-Gated Specialist Selector | 1 | 1 | 0 | 0 | 0 | 17545 |
| 0032_residual_failure_cluster_discovery | Residual Failure Cluster Discovery | 1 | 1 | 0 | 0 | 0 | 30201 |
| 0033_smooth_residual_archetype_specialists | Smooth Residual Archetype Specialists | 1 | 1 | 0 | 0 | 0 | 34989 |
| 0034_cluster_centroid_soft_gating | Cluster-Centroid Soft Gating | 1 | 1 | 0 | 0 | 0 | 22719 |
| 0035_forecast_revision_momentum_deep_dive | Forecast Revision Momentum Deep Dive | 1 | 1 | 0 | 0 | 0 | 57302 |
| 0036_revision_centroid_stack_ablation | Revision-Centroid Stack Ablation | 1 | 1 | 0 | 0 | 0 | 27249 |
| 0037_stack_trust_meta_features | Stack Trust Meta-Features | 1 | 1 | 0 | 0 | 0 | 37841 |
| 0038_forecast_history_state_deepening | Forecast-History State Deepening | 1 | 1 | 0 | 0 | 0 | 57739 |
| 0039_station_network_forecast_residual_interaction_mining |  | 0 | 0 | 0 | 0 | 0 | 0 |
| 0039_station_network_residuals | Station-Network Forecast Residual Interaction Mining | 1 | 1 | 0 | 0 | 0 | 72336 |
| 0040_station_network_smooth_residuals | Station-Network Smooth Local Residual Models | 1 | 1 | 0 | 0 | 0 | 39251 |
| 0041_station_network_forecast_stack | Station-Network Forecast Stack | 1 | 1 | 0 | 0 | 0 | 54375 |
| 0042_trust_router_sensitivity | Trust Router Sensitivity | 1 | 1 | 0 | 0 | 0 | 126943 |
| 0043_router_simplification_or_archive_refresh_decision | Router Simplification Or Archive Refresh Decision | 1 | 1 | 0 | 0 | 0 | 14487 |
| 0044_forecast_archive_continuous_scored_export | Forecast Archive Continuous Scored Export Promotion | 1 | 1 | 0 | 0 | 0 | 5950 |
| 0045_missing_press_raw_detail_backfill_or_blocker | Missing Press Raw-Detail Backfill Or Blocker | 1 | 1 | 0 | 0 | 0 | 16599 |
| 0046_long_history_cross_family_interaction_atlas | Long-History Cross-Family Interaction Atlas | 1 | 1 | 0 | 0 | 0 | 18094 |
| 0047_station_contribution_atlas | Station Contribution Atlas | 1 | 1 | 0 | 0 | 0 | 20884 |
| 0048_gated_residual_specialist_screen | Gated Residual Specialist Screen | 1 | 1 | 0 | 0 | 0 | 13706 |
| 0049_router_gate_stack_screen | 0049 Router Gate Stack Screen | 1 | 1 | 0 | 0 | 0 | 99118 |
| 0050_station_lag_slope_information_atlas | Station Lag/Slope Information Atlas | 1 | 1 | 0 | 0 | 0 | 93287 |
| 0051_station_regime_interaction_atlas | Station Regime Interaction Atlas | 1 | 1 | 0 | 0 | 0 | 64780 |
| 0052_candidate_residual_feature_design_notes | Candidate Residual Feature Design Notes | 1 | 1 | 0 | 0 | 0 | 73451 |
| 0053_candidate_timestamp_eligibility_audit | Candidate Timestamp Eligibility Audit | 1 | 1 | 0 | 0 | 0 | 23371 |
| 0054_station_only_walkforward_matrix_audit | Station-Only Walk-Forward Matrix Audit | 1 | 1 | 0 | 0 | 0 | 88812 |
| 0055_station_only_walkforward_benchmark | Station-Only Walk-Forward Benchmark | 1 | 1 | 0 | 0 | 0 | 33737 |
| 0056_station_only_failure_mode_analysis | Station-Only Failure Mode Analysis | 1 | 1 | 0 | 0 | 0 | 52268 |
| 0057_station_only_residual_specialist_design_queue | Station-Only Residual Specialist Design Queue | 1 | 1 | 0 | 0 | 0 | 16183 |
| 0058_station_only_late_period_bias_repair | Station-Only Late-Period Bias Repair | 1 | 1 | 0 | 0 | 0 | 14588 |
| 0059_station_only_february_march_transition_specialist | Station-Only February/March Transition Specialist | 1 | 1 | 0 | 0 | 0 | 17083 |
| 0060_station_only_spring_transition_pressure_dew_specialist | Station-Only Spring Pressure/Dew Specialist | 1 | 1 | 0 | 0 | 0 | 16735 |
| 0061_station_only_pressure_high_uncertainty_guard | Station-Only Pressure-High Uncertainty Guard | 1 | 1 | 0 | 0 | 0 | 14197 |
| 0062_station_only_nearby_temperature_level_error_scale | Station-Only Nearby Temperature-Level Error Scale | 1 | 1 | 0 | 0 | 0 | 14160 |
| 0063_station_only_guarded_candidate_stack | Station-Only Guarded Candidate Stack | 1 | 1 | 0 | 0 | 0 | 37947 |
| 0064_station_only_heat_proxy_specialist_validation | Station-Only Heat Proxy Specialist Validation | 1 | 1 | 0 | 0 | 0 | 20468 |
| 0065_station_feature_bucket_residual_mining | Station-Feature Bucket Residual Mining | 1 | 1 | 0 | 0 | 0 | 262468 |
| 0066_station_feature_guarded_stack | Station-Feature Guarded Stack | 1 | 1 | 0 | 0 | 0 | 36963 |
| 0067_station_official_family_router | Station/Official Family Router | 1 | 1 | 0 | 0 | 0 | 59964 |
| 0068_prior_calibrated_fusion_screen | Prior-Calibrated Fusion Screen | 1 | 1 | 0 | 0 | 0 | 105686 |
| 0069_era_source_aware_fusion_model | Era/Source-Aware Fusion Model | 1 | 1 | 0 | 0 | 0 | 120827 |
| 0070_nonlinear_local_residual_fusion_lab | Nonlinear Local Residual-Fusion Lab | 1 | 1 | 0 | 0 | 0 | 106109 |
| 0071_sparse_specialist_delta_stack | Sparse Specialist Delta Stack | 1 | 1 | 0 | 0 | 0 | 80556 |
| 0072_cell_robustness_smooth_shrinkage | Cell Robustness And Smooth Shrinkage | 1 | 1 | 0 | 0 | 0 | 145818 |
| 0073_source_era_specific_shrinkage | Source-Era-Specific Shrinkage | 1 | 1 | 0 | 0 | 0 | 135270 |
| 0074_online_residual_memory_halflife | Online Residual Memory Half-Life | 1 | 1 | 0 | 0 | 0 | 74242 |
| 0075_online_residual_memory_refinement | Online Residual Memory Refinement | 1 | 1 | 0 | 0 | 0 | 79649 |
| 0076_online_no_regret_trust_router | Online No-Regret Trust Router | 1 | 1 | 0 | 0 | 0 | 42245 |
| 0077_remaining_0075_error_feature_mining | Remaining 0075 Error Feature Mining | 1 | 1 | 0 | 0 | 0 | 119837 |
| 0078_prior_only_residual_specialists | Prior-Only Residual Specialists | 1 | 1 | 0 | 0 | 0 | 56416 |
| 0079_guarded_specialist_combination | Guarded Specialist Combination | 1 | 1 | 0 | 0 | 0 | 56513 |
| 0080_source_era_hardened_specialist_gate | Source/Era Hardened Specialist Gate | 1 | 1 | 0 | 0 | 0 | 82392 |
| 0081_rss_gate_stability_stress | RSS Gate Stability Stress | 1 | 1 | 0 | 0 | 0 | 38034 |
| 0082_refreshed_archive_replay_readiness_audit | 0082 Refreshed Archive Replay Readiness Audit | 1 | 1 | 0 | 0 | 0 | 12132 |
| 0083_expanded_frame_official_anchor_replay | 0083 Expanded-Frame Official-Anchor Replay | 1 | 1 | 0 | 0 | 0 | 18200 |
| 0084_expanded_frame_hardened_official_specialists | 0084 Expanded-Frame Hardened Official Specialists | 1 | 1 | 0 | 0 | 0 | 30267 |
| 0085_long_history_feature_station_residual_bridge | 0085 Long-History Feature And Station Residual Bridge | 1 | 1 | 0 | 0 | 0 | 29908 |
| 0086_guarded_long_history_residual_specialists | 0086 Guarded Long-History Residual Specialists | 1 | 1 | 0 | 0 | 0 | 27866 |
| 0087_long_history_signal_interaction_specialists | 0087 Long-History Signal Interaction Specialists | 1 | 1 | 0 | 0 | 0 | 30579 |
| 0088_prior_gated_specialist_stack | 0088 Prior-Gated Specialist Stack | 1 | 1 | 0 | 0 | 0 | 21403 |
| 0089_remaining_error_regime_autopsy | 0089 Remaining Error Regime Autopsy | 1 | 1 | 0 | 0 | 0 | 61845 |
| 0090_guarded_specialists_from_error_autopsy | 0090 Guarded Specialists From Error Autopsy | 1 | 1 | 0 | 0 | 0 | 66392 |
| 0091_near_miss_specialist_failure_analysis | 0091 Near-Miss Specialist Failure Analysis | 1 | 1 | 0 | 0 | 0 | 18693 |
| 0092_blocking_slice_guarded_specialists | 0092 Blocking-Slice Guarded Specialists | 1 | 1 | 0 | 0 | 0 | 14705 |
| 0093_guarded_champion_sensitivity_check | 0093 Guarded Champion Sensitivity Check | 1 | 1 | 0 | 0 | 0 | 65142 |
| 0094_expanded_high_error_interaction_lab | 0094 Expanded High-Error Interaction Lab | 1 | 1 | 0 | 0 | 0 | 80167 |
| 0095_mam_error_direction_split_lab | 0095 MAM Error-Direction Split Lab | 1 | 1 | 0 | 0 | 0 | 74108 |
| 0096_directional_cell_failure_audit | 0096 Directional Cell Failure Audit | 1 | 1 | 0 | 0 | 0 | 27229 |
| 0097_stable_directional_cell_specialist | 0097 Stable Directional Cell Specialist | 1 | 1 | 0 | 0 | 0 | 31208 |
| 0098_source_submonth_stable_cell_specialist | 0098 Source/Submonth Stable Cell Specialist | 1 | 1 | 0 | 0 | 0 | 46507 |
| 0099_mam_cell_policy_sensitivity | 0099 MAM Cell Policy Sensitivity | 1 | 1 | 0 | 0 | 0 | 103395 |
| 0100_stable_mam_cell_feature_atlas | 0100 Stable MAM Cell Feature Atlas | 1 | 1 | 0 | 0 | 0 | 43589 |
| 0101_stable_mam_cell_feature_specialists | 0101 Stable MAM Cell Feature Specialists | 1 | 1 | 0 | 0 | 0 | 48134 |
| 0102_timestamp_proof_unlock_queue | 0102 Timestamp Proof Unlock Queue | 1 | 1 | 0 | 0 | 0 | 18442 |
| 0103_current_rss_continuation_without_blocked_sources |  | 0 | 0 | 0 | 0 | 0 | 0 |
| 0103_current_rss_safe_continuation | 0103 Current RSS Continuation Without Blocked Sources | 1 | 1 | 0 | 0 | 0 | 14005 |
| EXP-0046-HKG-T24-R14 | EXP-0046 HKG-T24-R14 Eligible Upper-Air Thermal Potential and Inversion Structure | 19 | 12 | 1 | 2 | 4 | 246 |
| EXP-0047-HKG-T24-R15 | EXP-0047 HKG-T24-R15 Surface-Upper-Air Coupling and Mixing-Potential Experiment | 19 | 12 | 1 | 2 | 4 | 244 |
| EXP-0048-HKG-T24-R16 | EXP-0048 HKG-T24-R16 Fifty-Year Regional ISD Surface Core | 19 | 12 | 1 | 2 | 4 | 222 |
| EXP-0049-HKG-T24-R17 | EXP-0049 HKG-T24-R17 Station Metadata Breaks, Urbanization, and Era Transfer | 19 | 12 | 1 | 2 | 4 | 241 |
| EXP-0050 | EXP-0050 — Corrected Common-Row Benchmark and Fold-Geometry Repair | 27 | 12 | 8 | 3 | 4 | 357 |
| EXP-0051 | EXP-0051 — Fold-Safe Multi-Timescale Dynamic Climatology | 27 | 12 | 8 | 3 | 4 | 322 |
| EXP-0052 | EXP-0052 — Training-Only Change-Point and Era-Adaptive Target Model | 27 | 12 | 8 | 3 | 4 | 293 |
| EXP-0053 | EXP-0053 — Lagged Tmax Trajectory Shape and Dynamic-Time-Warping Analogues | 27 | 12 | 8 | 3 | 4 | 325 |
| EXP-0054 | EXP-0054 — Causal Intraseasonal Spectral, SSA, and Wavelet State | 27 | 12 | 8 | 3 | 4 | 295 |
| EXP-0055 | EXP-0055 — Hot/Cold Spell Duration and Reversal Hazard | 27 | 12 | 8 | 3 | 4 | 275 |
| EXP-0056 | EXP-0056 — Volatility, Entropy, and Conditional Forecastability | 27 | 12 | 8 | 3 | 4 | 295 |
| EXP-0057 | EXP-0057 — Full/50/30/15-Year Recency-Window Expert Ensemble | 27 | 12 | 8 | 3 | 4 | 307 |
| EXP-0058 | EXP-0058 — Potential-Temperature Mixed-Layer Ceiling | 27 | 12 | 8 | 3 | 4 | 301 |
| EXP-0059 | EXP-0059 — Dry-Adiabatic Parcel Descent and Surface Realization Fraction | 27 | 12 | 8 | 3 | 4 | 353 |
| EXP-0060 | EXP-0060 — Full Inversion Geometry, Not Single-Level Difference | 27 | 12 | 8 | 3 | 4 | 328 |
| EXP-0061 | EXP-0061 — Integrated Lower-Tropospheric Heat Content and Layer Thickness | 27 | 12 | 8 | 3 | 4 | 332 |
| EXP-0062 | EXP-0062 — Equivalent Potential Temperature and Moist Static Energy Profile | 27 | 12 | 8 | 3 | 4 | 318 |
| EXP-0063 | EXP-0063 — Dry-Layer Entrainment and Evaporative-Cooling Potential | 27 | 12 | 8 | 3 | 4 | 320 |
| EXP-0064 | EXP-0064 — Vertical Wind Shear, Veering/Backing, and Air-Mass Advection | 27 | 12 | 8 | 3 | 4 | 331 |
| EXP-0065 | EXP-0065 — Geopotential Thickness, Ridge Strength, and Persistence Proxy | 27 | 12 | 8 | 3 | 4 | 337 |
| EXP-0066 | EXP-0066 — Twenty-Four/48-Hour Sounding Evolution and Air-Mass Tendency | 27 | 12 | 8 | 3 | 4 | 328 |
| EXP-0067 | EXP-0067 — Functional Profile PCA and Sounding-Shape Analogues | 27 | 12 | 8 | 3 | 4 | 312 |
| EXP-0068 | EXP-0068 — Soft Upper-Air Regime Mixture of Experts | 27 | 12 | 8 | 3 | 4 | 259 |
| EXP-0069 | EXP-0069 — Sounding Quality, Missing-Level Geometry, and Reliability Shrinkage | 27 | 12 | 8 | 3 | 4 | 339 |
| EXP-0070 | EXP-0070 — Raw ISD Intraday Thermal Trajectory Before 15:00 HKT | 27 | 12 | 8 | 3 | 4 | 324 |
| EXP-0071 | EXP-0071 — Intraday Dewpoint, Pressure, and Wind-Tendency Tensor | 27 | 12 | 8 | 3 | 4 | 333 |
| EXP-0072 | EXP-0072 — Cold-Front and Monsoon-Surge Change-Point Detector | 27 | 12 | 8 | 3 | 4 | 309 |
| EXP-0073 | EXP-0073 — Flow-Relative Upwind Station Weighting | 27 | 12 | 8 | 3 | 4 | 321 |
| EXP-0074 | EXP-0074 — Regional Pressure-Gradient Vector and Tendency Field | 27 | 12 | 8 | 3 | 4 | 330 |
| EXP-0075 | EXP-0075 — Spatial Temperature/Dewpoint Plane, Curvature, and Thermal Centroid | 27 | 12 | 8 | 3 | 4 | 337 |
| EXP-0076 | EXP-0076 — Coastal–Inland Thermal Contrast and Sea-Breeze Precursor | 27 | 12 | 8 | 3 | 4 | 318 |
| EXP-0077 | EXP-0077 — North–South Continental Air-Mass Gradient and Surge Propagation | 27 | 12 | 8 | 3 | 4 | 323 |
| EXP-0078 | EXP-0078 — East–West Pearl River Estuary and Coastal-Flow Gradient | 27 | 12 | 8 | 3 | 4 | 347 |
| EXP-0079 | EXP-0079 — Graph-Laplacian Modes of the Long-History Station Network | 27 | 12 | 8 | 3 | 4 | 331 |
| EXP-0080 | EXP-0080 — Robust Spatial Distribution Shape, Entropy, and Hotspot Extremes | 27 | 12 | 8 | 3 | 4 | 311 |
| EXP-0081 | EXP-0081 — Station-to-HKO Lead–Lag Propagation Map | 27 | 12 | 8 | 3 | 4 | 299 |
| EXP-0082 | EXP-0082 — Station Homogenization, Dynamic Offsets, and HKO Domain Adaptation | 27 | 12 | 8 | 3 | 4 | 364 |
| EXP-0083 | EXP-0083 — Operational Station-Dropout and Missingness-Robust Learning | 27 | 12 | 8 | 3 | 4 | 301 |
| EXP-0084 | EXP-0084 — ISD Report-Type, QC, Observation-Age, and Precision Weighting | 27 | 12 | 8 | 3 | 4 | 322 |
| EXP-0085 | EXP-0085 — Antecedent Rainfall, Dry-Spell Length, and Surface-Wetness Memory | 27 | 12 | 8 | 3 | 4 | 343 |
| EXP-0086 | EXP-0086 — Cloud–Sunshine Persistence and Radiative-Regime Memory | 27 | 12 | 8 | 3 | 4 | 318 |
| EXP-0087 | EXP-0087 — Solar-Radiation-to-Tmax Conversion Efficiency | 27 | 12 | 8 | 3 | 4 | 326 |
| EXP-0088 | EXP-0088 — Grass Minimum, Evaporation, and Urban Heat-Storage Memory | 27 | 12 | 8 | 3 | 4 | 316 |
| EXP-0089 | EXP-0089 — North Point Sea–Air Contrast and Marine Moderation Memory | 27 | 12 | 8 | 3 | 4 | 315 |
| EXP-0090 | EXP-0090 — Waglan Wind Persistence and Marine-Ventilation State | 27 | 12 | 8 | 3 | 4 | 332 |
| EXP-0091 | EXP-0091 — Long-History Daily-Climate Latent State | 27 | 12 | 8 | 3 | 4 | 336 |
| EXP-0092 | EXP-0092 — Multivariate Historical Climate-Trajectory Analogues | 27 | 12 | 8 | 3 | 4 | 336 |
| EXP-0093 | EXP-0093 — Tropical-Cyclone Subsidence Teacher and Operational Student | 27 | 12 | 8 | 3 | 4 | 366 |
| EXP-0094 | EXP-0094 — Cloud/Rain/Sunshine Suppression Teacher and Pre-Cutoff Student | 27 | 12 | 8 | 3 | 4 | 337 |
| EXP-0095 | EXP-0095 — Full-Day Synoptic Archetype Distillation | 27 | 12 | 8 | 3 | 4 | 336 |
| EXP-0096 | EXP-0096 — Conditional Champion Gate Across R14/R15/R16/R17 and New Experts | 27 | 12 | 8 | 3 | 4 | 308 |
| EXP-0097 | EXP-0097 — Nested Residual Stacking with Orthogonal Information Blocks | 27 | 12 | 8 | 3 | 4 | 328 |
| EXP-0098 | EXP-0098 — Heteroscedastic Student-t Distribution and Forecastability Calibration | 27 | 12 | 8 | 3 | 4 | 330 |
| EXP-0099 | EXP-0099 — Tail-Specialist Quantile Mixture, 0.1°C CDF, and Final Pre-Confirmation Freeze | 29 | 12 | 9 | 4 | 4 | 354 |
| _analysis_manifests |  | 0 | 0 | 0 | 0 | 0 | 0 |
| _template | {{EXPERIMENT_ID}} - {{TITLE}} | 11 | 7 | 0 | 1 | 3 | 129 |
| hko_official_backfill_monitor |  | 0 | 0 | 0 | 0 | 0 | 0 |

## Appendix B. Formal EXP-0050–EXP-0099 outcome matrix

| folder | title | decision | candidate_model | candidate_n | candidate_mae | core_mae | candidate_minus_core_mae | candidate_rmse | core_rmse | candidate_minus_core_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0050 | EXP-0050 — Corrected Common-Row Benchmark and Fold-Geometry Repair | PROMOTE | exp-0050_equal_weight_diagnostic | 1460.0 | 1.317644772000457 | 1.2365984893663038 | 0.08104628263415314 | 1.6347518455602836 | 1.571376929571448 | 0.06337491598883571 |
| EXP-0051 | EXP-0051 — Fold-Safe Multi-Timescale Dynamic Climatology | REJECT | exp-0051_candidate | 1460.0 | 1.4160196100721507 | 1.2365984893663038 | 0.17942112070584693 | 1.7802262403926823 | 1.571376929571448 | 0.20884931082123437 |
| EXP-0052 | EXP-0052 — Training-Only Change-Point and Era-Adaptive Target Model | REJECT | exp-0052_candidate | 1460.0 | 1.576263492560994 | 1.2365984893663038 | 0.3396650031946902 | 1.9740585654581604 | 1.571376929571448 | 0.40268163588671246 |
| EXP-0053 | EXP-0053 — Lagged Tmax Trajectory Shape and Dynamic-Time-Warping Analogues | REJECT | exp-0053_candidate | 1460.0 | 1.894881108956418 | 1.2365984893663038 | 0.6582826195901141 | 2.3656144481619683 | 1.571376929571448 | 0.7942375185905204 |
| EXP-0054 | EXP-0054 — Causal Intraseasonal Spectral, SSA, and Wavelet State | REJECT | exp-0054_candidate | 1460.0 | 1.4450387171561008 | 1.2365984893663038 | 0.20844022778979698 | 1.8628019515144416 | 1.571376929571448 | 0.2914250219429937 |
| EXP-0055 | EXP-0055 — Hot/Cold Spell Duration and Reversal Hazard | REJECT | exp-0055_candidate | 1460.0 | 1.9087638080734732 | 1.2365984893663038 | 0.6721653187071694 | 2.409861768567915 | 1.571376929571448 | 0.838484838996467 |
| EXP-0056 | EXP-0056 — Volatility, Entropy, and Conditional Forecastability | REJECT | exp-0056_candidate | 1460.0 | 1.4646639378075903 | 1.2365984893663038 | 0.2280654484412865 | 1.875567472541199 | 1.571376929571448 | 0.3041905429697511 |
| EXP-0057 | EXP-0057 — Full/50/30/15-Year Recency-Window Expert Ensemble | REJECT | exp-0057_full_window_control | 1460.0 | 1.4015178572290306 | 1.2365984893663038 | 0.16491936786272676 | 1.7669478986921645 | 1.571376929571448 | 0.1955709691207166 |
| EXP-0058 | EXP-0058 — Potential-Temperature Mixed-Layer Ceiling | REJECT | exp-0058_candidate | 1460.0 | 1.4761535468291975 | 1.2365984893663038 | 0.23955505746289374 | 1.8815531269146712 | 1.571376929571448 | 0.3101761973432233 |
| EXP-0059 | EXP-0059 — Dry-Adiabatic Parcel Descent and Surface Realization Fraction | REJECT | exp-0059_candidate | 1460.0 | 1.4447769691194137 | 1.2365984893663038 | 0.20817847975310988 | 1.840459798806645 | 1.571376929571448 | 0.269082869235197 |
| EXP-0060 | EXP-0060 — Full Inversion Geometry, Not Single-Level Difference | REJECT | exp-0060_candidate | 1460.0 | 1.4749493969138556 | 1.2365984893663038 | 0.23835090754755184 | 1.88180647339342 | 1.571376929571448 | 0.3104295438219722 |
| EXP-0061 | EXP-0061 — Integrated Lower-Tropospheric Heat Content and Layer Thickness | REJECT | exp-0061_candidate | 1460.0 | 1.468217493201731 | 1.2365984893663038 | 0.2316190038354271 | 1.874156964700224 | 1.571376929571448 | 0.30278003512877616 |
| EXP-0062 | EXP-0062 — Equivalent Potential Temperature and Moist Static Energy Profile | REJECT | exp-0062_candidate | 1460.0 | 1.4601472598468592 | 1.2365984893663038 | 0.22354877048055544 | 1.8684850703880644 | 1.571376929571448 | 0.2971081408166165 |
| EXP-0063 | EXP-0063 — Dry-Layer Entrainment and Evaporative-Cooling Potential | REJECT | exp-0063_candidate | 1460.0 | 1.4631821893402877 | 1.2365984893663038 | 0.22658369997398387 | 1.8763253585296653 | 1.571376929571448 | 0.3049484289582174 |
| EXP-0064 | EXP-0064 — Vertical Wind Shear, Veering/Backing, and Air-Mass Advection | REJECT | exp-0064_candidate | 1460.0 | 1.4621697265016014 | 1.2365984893663038 | 0.2255712371352976 | 1.866795716740952 | 1.571376929571448 | 0.295418787169504 |
| EXP-0065 | EXP-0065 — Geopotential Thickness, Ridge Strength, and Persistence Proxy | REJECT | exp-0065_candidate | 1460.0 | 1.4979075413156262 | 1.2365984893663038 | 0.2613090519493224 | 1.9048462966168496 | 1.571376929571448 | 0.33346936704540164 |
| EXP-0066 | EXP-0066 — Twenty-Four/48-Hour Sounding Evolution and Air-Mass Tendency | REJECT | exp-0066_candidate | 1460.0 | 1.4610141819092235 | 1.2365984893663038 | 0.2244156925429197 | 1.8705430526639448 | 1.571376929571448 | 0.2991661230924969 |
| EXP-0067 | EXP-0067 — Functional Profile PCA and Sounding-Shape Analogues | REJECT | exp-0067_candidate | 1460.0 | 1.472469740477716 | 1.2365984893663038 | 0.23587125111141227 | 1.881743172014945 | 1.571376929571448 | 0.310366242443497 |
| EXP-0068 | EXP-0068 — Soft Upper-Air Regime Mixture of Experts | REJECT | exp-0068_candidate | 1460.0 | 1.340401505915051 | 1.2365984893663038 | 0.10380301654874713 | 1.690717305348849 | 1.571376929571448 | 0.11934037577740098 |
| EXP-0069 | EXP-0069 — Sounding Quality, Missing-Level Geometry, and Reliability Shrinkage | REJECT | exp-0069_candidate | 1460.0 | 1.528884726118778 | 1.2365984893663038 | 0.2922862367524741 | 1.885664422001238 | 1.571376929571448 | 0.31428749242979004 |
| EXP-0070 | EXP-0070 — Raw ISD Intraday Thermal Trajectory Before 15:00 HKT | REJECT | exp-0070_candidate | 1460.0 | 1.4581025552698907 | 1.2365984893663038 | 0.22150406590358696 | 1.8616654524413168 | 1.571376929571448 | 0.29028852286986884 |
| EXP-0071 | EXP-0071 — Intraday Dewpoint, Pressure, and Wind-Tendency Tensor | REJECT | exp-0071_candidate | 1460.0 | 1.4531096728628112 | 1.2365984893663038 | 0.21651118349650744 | 1.862089394877485 | 1.571376929571448 | 0.2907124653060371 |
| EXP-0072 | EXP-0072 — Cold-Front and Monsoon-Surge Change-Point Detector | REJECT | exp-0072_candidate | 1460.0 | 1.4239068362938845 | 1.2365984893663038 | 0.1873083469275807 | 1.8033613351518931 | 1.571376929571448 | 0.2319844055804452 |
| EXP-0073 | EXP-0073 — Flow-Relative Upwind Station Weighting | REJECT | exp-0073_candidate | 1460.0 | 1.4549164311341796 | 1.2365984893663038 | 0.21831794176787578 | 1.846968065774254 | 1.571376929571448 | 0.27559113620280606 |
| EXP-0074 | EXP-0074 — Regional Pressure-Gradient Vector and Tendency Field | REJECT | exp-0074_candidate | 1460.0 | 1.4541846781897794 | 1.2365984893663038 | 0.21758618882347558 | 1.853610202080414 | 1.571376929571448 | 0.282233272508966 |
| EXP-0075 | EXP-0075 — Spatial Temperature/Dewpoint Plane, Curvature, and Thermal Centroid | REJECT | exp-0075_candidate | 1460.0 | 1.450048199024415 | 1.2365984893663038 | 0.21344970965811116 | 1.856730913849122 | 1.571376929571448 | 0.285353984277674 |
| EXP-0076 | EXP-0076 — Coastal–Inland Thermal Contrast and Sea-Breeze Precursor | REJECT | exp-0076_candidate | 1460.0 | 1.466202383231849 | 1.2365984893663038 | 0.22960389386554514 | 1.873561705689672 | 1.571376929571448 | 0.30218477611822414 |
| EXP-0077 | EXP-0077 — North–South Continental Air-Mass Gradient and Surge Propagation | REJECT | exp-0077_candidate | 1460.0 | 1.4513455877325632 | 1.2365984893663038 | 0.21474709836625938 | 1.8302985434539856 | 1.571376929571448 | 0.25892161388253765 |
| EXP-0078 | EXP-0078 — East–West Pearl River Estuary and Coastal-Flow Gradient | REJECT | exp-0078_candidate | 1460.0 | 1.5317337586225246 | 1.2365984893663038 | 0.2951352692562208 | 1.913314480572569 | 1.571376929571448 | 0.3419375510011211 |
| EXP-0079 | EXP-0079 — Graph-Laplacian Modes of the Long-History Station Network | REJECT | exp-0079_candidate | 1460.0 | 1.4376600363991956 | 1.2365984893663038 | 0.2010615470328918 | 1.825648128370613 | 1.571376929571448 | 0.25427119879916504 |
| EXP-0080 | EXP-0080 — Robust Spatial Distribution Shape, Entropy, and Hotspot Extremes | REJECT | exp-0080_candidate | 1460.0 | 1.4632949393510366 | 1.2365984893663038 | 0.22669644998473282 | 1.8739219901281348 | 1.571376929571448 | 0.30254506055668684 |
| EXP-0081 | EXP-0081 — Station-to-HKO Lead–Lag Propagation Map | REJECT | exp-0081_candidate | 1460.0 | 1.4703839319698964 | 1.2365984893663038 | 0.23378544260359257 | 1.880104091395013 | 1.571376929571448 | 0.3087271618235652 |
| EXP-0082 | EXP-0082 — Station Homogenization, Dynamic Offsets, and HKO Domain Adaptation | REJECT | exp-0082_candidate | 1460.0 | 1.4327412712582692 | 1.2365984893663038 | 0.19614278189196543 | 1.8316045785669255 | 1.571376929571448 | 0.26022764899547757 |
| EXP-0083 | EXP-0083 — Operational Station-Dropout and Missingness-Robust Learning | REJECT | exp-0083_candidate | 1460.0 | 1.4392042763784076 | 1.2365984893663038 | 0.20260578701210386 | 1.8371041961099903 | 1.571376929571448 | 0.2657272665385424 |
| EXP-0084 | EXP-0084 — ISD Report-Type, QC, Observation-Age, and Precision Weighting | REJECT | exp-0084_candidate | 1460.0 | 1.4657099352008245 | 1.2365984893663038 | 0.2291114458345207 | 1.875646900225027 | 1.571376929571448 | 0.30426997065357897 |
| EXP-0085 | EXP-0085 — Antecedent Rainfall, Dry-Spell Length, and Surface-Wetness Memory | REJECT | exp-0085_candidate | 1460.0 | 1.4612800473807588 | 1.2365984893663038 | 0.224681558014455 | 1.8737628977813163 | 1.571376929571448 | 0.3023859682098684 |
| EXP-0086 | EXP-0086 — Cloud–Sunshine Persistence and Radiative-Regime Memory | REJECT | exp-0086_candidate | 1460.0 | 1.3867128311744794 | 1.2365984893663038 | 0.15011434180817562 | 1.7628350020138617 | 1.571376929571448 | 0.19145807244241375 |
| EXP-0087 | EXP-0087 — Solar-Radiation-to-Tmax Conversion Efficiency | REJECT | exp-0087_candidate | 1460.0 | 1.4719620004596896 | 1.2365984893663038 | 0.23536351109338582 | 1.8765687099464567 | 1.571376929571448 | 0.3051917803750088 |
| EXP-0088 | EXP-0088 — Grass Minimum, Evaporation, and Urban Heat-Storage Memory | REJECT | exp-0088_candidate | 1460.0 | 1.4597093703756874 | 1.2365984893663038 | 0.2231108810093836 | 1.85988098337828 | 1.571376929571448 | 0.288504053806832 |
| EXP-0089 | EXP-0089 — North Point Sea–Air Contrast and Marine Moderation Memory | REJECT | exp-0089_candidate | 1460.0 | 1.453305885409278 | 1.2365984893663038 | 0.21670739604297418 | 1.8440462268564828 | 1.571376929571448 | 0.27266929728503486 |
| EXP-0090 | EXP-0090 — Waglan Wind Persistence and Marine-Ventilation State | REJECT | exp-0090_candidate | 1460.0 | 1.3823969660131945 | 1.2365984893663038 | 0.14579847664689072 | 1.7347543286758589 | 1.571376929571448 | 0.16337739910441096 |
| EXP-0091 | EXP-0091 — Long-History Daily-Climate Latent State | REJECT | exp-0091_candidate | 1460.0 | 1.4674131595333533 | 1.2365984893663038 | 0.23081467016704948 | 1.848520851400566 | 1.571376929571448 | 0.27714392182911807 |
| EXP-0092 | EXP-0092 — Multivariate Historical Climate-Trajectory Analogues | REJECT | exp-0092_candidate | 1460.0 | 1.8589929718292544 | 1.2365984893663038 | 0.6223944824629506 | 2.304656673372427 | 1.571376929571448 | 0.7332797438009793 |
| EXP-0093 | EXP-0093 — Tropical-Cyclone Subsidence Teacher and Operational Student | MECHANISM_ONLY | exp-0093_candidate | 1460.0 | 1.4794574365397677 | 1.2365984893663038 | 0.2428589471734639 | 1.90043429563564 | 1.571376929571448 | 0.329057366064192 |
| EXP-0094 | EXP-0094 — Cloud/Rain/Sunshine Suppression Teacher and Pre-Cutoff Student | MECHANISM_ONLY | exp-0094_candidate | 1460.0 | 1.5004632580255468 | 1.2365984893663038 | 0.263864768659243 | 1.901105673836376 | 1.571376929571448 | 0.32972874426492815 |
| EXP-0095 | EXP-0095 — Full-Day Synoptic Archetype Distillation | MECHANISM_ONLY | exp-0095_candidate | 1460.0 | 1.3433273438689408 | 1.2365984893663038 | 0.10672885450263703 | 1.6831179236862224 | 1.571376929571448 | 0.11174099411477445 |
| EXP-0096 | EXP-0096 — Conditional Champion Gate Across R14/R15/R16/R17 and New Experts | REJECT | exp-0096_candidate | 1460.0 | 1.3187917256904038 | 1.2365984893663038 | 0.08219323632410003 | 1.635764865417446 | 1.571376929571448 | 0.064387935845998 |
| EXP-0097 | EXP-0097 — Nested Residual Stacking with Orthogonal Information Blocks | REJECT | exp-0097_candidate | 1460.0 | 1.4142209878136496 | 1.2365984893663038 | 0.1776224984473458 | 1.8091209523925609 | 1.571376929571448 | 0.23774402282111295 |
| EXP-0098 | EXP-0098 — Heteroscedastic Student-t Distribution and Forecastability Calibration | REJECT | exp-0098_candidate | 1460.0 | 1.4714673042226505 | 1.2365984893663038 | 0.2348688148563467 | 1.8816988648795931 | 1.571376929571448 | 0.3103219353081452 |
| EXP-0099 | EXP-0099 — Tail-Specialist Quantile Mixture, 0.1°C CDF, and Final Pre-Confirmation Freeze | REJECT | exp-0099_candidate | 1460.0 | 1.4258451639726415 | 1.2365984893663038 | 0.1892466746063377 | 1.786794901639361 | 1.571376929571448 | 0.21541797206791302 |


**Interpretation note:** positive `candidate_minus_core_mae` means the candidate was worse. `EXP-0050` promoted the corrected core/common-row benchmark and fold repair; its equal-weight diagnostic row is not the promoted champion. The repeated candidate underperformance is preserved as a guard against rerunning the same broad models.

## Appendix C. Numbered experiment metric progression from manifests

| folder | status | rows | metric_key | best_metric | candidate | next |
| --- | --- | --- | --- | --- | --- | --- |
| 0016_past_only_official_anchor_correction_ladder |  | 2670.0 | best_mae | 0.9958893622716024 |  |  |
| 0017_past_only_official_residual_analog_screen |  | 2670.0 | best_mae | 0.9921063358440408 |  |  |
| 0018_past_only_official_expert_blend_screen |  |  | best_mae | 0.9974056405474928 |  |  |
| 0020_regime_experts |  | 2670.0 | best_blend_mae | 1.0046105796178564 | regime_blend_inverse_mae_same_source |  |
| 0022_failure_specialists |  | 2670.0 | best_blend_mae | 1.0074137003747845 | failure_blend_best_same_source |  |
| 0023_composite_expert_stack |  | 2670.0 | best_mae | 1.0002082115362154 | composite_inverse_mae_all_prior_min120 |  |
| 0026_pressure_gradient_experts |  | 2670.0 | best_blend_mae | 0.9987914296652666 | pressure_blend_inverse_mae_same_source |  |
| 0027_pressure_regime_interaction_atlas |  | 2670.0 | best_blend_mae | 1.0041104807938863 | pressure_interaction_blend_inverse_mae_all_prior |  |
| 0028_smooth_gated_pressure_experts |  | 2670.0 | best_blend_mae | 1.00253292969718 | smooth_pressure_blend_best_all_prior |  |
| 0029_official_residual_source_text_range_dynamics |  | 2670.0 | best_blend_mae | 1.0043304080389488 | official_residual_blend_inverse_mae_same_source |  |
| 0030_multi_signal_local_residual_lab |  | 2670.0 | best_blend_mae | 1.0022578556630786 | multi_signal_blend_best_same_source |  |
| 0031_regime_gated_specialist_selector |  | 2670.0 | best_selector_mae | 1.0014482962991802 |  |  |
| 0032_residual_failure_cluster_discovery |  | 2670.0 | best_blend_mae | 0.9964415441014096 |  |  |
| 0033_smooth_residual_archetype_specialists |  | 2670.0 | best_blend_mae | 0.9937461382773508 |  |  |
| 0034_cluster_centroid_soft_gating |  | 2670.0 | best_blend_mae | 0.9933641814307462 |  |  |
| 0035_forecast_revision_momentum_deep_dive |  | 2670.0 | best_blend_mae | 0.9945672538248896 |  |  |
| 0036_revision_centroid_stack_ablation |  | 2670.0 | current_overall_best_mae | 0.99269220944966 |  |  |
| 0037_stack_trust_meta_features |  | 2670.0 | current_overall_best_mae | 0.9897578401276024 |  |  |
| 0038_forecast_history_state_deepening |  | 2670.0 | current_overall_best_mae | 0.9897578401276024 |  |  |
| 0039_station_network_residuals |  | 2670.0 | best_full_mae | 0.9880267255624336 | station_network_bucket_16_isd_pressure_mean_hpa_change_1d_text_signal_state_same_source |  |
| 0040_station_network_smooth_residuals |  | 2670.0 | best_full_mae | 0.984588792413364 | smooth_station_f04_target_lag60_tmax_c_plus_forecast_max_change_1_source_c_feature_only_same_source_k60_hl_non |  |
| 0041_station_network_forecast_stack |  | 2670.0 | best_full_mae | 0.9831191862562138 | stack_0041_core_source_forecast_text_positive_lift_same_source |  |
| 0042_trust_router_sensitivity |  | 2670.0 | best_full_mae | 0.982991504998032 | sensitivity_0042_history_threshold_source_forecast_text_g80_b30_core_source_forecast_text_positive_lift_same_source_g80_b30 |  |
| 0048_gated_residual_specialist_screen |  | 3846.0 | best_full_mae | 1.0222528509771205 |  |  |
| 0049_router_gate_stack_screen |  |  | best_full_mae | 0.980343427532767 | 0049_fixed_gate_residual_all_prior_h0_s0p5_r_sensitivity_0042_history_threshold_source_forecast_text_g80_b30_core_source_forecast_text_positive_lift_same_source_g80_b30_g_17154a62a0 |  |
| 0055_station_only_walkforward_benchmark |  |  | best_mae | 1.4024457575367009 |  |  |
| 0058_station_only_late_period_bias_repair |  |  | best_mae | 1.3351228124211925 | global_halflife365_min180_shrink365_cap1p5 |  |
| 0059_station_only_february_march_transition_specialist |  |  | best_mae | 1.3862996344237026 | month_prior_mean_min45_shrink120_cap1p25 |  |
| 0060_station_only_spring_transition_pressure_dew_specialist |  |  | best_mae | 1.3293548927910477 | spring_phase_min35_shrink100_cap1p0 |  |
| 0061_station_only_pressure_high_uncertainty_guard |  |  | best_mae | 1.333693585463042 |  |  |
| 0062_station_only_nearby_temperature_level_error_scale |  |  | best_mae | 1.33495344436846 |  |  |
| 0063_station_only_guarded_candidate_stack |  |  | best_mae | 1.3248606042869946 |  |  |
| 0064_station_only_heat_proxy_specialist_validation |  |  | best_mae | 1.321015699466693 |  |  |
| 0065_station_feature_bucket_residual_mining |  |  | best_mae | 1.3148387089689249 | term_son_traj_450110_99999_dew_point_c_latest_before_1500_current_minus_rolling_mean_14d_low_halflife730_min60_shrink120_cap1p0 |  |
| 0066_station_feature_guarded_stack |  |  | best_stack_mae | 1.3096660198534358 |  |  |
| 0067_station_official_family_router |  | 2670.0 | best_mae | 0.9521316826180168 | fixed_station_blend_0p25 |  |
| 0068_prior_calibrated_fusion_screen |  | 2670.0 | best_mae | 0.950434934268158 | prior_soft_source_h120_t0p02_fb0p15 |  |
| 0069_era_source_aware_fusion_model |  | 2670.0 | best_mae | 0.949299406897959 | causal_prior_selector_best_source_signeddiff_active_h120_fbconstant0p22_a0p75_0p25_0p0 |  |
| 0070_nonlinear_local_residual_fusion_lab |  | 2670.0 | best_mae | 0.9486180151523014 | causal_delta_soft0p02_source_signeddiff_range_h30 |  |
| 0071_sparse_specialist_delta_stack |  | 2670.0 | best_mae | 0.9461507103179788 | diagnostic_cell_atlas_rss_warm_top10_activem0p01 |  |
| 0072_cell_robustness_smooth_shrinkage |  | 2670.0 | best_mae | 0.947359490166776 | diagnostic_smooth_atlas_rss_warm_top10_nshrink60p0 |  |
| 0073_source_era_specific_shrinkage |  | 2670.0 | best_mae | 0.9451065682801856 | diagnostic_sourceera_fixed_all |  |
| 0074_online_residual_memory_halflife |  | 2670.0 | best_mae | 0.946580543582804 | causal_onmem_seasonal_behavior_h45_n20_lift0p0_cap0p2_lift_weighted |  |
| 0075_online_residual_memory_refinement |  | 2670.0 | best_mae | 0.9456033267531754 | causal_onmem_refine_all_h20_n10_cap0p2_lift_weighted |  |
| 0076_online_no_regret_trust_router |  | 2670.0 | best_mae | 0.9463955289121396 | trust_router_all_h20_n10_inverse_mae_blend |  |
| 0078_prior_only_residual_specialists |  | 2670.0 | best_mae | 0.9448508210099432 | specialist_weak_morning_warming_low1p5_feature_source_season_h30 |  |
| 0079_guarded_specialist_combination |  | 2670.0 | best_mae | 0.9439288177079488 | guarded_full_positive_m0078_candidate_lift_weighted_w1p0_lift0p0 |  |
| 0080_source_era_hardened_specialist_gate |  | 2670.0 | best_mae | 0.9437550087964722 | sourceera_rss_2022plus_full-positive-m0078-prior-lift-weighted-w1p0-lift0p0 |  |
| 0081_rss_gate_stability_stress |  | 2670.0 | best_mae | 0.9437550087964722 | rssgate_start20220101_full-positive-m0078-prior-lift-weighted-w1p0-lift0p0 |  |
| 0083_expanded_frame_official_anchor_replay | expanded_frame_baseline_not_champion_replacement | 5265.0 | best_mae | 1.059859166323171 | prior_blend_source_top5_min90 | Run 0084 to convert the best 0083 expanded-frame official-anchor correction into fold/source hardened specialists, then compare against 0081 on the old frame and against official raw on the newly available 2004-08-06 to 2011-09-13 press segment. Keep 2024+ sealed. |
| 0084_expanded_frame_hardened_official_specialists | hardened_gate_screen_complete | 5265.0 | best_mae | 1.059859166323171 | 0083_prior_blend_source_top5_min90 | Run 0085 as a long-history feature and station information-gain bridge on the same expanded official forecast frame: join the 1949-2026 feature matrix to 0084 residuals, rank station/upper-air/marine attributes by residual information gain, then design guarded local specialists for the highest-signal failure regimes. Keep 2024+ sealed. |
| 0086_guarded_long_history_residual_specialists | guarded_long_history_specialist_screen_complete | 5265.0 | best_mae | 1.0540483032721892 | specialist_isd-morning-to-midday-temp-rise-c_feature_m90 | Run 0087 to combine only hardened-passing long-history specialists with the official-anchor base, or, if 0086 has no hardened pass, deepen feature engineering around the top 0085 residual signals before attempting another specialist stack. Keep 2024+ sealed. |
| 0087_long_history_signal_interaction_specialists | long_history_interaction_specialist_screen_complete | 5265.0 | best_mae | 1.0517215888652507 | interaction_target-roll90-mean-lag7-c__x__ua-layer-925-850-ceiling-minus-isd-temp-c_interaction_m60 | Run 0088 to stack only hardened-passing single-feature and interaction specialists with prior-only source/frame gates, then compare against 0086 and the older 0081 partial-frame champion. Keep 2024+ sealed. |
| 0088_prior_gated_specialist_stack | prior_gated_specialist_stack_complete | 5265.0 | best_mae | 1.0517215888652507 | interaction_target-roll90-mean-lag7-c__x__ua-layer-925-850-ceiling-minus-isd-temp-c_interaction_m60 | Run 0089 to inspect remaining high-error regimes of the 0088/0087 champion, then mine station-specific and source-era interactions targeted only at those failures. Keep 2024+ sealed. |
| 0090_guarded_specialists_from_error_autopsy | guarded_specialists_from_error_autopsy_complete | 5265.0 | best_mae | 1.0517215888652507 | 0088_0087_interaction_champion | Run 0091 to inspect the best non-hardened 0090 candidates by failing slice, then design either narrower source-season specialists or a conservative ensemble constrained by the 0090 no-regression gate. |
| 0092_blocking_slice_guarded_specialists | blocking_slice_guarded_specialists_complete | 5265.0 | best_mae | 1.050706845255926 | guarded_failed_slices__autopsy_isd-morning-to-midday-temp-rise-c_source_season_feature_m120_cap0p55 | Run 0093_guarded_champion_sensitivity_check: stress-test the 0092 guarded champion with alternative guard definitions, guard subsets, correction caps, and min-history settings before treating it as the new pre-2024 research champion. |
| 0093_guarded_champion_sensitivity_check | guarded_champion_sensitivity_check_complete | 5265.0 | best_mae | 1.050660939385909 | sens_isd-morning-to-midday-temp-rise-c_source_season_feature_m120_s75_cap0p35_guard_all_0092_failed | Run 0094_expanded_high_error_interaction_lab: target the persistent MAM/new-press high-error regime with interaction specialists using station-network, target-memory, upper-air ceiling, and marine features, while keeping the 0093/0092 guarded champion as the baseline. |
| 0094_expanded_high_error_interaction_lab | expanded_high_error_interaction_lab_complete | 5265.0 | best_mae | 1.0498400990116323 | mamint_d07a699c_target-roll120-mean-lag7-c__x__isd-morning-to-midday-temp-rise-c_mam_all_m40 | Run 0095_mam_error_direction_split_lab: split the persistent MAM error regime into underforecast and overforecast sub-regimes, then test asymmetric guarded residual corrections using the strongest 0094 pair definitions and the current 0093/0094 champion baseline. |
| 0095_mam_error_direction_split_lab | mam_error_direction_split_lab_complete | 5265.0 | best_mae | 1.0495173644250873 | dirsplit_93376c53_target-roll120-mean-lag7-c__x__ua-layer-1000-925-ceiling-minus-is_mam_all_overforecast_only_m80_t0p1_cap0p25 | Run 0096_directional_cell_failure_audit: analyze where 0095 direction-split candidates helped or failed by prior-direction, pair bucket, source family, and MAM sub-month, then design the next bounded specialist from only the stable improving cells. |
| 0097_stable_directional_cell_specialist | stable_directional_cell_specialist_complete | 5265.0 | best_mae | 1.0495173644250873 | dirsplit_93376c53_target-roll120-mean-lag7-c__x__ua-layer-1000-925-ceiling-minus-is_mam_all_overforecast_only_m80_t0p1_cap0p25 | Run 0098_source_submonth_stable_cell_specialist: test whether the 0096 source/submonth stable cells can recover the small robust MAM gain lost by bucket-only guarding, still using no 2024+ rows and only the currently available forecast archive. |
| 0098_source_submonth_stable_cell_specialist | source_submonth_stable_cell_specialist_complete | 5265.0 | best_mae | 1.0494983868711296 | srcsub_b035802a_bucket-and-source-submonth_target-roll120-mean-lag7-c--x--ua- | Run 0099_mam_cell_policy_sensitivity: stress-test the best 0098 source/submonth policy across adjacent min-history, direction-threshold, and cap settings, while keeping 2024+ rows sealed. |
| 0099_mam_cell_policy_sensitivity | mam_cell_policy_sensitivity_complete | 5265.0 | best_mae | 1.049472831428368 | mampol_052ffb8d_m60_t0p05_c0p2 | Run 0100_stable_mam_cell_feature_atlas: inside the confirmed MAM bucket/source/submonth agreement cells, rank all leakage-eligible long-history station, upper-air, marine, and target-memory features for residual separation and information gain with 2024+ rows still sealed. |
| 0101_stable_mam_cell_feature_specialists | stable_mam_cell_feature_specialists_complete | 5265.0 | best_mae | 1.0490737910402703 | featcell_22ab56a6_trajectory-7-30-slope-c-per-_q3_agreement_m20_c0p25 | Run 0102_timestamp_proof_unlock_queue: attach issue/available-at proof for the high-scoring 0100 upper-air and daily marine features, then rerun 0101 with newly eligible families only if the timestamp audit passes. |
| 0102_timestamp_proof_unlock_queue | timestamp_proof_unlock_queue_complete_no_unlocks |  | input_0101_best_mae | 1.0490737910402703 | featcell_22ab56a6_trajectory-7-30-slope-c-per-_q3_agreement_m20_c0p25 | Run 0103_current_rss_continuation_without_blocked_sources: continue leakage-free analysis using the current scoreable RSS/press archive plus already future-allowed station and target-memory features while the forecast backfill continues; keep upper-air, daily climate, and marine proxies diagnostic-only until provider timestamp proof arrives. |
| 0103_current_rss_safe_continuation | current_rss_continuation_analysis_only_complete | 5265.0 | overall_candidate_mae | 1.0490737910402703 | featcell_22ab56a6_trajectory-7-30-slope-c-per-_q3_agreement_m20_c0p25 | Run 0104_safe_feature_interaction_stability_lab: use only future-allowed calendar, station, and target-memory features to test smoother interaction/stack stability by source and season; do not use upper-air, HKO daily climate, or marine proxies until 0102 is unlocked. |

## Appendix D. New 80-experiment queue summary

| experiment_id | title | priority | research_mode | eligibility | dependencies |
| --- | --- | --- | --- | --- | --- |
| 0104 | Canonical Frame, Evidence, and Scoreboard Registry | P0 | Foundation / audit | Deployable now | None; this is the organizing prerequisite |
| 0105 | Station Dossier, Identity, Geography, and Coverage Forensics | P0 | Foundation / data quality | Deployable now after deterministic metadata derivation | 0104 frame registry |
| 0106 | Per-Feature Availability, Eligibility, and Lineage Graph | P0 | Foundation / leakage audit | Deployable now | 0104 and source manifests |
| 0107 | Timezone, Daily Boundary, Unit, Duplicate, and Observation-Age Audit | P0 | Foundation / data quality | Deployable now | 0104–0106 |
| 0108 | Canonical Multi-Response Label and Residual Library | P0 | Foundation / target engineering | Deployable now | 0104 |
| 0109 | Temporal Multiple-Testing, Stability, and Negative-Control Harness | P0 | Foundation / statistical validity | Deployable now | 0104, 0108 |
| 0110 | One-Command Walk-Forward Replay and Artifact Harness | P0 | Foundation / implementation | Deployable now | 0104, 0106, 0108, 0109 |
| 0111 | Cross-Fitted Conditional Information-Gain Atlas | P1 | Exploratory information-gain | Deployable now for safe features; blocked families diagnostic-only | 0104–0110 |
| 0112 | Nonlinear Monotonicity and Conditional Response-Curve Atlas | P1 | Exploratory information-gain | Deployable now for safe features | 0111 |
| 0113 | Asymmetric Tail Dependence and Copula-Style Error Atlas | P1 | Exploratory information-gain / tail risk | Deployable now for safe features | 0108, 0109, 0111 |
| 0114 | Feature Redundancy Graph and Minimal Orthogonal Signal Set | P1 | Exploratory information-gain / dimensionality discipline | Deployable now for safe features | 0111–0113 |
| 0115 | Hierarchical Two- and Three-Feature Interaction ANOVA | P1 | Exploratory information-gain | Deployable now for safe inputs | 0111, 0112, 0114 |
| 0116 | Information-Gain Drift, Era Transfer, and Effect-Lifetime Atlas | P1 | Exploratory information-gain / robustness | Deployable now | 0111–0115 |
| 0117 | Correction Uplift and No-Harm Opportunity Atlas | P1 | Promotion-oriented information gain | Deployable now | 0108, 0111–0116 |
| 0118 | Value of Information Under Missingness, Age, and Source Dropout | P2 | Exploratory information-gain / operational robustness | Deployable now | 0105–0111 |
| 0119 | Causal Target Level–Slope–Curvature State Space | P1 | Exploratory then promotion-oriented | Deployable now | 0108–0111 |
| 0120 | Robust Local Derivatives, Reversal Hazard, and Trend Exhaustion | P1 | Exploratory information-gain | Deployable now | 0119 |
| 0121 | Volatility Compression–Expansion Transition Atlas | P1 | Exploratory information-gain / uncertainty | Deployable now | 0108, 0119 |
| 0122 | Causal Seasonal Phase-Boundary and Monsoon-Transition Index | P1 | Exploratory information-gain | Deployable now with safe inputs | 0119, 0121 |
| 0123 | Target-Memory × Station-Network Coherence Atlas | P1 | Exploratory information-gain then promotion | Deployable now | 0105, 0119, 0111 |
| 0124 | Spell Termination with Spatial Confirmation | P2 | Exploratory specialist | Deployable now | 0120, 0123 |
| 0125 | Phase-Aligned Year-over-Year and Submonth Analog Residuals | P2 | Exploratory information-gain | Deployable now | 0122, 0114 |
| 0126 | Causal Thermal Constraint Envelope and Breakout Detector | P1 | Promotion-oriented diagnostic | Deployable now | 0119–0123 |
| 0127 | Station Rank, Rank-Reversal, and Permutation Motif Atlas | P1 | Exploratory information-gain | Deployable now | 0105, 0107, 0111 |
| 0128 | Robust Spatial Interpolation and HKO Counterfactual Field Estimate | P1 | Exploratory information-gain | Deployable now after 0105 geometry | 0105, 0107, 0144 later optional |
| 0129 | Thermal-Front Geometry, Orientation, and Distance-to-HKO | P1 | Exploratory information-gain | Deployable now after station geometry audit | 0105, 0127, 0128 |
| 0130 | Dewpoint-Front, Moisture-Boundary, and Dry-Air Intrusion Geometry | P1 | Exploratory information-gain | Deployable now with safe station dewpoint | 0105, 0129 |
| 0131 | Pressure-Tendency Wave Propagation and Arrival-Time Map | P1 | Exploratory information-gain | Deployable now | 0105, 0107 |
| 0132 | Wind-Field Divergence, Vorticity, Deformation, and Directional Persistence | P1 | Exploratory information-gain | Deployable now where station wind is cutoff-safe | 0105, 0107 |
| 0133 | Surface Moisture-Flux Convergence and Ventilation Potential | P1 | Exploratory information-gain | Deployable now with safe station data | 0130, 0132 |
| 0134 | Flow-Conditioned Transport and Station-to-HKO Arrival-Time Atlas | P1 | Exploratory information-gain | Deployable now | 0105, 0132 |
| 0135 | Sea-Breeze Penetration Phase and Marine-Suppression Index | P1 | Exploratory information-gain then specialist | Deployable now from station proxies; marine daily source remains diagnostic | 0105, 0127, 0129, 0132 |
| 0136 | Cool-Surge Propagation, Coastal Modification, and Breakdown Index | P1 | Exploratory information-gain then specialist | Deployable now | 0122, 0131, 0132, 0134 |
| 0137 | Spatial-Field Topology, Hotspot Connectivity, and Boundary Morphology | P2 | Exploratory information-gain | Deployable now after geometry/QC | 0105, 0128–0130 |
| 0138 | Response-Specific Station Contribution and Group-Shapley Leaderboard | P1 | Exploratory information-gain / simplification | Deployable now | 0105, 0111, 0127–0137 |
| 0139 | Deterministic Static Station Context Feature Store | P0 | Foundation / feature engineering | Deployable now after reproducible derivation | 0105 |
| 0140 | Physics-Informed Station-Role Clustering and Group Compression | P1 | Exploratory information-gain / simplification | Deployable now | 0105, 0139, 0138 |
| 0141 | Flow-Relative Land–Sea–Urban Fetch Index | P1 | Exploratory information-gain | Deployable now after static context | 0132, 0139 |
| 0142 | Elevation, Lapse-Rate, and Downslope Realization Index | P2 | Exploratory information-gain | Deployable now | 0139, 0141, 0132 |
| 0143 | Coastline Orientation and Onshore-Penetration Exposure Index | P2 | Exploratory information-gain | Deployable now | 0139, 0141 |
| 0144 | Physics-Informed Graph Kernels and Causal Graph Modes | P1 | Exploratory information-gain | Deployable now | 0139–0143, 0105 |
| 0145 | Surface Wet-Bulb, Enthalpy, and Moist-Heat Network State | P1 | Exploratory information-gain | Deployable now from station temperature/dewpoint/pressure | 0107, 0130 |
| 0146 | Temperature–Dewpoint Spread and Dry-Heating Potential Atlas | P1 | Exploratory then promotion-oriented | Deployable now | 0145, 0112 |
| 0147 | Dewpoint Tendency × Wind-Advection Interaction Lab | P1 | Exploratory information-gain | Deployable now | 0132, 0134, 0145 |
| 0148 | Antecedent Wetness, Drying Recovery, and Nonlinear Surface-Memory Atlas | P2 | Diagnostic-to-safe-proxy research | Safe proxy inputs deployable; finalized rainfall/daily climate diagnostic-only without publication proof | 0106, 0145 |
| 0149 | Radiative-Suppression Safe Proxy Distillation | P1 | Diagnostic-to-deployable proxy conversion | Teacher diagnostic-only; student deployable if safe | 0106, 0145–0148 |
| 0150 | State-Dependent Solar-to-Tmax Conversion Efficiency | P2 | Diagnostic mechanism and safe-proxy research | Long daily solar diagnostic-only; short high-frequency pre-2024 diagnostic/prospective | 0149, 0179 later |
| 0151 | Visibility, Haze, and Aerosol-Cloud Suppression Proxy Atlas | P2 | Diagnostic-to-safe research | Eligibility must be proven per source/lag | 0106, 0149 |
| 0152 | Urban Heat-Storage Hysteresis and Nocturnal Thermal Memory | P2 | Exploratory information-gain | Long target/station proxies deployable; detailed high-frequency later | 0105, 0140, 0119 |
| 0153 | 1000-hPa Geopotential Height Diagnostic-to-Safe Proxy Conversion | P1 | Diagnostic teacher → deployable student | IGRA teacher diagnostic-only; safe student deployable | 0102, 0111, 0131–0134 |
| 0154 | Low-Level Inversion and Mixing-Cap Safe Proxy Conversion | P1 | Diagnostic teacher → deployable student | Upper-air teacher diagnostic-only; station student deployable | 0102, 0145–0147, 0132 |
| 0155 | Lower-Tropospheric Heat-Content and Thickness Safe Proxy Conversion | P1 | Diagnostic teacher → deployable student | Upper-air teacher diagnostic-only | 0153, 0154, 0142 |
| 0156 | Moisture-Profile and Stability Safe Proxy Conversion | P1 | Diagnostic teacher → deployable student | Upper-air teacher diagnostic-only | 0145–0149, 0154 |
| 0157 | Vertical Shear, Veering, and Advection Safe Proxy Conversion | P2 | Diagnostic teacher → deployable student | Upper-air teacher diagnostic-only | 0132, 0134, 0102 |
| 0158 | Sea-Temperature and Marine-Moderation Safe Proxy Conversion | P1 | Diagnostic teacher → deployable student | Daily marine teacher diagnostic-only; live feeds prospective | 0135, 0141, 0143, 0102 |
| 0159 | Tropical-Cyclone Subsidence, Cloud-Shield, and Flow-Regime Safe Proxy | P1 | Diagnostic teacher → deployable student | Best-track teacher diagnostic-only; safe student deployable | 0132–0135, 0149 |
| 0160 | Timestamp-Proof Acquisition and Prospective Shadow-Latency Experiment | P0 | Data unblock / prospective audit | Prospective only; does not retroactively unlock history without proof | 0102, 0106 |
| 0161 | Exact 0075/0081 Online-Memory Replay on the 5,265-Row Expanded Frame | P0 | Promotion-oriented benchmark | Deployable now | 0104, 0110 |
| 0162 | Hierarchical Source–Era–Season Online Residual Memory | P1 | Promotion-oriented | Deployable now | 0161, 0116 |
| 0163 | Signed Residual Streak, Bias Momentum, and Multi-Halflife State | P1 | Promotion-oriented | Deployable now | 0161, 0162 |
| 0164 | Full Forecast-Vintage Revision Path and Shape Features | P1 | Exploratory then promotion-oriented | Deployable only where all revisions are timestamp-eligible | 0104, 0106, official archive completeness |
| 0165 | Causal Forecast-Text Ontology and Meteorological State Extraction | P1 | Exploratory information-gain | Deployable where exact pre-cutoff text vintage exists | 0104, 0106, 0164 |
| 0166 | Numeric–Text–Station Consistency and Contradiction Index | P1 | Promotion-oriented trust research | Deployable on exact text/numeric vintages | 0165, 0123, 0149 |
| 0167 | Forecast Range, Predictive-Scale, and Trust Calibration | P1 | Uncertainty and routing | Deployable where exact range fields exist | 0108, 0164–0166 |
| 0168 | Forecast Staleness, Null Fields, Parser Confidence, and Product-Quality State | P1 | Data quality + trust routing | Deployable now | 0104, 0106, 0164 |
| 0169 | Invariant Press–RSS Cross-Era Calibration and Source Bridge | P1 | Promotion-oriented robustness | Deployable now on current disjoint eras; stronger after backfill | 0104, 0161–0168 |
| 0170 | Uplift-Based Abstaining Trust Router Across Credible Experts | P1 | Promotion-oriented integration | Deployable now after expert qualification | 0117, 0161–0169, selected station/target specialists |
| 0171 | MAM Latent Transition-Phase Classifier and Error Map | P1 | Exploratory then promotion-oriented | Deployable now with safe features | 0122, 0123, 0130–0136, 0162 |
| 0172 | Cool-Surge Breakdown and Rebound-Warming Specialist | P1 | Promotion-oriented specialist | Deployable now | 0136, 0120, 0171 |
| 0173 | Humid Maritime Suppression versus Cloud-Break Warming Specialist | P1 | Promotion-oriented specialist | Safe proxy deployable; blocked teachers diagnostic-only | 0149, 0150, 0171, 0135 |
| 0174 | Weak-Wind Heat-Buildup and Dry-Subsidence Specialist | P1 | Promotion-oriented specialist | Deployable now from safe proxies | 0141, 0142, 0146, 0153, 0155 |
| 0175 | Marine Suppression versus Inland Heat-Potential Duel Specialist | P1 | Promotion-oriented specialist | Deployable now from station/geospatial proxies | 0135, 0141, 0143, 0158 |
| 0176 | TC Quadrant Subsidence–Cloud Transition Specialist | P2 | Exploratory tail specialist | Safe student only; best track diagnostic | 0159, 0173, 0174 |
| 0177 | Extreme-Error Precursor and Catastrophic-Miss Prevention Model | P1 | Tail-risk promotion | Deployable now with safe features | 0113, 0121, 0167, 0171–0176 |
| 0178 | Signed Hot-Underforecast and Cold-Overforecast Action Specialists | P1 | Promotion-oriented tail specialists | Deployable now | 0113, 0117, 0171–0177 |
| 0179 | High-Frequency Morning-Heating Curve Shape Atlas | P1 later-track | Short-history diagnostic / prospective layer | Pre-2024 development only; 2024+ sealed | 0107, high-frequency path manifest, 0108 |
| 0180 | High-Frequency Spatial Propagation, Convergence, and Boundary Timing | P1 later-track | Short-history diagnostic / prospective layer | Pre-2024 development only | 0179, high-frequency wind/pressure/humidity feeds |
| 0181 | Since-Midnight Tmax Trajectory, Ceiling, and Remaining-Upside Model | P1 later-track | Short-history live/prospective research | Pre-2024 development only; operational timing must match market/forecast cutoff | 0179, since-midnight archive |
| 0182 | UV–Solar–Humidity–Wind Cloud-Suppression and Cloud-Break Proxy | P1 later-track | Short-history diagnostic / prospective layer | Pre-2024 development only | 0149, 0150, 0179 |
| 0183 | Short-History Teacher to Long-History Deployable Feature Distillation | P1 synthesis | Diagnostic-to-deployable bridge | Students deployable only if built from long-history safe inputs | 0179–0182 plus long-history proxy experiments |


## Appendix E. External meteorological cross-check used for hypothesis discipline

The experiment directions were cross-checked against authoritative source descriptions rather than generic modelling intuition:

- Hong Kong Observatory climate material describes the humid/fog-prone spring transition, hot humid May–August conditions, and the potential for exceptionally hot conditions under certain tropical-cyclone positions. This supports separate MAM phase, radiative-suppression, cloud-break, and TC subsidence/cloud-shield hypotheses.
- HKO automatic-weather-station documentation confirms a network with temperature, humidity, wind, pressure, rainfall and related observations, supporting the spatial-sensor-array treatment while still requiring station-specific metadata and exposure controls.
- HKO educational/case material on sea breezes and monsoon/cold-front changes supports convergence, coastal penetration, pressure-wave, wind-shift, and cool-surge experiments rather than simple station means.
- NOAA’s ISD documentation supports the breadth of surface variables and report/QC metadata, while the repository’s timestamp policy correctly remains more conservative than merely knowing the archive contains those values.

These sources support physical hypotheses only. They do not override the repository’s exact point-in-time eligibility rules or prove a feature will improve MAE.

## Appendix F. Artifact acceptance checklist for every future experiment

- [ ] Exact experiment folder under `experiments/` with immutable ID and title.
- [ ] `README.md` states purpose, hypothesis, prior related folders, and why the work is non-duplicate.
- [ ] Input paths and cryptographic hashes recorded.
- [ ] Data range, row count, coverage, missingness, station/source counts recorded.
- [ ] Feature formulas, units, lag, valid time, available-at rule, QC and fallback recorded.
- [ ] 2024+ confirmation access explicitly false.
- [ ] Fold geometry and all training-only transformations documented.
- [ ] Baselines include raw official where available, global/source bias, online memory, and simpler parent feature/model.
- [ ] MAE, RMSE, bias, median AE, P80/P90/P95, >2°C, >3°C and signed tails reported.
- [ ] Year, season, month, source, era, residual sign, high-error and late-window stability reported.
- [ ] Multiple-testing and temporal block confidence included for atlas work.
- [ ] Row-level OOF predictions included for model work.
- [ ] Negative results and failed candidates preserved, not overwritten.
- [ ] Acceptance/rejection decision uses predeclared gates.
- [ ] Next recommendation is executable and does not open confirmation data.

## Final research-director conclusion

The repository is not short of variables; it is short of a unified way to identify **orthogonal, causal, response-specific information** and promote it without frame confusion. The strongest current evidence says:

1. The official forecast remains the anchor to beat.
2. Online source-aware residual memory is the most credible proven correction mechanism, but it must be replayed on the expanded frame.
3. Station data is most promising as a spatial regime detector, residual corrector, and uncertainty scaler—not as a stand-alone replacement.
4. Target-memory shape, volatility, and transition state are safe and useful, especially when the station network confirms or contradicts them.
5. Upper-air, daily-climate, marine and TC archives contain valuable physical information but remain diagnostic until timestamp proof or safe proxy conversion succeeds.
6. The next competitive edge is likely a collection of small, bounded, physically justified actions with aggressive abstention—not one giant model trained on everything.

The program above is intentionally aggressive in signal discovery and conservative in claims. It provides enough breadth to probe every acquired data family, but enough statistical and leakage discipline to prevent a 150,000-character wishlist from becoming 150,000 characters of overfitting.
