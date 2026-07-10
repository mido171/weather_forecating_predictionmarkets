# Station Network Atlas Protocol

## Purpose

Treat every surrounding station as one sensor in a spatial array. Do not reduce the array to an unqualified mean, and do not interpret anonymous station IDs without verified metadata.

## Discovery and identity

The station list must be discovered from repository files and reconciled with metadata. The bundled seed list is a recall aid, not a closed universe. For each raw ID:

1. find every appearance in datasets, feature columns, and experiment artifacts;
2. locate station metadata;
3. reconcile aliases;
4. verify coordinates and elevation;
5. calculate distance and bearing from HKO;
6. assign role labels only when evidence supports them;
7. quarantine contradictory metadata.

Station `450090-99999` has appeared with suspicious coordinate evidence in prior repository material and must remain quarantined until verified.

## Mandatory dossier columns

`station_id`, `aliases`, `station_name`, `name_source`, `latitude`, `longitude`, `coordinate_source`, `elevation_m`, `distance_to_hko_km`, `bearing_from_hko_deg`, `distance_to_coast_km`, `role_labels`, `role_confidence`, `first_date`, `last_date`, `variables`, `coverage_rate`, `longest_gap_days`, `quality_flags`, `known_experiments`, `eligibility`, `notes`.

## Per-station feature families

For every eligible variable at every station, consider:

- latest safe level before cutoff;
- previous-day or previous-cycle level;
- one-day and multi-day change;
- short and long slope;
- acceleration and reversal;
- anomaly versus station rolling baseline;
- anomaly versus causal day-of-year baseline;
- percentile/rank versus own history;
- rank among contemporaneous peers;
- rolling volatility and range;
- missingness/quality state;
- persistence and sign streak.

Variables may include temperature, dew point, pressure, wind direction/speed/gust, visibility, precipitation, cloud/weather codes, and other actual catalog fields.

## Pairwise feature families

For every physically or statistically justified pair:

- temperature spread and spread change;
- anomaly spread;
- dew-point spread and change;
- temperature-dew-point-spread contrast;
- pressure difference and tendency difference;
- wind-vector difference;
- circular wind-direction difference;
- warming-slope difference;
- rank reversal;
- missingness asymmetry;
- distance-normalized gradient;
- projection along HKO bearing or prevailing wind.

Do not blindly promote all pairs. A broad atlas may screen them under temporal folds and support thresholds. Follow-up experiments must pre-register specific pairs or role abstractions.

## Station groups

Build groups from verified metadata and test sensitivity:

- coastal;
- inland;
- urban;
- open-exposure or airport;
- marine/island;
- northern;
- southern;
- eastern;
- western;
- elevated/hill;
- distance rings from HKO;
- data-driven clusters fitted on training history only.

For each group, derive robust mean, median, trimmed mean, range, IQR, warmest/coolest rank, fraction warming, moisture state, pressure tendency, and wind vector.

## Spatial modes

Candidate spatial summaries:

- north-south temperature/dew-point/pressure gradient;
- east-west gradient;
- coastal-inland spread;
- urban-open spread;
- marine-inland spread;
- HKO-centered radial gradient;
- station disagreement index;
- first few PCA modes;
- graph Laplacian modes;
- kriged or inverse-distance anomaly estimates;
- upwind-weighted state.

PCA, graph weights learned from data, and cluster assignments must be fitted within prior/fold-training data. Static geometry can be derived once from versioned metadata.

## Wind-conditioned relevance

Convert wind direction to `u` and `v` components. Define onshore/offshore and sector states using verified geography. Candidate logic:

- select or weight stations upwind of HKO;
- compare upwind and downwind anomalies;
- detect onshore marine penetration;
- detect northerly cool-surge propagation;
- detect westerly or downslope heating;
- evaluate wind persistence and shifts;
- examine mismatch between station wind fields.

Use the wind source available at cutoff. Do not select upwind stations using target-day later winds.

## Propagation experiments

Test whether regional state changes lead HKO target or residual:

- temperature anomaly arrival;
- dew-point surge;
- pressure rise/fall;
- wind shift;
- rank reversal;
- station group transition.

Use predeclared lags, exact timestamps, and temporal folds. Record whether propagation speed/direction is stable by season. Do not use full-history cross-correlation to choose a lag and score on the same data without nested validation.

## Contribution leaderboards

Produce separate station contribution rankings for:

- raw target Tmax;
- official residual;
- absolute official error;
- hot underforecast;
- cold overforecast;
- MAM high error;
- station-only residual;
- online-memory residual;
- uncertainty/trust.

Stratify by season, month, source, pressure state, moisture state, wind state, target-memory state, and error sign where support allows. Every leaderboard entry must report sample size, date range, fold consistency, effect size, and eligibility.

## Simplification and robustness

For each promising named station:

- leave it out;
- replace it with a role-group aggregate;
- replace it with nearest similar station;
- test station outage behavior;
- test era coverage;
- test whether identity remains meaningful after anomaly normalization.

Prefer stable group mechanisms over fragile station quirks unless the unique station role is verified and operationally reliable.

## Minimum support

Defaults unless the experiment pre-registers stricter rules:

- global univariate screen: at least 500 common rows;
- station-season signal: at least 250 rows;
- two-feature cell: at least 200 rows per promoted cell and at least three temporal folds;
- rare high-error specialist: at least 100 activations overall and 20 in each of three folds;
- station pair promotion: meaningful coverage in at least four years and no single year contributing more than 35% of activations.

Exploratory results below these thresholds must be labeled as such.

## Output artifacts

Every station-network atlas experiment should save:

- `station_dossier_snapshot.csv`
- `station_variable_coverage.csv`
- `station_contribution_leaderboard.csv`
- `station_pair_leaderboard.csv`
- `station_group_features.csv`
- `interaction_cells.csv`
- `fold_stability.csv`
- `station_dropout_ablation.csv`
- `metadata_conflicts.csv`
- `physical_annotations.md`
