# Canonical Dataset Family Atlas

## How to use this atlas

This atlas enumerates every currently known source family and the attributes the Director must look for. The repository census must still discover the exact file paths and columns. A listed attribute is a search obligation, not proof that the repository contains it or that it is operationally eligible.

For every family, determine:

- exact normalized and raw paths;
- date range;
- cadence;
- stations or spatial coverage;
- exact columns;
- units and quality flags;
- valid, issue, available-at, retrieval, and processing timestamps;
- point-in-time eligibility;
- current experiments using it;
- untested transformations and response roles.

## 1. HKO target daily Tmax history

Primary role:

- target label;
- causal target-memory predictors using only T−1 or earlier;
- climatology and regime-state construction;
- residual and error labels after the target resolves.

Attributes to inventory:

- target local date;
- target Tmax in °C;
- station identifier and target mapping;
- source/version;
- quality or missing flags;
- publication/revision metadata if present.

Required derived opportunities:

- lags 1, 2, 3, 7, 14, 30, 60, 90, 365 and physically justified alternatives;
- prior-only rolling means, medians, minima, maxima, ranges;
- rolling standard deviation, MAD, IQR;
- short/long slopes and slope contrast;
- acceleration and reversal;
- anomaly versus causal day-of-year climatology;
- year-over-year analogs;
- hot/cold spell duration;
- envelope breakout;
- volatility compression/expansion;
- intra-season percentile;
- transition and persistence states;
- climate trend and era drift using causal fitting.

Target-day T values are never predictors.

## 2. HKO daily climate elements

Known broad categories:

- temperature elements other than target Tmax;
- mean and minimum temperature;
- pressure;
- rainfall and rainy-day indicators;
- relative humidity;
- wet-bulb or moisture elements;
- cloud amount;
- sunshine duration;
- solar radiation;
- sea temperature;
- wind;
- lightning;
- visibility;
- other daily climate observations in normalized files.

Required exact-column census:

- element name/code;
- value and unit;
- date;
- station/site;
- quality flags;
- final/revised status;
- publication or available-at timestamp if any.

Current default status:

- target-side or diagnostic-only unless a safe lag and publication contract is proven;
- finalized historical daily values do not establish T−24 availability.

Research roles:

- long-history physical diagnostics;
- lagged climate-memory features when proof exists;
- latent mechanism targets for proxy conversion;
- seasonal and regime labels;
- comparison against safe station proxies.

## 3. HKO official RSS forecast archive

Attributes to inventory:

- target date;
- issue/published time in HKT;
- available-at time;
- source/product identifier;
- forecast maximum and minimum;
- range;
- forecast text;
- parsed weather, wind, rain, cloud, temperature, and confidence concepts;
- parser status and null fields;
- revision/vintage sequence;
- retrieval and archive metadata.

Eligibility:

- deployable where exact vintage is available before cutoff and target mapping is valid.

Research roles:

- official anchor;
- residual labels;
- source-aware online memory;
- forecast range and trust;
- revision momentum;
- numeric-text contradiction;
- staleness;
- source-era calibration;
- router and uncertainty features.

Known current scoreable RSS history is shorter than target/station history; the latest documented current slice was 2021-04-14 through 2023-12-31 with 992 rows in the 0103 frame.

## 4. HKO historical press forecast archive

Attributes to inventory:

- article/product date;
- exact issue or publication timestamp;
- target date resolved from text;
- forecast min/max;
- weather text;
- wind and rain wording;
- source/product era;
- parser confidence;
- null or ambiguous fields;
- duplicate or revised forecasts;
- archive retrieval metadata.

Eligibility:

- deployable only when issue time is before cutoff and parser mapping is validated.

Research roles:

- older official anchor;
- source-era residual memory;
- press-to-RSS bridge;
- text ontology;
- parser uncertainty;
- archive continuity.

Known clean coverage is approximately 2000–2011, with a major current gap between press and RSS eras.

## 5. NOAA ISD regional surface stations

Raw/normalized attributes to search:

- USAF/WBAN or canonical station ID;
- station name and metadata;
- latitude, longitude, elevation;
- observation valid time;
- air temperature;
- dew point;
- sea-level pressure;
- station pressure when present;
- wind direction;
- wind speed;
- wind gust;
- visibility;
- precipitation fields;
- cloud or sky-condition fields;
- weather codes;
- quality-control flags;
- report type;
- source and processing metadata.

Derived opportunities:

- latest-before-cutoff values;
- pre-cutoff summaries;
- one-day and multi-day changes;
- station-specific anomalies;
- seasonal anomalies;
- temperature-dew-point spread;
- wet-bulb/enthalpy proxies;
- pressure tendency;
- wind vector and persistence;
- station rank;
- pairwise gradients;
- coastal-inland and directional group spreads;
- pressure/moisture fronts;
- upwind station selection;
- graph or principal modes fitted causally;
- missingness and coverage state.

Current eligibility warning:

- quality-controlled retrospective ISD is not automatically exact operational vintage. Promotion requires release-latency proof or a validated lag contract. Otherwise use as diagnostic/proxy research.

## 6. NOAA IGRA upper-air archive for HKM00045004

Attributes to inventory by launch/profile/level:

- station ID;
- launch or observation time;
- valid time;
- pressure level;
- geopotential height;
- temperature;
- dew-point depression or moisture;
- wind direction;
- wind speed;
- reported/derived level type;
- quality flags;
- source/release metadata.

Derived diagnostic opportunities:

- 1000/925/850/700/500 hPa levels;
- layer thickness;
- lower-tropospheric mean temperature;
- lapse rate and inversion strength;
- moisture profile;
- precipitable-water proxy;
- stability indices;
- vertical wind shear;
- ridge/subsidence proxies;
- mixing-depth proxies;
- launch-to-launch changes.

Current status:

- diagnostic-only unless provider release/available-at latency is proven for all scored rows. The 0102 audit reportedly unlocked zero blocked upper-air features.

Primary value:

- identify physical mechanisms;
- define safe surface or operational-NWP proxy targets;
- evaluate whether blocked signals explain residuals beyond surface features.

## 7. HKO tropical cyclone best track

Attributes to search:

- observation/advisory time;
- cyclone identifier and name;
- latitude and longitude;
- central pressure;
- maximum sustained wind/intensity;
- motion direction and speed;
- classification;
- revision or best-track status.

Derived diagnostic opportunities:

- distance and bearing from HKO;
- quadrant;
- intensity;
- motion vector;
- closest approach timing;
- lagged effects;
- interaction with wind, cloud/rain proxies, and pressure.

Current status:

- retrospective best track is diagnostic-only for live prediction unless an operational advisory vintage is separately available and proven. Do not use finalized path/intensity as if known at T−24.

## 8. Radar, satellite, lightning, and nowcast sources

Attributes to search based on actual files:

- product type;
- image or grid timestamp;
- issue and available-at time;
- spatial domain and resolution;
- reflectivity/rain estimates;
- cloud-top or infrared proxies;
- lightning count/location/intensity;
- motion vectors;
- nowcast lead and valid time;
- retrieval success and missingness.

Current role:

- primarily prospective/live and short-history;
- physical feature design;
- recent-period diagnostics;
- future cloud/rain suppression and uncertainty layer.

Promotion requires exact historical vintage or prospective collection with enough validation history.

## 9. HKO marine, tide, and coastal-waters data

Attributes to search:

- site/station;
- observation time;
- sea temperature;
- tide level and phase;
- wave height/period/direction;
- current;
- water level;
- coastal wind;
- visibility or weather;
- issue/publication/available-at metadata;
- quality flags.

Potential mechanisms:

- marine thermal suppression;
- sea-breeze potential;
- coastal boundary-layer moisture;
- tide or coastal-flow modulation where physically plausible.

Current status:

- long historical finalized values without first-publication proof remain diagnostic-only;
- short live feeds are prospective once retrieval is instrumented.

Safe proxies include coastal-inland station spread, coastal dew-point gradient, and onshore wind.

## 10. HKO ARWF station forecasts

Attributes to inventory:

- model cycle/issue time;
- available-at time;
- station;
- lead and valid time;
- forecast temperature, humidity, wind, rainfall, cloud, or pressure fields;
- ensemble/member if present;
- version and retrieval metadata.

Potential role:

- future operational anchor or expert;
- station-network forecast gradients;
- model disagreement;
- uncertainty and routing.

Current limitation:

- minimal back-history. Do not treat prospective records as long-history evidence.

## 11. NCEP operational inventory/subsets

Inventory attributes:

- model/product;
- GRIB file;
- cycle/issue time;
- forecast lead;
- valid time;
- variable;
- level;
- grid/domain/resolution;
- member;
- byte size;
- download/retrieval state;
- checksum.

Potential focused variables if approved:

- 2 m temperature/dew point;
- 10 m wind;
- surface pressure/mean sea-level pressure;
- total cloud;
- precipitation;
- boundary-layer height;
- solar flux;
- 925/850/700 hPa temperature, humidity, height, and wind.

Current status:

- inventory is not a decoded feature source. No experiment may use it until exact extraction, cycle eligibility, legal/byte policy, and point-in-time contract are approved.

## 12. Static geospatial context

Attributes to derive or inventory:

- station coordinates;
- elevation;
- HKO distance and bearing;
- pairwise distances and bearings;
- coastline distance and orientation;
- land/sea classification;
- terrain exposure;
- slope/aspect where available;
- urban/land-use context;
- airport/open exposure;
- island/coastal/inland/hill group;
- graph adjacency and weights.

Eligibility:

- deployable when derived deterministically from versioned static data and verified station mapping.

Role:

- structure station groups;
- define upwind/downwind relationships;
- construct graph modes;
- impose physical priors;
- improve transferability and station dropout robustness.

## 13. Robust experiment-output datasets

Known files include:

- expanded feature matrices;
- R14/R15/R16/R17 feature matrices;
- OOF predictions;
- scoreboards;
- fold deltas;
- feature diagnostics.

Known documented dimensions:

- expanded EXP0050–0099 matrix: 48,577 rows, 566 columns, 1884-01-01 through 2023-12-31;
- R14: 26,632 rows, 120 columns;
- R15/R17: 23,943 rows, 120 columns;
- R16: 25,202 rows, 120 columns;
- OOF predictions from 1965 through 2023 in these screens.

Role:

- prior evidence;
- fast diagnostic reuse;
- comparable replay when construction is safe;
- feature lineage and negative-result preservation.

Caution:

- derived matrices inherit source eligibility. A column is not deployable merely because it is in a feature matrix. Verify lineage, timestamp status, and fold construction.

## 14. HKO high-frequency station archives

Known downloaded feeds:

- 1-minute temperature;
- since-midnight max/min;
- 1-minute humidity;
- 15-minute UV;
- 1-minute pressure;
- 1-minute solar;
- 10-minute wind.

Known broad start periods:

- temperature/humidity around mid-2020;
- pressure/solar/wind around mid-2021;
- continuation into 2026 in the broader raw archive.

Attributes to inventory:

- station;
- observation time;
- value and unit;
- quality or missing state;
- retrieval time;
- since-midnight aggregation definition;
- exact cutoff coverage.

Potential role:

- recent/live morning heating curves;
- spatial propagation;
- cloud/solar suppression;
- wind shifts;
- remaining-upside models;
- short-history teacher models;
- prospective trading layer.

Restriction:

- do not force these into a multi-decade backbone. Validate on recent history separately, keep confirmation rules explicit, and avoid target-day observations that occur after the T−24 cutoff.

## 15. Experiment corpus

Every experiment folder is itself a data source for research decisions. Inventory:

- ID/folder;
- hypothesis;
- specification;
- data families;
- stations;
- attributes/features;
- response;
- frame;
- rows/date range;
- baseline/candidate;
- MAE/RMSE/bias/tails;
- folds/slices;
- leakage status;
- conclusion;
- promotion decision;
- negative evidence;
- code and prediction artifacts.

The corpus determines novelty and prevents duplicated work. It must be read in full before the Director selects a new lane.
