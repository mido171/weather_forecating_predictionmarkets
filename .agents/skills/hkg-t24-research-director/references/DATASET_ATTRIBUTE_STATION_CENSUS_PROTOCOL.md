# Dataset, Attribute, and Station Census Protocol

## Non-negotiable completeness standard

The Director may not claim that “all data,” “all attributes,” or “all stations” were considered until the current repository has been scanned and the artifacts in this protocol have been produced. Documentation is useful but not authoritative enough by itself. The actual files and schemas are the current evidence base.

The census must be refreshed when:

- a dataset is added or normalized;
- an archive backfill changes coverage;
- a parser changes;
- station metadata changes;
- a new experiment materializes derived features;
- a source timestamp status is upgraded;
- the current census is older than the newest dataset or experiment artifact.

## Required outputs

Write under `<repo-root>/.hkg_t24_research/census/`:

1. `repository_file_inventory.csv`
2. `dataset_file_inventory.csv`
3. `table_inventory.csv`
4. `attribute_catalog.csv`
5. `timestamp_field_catalog.csv`
6. `station_ids.csv`
7. `station_variable_coverage.csv`
8. `station_year_coverage.csv`
9. `station_dossier.csv`
10. `source_eligibility_matrix.csv`
11. `data_quality_findings.csv`
12. `unreadable_files.csv`
13. `census_summary.md`
14. `census_manifest.json`

## Repository file inventory

For every file beneath `data/datasets` and `experiments`, record:

- relative path;
- top-level source or experiment folder;
- extension;
- file size;
- modified time;
- whether raw, normalized, derived, prediction, score, documentation, or code;
- optional SHA-256 for manageable files;
- reason when hashing is skipped;
- whether the file was opened successfully.

Do not hash multi-gigabyte files by default if it creates unnecessary cost; record size and a sampled or metadata signature. Hash every specification, code file, score table, and small normalized artifact used in an experiment.

## Table inventory

For every readable tabular file, record:

- format;
- row count;
- column count;
- column names;
- partition information;
- apparent primary key;
- duplicate-key count;
- date and timestamp ranges;
- station count;
- null rate;
- sample method if full scan is impractical;
- parser and library used;
- schema warnings.

Supported formats should include CSV, CSV.GZ, Parquet, JSON, JSONL/NDJSON, and other project formats discovered during inventory. Unknown or unreadable formats go into `unreadable_files.csv`; they are not silently ignored.

## Attribute catalog

Create one row per source-file-column or normalized logical attribute. Required fields:

- source family;
- source ID;
- file path;
- table or partition;
- column name;
- normalized semantic name when known;
- data type;
- unit;
- cadence;
- station-scoped yes/no;
- target-derived yes/no;
- timestamp-derived yes/no;
- valid-time field;
- issue-time field;
- available-at field;
- retrieval-time field;
- quality-flag field;
- date range;
- non-null count and rate;
- unique count;
- minimum, maximum, mean, standard deviation, and selected quantiles for numeric data;
- most common values for categorical data;
- suspicious constants or sentinel values;
- current eligibility class;
- blocker;
- plausible meteorological mechanism;
- plausible response roles;
- transformations already tested;
- transformations not yet tested;
- experiment IDs using the attribute.

The exact repository columns must be enumerated. The canonical family atlas is an ontology and checklist, not a substitute for this actual column catalog.

## Timestamp census

Every time-like column must be classified as one of:

- target local date;
- observation valid time;
- model cycle or issue time;
- forecast valid time;
- publication time;
- available-at time;
- retrieval time;
- archive file date;
- processing time;
- unknown time.

Record timezone, naive/aware status, resolution, monotonicity, and relationship to the target cutoff. An attribute cannot receive `DEPLOYABLE_PROVEN` merely because a date column exists. The census must identify the field or audited latency contract that proves availability.

For daily aggregates, document the observation window and whether it includes target-day post-cutoff observations. For forecast archives, document vintage selection. For upper-air and model fields, separate cycle, valid time, and archive release. For station observations, record latest-before-cutoff aggregation rules.

## Station ID discovery

Do not begin with a hardcoded list only. Discover every station-like identifier from:

- station metadata files;
- ISD identifiers;
- HKO high-frequency station names or codes;
- feature column prefixes;
- experiment specifications;
- score and diagnostic files;
- static geospatial packages;
- source documentation.

Normalize aliases but preserve raw IDs. If two identifiers may represent the same station, record the mapping confidence and evidence. Do not merge without proof.

## Station-variable coverage

For each station and raw variable, record:

- first and last observation;
- total rows;
- expected rows based on cadence;
- observed coverage rate;
- coverage by year;
- null rate;
- duplicate timestamps;
- quality-flag counts;
- longest gap;
- suspicious constant runs;
- units;
- timezone;
- operational availability status.

Coverage must be computed for the exact pre-cutoff summaries used by experiments, not only raw files. A station with long daily coverage may still lack the required observation before the decision cutoff.

## Station dossier

Every station considered for a promoted feature must have:

- canonical station ID;
- all aliases;
- verified station name;
- latitude and longitude;
- elevation;
- distance and bearing from HKO;
- distance to coast if available;
- urban, coastal, inland, island, airport/open-exposure, hill, or unknown classification;
- source of each metadata field;
- date ranges by variable;
- missingness;
- likely meteorological role;
- role confidence;
- top prior experiment appearances;
- known data-quality concerns.

Unknown roles remain `UNKNOWN`. Never infer a station name or geography from ID alone.

## Data-quality checks

At minimum, test:

- duplicated primary keys;
- duplicated timestamps;
- non-monotonic sequences;
- timezone shifts;
- off-by-one target-date alignment;
- impossible Tmax/Tmin ordering;
- temperature unit mistakes;
- pressure unit mistakes;
- wind direction outside range;
- wind speed negatives;
- dew point materially above temperature;
- impossible humidity;
- coordinate changes;
- station-ID collisions;
- suspicious zeros or sentinel values;
- long constant runs;
- daily aggregates with incomplete source cadence;
- mixed target and confirmation rows;
- values generated after cutoff;
- revisions selected after cutoff;
- parser null clusters by source era.

A quality finding must state affected files, rows, dates, stations, attributes, severity, and whether prior experiments may be invalidated.

## Eligibility disposition

Each source and attribute receives one:

- `DEPLOYABLE_PROVEN`
- `DEPLOYABLE_LAGGED_ONLY`
- `DIAGNOSTIC_ONLY`
- `PROSPECTIVE_ONLY`
- `BLOCKED`
- `REJECTED`

The disposition must cite evidence. Unknown issue time or publication latency is not “probably safe.” It is diagnostic or blocked.

## Transformation opportunity census

For every eligible numeric attribute, consider and record disposition for:

- latest-before-cutoff level;
- one-step and multi-step lags;
- first difference;
- percentage or standardized change where meaningful;
- slope;
- acceleration or curvature;
- exponentially weighted level and change;
- rolling mean, median, minimum, maximum, range;
- standard deviation, MAD, IQR;
- anomaly versus recent baseline;
- anomaly versus causal seasonal baseline;
- quantile and rank;
- threshold exceedance with prior-derived threshold;
- spell duration;
- sign streak;
- reversal;
- missingness;
- interaction candidates.

For circular wind direction, use vectors and circular differences. For precipitation or event counts, consider occurrence, accumulation, recency, and dry/wet spell. For categorical text, use exact vintage, parser confidence, stable ontology, and fold-local encoding.

The census is not an instruction to fit every transformation. It is a requirement to explicitly consider, prioritize, or reject each family.

## Cross-source joins

Document all join keys and temporal alignment:

- target date versus observation date;
- HKT versus UTC;
- station-day versus target-day;
- forecast issue and target date;
- upper-air launch time and target horizon;
- source availability and cutoff;
- duplicate rows from multiple vintages.

Before experiments, verify join cardinality and row changes. Many-to-many joins are rejected unless explicitly intended and aggregated before scoring.

## Census acceptance criteria

The census passes only when:

- every file has an inventory disposition;
- every readable table has a schema record;
- every column has an attribute record;
- every station ID is cataloged;
- every promoted station has a dossier;
- every time field is classified;
- unreadable files are explicit;
- eligibility is evidence-backed;
- the manifest records tool versions and run time;
- no 2024+ target-dependent summaries are computed during development.

If the census cannot be completed, the Director must state which portions are incomplete and whether the omission could change experiment selection.
