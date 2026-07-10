# NCEP Operational GRIB Inventory

## Dataset identity and use

| Field | Value |
| --- | --- |
| dataset_id | 10_ncep_operational_grib_inventory |
| recommended_db_inclusion | YES-object metadata only until decoded |
| recommended_layer | nwp_object_catalog |
| current_operational_predictor_value_0_100 | 0 |
| diagnostic_research_value_0_100 | 15 |
| future_potential_0_100 | 100 |
| verdict | Very high future potential; current table is only an incomplete inventory with empty cycle fields. |
| source_tables_or_files | 1 |
| audited_attributes | 12 |
| profiled_rows_across_files | 3400 |
| data_min |  |
| data_max |  |

## Source tables/files

| source_file | type | rows | attributes | data_min | data_max | db_action | db_layer | model_status | priority | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | parquet | 3400 | 12 |  |  | REGISTER_OBJECT_METADATA_AND_REBUILD_INVENTORY | nwp_object_catalog | NOT_YET_DECODED | CRITICAL | Cycle date/hour are empty; decode meteorological fields before modelling. |

## Attribute nature summary

### Semantic classes

| semantic_class | attribute_count |
| --- | --- |
| boolean | 1 |
| datetime_or_date | 1 |
| free_natural_text | 1 |
| hash_identifier | 1 |
| identifier_or_code | 1 |
| nominal_text | 4 |
| numeric | 2 |
| path_or_file_reference | 1 |

### Model roles

| model_role | attribute_count |
| --- | --- |
| FILE_QC | 1 |
| LEAKAGE_POLICY_METADATA | 3 |
| MODEL_CYCLE_TIME | 2 |
| NWP_LEAD_OR_ENSEMBLE_MEMBER | 2 |
| OBJECT_OR_SOURCE_REFERENCE | 1 |
| PROVENANCE_ONLY | 1 |
| PROVENANCE_TIMESTAMP | 1 |
| SOURCE_OR_CATALOG_METADATA | 1 |

### Operational status

| operational_status | attribute_count |
| --- | --- |
| CONDITIONAL_CATEGORICAL_ONLY | 1 |
| FUTURE_PREDICTOR_CONTEXT | 2 |
| GATING_ONLY | 3 |
| GATING_OR_AUDIT_ONLY | 1 |
| NOT_A_PREDICTOR | 2 |
| QUALITY_ONLY | 1 |
| REQUIRED_BEFORE_ANY_USE | 2 |

### Storage decisions

| storage_decision | attribute_count |
| --- | --- |
| KEEP | 9 |
| KEEP_REFERENCE_ONLY | 1 |
| REBUILD_AND_POPULATE | 2 |

## Dataset-specific quality issues

| severity | source_table | attributes | evidence | required_action |
| --- | --- | --- | --- | --- |
| CRITICAL | ncep_operational_grib2_inventory.parquet | cycle_date; cycle_hour_utc | Both cycle fields are empty for all 3,400 rows. | Rebuild inventory from filename and GRIB metadata with cycle_time_utc and valid_time_utc before any modelling. |

## Complete audited attribute dictionary

This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.

| source_file | attribute | source_dtype | semantic_class | row_count | non_null_count | null_count | null_pct | profile_min | profile_max | storage_decision | db_layer | model_role | operational_status | quality_action | usefulness_score_0_100 | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | source_id | string | identifier_or_code | 3400 | 3400 | 0 | 0.0 |  |  | KEEP | nwp_object_catalog | SOURCE_OR_CATALOG_METADATA | CONDITIONAL_CATEGORICAL_ONLY | NONE | 15 | Useful for lineage/source-era grouping, not meteorology by itself. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | content_sha256 | string | hash_identifier | 3400 | 3400 | 0 | 0.0 |  |  | KEEP | nwp_object_catalog | PROVENANCE_ONLY | NOT_A_PREDICTOR | NONE | 0 | Integrity/deduplication key; essential for provenance, not forecasting. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | raw_retrieved_at_utc | string | datetime_or_date | 3400 | 3400 | 0 | 0.0 | 2026-06-19T06:49:13.626499+00:00 | 2026-06-20T06:25:56.077690+00:00 | KEEP | nwp_object_catalog | PROVENANCE_TIMESTAMP | GATING_OR_AUDIT_ONLY | VALIDATE_TIMEZONE | 95 | Needed to prove retrieval/availability and reproduce vintages; not a weather variable. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | cycle_date | string | nominal_text | 3400 | 3400 | 0 | 0.0 |  |  | REBUILD_AND_POPULATE | nwp_object_catalog | MODEL_CYCLE_TIME | REQUIRED_BEFORE_ANY_USE | CURRENTLY_EMPTY_PARSE_FROM_FILENAME_OR_GRIB_METADATA | 100 | No NWP field can be used safely without cycle time. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | cycle_hour_utc | string | nominal_text | 3400 | 3400 | 0 | 0.0 |  |  | REBUILD_AND_POPULATE | nwp_object_catalog | MODEL_CYCLE_TIME | REQUIRED_BEFORE_ANY_USE | CURRENTLY_EMPTY_PARSE_FROM_FILENAME_OR_GRIB_METADATA | 100 | No NWP field can be used safely without cycle time. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | forecast_hour | int64 | numeric | 3400 | 3400 | 0 | 0.0 | 6.0 | 120.0 | KEEP | nwp_object_catalog | NWP_LEAD_OR_ENSEMBLE_MEMBER | FUTURE_PREDICTOR_CONTEXT | DERIVE_VALID_TIME_AND_VERIFY_MEMBER_SET | 85 | Essential for lead alignment and ensemble statistics. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | member | string | nominal_text | 3400 | 3400 | 0 | 0.0 |  |  | KEEP | nwp_object_catalog | NWP_LEAD_OR_ENSEMBLE_MEMBER | FUTURE_PREDICTOR_CONTEXT | DERIVE_VALID_TIME_AND_VERIFY_MEMBER_SET | 85 | Essential for lead alignment and ensemble statistics. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | content_length | int64 | numeric | 3400 | 3400 | 0 | 0.0 | 7406.0 | 38750.0 | KEEP | nwp_object_catalog | FILE_QC | QUALITY_ONLY | VERIFY_SUBSET_CONTENT_AND_TRUNCATION | 25 | Small files require verification that intended variables are present. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | content_path | string | path_or_file_reference | 3400 | 3400 | 0 | 0.0 |  |  | KEEP_REFERENCE_ONLY | nwp_object_catalog | OBJECT_OR_SOURCE_REFERENCE | NOT_A_PREDICTOR | REPLACE_MACHINE_LOCAL_PATH_WITH_OBJECT_URI | 5 | Keep as lineage/object reference; avoid machine-specific absolute paths in canonical DB. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | availability_tier | string | nominal_text | 3400 | 3400 | 0 | 0.0 |  |  | KEEP | nwp_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | operational_input_allowed | bool | boolean | 3400 | 3400 | 0 | 0.0 |  |  | KEEP | nwp_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
| 10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet | source_time_policy | string | free_natural_text | 3400 | 3400 | 0 | 0.0 |  |  | KEEP | nwp_object_catalog | LEAKAGE_POLICY_METADATA | GATING_ONLY | NONE | 100 | Critical eligibility gate; never feed blindly as a meteorological predictor. |
