# Full Attribute Dictionary

`ATTRIBUTE_DICTIONARY_FULL.csv` is the complete machine-readable attribute dictionary for all 1,869 audited attributes. The per-dataset Markdown files under `datasets/` render the same attribute evidence in human-readable form.

Important columns:

| Column | Meaning |
| --- | --- |
| dataset_id | Owning dataset. |
| source_file | Physical source file under data/datasets. |
| attribute | Source attribute name. |
| source_dtype | Observed source dtype during audit. |
| semantic_class | Audited interpretation of the field's nature. |
| row_count/non_null_count/null_count/null_pct | Coverage and missingness. |
| profile_min/profile_max | Observed min/max where a value range was available. |
| storage_decision | Whether/how the field should be stored. |
| db_layer | Recommended database layer. |
| model_role | Allowed or disallowed modeling role. |
| operational_status | Operational eligibility status. |
| quality_action | Cleanup/audit action required before use. |
| usefulness_score_0_100 | Audit usefulness score for research or operations. |
| rationale | Reasoning behind the attribute decision. |
