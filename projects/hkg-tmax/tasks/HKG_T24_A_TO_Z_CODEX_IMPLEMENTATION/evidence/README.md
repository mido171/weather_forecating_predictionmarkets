# HKG T+24 Tmax Dataset Audit Bundle

This bundle is the complete disposition of the supplied metadata profile.

## Contents

- `HKG_TMAX_DATASET_DB_AND_MODEL_VALUE_AUDIT.md` — human-readable full audit, including all 52 table decisions and all 1,869 attribute decisions.
- `HKG_TMAX_DATASET_DECISION_MATRIX.csv` — one row per dataset.
- `HKG_TMAX_TABLE_DECISIONS_ALL_52.csv` — one row per profiled table/file.
- `HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv` — one row per profiled attribute.
- `HKG_TMAX_DATA_QUALITY_ISSUES.csv` — 22 identified blockers/quality issues.
- `HKG_TMAX_ISD_STATION_DOSSIER_36.csv` — all 36 ISD stations mapped and physically classified.
- `HKG_TMAX_DB_SCHEMA_BLUEPRINT.sql` — PostgreSQL reference architecture enforcing data-layer and leakage separation.
- `AUDIT_SUMMARY.json` — counts, hashes and headline decision.

## Headline

Preserve almost every source, but separate operational, diagnostic, live, research, object and quarantine layers. Do not promote data merely because it exists in a retrospective archive.

Report size: 916,916 characters.
