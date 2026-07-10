# Database Usage and Layer Guide

The audit corpus was loaded into the local research database after the 2026-06-23 audit. The ingestion is intended for queryability, provenance, and safe feature selection, not for blindly treating every raw source as model-ready.

## Connection

```powershell
$env:PGPASSWORD="root"; & "C:\Program Files\PostgreSQL\16\bin\psql.exe" -h 127.0.0.1 -p 5432 -U postgres -d hkg_tmax_research
```

## Ingestion status

| Metric | Value |
| --- | --- |
| status | PASS |
| database_engine | postgresql |
| migration_version | 20260623_0001_audit_driven_ingestion |
| ingestion_batch_id | audit-ingest-bdbc1fce90c0-primary |
| datasets_accounted | 13 |
| tables_accounted | 52 |
| attributes_accounted | 1869 |
| quality_issues_accounted | 22 |
| rows_quarantined | 0 |
| objects_registered | 10 |
| duplicate_formats_skipped | 2 |
| strict_validation_passed | True |
| idempotency_passed | True |

## Rows loaded by layer

| db_layer | rows_loaded |
| --- | --- |
| acquisition_provenance | 522 |
| acquisition_quality | 93 |
| catalog | 39 |
| diagnostic_physics | 1122320 |
| diagnostic_regime_labels | 26189 |
| diagnostic_station_network | 4346780 |
| label_core | 48577 |
| live_exact_vintage | 278 |
| live_nwp_anchor | 530 |
| operational_anchor | 68513 |
| operational_archive_normalized | 452668 |
| operational_archive_raw | 486497 |
| quality_monitoring | 0 |
| raw_audit | 49628 |
| research_metrics | 372 |
| research_oof_predictions | 231564 |
| research_supervised | 5265 |
| sealed_confirmation | 882 |

## Recommended query starting points

| Schema/table or view | Use |
| --- | --- |
| feature_safe.hko_t24_official_anchor | Leakage-controlled official forecast anchor rows for T-24 style workflows. |
| feature_safe.hko_target_history_pre2024 | Pre-2024 official Tmax target history suitable for training labels and lagged target-memory features. |
| label_core.hko_daily_tmax | Canonical pre-2024 label table plus metadata. |
| sealed_confirmation.hko_daily_tmax | 2024+ holdout/confirmation labels; keep sealed from model training. |
| catalog.* | Dataset, file, attribute, profile, and station metadata. |
| governance.quality_issue | Open data quality blockers and required remediation actions. |
| ingestion.* | Batch, file-result, reconciliation, and row-rejection evidence. |

## Example model-safe join

```sql
select
  a.target_date,
  a.official_tmin_c,
  a.official_tmax_c,
  y.target_tmax_c as observed_tmax_c,
  a.source_product,
  a.issue_time_utc,
  a.available_at_utc
from feature_safe.hko_t24_official_anchor a
join feature_safe.hko_target_history_pre2024 y
  on y.local_date = a.target_date
where a.target_date < date '2024-01-01';
```

## Guardrails

- Do not train from `sealed_confirmation.*` unless explicitly running a holdout/confirmation evaluation.
- Do not promote raw diagnostic tables to operational predictors until their quality issues and point-in-time availability contracts are resolved.
- Treat object/catalog payloads as registered assets unless a parser-specific table exists.
- Use `governance.quality_issue` before adding any new feature family.
