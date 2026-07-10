# 00 Foundation Universal Ingestion Contract And Availability Ledger

Source spec:

```text
00_universal_ingestion_contract_and_availability_ledger.md
```

Execution role:

This is the first task because every provider ingest must share the same availability, as-of, lineage, raw-response, checksum, and source-gap conventions before data starts landing in PostgreSQL.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Use this task to create the shared ingestion contract tables and common metadata rules before any provider-specific backfill.
