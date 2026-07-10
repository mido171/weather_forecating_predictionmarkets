# Raw Archive Audit

- status: `PASS`
- data root: `C:\hkg_tmax_data`
- retrieval ledger rows: `11,025`
- successful retrieval rows: `11,023`
- failed retrieval rows: `2`
- unique successful content hashes: `10,483`
- file manifest rows: `10,483`
- dataset lineage rows: `11,023`
- audited unique content objects: `10,483`
- errors: `0`

## Verified

- every successful ledger row points to an existing content object;
- every audited content object hash matches its digest filename and ledger hash;
- every successful ledger row length matches the object length;
- every successful ledger row has a metadata sidecar with HTTP metadata;
- file manifest hashes match successful ledger hashes;
- dataset lineage covers every successful ledger row.
