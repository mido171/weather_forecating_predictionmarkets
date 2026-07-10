# Protocol

For each UTC date, construct 68 GFS issues, 68 GEFS-control issues, and 144 Himawari B13/S0510
scan issues. For each issue, download only the configured source messages/segment, validate the
payload, normalize HKO station and area features, write source metadata and features to Postgres,
commit, and then delete the raw payload. Every feature must join to a source issue with non-null
`available_at_utc`.

Optimized mode uses bounded fetch/normalize workers, keeps DB writes serialized in the main
process, and deletes each raw payload only after the DB commit or recorded failure handling.

Validation sequence:

1. Dry-run one-day inventory.
2. One-day optimized live smoke.
3. Idempotency rerun with completed-issue skipping.
4. Seven-day two-worker rehearsal.
5. Corrected 29-day robustness run.
6. Targeted retries for transient model failures.
7. DB audit for issue inventory, feature presence, leakage timestamps, duplicates, and errors.
8. Raw residue and staging audit.
