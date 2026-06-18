# Reproduce

Do not reproduce EXP-0032 as a polling experiment. The work was superseded.

For future routine collection, use:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax acquisition run-due
```

That command uses `config/collector_schedules.yaml`, appends retrieval-ledger
rows, and deduplicates unchanged hashes without creating experiments.
