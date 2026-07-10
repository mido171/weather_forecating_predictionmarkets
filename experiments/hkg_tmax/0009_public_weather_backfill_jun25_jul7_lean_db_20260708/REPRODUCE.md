# Reproduce

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local postgres url>'
.\.venv\Scripts\python.exe scripts\backfill_public_weather_to_postgres.py --start-date 2026-07-07 --end-date 2026-07-07
```
