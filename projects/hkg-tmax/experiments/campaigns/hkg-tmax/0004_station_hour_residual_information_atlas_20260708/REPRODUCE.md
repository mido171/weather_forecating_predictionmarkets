# Reproduce

```powershell
cd C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_0004_station_hour_residual_information_atlas.py
```

The runner creates temporary Postgres tables and writes local experiment artifacts. It does not mutate persistent database tables.
