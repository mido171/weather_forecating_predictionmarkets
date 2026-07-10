# Reproduce

```powershell
cd C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py
```

Optional:

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py --database-url postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research
```
