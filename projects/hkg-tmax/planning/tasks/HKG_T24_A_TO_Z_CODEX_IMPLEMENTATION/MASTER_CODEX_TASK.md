# Master Codex Orchestration Task

Read this package in the mandatory order from `START_HERE.md`. Execute the 40 tasks in `TASK_INDEX.csv` dependency order. Do not collapse the tasks into one undocumented script and do not skip a task because it appears administrative. Each task must create its own numbered folder under `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\experiments` and a passing `handoff_manifest.json` before dependent tasks begin.

Start immediately with T00. Launch T11 and T13 prospective collectors as early as their dependencies allow because their source history is short. Do not open 2024+ outcomes until T35 has frozen a candidate and T36 authorizes one-time access.

At the end, run:

```text
python tools/validate_bundle.py
```

Then create `FINAL_EXECUTION_STATUS.csv` with one row per task and link it from T39.
