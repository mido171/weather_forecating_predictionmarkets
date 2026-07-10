# Incomplete historical backfill tools

These scripts are retained for provenance, not presented as runnable backfills.

- `fetch_rap_tmax.py` imports the absent tracked `rap.rap_sources` package. Its former
  `tools.rap_has_downloader` path also predates the canonical discovery-tool layout.
- `run_station_full_flow.py` orchestrates absent historical `ml` and backtesting scripts,
  along with paths from the pre-reorganization layout.

Do not execute these files. Promote required behavior into tested packages and rebuild a
bounded `tools/backfills` entrypoint before use.
