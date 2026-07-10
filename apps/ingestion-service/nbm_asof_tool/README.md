# NBM As-Of Daily Tmax Fetcher

This tool fetches KMIA (Miami International) NBM "as-of" daily max temperature forecasts.

## Dependencies (conda-forge recommended)
```
conda create -n nbm_fetch python=3.11 -y
conda activate nbm_fetch
conda install -c conda-forge herbie-data cfgrib eccodes xarray pandas numpy scikit-learn tqdm -y
```

## Run
From this directory:
```
python fetch_nbm_tmax_asof.py
```

Defaults:
- Targets 2026-07-01 through 2026-07-10.
- Uses as-of (T-1) 12Z per target date.
- Outputs `nbm_kmia_tmax_asof_20260701_20260710.csv`.

## Notes
- The script is resilient to missing data and will output status flags.
- For future dates, missing data is expected until the NBM runs exist.
