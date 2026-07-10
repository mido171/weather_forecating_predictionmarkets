# KMIA As-Of Multi-Model Tmax Fetcher

This tool fetches KMIA daily Tmax forecasts using strict as-of runs:
- GFS and GEFS: (T-1) 12Z
- RAP: (T-1) 09Z (optional)
- HRRR: (T-1) 12Z (optional)

It enforces a daytime coverage gate (default: must reach 18:00 local) to avoid
silently wrong "daily Tmax" values for short-horizon models.

## Dependencies (conda-forge recommended)
```
conda create -n asof_fetch python=3.11 -y
conda activate asof_fetch
conda install -c conda-forge herbie-data cfgrib eccodes xarray pandas numpy scikit-learn tqdm -y
```

## Run (March 1–10, 2017 default)
```
python fetch_kmia_tmax_asof_multi_model.py --start 2017-03-01 --end 2017-03-10 --out kmia_asof_tmax_20170301_20170310.csv
```

Optional HRRR/RAP attempts:
```
python fetch_kmia_tmax_asof_multi_model.py --start 2017-03-01 --end 2017-03-10 --include-hrrr --include-rap --out kmia_asof_tmax_20170301_20170310_with_hrrr_rap.csv
```
