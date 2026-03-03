# KNYC Quantile + KNN + Gate + Rolling Conformal

This track documents the KNYC same-day Tmax point/quantile system implemented under `pipelines/quantile_knn_conformal`.

## Scope

- Target: KNYC same-day daily maximum temperature (`y_tmax`) in whole degF.
- Inputs: 30m/hourly observation export, station universe, and KNYC settled truth CSV.
- Core stack:
  - LightGBM quantile grid (15 quantiles)
  - KNN analog quantile expert
  - Learned gate (`alpha`) blending ML and KNN quantiles
  - Rolling conformal calibration for trade-time decision rows
  - Quantile-to-CDF conversion, integer PMF, and bucket probability outputs

## Canonical Code

- Pipeline entrypoint: `pipelines/quantile_knn_conformal/run_pipeline.py`
- Config snapshot: `configs/quantile_knn_conformal_KNYC.yaml`
- Core modules:
  - `data_loading.py`, `dataset_builder.py`, `feature_builder.py`
  - `train_quantiles.py`, `knn_analog.py`, `train_gate.py`, `conformal.py`
  - `cdf_bucket_mapper.py`, `evaluate.py`, `leakage_audit.py`, `live_infer.py`, `reporting.py`

## Latest Run (2026-02-28)

See `documentation/knyc_quantile_knn_conformal/00_run_record_2026-02-28.md`.

## MOS-First Runtime Matrix Run (2026-03-01)

See `documentation/knyc_quantile_knn_conformal/01_run_record_2026-03-01_mos_first_plan.md`.

## Artifact Root

- `D:\Ahmed\data\kalshi\Experiments\point_foreacast\E1`

Contains:

- `00_config_snapshot` through `10_bundle`
- stage manifests for reproducibility
- leak audit, model comparison, calibration diagnostics, prediction exports, and deploy bundle
