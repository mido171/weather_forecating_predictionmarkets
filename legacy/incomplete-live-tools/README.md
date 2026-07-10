# Incomplete historical live tools

These scripts are retained for provenance but are not current runnable entrypoints. They
import calibration, feature, and model-training modules that were never present in the tracked
`ml_live` package. Presenting them under active `tools/live` implied a working contract that
the repository could not satisfy.

- `run_kmia_live.py`: imports calibration, feature, and model-training modules that are absent.
- `run_station_live.py`: wrapper around the incomplete runner above.
- `train_e92_models.py`: references absent E92 modeling modules.
- `run_kmia_live_v5plus8.py`: imports the absent `ml.run_mos_45_suite` module in its
  training and inference paths.
- `mos_quantile_live_inference.py`: imports the absent historical `ml` training tree and
  retains paths from the pre-reorganization layout.
- `mos_blend12_bundle.py`: private helper retained beside the incomplete quantile runner;
  it has no supported standalone entrypoint.
- `eval_kalshi_bridge.py`: references undefined runtime constants and an absent prediction
  artifact contract, so its old application path was not its only release blocker.

Do not execute them without first promoting the missing behavior into a tested package and
passing the root live-tool safety rules.
