# Model Persistence & Reproducibility Rules

## 1) Default persistence mechanism
Use scikit-learn recommended persistence patterns (joblib is typical for sklearn objects):
https://scikit-learn.org/stable/model_persistence.html

## 2) Security warning (MUST DOCUMENT AND ENFORCE)
joblib is pickle-based; loading untrusted models can execute arbitrary code:
https://joblib.readthedocs.io/en/latest/persistence.html

Therefore:
- Only load artifacts produced by your own training pipeline.
- Store strong hashes for artifacts and validate them.
- Store environment versions used during training (python, sklearn).

## 3) Version compatibility warning
scikit-learn notes environment compatibility constraints; persisted models may break if versions differ.
Implementation must store:
- python version
- scikit-learn version
- numpy version
- git commit hash (if available)

## 4) Artifact layout (required)
artifacts/
  <run_id>/
    model_mean.joblib
    model_sigma.joblib
    calibrators/...
    metrics.json
    report.md
    plots/...
    metadata.json

## 5) Determinism
Training must allow a global random seed. The seed must be stored in metadata.json.

