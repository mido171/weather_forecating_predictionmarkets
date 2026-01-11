# WX-206 — Sigma model training (σ̂) via squared residual prediction + positivity guarantees

## Objective
Train a model to predict uncertainty for each day T.

## Definition (must follow exactly)
1) Fit mean model on training data → produces μ̂(T).
2) Compute residual on training rows:
   r(T) = y(T) − μ̂(T)
3) Define sigma-target:
   v(T) = r(T)^2   (squared residual)
4) Train sigma model to predict v(T) from sigma features.

Sigma features (minimum):
- disagreement(T) = stddev([gfs, mex, nam, nbs, nbe])
- mu_hat(T) = μ̂(T)

Sigma model output:
- v̂(T) (predicted squared error)
Then:
- σ̂(T) = sqrt(max(v̂(T), eps))

## Requirements
- sigma model must never output negative variance:
  - clamp with eps or train on log(v+eps).
- Persist sigma model artifact:
  - `artifacts/<run_id>/<station>/model_sigma.joblib`

## Acceptance Criteria
- [ ] sigma training runs for each station.
- [ ] σ̂(T) is always finite and > 0.
- [ ] Evaluate sigma quality:
  - coverage of 68%/90% intervals
  - residual standardization checks (optional)
- [ ] Persisted artifacts include metadata.

