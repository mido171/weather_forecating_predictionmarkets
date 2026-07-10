# Expert Model Catalog

## Principle

Train independent experts on coherent data families. Each expert outputs a point forecast, expected absolute error, uncertainty information where available, and a genuine out-of-fold prediction history.

## Experts

1. Official raw HKO anchor.
2. Official residual-memory baseline.
3. Official forecast residual ML expert.
4. Long-history target-memory/climatology expert.
5. Station-network microclimate expert.
6. GFS direct and GFS MOS experts.
7. GEFS direct median and calibrated probabilistic MOS experts.
8. IFS deterministic MOS challenger.
9. IFS ensemble challenger.
10. ARWF direct/MOS challenger after prospective collection.
11. CWA WRF regional challenger after prospective collection.
12. AI weather-model challengers.
13. Diagnostic-physics safe-student expert.
14. Specialist-corrected expert variants.

## Training targets

Direct experts may predict target Tmax. Anchor-centered experts predict residual:

```text
actual_tmax - anchor_forecast
```

Expected-error models predict absolute OOF error. Distributional experts predict quantiles or calibrated residual distributions.

## Candidate model classes

Always establish regularized linear/GAM baselines. Then compare constrained gradient boosting, CatBoost/LightGBM, quantile boosting, random forest only where justified, and small neural models only after tree/GAM baselines. Hyperparameters are selected inside nested temporal folds.

## Short-history model policy

IFS, AI models, CWA WRF and ARWF cannot receive unrestricted router weight. They begin as shadow/challenger experts with weight caps and hierarchical shrinkage toward the core router. Their caps rise only after untouched seasonal evidence.
