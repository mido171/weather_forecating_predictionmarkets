# EPIC #2 — ML Training & Evaluation (μ/σ probabilistic settlement predictor)

## Goal (what “done” looks like)
Build a **Python** training pipeline that learns to predict the **Kalshi settlement value** for “Daily High Temperature” markets:

- **Target (label):** Daily maximum temperature used by Kalshi settlement (**NWS CLI daily climate report** value for the settlement station).
- **Inputs (features, as-of T-1):** MOS as-of forecasts (from Epic #1) for multiple models:
  - `GFS`, `MEX`, `NAM`, `NBS`, `NBE` daily-max guidance for target day **T** chosen from runs **≤ asOfUtc** (no leakage).

The pipeline must output:
1) **Mean model** μ̂(T): point prediction of settlement Tmax.
2) **Uncertainty model** σ̂(T): predicted standard deviation of the forecast error for day T, used to produce a probability distribution over integer temperatures and Kalshi bins.

It must also:
- produce **high-quality metrics and calibration reports** (probability quality matters for trading)
- persist models + metadata for later use (Epic #3 and live system)
- be repeatable and auditable (dataset versioning + run metadata)
- support multiple stations (KMIA, KNYC, KMDW, KPHL, KLAX) and arbitrary future Kalshi stations ingested by Epic #1.

## Scope boundary (Epic #2 ONLY)
✅ Included:
- Python project scaffolding for training/evaluation
- MySQL dataset extraction (from Epic #1 tables)
- Data validation & leakage audits
- Feature engineering (minimal + robust)
- Train μ model + σ model
- Build predictive distribution and bin probabilities
- Probability calibration layer (bin-level)
- Model persistence + model registry metadata
- Detailed metrics + reports (MAE/RMSE + probabilistic scoring + calibration)

❌ Not included:
- Pulling Kalshi historical orderbooks / price backtests (Epic #3)
- Live trading execution

## Why the μ/σ approach is recommended (for Kalshi)
Kalshi contracts trade on **probabilities of discrete outcomes** (temperature buckets). A point estimate alone is insufficient. By producing (μ̂, σ̂) you can generate a full distribution:

`Tmax_settle(T) ~ Normal(μ̂(T), σ̂(T))` (or other distribution later)

Then you can compute:
- `P(bin)` for each Kalshi bucket (e.g., 81–82)
- expected value vs market prices (Epic #3)

## External references (must be consistent with implementation)
The ML evaluation and persistence approach uses standard scikit-learn tools:
- Time series splitting to avoid leakage: TimeSeriesSplit (scikit-learn docs)
- Probabilistic scoring: Brier score loss, log loss (scikit-learn docs)
- Calibration diagnostics: calibration_curve (reliability diagram), CalibrationDisplay (scikit-learn docs)
- Calibration method: IsotonicRegression (scikit-learn docs)
- Persistence: scikit-learn model persistence guidance + joblib persistence warning (pickle security)

(See `docs/references.md` in this epic package for exact URLs.)

## Deliverables
1) Python package/module (inside repo) with:
   - dataset extraction (MySQL → Parquet/CSV snapshots)
   - training pipeline
   - evaluation suite
   - artifact writer (models + metadata)
2) Model artifacts:
   - μ model file
   - σ model file
   - calibration artifacts
   - JSON metadata (feature list, date range, versions, metrics summary)
3) Reports:
   - Markdown report (human readable)
   - JSON report (machine readable)
   - Plots (calibration curves, residuals, etc.)
4) Optional (but recommended) DB tables for tracking training runs and artifact locations:
   - created via Flyway migrations in the Java `models` module

## Definition of Done (Epic-level)
- Given a station list and date range:
  - pipeline builds dataset from MySQL
  - trains μ and σ models with leakage-safe time-based splits
  - produces probabilities for each integer temperature and configured bins
  - produces and saves metrics reports and plots
  - persists artifacts and writes run metadata
- A re-run with the same config produces:
  - identical dataset snapshot hash (if DB hasn’t changed)
  - deterministic model outputs (within floating tolerance) due to fixed random seeds
- Documentation explains exactly how μ/σ are trained and used.
