# WFPM-EPIC-EXTDATA — External Observations + Climate Context for KMIA Day‑Ahead Tmax Trading Model

## Background

We are building an operational, leakage‑free machine learning forecasting system whose single job is: at **day T‑1 12:00Z**, produce a **sharp point forecast and a calibrated probability distribution** for **KMIA (Miami International Airport) maximum temperature (Tmax) on local day T**. The output is used for daily decision‑making in Kalshi temperature markets (bucketed ranges and thresholds), so the model must be both *accurate* (low MAE/RMSE) and *honest* (probabilities that match real frequencies).

Two model “tracks” already exist in this project:

1) **MOS track (GFS/NAM, 2007–present)** — long history but fewer modern fields.
2) **Gribstream track (multi‑model, ~2021–present)** — richer guidance (more models/derived features) but shorter history.

The purpose of this epic is to add **high‑signal, freely available observational/context data** that improve both tracks (and especially the gribstream track) without breaking the strict “as‑of” forecasting rule.

## Why external observations matter for Miami Tmax

Miami Tmax is heavily driven by mesoscale processes that are only partially captured by coarse guidance and may be mis‑timed even by high‑resolution models:

- **Sea‑breeze strength/timing**: marine winds/pressure and coastal vs inland gradients strongly govern afternoon mixing and peak temperatures.
- **Boundary‑layer moisture & stability**: morning dewpoint, inversion strength, precipitable water, and CAPE/CIN influence cloudiness and convective timing, which in turn caps Tmax.
- **Antecedent wetness**: rainfall and soil/near‑surface temperature change sensible vs latent heat partitioning.
- **Ocean boundary condition**: SST anomalies affect marine air mass properties and the sea‑breeze temperature contrast.

These factors are observable in real time (or near‑real time) and provide “truth anchoring” that reduces model busts, improves bias control, and enhances probability calibration.

## Epic scope

We will ingest, normalize, and feature‑engineer five new data sources:

1) **ASOS/METAR surface observations** (airport ring around KMIA) from Iowa Environmental Mesonet (IEM) `asos.py`.
2) **FAWN mesonet observations** (South Florida stations) from the FAWN “today/last96” API.
3) **NDBC/NOS coastal marine observations** (Virginia Key, Port Everglades) from NOAA NDBC text feeds and historical archives.
4) **IGRA v2.2 sounding‑derived parameters** (Miami radiosonde) from NOAA NCEI (PW, CAPE/CIN, stability/inversion metrics).
5) **OISST v2.1 daily SST and anomaly** near South Florida from NOAA/NCEI THREDDS.

Each source will be stored in dedicated DB tables with raw provenance (retrieval timestamps, source URLs, raw payload hashes when feasible) and will be converted into daily “as‑of” features usable at T‑1 12Z.

## As‑of discipline (no leakage)

For every training/example row corresponding to **target_date_local = T**, we define:

- `asof_utc = (T − 1 day) at 12:00Z`
- Features must use **only information that would have been available at or before `asof_utc`**, allowing a conservative latency buffer per source.

Rules:

- **Surface obs (ASOS/FAWN/NDBC)**: use observations with `obs_time_utc <= asof_utc − 15 minutes`.
- **IGRA derived**: update is daily and not guaranteed intra‑day; use most recent sounding record with `sounding_time_utc <= asof_utc − 12 hours` (fallback to <= 36 hours if missing).
- **OISST**: published with ~24h latency and may be revised for ~2 weeks; for strict “live‑replica” we use SST through `T−2` (i.e., `sst_day <= asof_utc.date − 1 day`), and in features we prefer 7‑/14‑day means to reduce revision sensitivity.

These buffers are intentionally conservative so the backtest is not optimistically biased.

## System architecture

1) **Ingest → Raw**: download source payloads (CSV/JSON/TXT/NetCDF/ZIP), store raw bytes or a hash reference, record retrieval metadata.
2) **Parse → Normalized tables**: parse to typed columns, convert units once, store in DB with strong keys and idempotent upserts.
3) **Feature store builder**: for each target day T, query each table for the appropriate as‑of window and compute engineered daily aggregates (means, slopes, gradients, quantiles, flags).
4) **Model training**: retrain the existing tree model pipelines (XGBoost/LightGBM) with these added features; keep the same train/val/test splits and rolling‑origin evaluation.
5) **Scoring & reporting**: compute point metrics (MAE/RMSE/bias) and probabilistic metrics (pinball/CRPS‑approx, interval coverage/width, Brier/logloss for Kalshi‑style events, reliability/ECE/MCE).

## Definition of success

We will consider this epic successful when:

- The end‑to‑end pipeline can backfill and update daily without manual intervention.
- Feature generation is deterministic and strictly as‑of.
- The new features produce measurable skill on 2024–2025 test vs the current gribstream baseline (target: **≥ 0.05°F MAE reduction** or equivalent improvement in CRPS/Brier without degrading calibration).
- A report artifact is generated showing ablations (baseline vs +ASOS vs +marine vs +FAWN vs +IGRA vs +SST vs all).

## Out of scope

- Purchasing proprietary datasets.
- Full reanalysis ingestion (ERA5, etc.).
- Building a separate model for other stations; additional stations are used only as **features** to improve KMIA.

## References (for Codex implementation)

- IEM ASOS request API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=
- FAWN API (“today/obs” and “today/last96”): https://fawn.ifas.ufl.edu/controller.php/today/
- NDBC formatting and measurement descriptions: https://www.ndbc.noaa.gov/faq/5day.shtml and https://www.ndbc.noaa.gov/faq/measdes.shtml
- IGRA readme and derived format: https://www.ncei.noaa.gov/pub/data/igra/igra2-readme.txt and https://www.ncei.noaa.gov/pub/data/igra/derived/igra2-derived-format.txt
- OISST v2.1 THREDDS catalog: https://www.ncei.noaa.gov/thredds/catalog/OisstBase/NetCDF/V2.1/AVHRR/catalog.html


## Operational and reproducibility requirements

This system is not a research notebook; it must behave like a trading engine component. That implies:

- **Deterministic rebuilds**: given the same as‑of cutoff and the same raw inputs, the feature rows must be identical.
- **Idempotent ingestion**: running the same backfill job twice must not double‑insert data. Upserts must be keyed by `(source, station_id, valid_time_utc)` (or equivalent) with clear conflict behavior.
- **Conservative time handling**: all timestamps are stored in UTC. Only the target label and “day buckets” are computed in `America/New_York` (KMIA local time). DST transitions must be handled explicitly.
- **Graceful missingness**: data outages happen. Feature builders must tolerate missing station hours by computing robust aggregates (e.g., “use last available within the window”; include coverage features like `obs_count_6h` so the model can down‑weight weak inputs).

## Where the edge comes from (Kalshi framing)

Kalshi markets typically pay based on whether Tmax falls into a bucket (e.g., 85–89°F) or crosses a threshold (e.g., ≥ 90°F). To trade rationally, we need:

1) A **distribution** over Tmax for day T, not just a point estimate.
2) A **calibration check** so that “65%” means “about 65% in the long run.”

The external observations in this epic improve the *shape and calibration* of the distribution because they provide the model with state variables that control “tail” behavior:
- A moist, unstable morning with high PW/CAPE increases storm probability, often suppressing Tmax (shifts probability mass downward).
- A strong onshore marine wind signal can cap Tmax even when guidance is hot (reduces upper tail).
- A dry, deeply mixed morning regime increases the chance of a hot outcome (fattens upper tail).

So the epic is directly tied to capturing the *reasons* that guidance misses, which is how you generate +EV vs an order book that is often anchored to a single deterministic forecast.

## Data source selection: exact South Florida locations

The goal is not “more stations”; it is “the minimum set of stations that capture the dominant gradients around KMIA.”

Surface airport ring (ASOS/METAR):
- KMIA (target station, anchor)
- KFLL (Fort Lauderdale coastal)
- KOPF (Opa‑locka, north/nearby)
- KTMB (Miami Executive, southwest)
- KHST (Homestead, south)
- KPBI (West Palm Beach, farther north gradient check)

FAWN mesonet (South Florida):
- 420 Ft. Lauderdale (Broward; coastal urban)
- 440 Homestead (Miami‑Dade; south/inland‑edge)
- 410 Belle Glade (interior Everglades influence)
- 425 Wellington (Palm Beach inland)

Marine (NDBC/NOS):
- VAKF1 Virginia Key (very near the coast east of Miami)
- PEGF1 Port Everglades (north coastal reference)
Optional additions if needed later: nearby offshore buoys (if they report WTMP) and/or NOAA CO‑OPS stations for water temperature.

Upper air (IGRA derived):
- USM00072202 (Miami radiosonde station identifier in IGRA station list)

Sea surface temperature (OISST):
- 0.25° grid cells in a small box around Miami (e.g., lat 24.5–26.5, lon 279.0–281.0 degrees_east), summarized to a single daily value.

## Quality control expectations

Because trading is unforgiving, we explicitly add QC and sanity checks:

- Physical bounds (e.g., dewpoint cannot exceed temperature; wind direction 0–360; pressures in plausible ranges).
- Missing value handling: the sources use different missing codes (IEM: “M”; NDBC: “MM”; IGRA: -99999; OISST: -999).
- Duplicate detection by timestamp.
- “Coverage features” that quantify how much data was present in each lookback window.

## Deliverables

This epic is complete only when:

- All five ingestion pipelines can run (backfill + daily incremental) and persist to DB.
- A feature builder can generate a single daily row for KMIA that includes these new features with explicit as‑of cutoffs.
- Training + evaluation can be rerun end‑to‑end to produce an artifacts folder with metrics, reliability tables, and ablation comparisons, and a one‑click rebuild script.
