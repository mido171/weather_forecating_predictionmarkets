# EPIC #1 — Kalshi Weather Data Ingestion (MOS “as-of” forecasts + CLI settlement truth)

## Goal (what “done” looks like)
Build a **Java Spring Boot + MySQL** ingestion subsystem that can, for any supported **Kalshi daily high-temperature series** (e.g., KXHIGHMIA), reliably produce a historical dataset of:

- **Forecast features available “as-of” day T-1** (MOS-based) for predicting day **T**’s daily max temperature
- **Settlement truth for day T** from the **NWS Daily Climate Report (CLI)** (same source Kalshi uses)

The output must be **apples-to-apples** across:
- station
- target date (local date for the station)
- climate-report “day” definition (local standard time)
- as-of (no forward-looking leakage)

It must also be:
- **idempotent** (safe to re-run, restart, partial resume)
- **scalable** (multi-station, multi-year)
- **auditable** (store raw payloads, run times, retrieval times, versions)

## Sources (authoritative)
- Kalshi API “Get Series” response includes settlement source URL(s) per series. Example: KXHIGHMIA settlement is NWS CLI issued by MIA and hosted at forecast.weather.gov.  
  - KXHIGHMIA: https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHMIA  
  - KXHIGHLAX: https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHLAX  
  - KXHIGHCHI: https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHCHI  
  - KXHIGHPHIL: https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHPHIL  
  - KXHIGHNY: https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHNY  
- Kalshi Help Center: weather markets settle on final NWS Daily Climate Report, and climate reports use **local standard time** (DST implies 1:00AM→12:59AM).  
  https://help.kalshi.com/markets/popular-markets/weather-markets
- IEM MOS archive: model availability, run cycles (including NBS/NBE), and MOS variable definitions (n_x = max/min temp, tmp = 2m temp).  
  https://mesonet.agron.iastate.edu/mos/  
  https://mesonet.agron.iastate.edu/mos/fe.phtml
- IEM MOS bulk CGI backend: /cgi-bin/request/mos.py (supports sts/ets window; csv/json/excel).  
  https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?help=
- IEM CLI JSON backend: /json/cli.py (station/year; json/csv).  
  https://mesonet.agron.iastate.edu/json/cli.py?help=
- IEM CLI notes on climate day window: midnight local standard time; during DST it maps to 1AM→1AM local daylight time.  
  https://mesonet.agron.iastate.edu/nws/clitable.php

## Scope boundary (Epic #1 ONLY)
✅ Included:
- Station/series discovery (Kalshi series metadata) for temperature-high markets
- Fetch + persist MOS data needed to construct T-1 “as-of” features for day T
- Fetch + persist CLI settlement values for day T
- Robust time semantics + leakage-proof “as-of” selection logic
- Idempotent backfill + checkpointing
- Operational concerns (retries, rate limits, raw payload storage, auditing)
- Works for: Miami, NYC, Chicago, Philadelphia, Los Angeles (and extensible)

❌ Not included:
- ML training (Epic #2)
- Kalshi market price backtesting (Epic #3)
- Trading engine integration

## Deliverables
1) MySQL schema + JPA entities in module `models`
2) Java Spring Boot ingestion service that:
   - resolves Kalshi series → settlement source URL → station identifier strategy
   - ingests CLI truth into `cli_daily`
   - ingests MOS runs into `mos_run` + derived features into `mos_asof_feature`
   - can run:
     - incremental daily updates
     - historical backfills
     - resume after abort
3) Documentation:
   - `docs/time-semantics.md`
   - `docs/station-mapping.md`
   - `docs/architecture.md`
   - `docs/runbook.md`
   - `agents.md` (Codex/agent handoff instructions)

## Definition of Done (Epic-level)
- For each of the 5 target series (NY, PHL, MIA, CHI, LAX), for any date range requested:
  - the system can materialize, per day T:
    - CLI settlement tmax (truth)
    - MOS as-of(T-1) features for models: GFS, MEX, NAM, NBS, NBE
    - metadata: asOfUtc, chosen MOS runtimeUtc per model, retrieval timestamps, raw payload references
- Proven no-leakage:
  - runtimeUtc chosen for features is always <= asOfUtc
  - asOfUtc is computed from station timezone and a configured local “decision time”
- Restartability:
  - killing the process mid-backfill and re-running continues without duplicates or gaps
- Build correctness:
  - `mvn clean install` succeeds from repo root (within Epic #1 modules)
