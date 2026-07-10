# Remaining Acquisition Action Plan

Polymarket, modelling, ML, and backtesting are excluded. This plan is only for
HKG/HKO Tmax weather-data acquisition and provenance.

## Current Constraint

The earlier network/data-root execution block was cleared during this reset.
The canonical raw archive remains `C:\hkg_tmax_data`, and all future fetches
must continue to use that content-addressed archive rather than a repo-local
replacement archive.

## Current Raw Archive State

- retrieval attempts: `8,352`
- successful retrieval attempts: `8,351`
- failed retrieval attempts: `1`
- logical source IDs: `235`
- successful archived bytes: `4,392,458,009`
- successful unique content hashes: `7,938`

## Continue Immediately

1. Keep the expanded HKO current satellite batch running prospectively.

   Command:

   ```powershell
   .\.venv\Scripts\python.exe -m hkg_tmax acquisition hko-backfill --batch satellite-current --continue-on-error --delay-seconds 0 --skip-existing-successes
   ```

   The resumed batch completed with no failed downloads and the raw archive
   audit passed. The HKO manifests are rolling operational windows, so
   manifest-listed filenames not present in the ledger must be treated as
   current-window/persistent-miss candidates unless a fresh live preflight
   proves they still return 2xx URLs.

2. Monitor the installed prospective collector for mutable/latest sources.

   Covered families:

   - HKO live regional feeds;
   - HKO latest JSON forecasts/warnings;
   - HKO RSS current forecast/warning feeds;
   - ARWF current station/grid forecasts and nowcasts;
   - NCEP GFS/GEFS rolling regional subsets;
   - periodic DATA.GOV.HK historical archive refresh;
   - periodic NOAA ISD/IGRA archive refresh.

   The Windows Task Scheduler task `HKG-Tmax-Collector` was installed and is
   enabled. Current Task Scheduler state after final checks: `Ready`.

3. Parse downloaded station/weather archives into bronze coverage reports.

   This is not predictive modelling. It is acquisition QA needed to prove which
   stations, variables, timestamps, and vintages are present.

   Current offline evidence:

   - `reports/station_weather_coverage.md` proves the downloaded HKO
     high-frequency historical ZIP archives contain multi-station rows and
     lists the station coverage by feed;
   - the same report lists NOAA ISD nearby station-year coverage.
   - `reports/source_family_coverage.md` groups successful raw retrievals by
     acquisition family and records remaining work by family.

   Required outputs:

   - HKO high-frequency station-variable coverage;
   - NOAA ISD station-year coverage;
   - HKO ARWF station/grid coverage;
   - HKO Daily Extract and D1 daily climate coverage;
   - forecast/RSS vintage coverage;
   - NCEP cycle/member/lead/variable coverage.

4. Continue deterministic static context products from archived official static
   raw sources.

   Required outputs:

   Completed acquisition-readiness outputs:

   - station registry;
   - station distance/bearing matrices;
   - solar-geometry tables.

   Remaining parser outputs:

   - terrain/elevation/slope/aspect station context;
   - coastline/land-water context;
   - LUHK land-utilization station context.

## Source-Discovery Work Still Needed

These should be investigated, but must not be treated as already available:

- older historical versions of HKO rainfall, visibility, and direct RHR JSON
  feeds beyond current/latest snapshots;
- older historical versions of HKO JSON forecast endpoints (`flw`, `fnd`,
  `warnsum`, `warningInfo`, `swt`) beyond RSS historical archives;
- alternate lawful official satellite archives if HKO's rolling current window
  is insufficient;
- gridded SST/ocean product choice and subset plan;
- ECMWF/DWD/AI forecast archive source policy and byte budget.

## Manual Approval / Huge / Credential-Gated

Do not bulk download these without an explicit approved subset and byte budget:

- full historical and continuous operational GFS/GEFS/ECMWF/DWD/AI forecast
  archives;
- ERA5/ERA5-Land reanalysis, which also requires CDS credentials and is
  retrospective-only;
- gridded SST products such as OISST/OSTIA;
- full PlanD 3D photo-realistic model tile payloads beyond the archived tile
  indexes;
- any alternate full satellite archive that is materially larger than HKO's
  rolling public current-window products.

## Acceptance Checks After Downloads

Run these before any commit or before declaring acquisition complete:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax doctor
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
```

Also rerun a full raw archive audit proving:

- every successful ledger row has a content object and HTTP metadata sidecar;
- every object content hash and length matches the ledger;
- file manifest and dataset lineage cover every successful retrieval;
- append-only/deduplicated storage did not overwrite prior immutable objects;
- all new source IDs and remaining blocker decisions are documented.

