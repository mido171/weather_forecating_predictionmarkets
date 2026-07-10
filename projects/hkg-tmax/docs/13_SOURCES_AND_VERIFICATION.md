# Official Sources and Verification Notes

**Reviewed:** 2026-06-18  
This is a research starting list. Archive exact source snapshots before relying on them.

## Polymarket

### Current Hong Kong event template

Official event page:

`https://polymarket.com/event/highest-temperature-in-hong-kong-on-june-19-2026`

At review time, its rules named:

- Hong Kong Observatory;
- Daily Extract;
- `Absolute Daily Max (deg. C)`;
- one-decimal source precision;
- later revisions excluded.

The repository does not hard-code this forever. Every event is archived and parsed.

### APIs

- Gamma event by slug:  
  `https://gamma-api.polymarket.com/events/slug/{slug}`
- CLOB order book:  
  `https://clob.polymarket.com/book?token_id={token_id}`
- Market WebSocket documentation:  
  `https://docs.polymarket.com/market-data/websocket/market-channel`
- Price history documentation:  
  `https://docs.polymarket.com/api-reference/markets/get-prices-history`
- Fees:  
  `https://docs.polymarket.com/trading/fees`

## Hong Kong Observatory

### Daily Extract

`https://www.hko.gov.hk/en/cis/dailyExtract.htm`

### Official daily maximum API

`https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO`

HKO’s API documentation identifies:

- `CLMMAXT` as Daily Maximum Temperature;
- `HKO` as Hong Kong Observatory;
- HKO history beginning in 1884 with a 1940–1946 exclusion/gap.

This does not by itself prove first-publication parity with the contract field.

### Open-data catalog

`https://www.hko.gov.hk/en/abouthko/opendata_intro.htm`

The catalog documents live products including one-minute mean temperature and max/min since midnight, commonly refreshed every ten minutes, plus other weather data.

### Direct live CSVs

- one-minute mean temperature:  
  `https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_temperature.csv`
- max/min since midnight:  
  `https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_since_midnight_maxmin.csv`

### Station information

`https://www.hko.gov.hk/en/cis/stn.htm`

### Historical data API specification

`https://data.gov.hk/en/help/api-spec#historicalAPI`

Test dataset-specific support. Do not assume every live feed is versioned.

## Operational NWP

- ECMWF open forecast data:  
  `https://www.ecmwf.int/en/forecasts/datasets/open-data`
- NOAA GFS operational files:  
  `https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/`
- NOAA GEFS operational files:  
  `https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/`
- DWD ICON:  
  `https://opendata.dwd.de/weather/nwp/icon/`
- DWD ICON-EPS:  
  `https://opendata.dwd.de/weather/nwp/icon-eps/`

Actual cycle availability must be observed; initialization time is not delivery time.

## Retrospective data

- ERA5 single levels:  
  `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels`
- ERA5-Land:  
  `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land`

These are retrospective by default and not operational predictors without a historically accurate release-lag design.

## Codex repository customization

Official OpenAI Codex docs:

- Subagents: `https://developers.openai.com/codex/subagents`
- Skills: `https://developers.openai.com/codex/skills`
- AGENTS.md: `https://developers.openai.com/codex/guides/agents-md`

The repository uses project custom agents under `.codex/agents/` and skills under `.agents/skills/`.

## Verification rule

A URL in this file is not enough. For every production dependency:

1. save the page/payload;
2. save retrieval time and hash;
3. record the exact relevant field/line/schema;
4. add a source contract;
5. test it.
