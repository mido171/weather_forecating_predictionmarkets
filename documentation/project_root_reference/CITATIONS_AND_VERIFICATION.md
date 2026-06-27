# Evidence Used to Design This Bootstrap

Reviewed on 2026-06-18.

## Settlement

The current Polymarket Hong Kong Tmax event page names the Hong Kong Observatory Daily Extract, field `Absolute Daily Max (deg. C)`, one-decimal source precision, and exclusion of later revisions:

`https://polymarket.com/event/highest-temperature-in-hong-kong-on-june-19-2026`

This repository treats that as event-specific and rechecks every market.

## HKO target proxy and station metadata

Official HKO resources:

- Daily Extract: `https://www.hko.gov.hk/en/cis/dailyExtract.htm`
- CLMMAXT station HKO:  
  `https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO`
- Open-data catalog: `https://www.hko.gov.hk/en/abouthko/opendata_intro.htm`
- Station information: `https://www.hko.gov.hk/en/cis/stn.htm`
- Daily climate download: `https://www.hko.gov.hk/en/cis/downloadpage.htm`

HKO documentation describes `CLMMAXT` as Daily Maximum Temperature, station code `HKO` as Hong Kong Observatory, and a history beginning in 1884 with a 1940–1946 exclusion/gap. This is strong evidence for a historical candidate label, not proof of first-publication settlement parity.

## Point-in-time live products

Official direct resources include:

- latest one-minute mean temperature:  
  `https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_temperature.csv`
- latest max/min since midnight:  
  `https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_since_midnight_maxmin.csv`

The HKO open-data catalog documents update frequencies and provisional status.

## Forecast/model data

- ECMWF open forecast data: `https://www.ecmwf.int/en/forecasts/datasets/open-data`
- NOAA GFS/GEFS operational files under NOMADS
- DWD ICON/ICON-EPS open data
- Copernicus ERA5/ERA5-Land datasets

Reanalysis is marked retrospective-only by default.

## Polymarket APIs

Official documentation covers event discovery, market WebSocket data, order books, price history, and fees. The repository archives exact event metadata and fee parameters instead of assuming permanent behavior.

## Codex structure

Official Codex documentation supports:

- project-scoped custom agents in `.codex/agents/*.toml`;
- global subagent limits under `[agents]`;
- project skills under `.agents/skills/<skill>/SKILL.md`;
- `name` and `description` frontmatter for skills.

Sources:

- `https://developers.openai.com/codex/subagents`
- `https://developers.openai.com/codex/skills`

## Honesty boundary

This bootstrap was validated structurally and with automated tests. It does not contain downloaded third-party weather history, a trained champion model, or evidence of profitable trading. Those outcomes depend on executing the goals and passing the research gates.
