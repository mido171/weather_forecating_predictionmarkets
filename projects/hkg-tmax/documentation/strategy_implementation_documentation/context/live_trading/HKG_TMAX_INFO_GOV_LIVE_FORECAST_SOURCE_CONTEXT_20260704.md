# HKG Tmax Live Forecast Source Context - Info.gov Apples-to-Apples Contract

Created: 2026-07-04 16:25:24 CEST / 2026-07-04 22:25:24 HKT

Purpose: preserve the exact conclusion from the July 4 live-source investigation: for live trading and live forecasting against the HKG Tmax Polymarket market, the forecast source must match the historical source in the local PostgreSQL table `public.hko_historical_forecasts_2000_2026`. The historical archive used by the current strategy experiment is not the HKO 9-day OpenData API. It is the HKSAR Government Info.gov press-release feed for HKO local weather forecasts.

## Executive Conclusion

The apples-to-apples source for live trading is:

- HKSAR Government Press Releases, weather section.
- Same-source daily index pattern: `https://www.info.gov.hk/gia/wr/YYYYMM/DD.htm`.
- Current-day redirect/index: `https://www.info.gov.hk/gia/wr/today.htm`.
- Forecast pages titled exactly like `PRESS WEATHER NO. ### - LOCAL WEATHER FORECAST`.
- Individual page pattern: `https://www.info.gov.hk/gia/wr/YYYYMM/DD/PYYYYMMDDNNNN.htm`.
- Forecast text begins with `LOCAL WEATHER FORECAST` and contains language such as `HERE IS THE LATEST WEATHER BULLETIN ISSUED BY THE HONG KONG OBSERVATORY.`
- Dispatch line contains the actual issue/as-of time, for example `DISPATCHED BY HONG KONG OBSERVATORY AT 21:45 HKT ON 04.07.2026`.

The source that should be treated as the live primary forecast input is not:

- The HKO `fnd` 9-day forecast API, even though it often contains a matching min/max forecast for tomorrow.
- The HKO webpage widget.
- The Data.gov dataset page around HKO local forecasts.
- Airport forecast pages.
- GribStream or any NWP archive.
- A general weather website, a scraped card, or any source that cannot be traced to the Info.gov local forecast press release.

The current live source and the historical DB source are the same source family when and only when the live source is one of these Info.gov local forecast press-release pages.

## Why This Matters

The forecasting system being built here is for Hong Kong Observatory daily absolute maximum temperature at the HKO station. The target for the immediate Polymarket example is the market titled `Highest Temperature in Hong Kong on July 5, 2026`. The market resolves against the Hong Kong Observatory Daily Extract field `Absolute Daily Max (deg. C)` for the specified date, to one decimal place. The official HKO local forecast is a forecast input, not the settlement target. Settlement is the later observed station Tmax.

The whole statistical edge depends on not mixing historical and live forecast sources. If the historical backtest learns from Info.gov local forecast bulletins, but live trading uses HKO 9-day API forecasts, the system is no longer measuring the same data-generating process. That would create an apples-to-oranges mismatch. It may still be useful as auxiliary context, but it is not the correct live production anchor for the strategy already evaluated.

The current best empirical strategy from experiment `0215_gpt_pro_point_forecast_strategy` is anchored on the latest official HKO local forecast available before a cutoff. The model then applies a small historical residual correction. That residual correction is only meaningful if the live forecast has the same source semantics, issuance behavior, and text parsing conventions as the historical forecast archive.

## Historical DB Source of Truth

Database:

- Local database: `hkg_tmax_research`.
- Redacted connection shape: `postgresql://postgres:***@127.0.0.1:5432/hkg_tmax_research`.
- Forecast archive table: `public.hko_historical_forecasts_2000_2026`.
- Important source fields:
  - `source`
  - `source_url`
  - `product_type`
  - `title`
  - `index_date`
  - `snapshot_at_hkt`
  - `issue_at_hkt`
  - `issue_at_utc`
  - `target_date`
  - `target_issue_lead_days`
  - `forecast_min_c`
  - `forecast_max_c`
  - `row_quality_status`
  - `full_text`
  - `raw_sha256`
  - `raw_path`

The historical archive rows used for the official forecast anchor are selected from:

- `source = 'info_gov'`
- `product_type = 'local'`
- `row_quality_status = 'usable_local_minmax'`
- `target_issue_lead_days = 1` for the T-1 next-day forecast archive
- non-null `forecast_min_c`
- non-null `forecast_max_c`
- non-null `issue_at_hkt`
- non-null `target_date`

Representative DB row from the same source family:

- Title: `PRESS WEATHER NO. 165 - LOCAL WEATHER FORECAST`
- Source URL: `https://www.info.gov.hk/gia/wr/202606/20/P2026062000831.htm`
- Issue time: `2026-06-20 23:45 HKT`
- Target date: `2026-06-21`
- Forecast min/max: `29.0 / 33.0 C`
- Product type: `local`
- Source: `info_gov`
- Row quality: `usable_local_minmax`
- Target lead: `1`

This directly proves that the historical archive is Info.gov local forecast press releases, not the 9-day API.

## Historical Forecast Archive Coverage and Usability

All counts below are from a direct read-only query of `public.hko_historical_forecasts_2000_2026` on 2026-07-04.

Total rows in `hko_historical_forecasts_2000_2026`:

- `324,179` rows.

Rows that are exactly usable as local min/max forecast anchors:

- `115,795` rows are `product_type='local'`, `source='info_gov'`, `row_quality_status='usable_local_minmax'`.
- These rows cover `9,667` distinct target dates.
- First target date: `2000-01-02`.
- Last target date currently in DB: `2026-06-21`.
- Null `forecast_min_c`: `0`.
- Null `forecast_max_c`: `0`.
- Null `issue_at_hkt`: `0`.
- Null `target_date`: `0`.
- Within this usable local min/max subset, the null/unusable rate for the required min/max/issue/target fields is `0.00%`.

Unusable or not-primary rows in the same table:

- `5day` Info.gov rows:
  - `6,193` rows.
  - All are `row_quality_status='bulletin_only_multiday_product'`.
  - Parsed target days: `0`.
  - Null forecast min rows: `6,193`.
  - Null forecast max rows: `6,193`.
  - Usable for this current local-min/max strategy: no.

- `7day` Info.gov rows:
  - `23,223` rows.
  - All are `row_quality_status='bulletin_only_multiday_product'`.
  - Parsed target days: `0`.
  - Null forecast min rows: `23,223`.
  - Null forecast max rows: `23,223`.
  - Null issue time rows: `5`.
  - Usable for this current local-min/max strategy: no.

- `9day` Info.gov rows:
  - `30,438` rows.
  - All are `row_quality_status='bulletin_only_multiday_product'`.
  - Parsed target days: `0`.
  - Null forecast min rows: `30,438`.
  - Null forecast max rows: `30,438`.
  - Usable for this current local-min/max strategy: no.

- `local` rows with `row_quality_status='usable_local_tmax_only'`:
  - `58,199` rows.
  - Cover `9,258` target dates.
  - First target date: `2000-01-01`.
  - Last target date: `2026-06-20`.
  - Forecast max is present.
  - Forecast min is null for all `58,199` rows.
  - These may be useful for future Tmax-only research, but they are not the strict current min/max official-anchor subset used in the most recent apples-to-apples strategy check.

- `local` rows with `row_quality_status='missing_forecast_max'`:
  - `29,383` rows.
  - Cover `9,645` target dates.
  - First target date: `2000-01-01`.
  - Last target date: `2026-06-20`.
  - Forecast max is null for all `29,383` rows.
  - Forecast min is null for `26,847` rows.
  - Not usable as Tmax forecast anchors.

- `local` rows with `row_quality_status='missing_target_date'`:
  - `60,940` rows.
  - Parsed target days: `0`.
  - Null target date rows: `60,940`.
  - Forecast min is null for `51,560` rows.
  - Forecast max is null for `51,666` rows.
  - Null issue time rows: `42`.
  - Not usable unless target-date parsing is repaired.

- `local` rows with `row_quality_status='invalid_target_lead'`:
  - `8` rows.
  - Cover `5` target dates.
  - First target date: `1990-11-03`.
  - Last target date: `2005-02-25`.
  - Forecast min/max present but target lead is invalid for the strategy contract.
  - Not usable for strict lead-0/lead-1 modeling.

Interpreting those percentages:

- Across the whole `hko_historical_forecasts_2000_2026` table, `115,795 / 324,179` rows are strict usable local min/max anchors. That is about `35.72%`.
- Across the whole table, about `64.28%` of rows are not usable as strict local min/max anchors for the current strategy. This large unusable percentage is expected because the table also contains bulletin-only multi-day products, target-date failures, Tmax-only rows, and rows without forecast max.
- Within local Info.gov products only, `115,795 / 264,325` rows are strict usable local min/max anchors. That is about `43.81%`.
- Within local Info.gov products only, about `56.19%` are not strict usable local min/max anchors.
- The strict usable local min/max subset itself has `0.00%` nulls in the required fields.

## What the Local Forecast Actually Forecasts

The Info.gov `LOCAL WEATHER FORECAST` product is a short HKO text forecast for Hong Kong local weather. The HKO RSS schedule describes it as providing local weather forecast for today and/or tomorrow. It is normally updated around 45 minutes past each hour, at `16:15 HKT`, at `23:15 HKT`, and as necessary. The 9-day forecast is a different product, normally updated around `11:30 HKT`, `16:30 HKT`, and as necessary.

In the historical DB, strict usable local min/max rows only have:

- Same-day target forecasts: `target_issue_lead_days = 0`.
- Next-day target forecasts: `target_issue_lead_days = 1`.

There is no strict usable local min/max forecast horizon beyond lead 1 in this table. This is why the strategy has been treated as a T-1 or early T forecast problem, not a multi-day forecast problem.

Lead distribution in the strict usable local min/max subset:

- Lead `0`:
  - `27,291` rows.
  - `8,424` distinct target dates.
  - First target date: `2000-02-24`.
  - Last target date: `2026-06-20`.
  - First issue time: `2000-02-24 00:05 HKT`.
  - Last issue time: `2026-06-20 03:45 HKT`.

- Lead `1`:
  - `88,504` rows.
  - `9,665` distinct target dates.
  - First target date: `2000-01-02`.
  - Last target date: `2026-06-21`.
  - First issue time: `2000-01-01 16:22 HKT`.
  - Last issue time: `2026-06-20 23:45 HKT`.

For target day T, the lead-1 forecast rows are issued on T-1 HKT. They are usually text forecasts framed as `tonight and tomorrow`. In this frame, `tomorrow` is target day T.

## Number of Issuances Per Target Day in the Historical Lead-1 Archive

For `product_type='local'`, `source='info_gov'`, `row_quality_status='usable_local_minmax'`, and `target_issue_lead_days=1`, the number of forecasts per target date is not always one. It varies by day. This is important because the strategy can choose a cutoff and then select the latest forecast available before that cutoff.

Historical lead-1 distribution:

- `1` forecast for target day: `29` target days.
- `2` forecasts for target day: `1,119` target days.
- `3` forecasts for target day: `136` target days.
- `4` forecasts for target day: `54` target days.
- `5` forecasts for target day: `30` target days.
- `6` forecasts for target day: `87` target days.
- `7` forecasts for target day: `36` target days.
- `8` forecasts for target day: `45` target days.
- `9` forecasts for target day: `70` target days.
- `10` forecasts for target day: `5,856` target days.
- `11` forecasts for target day: `1,593` target days.
- `12` forecasts for target day: `409` target days.
- `13` forecasts for target day: `140` target days.
- `14` forecasts for target day: `37` target days.
- `15` forecasts for target day: `16` target days.
- `16` forecasts for target day: `6` target days.
- `17` forecasts for target day: `2` target days.

The mode is `10` lead-1 local forecasts per target day. The second largest bucket is `11`. That matches the fact that the local forecast is often reissued repeatedly from late afternoon through late night HKT, but the actual count varies with warnings, tropical cyclone conditions, corrections, or operational schedule changes.

## Current Live July 5, 2026 State

Target market example:

- Polymarket event: `https://polymarket.com/event/highest-temperature-in-hong-kong-on-july-5-2026`.
- HKO target date: `2026-07-05` in Hong Kong local calendar time.
- Settlement source: HKO Daily Extract, field `Absolute Daily Max (deg. C)`.
- Settlement precision: one decimal place.
- Revisions after initial publication do not count for the market.

Clock conversion for this specific target:

- Hong Kong is UTC+8.
- Stockholm on 2026-07-04/05 is CEST, UTC+2.
- HKT is 6 hours ahead of Stockholm during CEST.
- HKG target day `2026-07-05 00:00 HKT` begins at `2026-07-04 18:00 CEST`.
- HKG target day `2026-07-05 23:59 HKT` ends at `2026-07-05 17:59 CEST`.
- The selected strategy cutoff `T-1 23:59 HKT` equals `2026-07-04 17:59 CEST`.
- That cutoff is exactly one minute before the HKG target day begins, and about 24 hours before the HKG target day ends.

Important current DB fact:

- The local Postgres historical archive currently has no rows for `target_date = 2026-07-05`.
- Direct query result for strict usable local min/max rows on `2026-07-05`: `0`.
- The latest strict usable local min/max target date currently in the DB is `2026-06-21`.
- Therefore, live July 5 trading cannot rely on the local DB already containing the July 5 live forecast. It must either ingest the live Info.gov pages first, or the live prediction code must parse Info.gov directly and then persist the parsed row.

## Current Same-Source Live Forecasts Verified on 2026-07-04

The current Info.gov weather press-release index for 2026-07-04 is:

- `https://www.info.gov.hk/gia/wr/202607/04.htm`
- Also visible through `https://www.info.gov.hk/gia/wr/today.htm` at the time of verification.

The latest same-source local forecast page verified at context creation time was:

- Title: `PRESS WEATHER NO. 290 - LOCAL WEATHER FORECAST`
- URL: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400858.htm`
- Dispatch time: `21:45 HKT on 04.07.2026`
- Stockholm time: `15:45 CEST on 04.07.2026`
- Forecast period: `Weather forecast for tonight and tomorrow`
- Forecast target interpretation: since the issue date is `2026-07-04 HKT`, `tomorrow` is `2026-07-05 HKT`.
- Forecast min: about `27 C`.
- Forecast max: around `31 C in the urban areas`.
- New Territories caveat: `a couple of degrees higher in the New Territories`.
- For this market, the urban/HKO-station max is the relevant forecast anchor. Do not use the New Territories add-on as the HKO station forecast max.

The latest three same-source local forecast pages verified were:

- `PRESS WEATHER NO. 290 - LOCAL WEATHER FORECAST`
  - URL: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400858.htm`
  - Dispatch: `21:45 HKT`, `2026-07-04`
  - Stockholm: `15:45 CEST`, `2026-07-04`
  - July 5 min/max: about `27 / 31 C`
  - Text includes: mainly cloudy, a few showers, heavier showers and squally thunderstorms tonight and at first tomorrow, bright periods during the day, max around 31 in urban areas.

- `PRESS WEATHER NO. 280 - LOCAL WEATHER FORECAST`
  - URL: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400831.htm`
  - Dispatch: `20:46 HKT`, `2026-07-04`
  - Stockholm: `14:46 CEST`, `2026-07-04`
  - July 5 min/max: about `27 / 31 C`
  - Same max anchor: around `31 C in the urban areas`.

- `PRESS WEATHER NO. 270 - LOCAL WEATHER FORECAST`
  - URL: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400800.htm`
  - Dispatch: `19:46 HKT`, `2026-07-04`
  - Stockholm: `13:46 CEST`, `2026-07-04`
  - July 5 min/max: about `27 / 31 C`
  - Same max anchor: around `31 C in the urban areas`.

The forecast had not changed across those latest three same-source bulletins: all carried a `27 / 31 C` local urban forecast range for July 5.

## HKO OpenData Cross-Check, But Not Primary Source

HKO OpenData `flw`:

- URL: `https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en`
- Data type: local weather forecast JSON.
- Verified update time: `2026-07-04T21:45:00+08:00`.
- Text matched the Info.gov `PRESS WEATHER NO. 290` local forecast:
  - Minimum temperature about `27 C`.
  - Maximum temperature around `31 C in the urban areas`.
- This is the same HKO product concept, and it is useful as a sanity check.
- However, the historical DB archive is specifically Info.gov press-release pages. For strict apples-to-apples live strategy operation, Info.gov remains the primary source and `flw` should be treated as secondary confirmation unless the ingestion code explicitly proves equivalence and stores source provenance.

HKO OpenData `fnd`:

- URL: `https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en`
- Data type: 9-day weather forecast JSON.
- Verified update time: `2026-07-04T18:50:00+08:00`.
- For `forecastDate = 20260705`, it showed:
  - Forecast min: `27 C`.
  - Forecast max: `31 C`.
  - PSR: `Medium High`.
- This is useful context only.
- It should not be the official anchor for the currently evaluated strategy because the DB source and product are different.

## Strategy Cutoff Contract From the Current Experiment

Experiment:

- ID: `0215`
- Slug: `gpt_pro_point_forecast_strategy`
- Main script: `scripts/run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py`
- Results directory: `experiments/0215_gpt_pro_point_forecast_strategy/results`
- Main report: `experiments/0215_gpt_pro_point_forecast_strategy/final_forecasting_strategy_report.md`

Main run summary:

- Target rows: `8,765`.
- Lead-1 forecast rows loaded into the experiment: `80,089`.
- Lead-0 forecast rows loaded into the experiment: `24,565`.
- Climate rows: `187,025`.
- Feature rows: `35,060`.
- Feature columns: `1,166`.
- Prediction rows: `265,888`.
- Ablation prediction rows: `227,904`.
- Leakage row audit failures: `0`.

Selected cutoff:

- `23:59 HKT on T-1`.
- In Stockholm summer time: `17:59 CEST on T-1`.
- In Stockholm winter time: `16:59 CET on T-1`.

For the July 5, 2026 market:

- Target day T: `2026-07-05 HKT`.
- Strategy cutoff: `2026-07-04 23:59 HKT`.
- Stockholm equivalent: `2026-07-04 17:59 CEST`.
- At that cutoff, choose the latest same-source Info.gov `LOCAL WEATHER FORECAST` whose dispatch time is less than or equal to cutoff.
- At context creation time, the latest verified same-source forecast was `21:45 HKT`. Later before cutoff, a `23:15 HKT` or `23:45 HKT` local forecast may appear. The live decision system must re-check the Info.gov source at or immediately before the cutoff.

Selected model/result:

- Selected model ID: `B3_grouped_residual_shrinkage`.
- Selected model family: `baseline`.
- Selected official-row-only score:
  - n: `4,747`.
  - MAE: `0.9216066007226759 C`.
  - RMSE: `1.1826760995241477 C`.
  - Median absolute error: `0.7372210878929693 C`.
  - Bias: `0.012704769322469249 C`.
  - p80 absolute error: `1.477812449139664 C`.
  - p90 absolute error: `1.9641225406986251 C`.
  - p95 absolute error: `2.3954636051070834 C`.
  - Max absolute error: `4.403362920320813 C`.
  - Evaluation date range: `2011-01-01` through `2023-12-31`.

Raw official baseline at the same cutoff:

- Model ID: `B1_raw_official_latest`.
- n: `4,747`.
- MAE: `0.9274910469770383 C`.
- RMSE: `1.1915190793489194 C`.
- Median absolute error: `0.7000000000000028 C`.
- Bias: `-0.12285654097324625 C`.
- p80 absolute error: `1.5 C`.
- p90 absolute error: `2.0 C`.
- p95 absolute error: `2.3999999999999986 C`.
- Max absolute error: `4.5 C`.

Improvement versus raw official at the same cutoff:

- MAE improvement: about `0.005884446254362463 C`.
- RMSE improvement: about `0.00884297982477178 C`.
- This improvement is real in the run artifacts, but it is extremely small.
- The hard promotion gates did not pass because the run required at least `0.035 C` improvement in MAE and RMSE versus raw official.
- Bias gate passed.
- Row-count gate passed.
- Overall hard gates did not pass.

Practical interpretation:

- The selected model is not a magical standalone ML engine.
- It is mainly the latest official HKO local forecast max before cutoff, plus a small historical grouped residual correction.
- It uses historical HKO forecast errors grouped by features such as month, season, forecast range bin, month crossed with integer forecast max, forecast max bin, and issue-hour family.
- It does not currently rely on GribStream/NWP for the selected live decision.
- It does not currently prove that a complex ML model beats the official local forecast by a meaningful margin.
- For live trading, the dominant signal is the latest same-source HKO local forecast max.

## Best Current Live Forecast Anchor for July 5, 2026

As of `2026-07-04 22:25 HKT / 16:25 CEST`, the same-source Info.gov local forecast anchor for July 5 was:

- Minimum: about `27 C`.
- Maximum: around `31 C in the urban areas`.
- Target station implication: use `31 C` as the official local forecast Tmax anchor for HKO station modeling.
- Weather regime: active southerly airstream, showers, squally thunderstorms, bright periods during the day.
- Operational caveat: the source should be re-polled at the actual strategy cutoff `2026-07-04 23:59 HKT / 17:59 CEST`, because a later local forecast can appear before cutoff.

Do not treat the `New Territories a couple of degrees higher` sentence as the HKO station max. Polymarket resolves to the Hong Kong Observatory station daily extract value, not a territory-wide maximum and not New Territories localized readings.

## Live Ingestion Requirements

For a production live-prediction path, implement or verify these exact steps:

1. Fetch the Info.gov weather press-release index for the relevant Hong Kong date:
   - Preferred historical-index URL: `https://www.info.gov.hk/gia/wr/YYYYMM/DD.htm`.
   - Current-day shortcut: `https://www.info.gov.hk/gia/wr/today.htm`.

2. Find only links whose title contains:
   - `LOCAL WEATHER FORECAST`.

3. Exclude:
   - `9-DAY WEATHER FORECAST`.
   - `HOURLY READINGS`.
   - `SOUTH CHINA COASTAL WATERS`.
   - `WEATHER OF OTHER CITIES`.
   - Warning-only bulletins.
   - Tropical cyclone bulletins unless they are embedded inside the local forecast page as context.

4. Fetch each candidate local forecast page.

5. Parse and persist:
   - `source = 'info_gov'`.
   - `source_url`.
   - `product_type = 'local'`.
   - `title`.
   - `index_date`.
   - `snapshot_at_hkt`.
   - `snapshot_at_utc`.
   - `issue_at_hkt` from the dispatch line.
   - `issue_at_utc`.
   - `forecast_period`.
   - `target_date`.
   - `target_issue_lead_days`.
   - `forecast_min_c`.
   - `forecast_max_c`.
   - `forecast_range_c`.
   - `forecast_midpoint_c`.
   - `temperature_text`.
   - `full_text`.
   - `raw_sha256`.
   - `raw_path` or equivalent raw HTML/text archive path.
   - `parse_status`.
   - `parse_notes`.
   - `row_quality_status`.

6. For a target day T and a T-1 cutoff:
   - Keep only forecasts issued at or before cutoff.
   - Keep only forecasts whose target date resolves to T.
   - Choose the latest issue time before cutoff as the official anchor.
   - If multiple pages have the same issue time, deduplicate by `raw_sha256` and prefer the canonical Info.gov URL.

7. For July 5, 2026:
   - Target date T: `2026-07-05 HKT`.
   - Strategy cutoff: `2026-07-04 23:59 HKT`.
   - Stockholm cutoff: `2026-07-04 17:59 CEST`.
   - Live pull should happen immediately before this cutoff, and again after the cutoff only for audit, not for the as-of prediction.

8. Hard fail if:
   - No Info.gov local forecast page can be fetched.
   - The latest page lacks a dispatch line.
   - The parser cannot resolve whether `tomorrow` maps to target day T.
   - The parser cannot extract `forecast_max_c`.
   - The latest available forecast was issued after the cutoff.
   - The code silently falls back to the 9-day API as if it were the same source.

## Source Hierarchy for Live Prediction

Primary source:

- Info.gov local forecast press-release pages.
- This is the only source that is confirmed to match the historical archive used in the current strategy.

Secondary confirmation source:

- HKO OpenData `flw`, local weather forecast JSON.
- It can be used to sanity-check that the text and update time match the latest Info.gov local forecast.
- It should not replace Info.gov unless the system explicitly logs the source substitution and the research is rerun using that source history.

Diagnostic-only source:

- HKO OpenData `fnd`, 9-day weather forecast JSON.
- Useful for context, but not apples-to-apples with the selected historical forecast archive.

Do-not-use-as-anchor sources:

- 5-day/7-day/9-day Info.gov bulletin-only rows currently present in the historical table.
- GribStream/NWP archive.
- Weather website widgets.
- Airport pages.
- Human-estimated New Territories add-on values.

## Testing Checklist for Future Implementation

A correct live-source implementation should be tested against these checks:

- Given the archived DB sample URL `https://www.info.gov.hk/gia/wr/202606/20/P2026062000831.htm`, the parser should output:
  - product type `local`
  - source `info_gov`
  - title `PRESS WEATHER NO. 165 - LOCAL WEATHER FORECAST`
  - issue time `2026-06-20 23:45 HKT`
  - target date `2026-06-21`
  - min/max `29 / 33 C`
  - target lead `1`

- Given live page `https://www.info.gov.hk/gia/wr/202607/04/P2026070400858.htm`, the parser should output:
  - product type `local`
  - source `info_gov`
  - title `PRESS WEATHER NO. 290 - LOCAL WEATHER FORECAST`
  - issue time `2026-07-04 21:45 HKT`
  - target date `2026-07-05`
  - min/max `27 / 31 C`
  - target lead `1`

- Given the 2026-07-04 index page, the link selector should identify the latest local forecast links in descending recency:
  - NO. 290
  - NO. 280
  - NO. 270
  - then earlier local forecasts on the same day.

- Given cutoff `2026-07-04 23:59 HKT`, the forecast selector should return the latest local forecast issued at or before that cutoff. If a later 23:15 or 23:45 page appears before the cutoff, it should supersede NO. 290. If no later page appears, NO. 290 remains the latest verified source.

- Given the HKO `fnd` API returns July 5 min/max `27 / 31 C`, the implementation may log it as cross-check context but must not label it as the primary historical-equivalent source.

- Given the local DB has no row for `target_date = 2026-07-05`, the live prediction code must not pretend that the historical DB already contains the live row. It must fetch and persist or fetch and parse live.

## Concrete Current Recommendation

For the July 5, 2026 Polymarket market, the next correct operational step is:

1. At or immediately before `2026-07-04 17:59 CEST` Stockholm time, fetch the Info.gov weather press-release index.
2. Identify the latest `LOCAL WEATHER FORECAST` page issued at or before `2026-07-04 23:59 HKT`.
3. Parse the July 5 urban/HKO-relevant max from that page.
4. Use that max as the official forecast anchor.
5. Apply only the model/residual correction that was trained/evaluated on the same Info.gov local source.
6. Log the raw source URL, dispatch time, parsed min/max, target date, cutoff, and final point forecast.
7. Do not use a post-cutoff page for the trading decision.

As of the creation of this context file, the latest verified same-source forecast said July 5 maximum temperature would be around `31 C` in the urban areas. That is the live same-source anchor until a later same-source Info.gov local forecast supersedes it before the selected cutoff.

## Source URLs Used for This Context

- Info.gov current/day index: `https://www.info.gov.hk/gia/wr/today.htm`
- Info.gov explicit July 4, 2026 index: `https://www.info.gov.hk/gia/wr/202607/04.htm`
- Latest verified same-source local forecast: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400858.htm`
- Previous same-source local forecast: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400831.htm`
- Previous same-source local forecast: `https://www.info.gov.hk/gia/wr/202607/04/P2026070400800.htm`
- HKO RSS schedule page: `https://rss.weather.gov.hk/rsse.html`
- HKO OpenData local weather forecast API: `https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en`
- HKO OpenData 9-day forecast API: `https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en`

