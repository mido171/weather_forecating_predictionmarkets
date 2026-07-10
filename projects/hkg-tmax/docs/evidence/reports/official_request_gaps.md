# Official Request Gaps

These gaps remain after public-source acquisition and bounded official-source discovery. They must not block other acquisition work.

## HKO Dense Headquarters / Regional Sub-Daily History Request Package

- requested station: HKO Headquarters / WMO 45005 plus all available regional automatic weather stations
- requested period: 1984-01-01 through 2019-12-31, or earliest available dense archive through start of public DATA.GOV.HK history
- requested cadence: one-, five-, or ten-minute observations, whichever is officially available
- requested variables: temperature, max/min, relative humidity, pressure, wind direction/speed/gust, rainfall, visibility, present weather/RHR fields, solar radiation, UV, station metadata and relocations
- required metadata: issue/publication timestamps, station IDs, units, missing-value codes, QC flags, revision policy, license and commercial-use terms
- current public counterpart: HKO live feeds and DATA.GOV.HK historical ZIP archives where available from 2020/2021 onward
- status: `official_request_required`

## Forecast / ARWF JSON Vintages

- requested products: FLW, FND, Warnings, SWT, ARWF station/grid forecasts, RHR/current-weather JSON vintages
- requested period: earliest retained provider archive through present
- status: `historically_unavailable_or_request_required`; RSS historical archives are already acquired where public
