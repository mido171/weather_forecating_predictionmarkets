# Weather Data Extraction

This branch is the extraction-focused cut of the original weather prediction markets repo.

Kept here:

- Weather.com/Wunderground historical observation ingestion and daily max derivation.
- IEM/Mesonet CLI, MOS, ASOS, AFOS, and Climodat fetchers.
- NCEI/NWS/ACIS/NDFD truth and grid fetchers.
- Gribstream history/forecast clients and parsers.
- NOAA model-grid extraction workers using Herbie, direct GRIB, and xarray.
- Kalshi market-data, candlestick, orderbook, and series metadata fetchers.
- Polymarket public event/price/trade downloaders.
- Small local readers/exporters used to pull extracted truth from DB tables.

Removed from this cut:

- Generated artifacts and model outputs.
- UI app code.
- Planning docs, Jiras, and long-form reports.
- Model training sweeps and calibration experiments.
- Backtest-only scripts that do not fetch or parse upstream data.

## Smoke Tests

Run the local smoke tests without hitting external APIs:

```powershell
python -m compileall -q .
python smoke_tests/smoke_extractors.py
```

Java compile smoke:

```powershell
mvn -q -DskipTests package
```

The smoke harness checks representative parser/normalizer paths for each upstream extraction family. Real live fetching still requires provider credentials and network access.
