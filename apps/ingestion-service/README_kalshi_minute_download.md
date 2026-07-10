# Kalshi Minute Bucket Downloader (KXHIGHMIA)

This script downloads minute-by-minute YES price candlesticks for each daily KXHIGHMIA event and writes one CSV per event day.

## What It Produces
- One CSV per day: `KMIA_YYYYMMDD.csv`
- Output rows only for minutes where **at least one** bucket has a price
- Column 1: `timestamp` as ISO8601 UTC with trailing `Z` (end-of-minute timestamps)
- Columns 2..N: Kalshi bucket labels (`market.subtitle`) including the `°` symbol
- Values: `price.mean_dollars * 100` (cents), empty cell if missing

## Install Dependencies

Windows PowerShell:
```powershell
python -m pip install -r ingestion-service\requirements.txt
```

Linux/macOS:
```bash
python -m pip install -r ingestion-service/requirements.txt
```

## Run (Default Output Path)

Windows PowerShell:
```powershell
python ingestion-service\scripts\kalshi_download_kxhighmia_minute.py --start-date 2025-01-01 --end-date 2025-12-31
```

Linux/macOS:
```bash
python ingestion-service/scripts/kalshi_download_kxhighmia_minute.py --start-date 2025-01-01 --end-date 2025-12-31
```

Default output directory:
```
data/kalshi_backtest_data
```

## Run (Custom Output Path)

Windows PowerShell:
```powershell
python ingestion-service\scripts\kalshi_download_kxhighmia_minute.py --start-date 2025-01-01 --end-date 2025-12-31 --out-dir C:\path\to\kalshi_backtest_data
```

Linux/macOS:
```bash
python ingestion-service/scripts/kalshi_download_kxhighmia_minute.py --start-date 2025-01-01 --end-date 2025-12-31 --out-dir /path/to/kalshi_backtest_data
```

## Notes
- The script respects Kalshi's historical cutoff and uses the correct historical endpoints when required.
- A `manifest.json` is written in the output directory with per-date metadata and any errors.
- The output format matches the expected `KMIA_YYYYMMDD.csv` structure (e.g., `KMIA_20251225.csv`).
- KXHIGHMIA event tickers are ambiguous between `DDMONYY` and `YYMONDD` (both can exist, but refer to different dates). The script resolves this by trying both and selecting the one whose event title date matches the requested day.
