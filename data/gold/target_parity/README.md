# Target Parity Gold Artifacts

This directory holds generated HKO target-parity artifacts. CSV/JSON payloads
are intentionally ignored by Git under the repository data policy; regenerate
them with:

```powershell
.\.venv\Scripts\python.exe scripts\build_target_parity.py --year 2026 --month 5 --daily-source-id hko_daily_extract_202605
```

Current G1 status: latest Daily Extract versus latest CLMMAXT can be measured,
but first-publication parity is not proven until archived first Daily Extract
payloads are available for target dates.
