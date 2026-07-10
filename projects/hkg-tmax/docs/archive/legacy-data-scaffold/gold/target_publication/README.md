# Target Publication Gold Artifacts

This directory holds generated Daily Extract publication ledgers. CSV payloads
are ignored by Git; regenerate the current checkpoint with:

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6
```

Rows marked `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST` are not accepted as
provider first-publication proof. G1 needs repeated near-publication polling
before first-published target parity can pass.
