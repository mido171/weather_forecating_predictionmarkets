# Conclusion

## Decision

`ACCEPTED`

## What The Evidence Supports

The Daily Extract poller now supports bounded repeat polling and candidate-date
gating. Unit tests prove that provider-first-publication candidate status
requires an active polling start marker and an explicit watched local date, and
that revisions override candidate status.

The bounded live run completed two poll iterations and exited normally. All 17
June 2026 rows remained `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST`.

## What The Evidence Does Not Support

This does not prove provider first-publication parity. No watched date was
observed appearing for the first time during this run.

This does not authorize G2, predictive modelling, machine learning, or market
backtesting.

## Leakage Review

PASS for this checkpoint. The output is target-publication evidence only and
does not enter a forecast feature table.

## Reproducibility Review

PASS after `pytest`, `validate all`, `ruff`, and `mypy`.

## Final Next Action

Run bounded polling around the next expected Daily Extract publication with:

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations <N> --interval-seconds <seconds> --active-polling-start-at now --watch-candidate-date <YYYY-MM-DD> --metrics experiments\EXP-0004-g1-daily-extract-bounded-polling-candidate-gating\results\metrics.json
```

Only review candidate status after the row actually appears under active
polling.
