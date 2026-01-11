# WX-110 — Data quality & leakage audit reports (CLI vs MOS availability, missingness, and asOf correctness)

## Objective
Provide automated verification reports that prove:
- no forward-looking leakage
- good coverage/completeness
- consistent station/date alignment

## Required checks (minimum)
1) No-leakage:
   - assert `chosen_runtime_utc <= asof_utc` for all rows
2) Station coverage:
   - each station has CLI rows for requested date range
3) MOS availability:
   - each station+model has mos_run coverage in expected eras
4) Feature coverage:
   - % missing tmax_f by station+model
5) CLI revisions:
   - count of days where tmax changed on re-ingest (optional)
6) Alignment:
   - feature rows exist only when target date exists within the MOS run forecast horizon

## Outputs
- A human-readable report (Markdown or HTML) saved to disk
- A machine-readable JSON report saved to disk

## Acceptance Criteria
- [ ] Running the report for a 1-year backfill produces both artifacts.
- [ ] The report contains clear PASS/FAIL sections.
- [ ] FAIL includes actionable diagnostics (e.g., sample offending rows).

