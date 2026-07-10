# Leakage Control

This file is a compatibility entry point for workflows that refer to
`docs/06_LEAKAGE_CONTROL.md`.

The canonical point-in-time and leakage specification is
`docs/03_ASOF_AND_LEAKAGE.md`. Its requirements govern this repository:

- every time-varying record must carry defensible timestamp semantics where
  applicable;
- feature eligibility is `available_at <= forecast_cutoff`;
- missing or ambiguous availability fails closed;
- revised observations, latest-only forecasts, reanalysis, and final best
  tracks cannot be used as operational point-in-time inputs unless their
  historical availability is proven;
- preprocessing, calibration, feature selection, and model selection must be
  fitted inside the appropriate training window;
- locked-test access must be logged and restricted.

For implementation and audits, read and follow `docs/03_ASOF_AND_LEAKAGE.md`.
Do not treat this file as a separate or weaker leakage contract.
