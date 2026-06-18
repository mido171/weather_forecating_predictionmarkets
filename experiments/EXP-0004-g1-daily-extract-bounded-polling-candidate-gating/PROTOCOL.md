# Predeclared Protocol

## Scope

G1 infrastructure only. No forecasts, predictive features, ML, market prices,
order books, trades, liquidity, execution, or market replay.

## Candidate

- Add bounded repeat polling flags to `scripts/poll_daily_extract.py`.
- Add active polling start and watched-date gating to publication ledger logic.
- Add tests proving conservative default behavior and candidate-date gating.

## Metrics

- poll iterations completed;
- ledger row count;
- evidence class counts;
- provider-first candidates, if any;
- revision count.

## Acceptance

- Bounded poll command exits on its own.
- Candidate evidence class cannot appear without explicit watched date and
  active start marker.
- Existing June 2026 rows remain archive-first-observed in the default run.
- `pytest`, `validate all`, `ruff`, and `mypy` pass.

## Rejection

Reject if candidate gating is ambiguous, unsafe by default, or untested.
