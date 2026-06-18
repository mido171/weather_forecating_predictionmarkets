# Predeclared Protocol

Complete before accepting polling results.

## Target And Horizon

- target version: `hko_daily_absolute_max_first_published_pending_g1`
- target source: HKO Daily Extract monthly backing payload
- horizon: not applicable; target publication capture only
- prediction unit: no forecasts or model predictions

## Sample

- development: current-month Daily Extract backing payloads available during
  this run
- validation: command rerun against archived sidecars and parser tests
- locked test: not applicable
- live shadow: future repeated polling after this infrastructure exists
- inclusion:
  - HKO Daily Extract catalog snapshot;
  - requested current-month Daily Extract backing payload;
  - all parsed numeric and nonnumeric target rows for the requested month.
- exclusion:
  - market prices/books/trades/liquidity/execution/replay;
  - predictive features or model fitting;
  - any claim of true provider first publication unless polling was active
    before the row appeared.

## Candidate

- feature/formula/model: polling command plus publication ledger builder
- transformations:
  - archive raw before parsing;
  - parse target rows with `Decimal`;
  - store first archive observation and latest archive observation by local date;
  - preserve raw hashes and sidecar paths;
  - classify evidence conservatively.
- allowed hyperparameters: requested year/month only
- seeds: none

## Metrics

- primary:
  - number of Daily Extract payloads archived;
  - number of parsed target rows;
  - number of ledger rows with raw hash and retrieved-at provenance.
- guardrails:
  - zero rows labelled provider-first-publication without evidence;
  - zero raw-overwrite events;
  - fail closed on parse/source/precision/date issues;
  - all generated ledgers declare whether evidence is archive-first-observed or
    provider-first-publication.

## Acceptance

- Command archives the catalog and monthly backing payload immutably.
- Command builds `data/gold/target_publication/daily_extract_first_seen.csv`.
- Command builds `reports/daily_extract_publication.md`.
- Ledger rows contain local date, target value, completeness/quality, first
  archive retrieved-at, latest archive retrieved-at, source hashes, source paths,
  and evidence class.
- Existing rows are not mislabelled as true first publication.
- `pytest`, `validate all`, `ruff`, and `mypy` pass before any commit.

## Rejection Or Block

Reject or block if the command cannot preserve raw provenance, creates
misleading first-publication claims, cannot parse current Daily Extract safely,
or weakens target/leakage checks.

## Locked-Test Decision

Not authorized. No forecast-performance data or model evaluation is touched.
