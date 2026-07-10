# T06 Completion Record

Task: Resumable GribStream Runs Client and Raw Landing Zone

Evidence folder: `experiments/0213_gribstream_resumable_runs_client`

## What Was Done

- Added the reusable `hkg_tmax.gribstream` client, selector resolver, planner, normalizer, and PostgreSQL store.
- Added the T06 runner and status checker.
- Resolved the exact GFS `TMP` / `2 m above ground` selector from the live shared-parameter catalog.
- Fetched a bounded GFS `/runs` smoke object as NDJSON gzip and loaded normalized values into the T04 lineage schema.

## Acceptance Finalization

- Duplicate request/value protection is enforced by canonical request SHA plus `nwp_core.point_value` upsert keys.
- Resume behavior is ledgered; `.part` files are retried under the same request SHA and completed raw objects are reused.
- Selector/run/valid/member lineage is stored in `catalog.variable_selector_snapshot`, `nwp_core.model_run`, and `nwp_core.point_value`.

## Open Blockers

- None
