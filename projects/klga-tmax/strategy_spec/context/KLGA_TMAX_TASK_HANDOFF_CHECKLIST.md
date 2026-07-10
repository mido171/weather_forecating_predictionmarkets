# KLGA Tmax Task Handoff Checklist

Use this checklist before calling any numbered KLGA task complete.

## Scope And Source Reading

- [ ] Read `AGENTS.md`.
- [ ] Read the main strategy spec.
- [ ] Read `supplemental_doc_1.md`.
- [ ] Read `supplemental_doc_1_patch_1.md`.
- [ ] Read the Postgres persistence context.
- [ ] Read previous implemented task deep dives that this task depends on.
- [ ] Read the exact source spec from the task folder.
- [ ] Confirm whether live fetching is authorized or not authorized.

## Implementation

- [ ] Code changes are under `implementation/src/klga_tmax`.
- [ ] Tests are under `implementation/tests`.
- [ ] New schema changes have an Alembic migration.
- [ ] Alembic `revision` value is 32 characters or fewer.
- [ ] Runtime SQL uses parameters for external values.
- [ ] Provider credentials are not hardcoded.
- [ ] Raw provider payload persistence exists before normalized persistence when the task fetches provider data.
- [ ] Availability timestamps are explicit and traceable.
- [ ] Feature eligibility follows `effective_available_at_utc <= cutoff_utc`.
- [ ] Station IDs, provider IDs, and coordinate tiers come from `station_universe.py` or `registry.station_registry`.
- [ ] `db inspect-contract` is updated when new objects become required contracts.
- [ ] A task-specific validation command exists when persistent data or a contract is introduced.
- [ ] CLI commands are audited through `audit.pipeline_runs` when they perform DB work.

## Tests

- [ ] Unit tests cover parsing, mapping, and validation rules.
- [ ] Negative tests cover malformed provider rows or missing required DB rows.
- [ ] Migration/schema tests cover new tables, constraints, and indexes.
- [ ] CLI tests cover command registration and failure exit codes.
- [ ] Fixture data is small and does not include secrets.
- [ ] Tests do not depend on live external APIs unless explicitly marked and skipped by default.

## Verification Commands

Run from `implementation`:

```powershell
python -m compileall -q src tests
python -m pytest -q
python -m klga_tmax.cli --help
python -m klga_tmax.cli validate --help
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli db migrate
python -m klga_tmax.cli db inspect-contract
python -m klga_tmax.cli validate foundation
python -m klga_tmax.cli validate station-universe
```

Also run:

- [ ] The task-specific CLI command with fixture or dry-run inputs.
- [ ] The task-specific validation command.
- [ ] A direct DB readback query for newly created tables or seeded rows.
- [ ] Documentation quality gate for the deep-dive document when the script is available.

## Documentation

- [ ] Create `strategy_spec/context/KLGA_TMAX_<NN>_<TASK_SLUG>_IMPLEMENTATION_DEEP_DIVE.md`.
- [ ] List every changed file.
- [ ] Explain schema/table/index definitions.
- [ ] Explain CLI commands and exit codes.
- [ ] Explain provider API assumptions and live-fetch boundary.
- [ ] Explain availability/leakage rules.
- [ ] Explain migration and rollback.
- [ ] Include exact verification commands and outputs.
- [ ] State known limitations and next handoff.
- [ ] Update `KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md` status.

## Final Response

The final response should include:

- [ ] What changed.
- [ ] Where the main files are.
- [ ] Which commands passed.
- [ ] Whether DB migration/validation passed.
- [ ] Whether live data was fetched.
- [ ] Any blocker, credential need, quota limit, or provider-data limitation.
