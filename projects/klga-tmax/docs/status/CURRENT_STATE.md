# Current state

Last structural verification: 2026-07-10.

- KLGA is a first-class project at `projects/klga-tmax`, not a bootstrap or
  nested implementation tree.
- Source/tests/config/scripts/experiments and Alembic assets live directly at
  the project root.
- Strategy, numbered task specifications, context, and implementation docs are
  consolidated under `docs/`.
- Runtime artifacts are external through `KLGA_RUN_ROOT`/`KLGA_ARTIFACT_ROOT`.
- Default concurrency is one and live/prod operation is fail-closed.
- Existing scientific/task completion claims were preserved but not
  independently revalidated by the repository-structure migration.
