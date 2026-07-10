# Ingestion service agent rules

The root `AGENTS.md` applies. Read this service's `README.md`, active Spring
profiles, bound configuration classes, runner conditions, and focused tests
before changing behavior.

- Ordinary startup must remain local, loopback-bound, and inert: no provider
  fetch, backfill, pilot job, scheduler, schema mutation, or worker process.
- Administrative/mutating controllers require explicit enablement and the
  local control-token filter. Never log or echo token/API-key values.
- Provider jobs require explicit execution, narrow location/date/task scope,
  hard request/runtime/retry/queue budgets, and one worker by default.
- Credentials come from environment variables; examples are blank. Never add a
  credential-bearing URL or literal password.
- Runtime data belongs under external/configured storage or ignored `var/`, not
  `src/main/resources`.
- Use focused Spring context/config/controller tests. Provider tests are mocked;
  no live request is an implicit verification step.
