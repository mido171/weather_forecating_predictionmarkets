# KLGA Tmax: start here

KLGA is the New York/LaGuardia Tmax research component of the weather-markets
monorepo. It combines point-in-time provider acquisition, Postgres contracts,
forecast evaluation, and weather-market research.

Safe orientation:

1. Read `../../AGENTS.md` and `AGENTS.md`.
2. Read `docs/status/CURRENT_STATE.md` and `docs/INDEX.md`.
3. Confirm Git identity and tracked-only status.
4. Configure ignored local environment values from `.env.example` without
   printing secrets.
5. Run only focused offline checks for the task.

Provider commands are dry/fail-closed without `--execute` and hard budgets.
Database migrations and live/prod mode are never startup actions.
