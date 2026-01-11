# WX-111 — Epic #1 documentation pack (hardcore) + developer onboarding checklist

## Objective
Produce serious documentation so implementation via agent tooling is straightforward.

## Required docs (must be committed)
- `agents.md` — rules + guardrails for future agents
- `docs/time-semantics.md` — final, canonical time definitions
- `docs/station-mapping.md` — Kalshi → NWS CLI → IEM mapping
- `docs/architecture.md` — modules, data flow, schemas
- `docs/runbook.md` — how to run backfills + common failures
- `docs/sql/` — example audit queries (no-leakage, completeness)

## Acceptance Criteria
- [ ] A brand-new dev can:
  - build repo
  - point to MySQL
  - run a 30-day backfill for one station
  - run the audit report
  - without reading code
- [ ] Documentation includes at least 2 worked examples:
  - one station in DST season
  - one station in standard time season
- [ ] Every external dependency URL is listed (Kalshi + IEM)

