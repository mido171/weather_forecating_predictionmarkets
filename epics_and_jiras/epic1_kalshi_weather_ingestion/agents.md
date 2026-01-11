# agents.md — Implementation Handoff Rules (for Codex / agent tooling)

## Scope for this repo (important)
This repository will implement:
- Epic #1: Java ingestion + MySQL persistence for MOS + CLI
Later epics will add:
- Epic #2: Python ML training
- Epic #3: Python backtesting vs Kalshi prices

## Non-negotiable rules
1) All JPA entities, enums, and shared DB models MUST live in module: `models`
2) All timestamps are stored in UTC, except:
   - `target_date_local` which is a DATE
   - `station_zoneid` stored as string
3) No forward-looking leakage:
   - MOS runtime_utc used for as-of features must be <= asof_utc
4) Idempotency:
   - Every ingestion job can be killed and restarted without manual cleanup
5) Auditing:
   - store raw payload hashes and retrieved timestamps
6) Build:
   - `mvn clean install` from root must pass at every commit

## Implementation style
- Prefer Flyway for schema migrations (versioned SQL)
- Prefer `WebClient` (Spring reactive) or `OkHttp` with retries
- Strict JSON parsing (Jackson), with schema versioning stored in DB

