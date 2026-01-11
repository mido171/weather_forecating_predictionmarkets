# WX-101 — Repo scaffold for Epic #1 (multi-module Maven + Spring Boot + models module)

## Objective
Create the new repository skeleton so that **`mvn clean install`** succeeds from root and provides the module structure required by later JIRAs.

## Requirements
- Root Maven parent `pom.xml` with modules:
  - `models` (JPA entities, enums, shared DTOs, Flyway migrations)
  - `common` (time helpers, HTTP clients, retry utils, hashing)
  - `ingestion-service` (Spring Boot app; scheduled jobs / batch jobs)
- Java version pinned (e.g., 21) + consistent toolchain
- Spring Boot configured for MySQL (but can run tests with H2)
- Add minimal `.editorconfig`, `checkstyle` or `spotless` (optional but recommended)
- Add `README.md` describing how to build and run.

## Acceptance Criteria
- [ ] Running `mvn clean install` at repo root completes successfully on a clean machine.
- [ ] Module `models` is present and contains an empty placeholder JPA entity package.
- [ ] Module `ingestion-service` starts (at least with an in-memory profile) without DB connectivity.
- [ ] A standard logging framework is configured (SLF4J + Logback).
- [ ] `docs/` folder committed (even if placeholder).

## Notes / Sources
- This epic ultimately consumes Kalshi API (public, no auth required for market data endpoints): {"base":"https://api.elections.kalshi.com/trade-api/v2"}.
- Kalshi provides settlement sources per series via `GET /series/{ticker}`.

