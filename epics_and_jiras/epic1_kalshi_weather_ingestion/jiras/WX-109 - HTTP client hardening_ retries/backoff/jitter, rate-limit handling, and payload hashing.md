# WX-109 — HTTP client hardening: retries/backoff/jitter, rate-limit handling, and payload hashing

## Objective
Implement hardened HTTP client utilities (in `common` module) for calling:
- Kalshi API endpoints
- IEM endpoints

## Required behaviors
- Timeouts:
  - connect timeout
  - read timeout
- Retries with jittered exponential backoff for transient failures:
  - 429
  - 503
  - network timeouts
- Circuit breaker or max retry cap:
  - avoid infinite retry storms

## Payload hashing
For every external response stored:
- compute SHA-256 over raw bytes
- store hash in DB
- optionally store raw payload only if new hash not seen before (dedupe)

## Acceptance Criteria
- [ ] Retries occur for 503 and eventually succeed in test with a mocked server.
- [ ] 429/503 cause exponential backoff (with jitter).
- [ ] Payload hash is stable and tested (same bytes → same hash).
- [ ] Logging includes:
  - endpoint
  - status code
  - retry count
  - backoff delay
  - correlation/job id

