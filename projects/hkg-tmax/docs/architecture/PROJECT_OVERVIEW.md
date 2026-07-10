# Project Overview

This repository is a Codex-operated research bootstrap for forecasting the
first-published, contract-authoritative Hong Kong Observatory daily maximum
temperature as a leakage-free probability distribution.

The project order is intentionally conservative:

1. Prove the repository, archive, and experiment ledger work.
2. Prove settlement target semantics and historical-label parity.
3. Select and freeze a point-in-time forecast cutoff.
4. Acquire and audit lawful source vintages.
5. Build transparent baselines before advanced models.

Authoritative starting points:

- `AGENTS.md` defines the operating constitution.
- `docs/status/NEXT_ACTIONS.md` defines the verified next-action queue.
- `docs/research/RESEARCH_CHARTER.md` defines the research hierarchy and scope.
- `docs/contracts/ASOF_AND_LEAKAGE.md` defines timestamp and leakage rules.
- `docs/operations/PRODUCTION_GATE.md` defines conditions for any production use.

Predictive modelling and machine learning are blocked until the upstream target,
archive, as-of, source, and baseline gates pass.
