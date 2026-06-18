# Results

## Run integrity

- command: see `RUN_CONFIG.yaml`
- start/end: 2026-06-18T16:05Z to 2026-06-18T16:10Z UTC for fetch/verification/documentation pass
- code commit: no `HEAD` existed before this first commit
- dirty state: expected dirty state before first immutable commit
- data manifest hash: recorded after `MANIFEST.json` regeneration
- rows: 7 `bootstrap_now` sources, 14 raw retrieval events
- failed rows: 0 accepted source payloads failed archive verification
- leakage validator: `validate all` PASS with expected G1/G2 warnings
- reproducibility precheck: local `.venv` base/dev install, doctor, tests, validation, lint, mypy all PASS

## Primary result

| Metric | Baseline | Candidate | Absolute delta | Relative delta | 95% CI |
|---|---:|---:|---:|---:|---|
| G0 smoke acceptance | — | PASS | — | — | deterministic |

## Guardrails

- `make doctor`: FAIL in this Windows shell because `make` is not installed.
- `.venv\Scripts\python.exe -m hkg_tmax doctor`: PASS.
- `.venv\Scripts\python.exe -m pytest`: PASS, 34 tests.
- `.venv\Scripts\python.exe -m hkg_tmax validate all`: PASS, with expected warnings that target parity and primary horizon remain open.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`: PASS.
- `.venv\Scripts\python.exe -m mypy src`: PASS.
- `.venv\Scripts\python.exe -m hkg_tmax sources fetch --tag bootstrap_now`: PASS twice.
- `.venv\Scripts\python.exe -m hkg_tmax sources report`: PASS, generated `reports/source_inventory.md`.
- Archive verification: PASS for 7 sources and 14 sidecars.

## Year-by-year

Not applicable.

## Regime breakdown

Not applicable.

## Calibration

Not applicable.

## Boundary and tail days

Not applicable.

## Ablation

Not applicable.

## Sensitivity

Not applicable.

## Negative controls

Added and passed focused tests:

- HTTP 500 payloads raise `FetchError` and are not archived.
- Empty HTTP 200 payloads raise `FetchError` and are not archived.
- Malformed HKO climate CSV without the required header raises `HKOParseError`.

## Worst cases and failure taxonomy

Issues found and handled:

- Requested file names `HKG_TMAX_FIRST_GOALS.md`, `docs/00_PROJECT_OVERVIEW.md`,
  and `docs/06_LEAKAGE_CONTROL.md` are absent from the repo. The available
  equivalents read were `FIRST_GOALS.md`, `docs/01_RESEARCH_CHARTER.md`, and
  `docs/03_ASOF_AND_LEAKAGE.md`.
- Compatibility entry points for those three requested paths were added and
  read after creation. They point to the canonical existing goal and leakage
  documents and do not create divergent governance.
- `make` is unavailable in this PowerShell environment, so Makefile targets
  were run by their direct Python equivalents.
- `scripts/bootstrap.ps1` attempted full `.[research,dev]` install and hit a
  Windows long-path error inside `statsmodels`. The same script also continued
  after failed native commands. The script was patched to fail loudly on native
  command failures. G0 checks were then run with base/dev dependencies, which
  are sufficient for doctor/test/validation/fetch.
- After `EXP-0001` was created, `tests/test_experiments.py` failed because it
  copied the live mutable registry and assumed `next_id: 1`. The test was
  patched to seed an isolated temp registry, then the focused and full suites
  passed.

## Missingness/common sample

No missing bootstrap source payloads. All seven configured `bootstrap_now`
sources produced two archived retrieval events.

## Compute and operational cost

One local virtual environment, two live HTTP fetch passes with configured
request pacing, and local tests/validation. No modelling compute.

## Unexpected findings

The PowerShell bootstrap script could report success after failure before the
patch. This was a reproducibility risk and is fixed in this experiment.

## Full artifact list

- `reports/source_inventory.md`
- `experiments/EXP-0001-g0-repository-and-archive-smoke-test/DATA_MANIFEST.yaml`
- `experiments/EXP-0001-g0-repository-and-archive-smoke-test/results/metrics.json`
- `data/raw/<source_id>/2026/06/18/*` raw snapshots and sidecars
- `MANIFEST.json`
