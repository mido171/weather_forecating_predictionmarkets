# Safe Commands and Verification

## Safe startup

```powershell
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git -c core.fsmonitor=false status --short --branch --untracked-files=no
python tools/repo/doctor.py
```

These commands do not enumerate every untracked file, contact providers, or run product code.

## Search narrowly

Preferred:

```powershell
rg --files projects/hkg-tmax/src
rg "pattern" projects/hkg-tmax/src projects/hkg-tmax/tests
git ls-files projects/hkg-tmax
```

Avoid root recursive PowerShell listing, `tree /F`, unrestricted hidden scans, junction
following, and full hashing. Expand scope only when the task requires it and state the reason.

## Test progressively

```powershell
python -m unittest discover -s tools/repo/tests -p "test_*.py"
python -m compileall -q projects/hkg-tmax/src
python -m compileall -q projects/klga-tmax/src projects/klga-tmax/tests
```

Project tests should target changed modules first. Do not run root `pytest`, `pytest -n auto`,
`compileall .`, or parallel Maven as a routine check. Provider/database/live tests require an
explicitly prepared environment and authorization.

## Git review

```powershell
git diff --check
git diff --name-status
git diff --stat
git diff --cached --name-status
git diff --cached --stat
```

Use explicit path staging. Never clean, reset, broadly restore, or rewrite history without a
recovery artifact and explicit user authorization.

## Resource defaults

- One worker by default; two maximum without explicit approval.
- BLAS/OpenMP/NumExpr threads set to one for agent-run research.
- Bounded date, station, model, request, byte, retry, row, and duration budgets.
- Serial Maven and scoped Python tests.
- No scheduled tasks or background services created by verification.

## Full maintenance checks

Full filesystem manifests, large-data reconciliation, `git fsck`, broad test suites, or
provider-backed integration checks belong in a declared maintenance window. Record expected
runtime/resources, use one owner, write logs outside Git, and define cancellation/rollback.
