#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: Git is required to verify the standalone repository root." >&2
  exit 2
fi
REPO_ROOT="$(git -c core.fsmonitor=false rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: Run bootstrap from the weather_data_extraction checkout." >&2
  exit 2
}
if [[ "$(basename "$REPO_ROOT")" != "weather_data_extraction" ]] \
  || [[ "$ROOT" != "$REPO_ROOT/projects/hkg-tmax" ]] \
  || [[ ! -d "$REPO_ROOT/.git" ]] \
  || [[ -L "$REPO_ROOT/.git" ]]; then
  echo "ERROR: Refusing bootstrap outside the canonical weather_data_extraction root." >&2
  exit 2
fi
if [[ "$(git -C "$REPO_ROOT" config --local --get core.fsmonitor || true)" != "false" ]]; then
  echo "ERROR: Local core.fsmonitor must be false before bootstrap." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ required; found {sys.version}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -e ".[research,dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add contact/API values where needed."
fi

python -m hkg_tmax doctor
python ../../tools/repo/doctor.py --root ../.. --scope projects/hkg-tmax
python -m pytest -q \
  tests/test_bootstrap_safety_contract.py \
  tests/test_config_and_sources.py \
  tests/test_experiments.py \
  tests/test_validation.py \
  tests/test_hko_backfill.py \
  tests/hkg_t24/test_h24n_contract_policy.py \
  tests/hkg_t24/test_schema_sql_contract.py \
  tests/test_demo_trading_migration.py
python -m hkg_tmax validate all
python scripts/manage_campaign_documentation.py check

echo
echo "Bootstrap complete."
echo "Next: follow AGENTS.md section 2, then read START_HERE.md and README.md."
