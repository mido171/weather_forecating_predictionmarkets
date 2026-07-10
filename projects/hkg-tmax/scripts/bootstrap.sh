#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
python -m pip install --upgrade pip
python -m pip install -e ".[research,dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add contact/API values where needed."
fi

if [[ ! -d .git ]] && command -v git >/dev/null 2>&1; then
  git init
fi

python -m hkg_tmax doctor
python -m pytest
python -m hkg_tmax validate all
python -m hkg_tmax manifest

echo
echo "Bootstrap complete."
echo "Next: read CODEX_START_HERE.md and execute FIRST_GOALS.md."
