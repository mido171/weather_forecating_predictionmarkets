#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m hkg_tmax doctor
python3 -m hkg_tmax validate all
python3 -m pytest
python3 -m hkg_tmax manifest
