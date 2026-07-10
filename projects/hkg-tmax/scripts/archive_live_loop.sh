#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${ARCHIVE_EXECUTE:-0}" != "1" ]]; then
  echo "Dry run only. Set ARCHIVE_EXECUTE=1 after reviewing sources and budgets." >&2
  exit 2
fi

INTERVAL_SECONDS="${ARCHIVE_INTERVAL_SECONDS:-600}"
MAX_ITERATIONS="${ARCHIVE_MAX_ITERATIONS:-1}"
if (( INTERVAL_SECONDS < 60 || MAX_ITERATIONS < 1 || MAX_ITERATIONS > 144 )); then
  echo "Unsafe budget: interval must be >=60 seconds and iterations must be 1..144." >&2
  exit 2
fi

SOURCES=(
  hko_latest_1min_temperature
  hko_since_midnight_maxmin
  hko_local_weather_forecast
  hko_nine_day_forecast
)

failures=0
for ((iteration=1; iteration<=MAX_ITERATIONS; iteration++)); do
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  args=()
  for source in "${SOURCES[@]}"; do
    args+=(--source-id "$source")
  done
  if ! python3 -m hkg_tmax acquisition collect "${args[@]}" \
    --max-sources "${#SOURCES[@]}" --continue-on-error --execute; then
    failures=$((failures + 1))
    if (( failures >= 3 )); then
      echo "Stopping after three consecutive failed collection rounds." >&2
      exit 1
    fi
  else
    failures=0
  fi
  if (( iteration < MAX_ITERATIONS )); then
    sleep "$INTERVAL_SECONDS"
  fi
done
