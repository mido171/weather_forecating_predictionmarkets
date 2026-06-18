#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

INTERVAL_SECONDS="${ARCHIVE_INTERVAL_SECONDS:-600}"
SOURCES=(
  hko_latest_1min_temperature
  hko_since_midnight_maxmin
  hko_local_weather_forecast
  hko_nine_day_forecast
)

echo "Starting append-only live archive loop. Interval: ${INTERVAL_SECONDS}s"
echo "Stop with Ctrl-C. Each fetch is a distinct timestamped snapshot."

while true; do
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  args=()
  for source in "${SOURCES[@]}"; do
    args+=(--id "$source")
  done
  python3 -m hkg_tmax sources fetch "${args[@]}" --continue-on-error || true
  sleep "$INTERVAL_SECONDS"
done
