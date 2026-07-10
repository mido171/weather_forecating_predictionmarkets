from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
T03_EXP = REPO_ROOT / "experiments/0210_gribstream_catalog_coverage_licence_quota_audit"
T04_EXP = REPO_ROOT / "experiments/0211_nwp_database_object_storage_migrations"
T05_EXP = REPO_ROOT / "experiments/0212_canonical_location_station_geospatial_registry"
STATUS_PATH = T03_EXP / "logs/t03_t05_background_status.json"
COVERAGE_PATH = T03_EXP / "coverage_probe_results.csv"
API_EVENT_LOG = T03_EXP / "logs/gribstream_api_events.jsonl"


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def status_counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get(column, "") or "blank"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_api_events() -> int:
    if not API_EVENT_LOG.exists():
        return 0
    with API_EVENT_LOG.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {}
    coverage = csv_rows(COVERAGE_PATH)
    payload = {
        "status_file": str(STATUS_PATH),
        "current_status": status,
        "coverage_probe_rows": len(coverage),
        "coverage_probe_status_counts": status_counts(coverage, "probe_status"),
        "http_status_counts": status_counts(coverage, "http_status"),
        "api_event_count": count_api_events(),
        "evidence_folders": {
            "T03": str(T03_EXP),
            "T04": str(T04_EXP),
            "T05": str(T05_EXP),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
