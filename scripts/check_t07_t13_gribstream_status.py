from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments/0214_t07_t13_gribstream_backfill"
STATUS_PATH = EXPERIMENT_ROOT / "logs/t07_t13_status.json"
LEDGER_PATH = EXPERIMENT_ROOT / "resume_ledger.jsonl"
API_EVENT_LOG = EXPERIMENT_ROOT / "logs/gribstream_api_events.jsonl"
MANIFEST_PATH = EXPERIMENT_ROOT / "executed_chunks.csv"
BLOCKERS_PATH = EXPERIMENT_ROOT / "blockers.csv"


def read_json(path: Path) -> object:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_tail(path: Path, count: int = 20) -> list[object]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-count:]]


def summarize_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by_status: dict[str, int] = {}
    by_task: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    row_total = 0
    point_total = 0
    for row in rows:
        by_status[row.get("status", "")] = by_status.get(row.get("status", ""), 0) + 1
        by_task[row.get("task_id", "")] = by_task.get(row.get("task_id", ""), 0) + 1
        by_dataset[row.get("dataset", "")] = by_dataset.get(row.get("dataset", ""), 0) + 1
        row_total += int(row.get("row_count") or 0)
        point_total += int(row.get("point_rows") or 0)
    return {
        "exists": True,
        "chunks": len(rows),
        "by_status": by_status,
        "by_task": by_task,
        "by_dataset": by_dataset,
        "row_count": row_total,
        "point_rows": point_total,
    }


def main() -> int:
    print(
        json.dumps(
            {
                "status": read_json(STATUS_PATH),
                "executed_summary": summarize_csv(MANIFEST_PATH),
                "blockers_summary": summarize_csv(BLOCKERS_PATH),
                "ledger_tail": read_jsonl_tail(LEDGER_PATH),
                "api_event_tail": read_jsonl_tail(API_EVENT_LOG),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
