#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors: list[str] = []


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exists(path: Path) -> bool:
    if path.exists():
        return True
    if os.name == "nt":
        return Path("\\\\?\\" + str(path.resolve())).exists()
    return False


task_index = read_csv(root / "TASK_INDEX.csv")
status_index_path = root / "TASK_STATUS_INDEX.csv"
status_rows = read_csv(status_index_path) if status_index_path.exists() else []
status_by_id = {row["task_id"]: row for row in status_rows}

if len(task_index) != 40:
    errors.append(f"expected 40 tasks, got {len(task_index)}")

ids = [row["task_id"] for row in task_index]
if len(set(ids)) != len(ids):
    errors.append("duplicate task ids")

if status_rows:
    if len(status_rows) != 40:
        errors.append(f"expected 40 status rows, got {len(status_rows)}")
    if set(status_by_id) != set(ids):
        errors.append("TASK_STATUS_INDEX task ids do not match TASK_INDEX")

for row in task_index:
    task_id = row["task_id"]
    live_row = status_by_id.get(task_id, row)
    task_file = root / live_row["task_file"]
    spec_file = root / row["spec_file"]
    if not exists(task_file):
        errors.append(f"missing task file {task_file}")
    if not exists(spec_file):
        errors.append(f"missing spec file {spec_file}")
    else:
        spec = json.loads(spec_file.read_text(encoding="utf-8"))
        if spec.get("id") != task_id:
            errors.append(f"spec id mismatch {task_id}")
    if live_row.get("status") == "completed":
        completion_record = live_row.get("completion_record", "")
        if not completion_record or not exists(root / completion_record):
            errors.append(f"missing completion record {task_id}")

for required in [
    "START_HERE.md",
    "CODEX_GLOBAL_EXECUTION_CONTRACT.md",
    "T24_POINT_IN_TIME_CONSTITUTION.md",
    "MASTER_EXECUTION_DAG.md",
    "GRIBSTREAM_MODEL_DISPOSITION_MATRIX.csv",
    "SEMANTIC_VARIABLE_REQUIREMENTS.csv",
    "TASK_WORKFLOW.md",
]:
    if not exists(root / required):
        errors.append(f"missing {required}")

status_counts: dict[str, int] = {}
for row in status_rows:
    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

print(json.dumps({"tasks": len(task_index), "status_counts": status_counts, "errors": errors}, indent=2))
sys.exit(1 if errors else 0)
