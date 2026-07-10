from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments/0213_gribstream_resumable_runs_client"
STATUS_PATH = EXPERIMENT_DIR / "logs/t06_status.json"
LEDGER_PATH = EXPERIMENT_DIR / "resume_ledger.jsonl"
API_EVENT_LOG = EXPERIMENT_DIR / "logs/gribstream_api_events.jsonl"


def read_json(path: Path) -> object:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_tail(path: Path, count: int = 20) -> list[object]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-count:]]


def main() -> int:
    print(
        json.dumps(
            {
                "status": read_json(STATUS_PATH),
                "resume_ledger_tail": read_jsonl_tail(LEDGER_PATH, 20),
                "api_event_tail": read_jsonl_tail(API_EVENT_LOG, 20),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
