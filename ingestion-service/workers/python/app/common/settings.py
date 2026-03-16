from __future__ import annotations

from pathlib import Path
import os


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "pom.xml").exists() and (parent / "ingestion-service").exists():
            return parent
    return Path.cwd()


def sqlite_root() -> Path:
    return Path(os.environ.get("PILOT_SQLITE_ROOT", r"D:\Ahmed\data\sqlite\MOS_aggregate_V2.0"))
