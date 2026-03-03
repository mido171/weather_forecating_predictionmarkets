#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "tools" / "ncei_truth" / "run.py"
    if not runner.exists():
        raise SystemExit(f"Missing runner: {runner}")
    cmd = [sys.executable, str(runner)] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

