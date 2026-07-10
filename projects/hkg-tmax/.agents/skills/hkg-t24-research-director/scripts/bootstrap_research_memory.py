#!/usr/bin/env python3
"""Initialize/refresh HKG T+24 persistent research memory in one command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--census-mode", choices=["metadata","sample","full"], default="sample")
    p.add_argument("--sample-rows", type=int, default=100000)
    p.add_argument("--max-full-file-mb", type=float, default=256.0)
    return p.parse_args()


def run(command: list[str], allowed: set[int] = {0,2}) -> None:
    print("+", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode not in allowed:
        raise SystemExit(result.returncode)


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    scripts = Path(__file__).resolve().parent
    python = sys.executable
    run([python, str(scripts / "research_state.py"), "--repo-root", str(repo), "init"], {0})
    run([
        python, str(scripts / "inventory_repository.py"),
        "--repo-root", str(repo),
    ])
    run([
        python, str(scripts / "build_data_census.py"),
        "--repo-root", str(repo),
        "--mode", args.census_mode,
        "--sample-rows", str(args.sample_rows),
        "--max-full-file-mb", str(args.max_full_file_mb),
    ])
    run([
        python, str(scripts / "index_experiments.py"),
        "--repo-root", str(repo),
    ])
    print(f"Research memory refreshed under: {repo / '.hkg_t24_research'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
