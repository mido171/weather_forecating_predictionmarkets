#!/usr/bin/env python3
"""Inventory every repository file relevant to HKG T+24 research."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from _common import sample_hash, sha256_file, utc_now, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--full-hash-max-mb", type=float, default=32.0)
    p.add_argument("--include", action="append", help="Relative root; repeatable")
    return p.parse_args()


def category(path: Path) -> str:
    name = path.name.lower()
    suffix = "".join(path.suffixes).lower()
    if name in {"readme.md", "results.md", "conclusion.md", "leakage_audit.md"}:
        return "documentation"
    if name in {"experiment_spec.json", "summary.json", "run_manifest.json"}:
        return "experiment_metadata"
    if suffix.endswith((".py", ".ps1", ".sh", ".sql", ".r")):
        return "code"
    if "prediction" in name:
        return "predictions"
    if "score" in name or "metric" in name:
        return "scores"
    if suffix.endswith((".parquet", ".csv", ".csv.gz", ".jsonl", ".ndjson", ".feather")):
        return "tabular_data"
    if suffix.endswith((".md", ".txt", ".pdf", ".html")):
        return "documentation"
    if suffix.endswith((".png", ".jpg", ".jpeg", ".svg")):
        return "figure"
    return "other"


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    output = (args.output_dir or repo / ".hkg_t24_research" / "census").resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = args.include or ["data/datasets", "experiments"]
    max_bytes = int(args.full_hash_max_mb * 1024 * 1024)
    rows: list[dict] = []
    failures: list[dict] = []
    extension_counts: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()

    for rel_root in roots:
        root = (repo / rel_root).resolve()
        if not root.exists():
            failures.append({"path": rel_root, "error": "ROOT_MISSING"})
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            try:
                stat = path.stat()
                rel = path.relative_to(repo).as_posix()
                root_rel = path.relative_to(root)
                top = root_rel.parts[0] if root_rel.parts else ""
                suffix = "".join(path.suffixes).lower() or "<none>"
                extension_counts[suffix] += 1
                folder_counts[f"{rel_root}/{top}"] += 1
                if stat.st_size <= max_bytes:
                    digest = sha256_file(path)
                    hash_mode = "sha256_full"
                else:
                    digest = sample_hash(path)
                    hash_mode = "sha256_size_first_last_sample"
                rows.append({
                    "root_kind": rel_root,
                    "relative_path": rel,
                    "top_folder": top,
                    "file_name": path.name,
                    "extension": suffix,
                    "size_bytes": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                    "category": category(path),
                    "content_hash": digest,
                    "hash_mode": hash_mode,
                })
            except Exception as exc:
                failures.append({"path": str(path), "error": repr(exc)})

    write_csv(output / "repository_file_inventory.csv", rows)
    write_csv(output / "unreadable_files.csv", failures, ["path", "error"])
    write_csv(
        output / "extension_counts.csv",
        [{"extension": key, "file_count": value} for key, value in sorted(extension_counts.items())],
    )
    write_csv(
        output / "folder_counts.csv",
        [{"folder": key, "file_count": value} for key, value in sorted(folder_counts.items())],
    )
    manifest = {
        "created_at_utc": utc_now(),
        "repo_root": str(repo),
        "roots": roots,
        "file_count": len(rows),
        "failure_count": len(failures),
        "full_hash_max_mb": args.full_hash_max_mb,
    }
    write_json(output / "repository_inventory_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
