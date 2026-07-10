"""Generate the exact changed-file inventory for the repository consolidation."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


PURPOSES = {
    "apps/": ("applications", "deployable service relocation or runtime hardening"),
    "packages/": ("shared-packages", "shared library relocation or safety hardening"),
    "projects/hkg-tmax/": ("hkg-tmax", "HKG project consolidation and portability"),
    "projects/klga-tmax/": ("klga-tmax", "KLGA project consolidation and safety"),
    "tools/": ("tooling", "bounded operational or repository tooling"),
    "tests/": ("root-tests", "cross-component smoke coverage"),
    "docs/": ("documentation", "canonical architecture, operations, security, or migration record"),
    ".github/": ("ci", "offline continuous-integration gate"),
    "legacy/": ("legacy", "non-authoritative historical retention"),
    "config/": ("root-config", "non-secret configuration example"),
}


def classify(path: str) -> tuple[str, str]:
    normalized = path.replace("\\", "/")
    for prefix, values in PURPOSES.items():
        if normalized.startswith(prefix):
            return values
    return "root-governance", "repository root structure or governance"


def git_lines(root: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.autocrlf=false",
            "-c",
            "diff.renameLimit=10000",
            *args,
        ],
        cwd=root,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def build_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    represented: set[str] = set()
    for line in git_lines(root, "diff", "--name-status", "--find-renames=50%", "HEAD", "--"):
        fields = line.split("\t")
        status = fields[0]
        if status.startswith(("R", "C")) and len(fields) >= 3:
            old_path, new_path = fields[1], fields[2]
        elif len(fields) >= 2:
            old_path, new_path = "", fields[1]
        else:
            continue
        effective_path = new_path or old_path
        area, purpose = classify(effective_path)
        rows.append(
            {
                "status": status,
                "old_path": old_path,
                "new_path": new_path,
                "area": area,
                "purpose": purpose,
            }
        )
        represented.add(effective_path)
    for path in git_lines(root, "ls-files", "--others", "--exclude-standard"):
        if path in represented:
            continue
        area, purpose = classify(path)
        rows.append(
            {
                "status": "A",
                "old_path": "",
                "new_path": path,
                "area": area,
                "purpose": purpose,
            }
        )
    return sorted(rows, key=lambda row: (row["area"], row["new_path"] or row["old_path"]))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "docs" / "migrations" / "2026-07-10-changed-files.csv"
    rows = build_rows(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("status", "old_path", "new_path", "area", "purpose"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} changed-file records to {output}")


if __name__ == "__main__":
    main()
