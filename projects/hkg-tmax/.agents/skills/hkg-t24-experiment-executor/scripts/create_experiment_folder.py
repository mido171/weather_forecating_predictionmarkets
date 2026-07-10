#!/usr/bin/env python3
r"""Create an immutable, self-contained HKG T+24 experiment folder.

Examples
--------
python create_experiment_folder.py --repo-root C:\path\to\repo \
    --spec C:\path\to\experiment_spec.json
python create_experiment_folder.py --repo-root C:\path\to\repo \
    --slug station_pair_regime_atlas --title "Station pair regime atlas"
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from _common import DirectoryLock, next_experiment_id, sha256_file, utc_now, write_json

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,100}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--spec", type=Path, help="Pre-registered experiment_spec.json")
    parser.add_argument("--slug", help="Used only when --spec is omitted")
    parser.add_argument("--title", help="Used only when --spec is omitted")
    parser.add_argument("--experiment-id", type=int, help="Reserve this exact four-digit ID")
    parser.add_argument("--experiments-dir", type=Path, help="Override <repo-root>/experiments")
    return parser.parse_args()


def load_or_build_spec(args: argparse.Namespace, template_path: Path) -> dict:
    if args.spec:
        if not args.spec.is_file():
            raise FileNotFoundError(f"Specification does not exist: {args.spec}")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    else:
        if not args.slug or not args.title:
            raise ValueError("Provide --spec or both --slug and --title")
        spec = json.loads(template_path.read_text(encoding="utf-8"))
        spec["slug"] = args.slug
        spec["title"] = args.title
    slug = str(spec.get("slug", "")).strip()
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"Invalid slug {slug!r}; use lowercase letters, numbers, _ or -")
    return spec


def render_template(source: Path, destination: Path, replacements: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    destination.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")
    experiments_root = (args.experiments_dir or repo_root / "experiments").resolve()
    experiments_root.mkdir(parents=True, exist_ok=True)

    script_root = Path(__file__).resolve().parent
    skill_root = script_root.parent
    template_root = skill_root / "assets" / "templates"
    spec = load_or_build_spec(args, template_root / "experiment_spec.json")

    lock_path = experiments_root / ".hkg_t24_experiment_id.lock"
    with DirectoryLock(lock_path):
        experiment_number = args.experiment_id or next_experiment_id(experiments_root)
        if not (0 <= experiment_number <= 9999):
            raise ValueError("Experiment ID must fit four digits")
        experiment_id = f"{experiment_number:04d}"
        folder_name = f"{experiment_id}_{spec['slug']}"
        final_dir = experiments_root / folder_name
        if final_dir.exists():
            raise FileExistsError(
                f"Experiment folder already exists and will not be overwritten: {final_dir}"
            )
        staging_dir = experiments_root / f".{folder_name}.creating"
        if staging_dir.exists():
            raise FileExistsError(f"Stale staging directory exists: {staging_dir}")
        staging_dir.mkdir()
        try:
            for name in ("src", "logs", "diagnostics", "figures", "artifacts"):
                (staging_dir / name).mkdir()

            spec["experiment_id"] = experiment_id
            spec["created_at_utc"] = utc_now()
            write_json(staging_dir / "experiment_spec.json", spec)

            replacements = {
                "EXPERIMENT_ID": experiment_id,
                "SLUG": spec["slug"],
                "TITLE": spec.get("title", ""),
                "STATUS": "PRE_REGISTERED",
                "CREATED_AT_UTC": spec["created_at_utc"],
                "HYPOTHESIS": spec.get("hypothesis", "TO BE COMPLETED BEFORE SCORING"),
            }
            for template_name in ("README.md", "RESULTS.md", "CONCLUSION.md"):
                render_template(
                    template_root / template_name,
                    staging_dir / template_name,
                    replacements,
                )

            (staging_dir / "leakage_audit.md").write_text(
                "# Leakage and Point-in-Time Audit\n\n"
                "Status: `PENDING`\n\n"
                "Complete every check before scoring. Unknown availability is not a pass.\n",
                encoding="utf-8",
            )
            (staging_dir / "REPRODUCE.md").write_text(
                "# Reproduction\n\n"
                "Record the exact environment, command, and expected artifacts here.\n",
                encoding="utf-8",
            )
            (staging_dir / "data_manifest.csv").write_text(
                "source_id,path,sha256,size_bytes,row_count,date_start,date_end,"
                "timestamp_fields,availability_class,notes\n",
                encoding="utf-8",
            )
            (staging_dir / "feature_definitions.csv").write_text(
                "feature_name,role,formula,input_columns,units,lag,window,"
                "fit_scope,availability_rule,missingness_policy\n",
                encoding="utf-8",
            )
            summary = {
                "experiment_id": experiment_id,
                "slug": spec["slug"],
                "status": "FAILED_RUNTIME",
                "created_at_utc": spec["created_at_utc"],
                "target": "HKO daily Tmax T-24",
                "frame_id": spec.get("frame", {}).get("frame_id", ""),
                "date_start": None,
                "date_end": None,
                "n_candidate": 0,
                "n_common": 0,
                "baseline_id": spec.get("baseline", {}).get("id") or None,
                "baseline_mae_c": None,
                "candidate_id": None,
                "candidate_mae_c": None,
                "mae_delta_c": None,
                "candidate_rmse_c": None,
                "candidate_bias_c": None,
                "leakage_status": "BLOCKED",
                "confirmation_rows_used": 0,
                "owner_authorized_confirmation": bool(
                    spec.get("owner_authorized_confirmation", False)
                ),
                "promotion_decision": "PENDING",
                "spec_sha256": sha256_file(staging_dir / "experiment_spec.json"),
                "code_sha256": None,
                "data_manifest_sha256": None,
                "common_row_hash": None,
                "baseline_n": None,
                "candidate_n": None,
                "development_gate_reached": False,
                "notes": "Scaffold only; replace status only after a complete run or documented rejection.",
            }
            write_json(staging_dir / "summary.json", summary)
            run_manifest = {
                "experiment_id": experiment_id,
                "folder_name": folder_name,
                "created_at_utc": spec["created_at_utc"],
                "repo_root": str(repo_root),
                "experiments_root": str(experiments_root),
                "scaffold_script": str(Path(__file__).resolve()),
                "spec_source": str(args.spec.resolve()) if args.spec else None,
                "state": "SCAFFOLDED",
            }
            write_json(staging_dir / "run_manifest.json", run_manifest)

            # Rename is atomic on one filesystem and avoids half-created visible experiments.
            staging_dir.replace(final_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    print(final_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
