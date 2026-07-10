#!/usr/bin/env python3
"""Index every experiment folder into a cumulative evidence registry."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from _common import read_text_if_exists, sha256_file, utc_now, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--experiments-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--max-text-chars", type=int, default=2_000_000)
    return p.parse_args()


def load_json_safe(path: Path) -> tuple[dict, str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}, "JSON_NOT_OBJECT"
        return value, None
    except Exception as exc:
        return {}, repr(exc)


def find_first(folder: Path, names: list[str]) -> Path | None:
    low_names = {name.lower() for name in names}
    direct = [p for p in folder.iterdir() if p.is_file() and p.name.lower() in low_names]
    if direct:
        return sorted(direct)[0]
    for path in folder.rglob("*"):
        if path.is_file() and path.name.lower() in low_names:
            return path
    return None


def regex_float(text: str, labels: list[str]) -> float | None:
    for label in labels:
        pattern = re.compile(
            rf"{label}\s*(?:[:=]|is)?\s*`?\s*(-?\d+(?:\.\d+)?)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def regex_int(text: str, labels: list[str]) -> int | None:
    value = regex_float(text, labels)
    return int(value) if value is not None else None


def detect_status(text: str) -> str:
    low = text.lower()
    if "rejected_leakage" in low or "target leakage" in low:
        return "REJECTED_LEAKAGE"
    if "rejected_timestamp" in low or "timestamp blocked" in low:
        return "REJECTED_TIMESTAMP"
    if "promotion candidate" in low:
        return "COMPLETED_PROMOTION_CANDIDATE"
    if "diagnostic-only" in low or "information gain only" in low:
        return "COMPLETED_INFORMATION_GAIN_ONLY"
    if "null" in low or "negative result" in low or "did not improve" in low:
        return "COMPLETED_NULL_OR_NEGATIVE"
    return "UNRESOLVED"


def extract_scoreboard(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        # Prefer explicitly marked candidate/best rows.
        chosen = None
        for row in rows:
            role = (row.get("role") or row.get("status") or "").lower()
            if role in {"candidate","best","champion"}:
                chosen = row
                break
        if chosen is None:
            chosen = rows[0]
        def get_float(*keys: str) -> float | None:
            for key in keys:
                value = chosen.get(key)
                if value not in (None, ""):
                    try:
                        return float(value)
                    except ValueError:
                        continue
            return None
        def get_int(*keys: str) -> int | None:
            value = get_float(*keys)
            return int(value) if value is not None else None
        return {
            "scoreboard_path": str(path),
            "scoreboard_rows": len(rows),
            "candidate_id_scoreboard": chosen.get("candidate_id") or chosen.get("model") or chosen.get("candidate"),
            "candidate_mae_scoreboard": get_float("mae_c","mae","MAE"),
            "candidate_rmse_scoreboard": get_float("rmse_c","rmse","RMSE"),
            "candidate_bias_scoreboard": get_float("bias_c","bias","Bias"),
            "n_scoreboard": get_int("n","row_count","n_common","N"),
            "delta_scoreboard": get_float("mae_delta_c","mae_delta_vs_baseline_c","delta"),
        }
    except Exception as exc:
        return {"scoreboard_error": repr(exc), "scoreboard_path": str(path)}


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    experiments = (args.experiments_root or repo / "experiments").resolve()
    output = (args.output_dir or repo / ".hkg_t24_research").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not experiments.is_dir():
        raise FileNotFoundError(experiments)

    evidence_rows: list[dict] = []
    file_rows: list[dict] = []
    unresolved: list[dict] = []
    negative_seed: list[dict] = []

    folders = sorted(p for p in experiments.iterdir() if p.is_dir() and not p.name.startswith("."))
    for folder in folders:
        spec, spec_error = load_json_safe(folder / "experiment_spec.json")
        summary, summary_error = load_json_safe(folder / "summary.json")
        readme_path = find_first(folder, ["README.md","report.md","analysis.md"])
        results_path = find_first(folder, ["RESULTS.md","results.txt","summary.md"])
        conclusion_path = find_first(folder, ["CONCLUSION.md","conclusion.txt"])
        leakage_path = find_first(folder, ["leakage_audit.md","leakage.md"])
        scoreboard_path = find_first(folder, ["scoreboard.csv","scoreboard_summary.csv"])
        readme = read_text_if_exists(readme_path, args.max_text_chars) if readme_path else ""
        results = read_text_if_exists(results_path, args.max_text_chars) if results_path else ""
        conclusion = read_text_if_exists(conclusion_path, args.max_text_chars) if conclusion_path else ""
        leakage = read_text_if_exists(leakage_path, args.max_text_chars) if leakage_path else ""
        combined = "\n".join((readme, results, conclusion, leakage))
        scoreboard = extract_scoreboard(scoreboard_path)

        experiment_id_match = re.match(r"^(\d{4})", folder.name)
        experiment_id = (
            str(summary.get("experiment_id") or spec.get("experiment_id") or
                (experiment_id_match.group(1) if experiment_id_match else folder.name))
        )
        status = summary.get("status") or detect_status(combined)
        candidate_mae = summary.get("candidate_mae_c")
        if candidate_mae is None:
            candidate_mae = scoreboard.get("candidate_mae_scoreboard")
        if candidate_mae is None:
            candidate_mae = regex_float(combined, [r"best\s+mae", r"candidate\s+mae", r"\bmae\b"])
        baseline_mae = summary.get("baseline_mae_c")
        if baseline_mae is None:
            baseline_mae = regex_float(combined, [r"baseline\s+mae", r"official\s+mae", r"raw\s+mae"])
        rmse = summary.get("candidate_rmse_c")
        if rmse is None:
            rmse = scoreboard.get("candidate_rmse_scoreboard")
        if rmse is None:
            rmse = regex_float(combined, [r"\brmse\b"])
        n_common = summary.get("n_common")
        if n_common is None:
            n_common = scoreboard.get("n_scoreboard")
        if n_common is None:
            n_common = regex_int(combined, [r"common\s+(?:scored\s+)?rows", r"\brows\b", r"\bn\b"])

        data_sources = spec.get("data_sources", [])
        source_ids = [
            str(source.get("source_id"))
            for source in data_sources if isinstance(source, dict) and source.get("source_id")
        ]
        features = spec.get("features", [])
        feature_names = [
            str(feature.get("name"))
            for feature in features if isinstance(feature, dict) and feature.get("name")
        ]

        row = {
            "experiment_id": experiment_id,
            "folder": folder.name,
            "relative_path": folder.relative_to(repo).as_posix(),
            "title": spec.get("title") or "",
            "mode": spec.get("mode") or "",
            "hypothesis": spec.get("hypothesis") or "",
            "status": status,
            "frame_id": summary.get("frame_id") or spec.get("frame", {}).get("frame_id") or "",
            "date_start": summary.get("date_start") or "",
            "date_end": summary.get("date_end") or "",
            "n_common": n_common,
            "baseline_id": summary.get("baseline_id") or spec.get("baseline", {}).get("id") or "",
            "baseline_mae_c": baseline_mae,
            "candidate_id": summary.get("candidate_id") or scoreboard.get("candidate_id_scoreboard") or "",
            "candidate_mae_c": candidate_mae,
            "candidate_rmse_c": rmse,
            "candidate_bias_c": summary.get("candidate_bias_c") or scoreboard.get("candidate_bias_scoreboard"),
            "mae_delta_c": summary.get("mae_delta_c") or scoreboard.get("delta_scoreboard"),
            "leakage_status": summary.get("leakage_status") or (
                "PASS" if re.search(r"\bleakage\b.{0,30}\bpass\b", leakage, re.I | re.S) else "UNRESOLVED"
            ),
            "confirmation_rows_used": summary.get("confirmation_rows_used"),
            "promotion_decision": summary.get("promotion_decision") or "",
            "source_ids": "|".join(source_ids),
            "feature_count_spec": len(feature_names),
            "feature_names": "|".join(feature_names[:500]),
            "response": spec.get("response", {}).get("name") or "",
            "readme_path": readme_path.relative_to(repo).as_posix() if readme_path else "",
            "results_path": results_path.relative_to(repo).as_posix() if results_path else "",
            "conclusion_path": conclusion_path.relative_to(repo).as_posix() if conclusion_path else "",
            "scoreboard_path": scoreboard_path.relative_to(repo).as_posix() if scoreboard_path else "",
            "spec_present": (folder / "experiment_spec.json").is_file(),
            "summary_present": (folder / "summary.json").is_file(),
            "spec_error": spec_error or "",
            "summary_error": summary_error or "",
            "text_chars_indexed": len(combined),
            "main_insight_excerpt": re.sub(r"\s+", " ", conclusion or results or readme)[:1000],
        }
        evidence_rows.append(row)

        if status in {
            "COMPLETED_NULL_OR_NEGATIVE","REJECTED_LEAKAGE","REJECTED_TIMESTAMP",
            "REJECTED_SPECIFICATION","REJECTED_DATA_QUALITY","BLOCKED_MISSING_DATA"
        } or re.search(r"\bnull\b|\bnegative\b|\bblocked\b|\bdid not improve\b", combined, re.I):
            negative_seed.append({
                "experiment_id": experiment_id,
                "folder": folder.name,
                "status": status,
                "hypothesis": row["hypothesis"],
                "observed_effect": row["main_insight_excerpt"],
                "failure_taxonomy": "",
                "retest_condition": "",
                "director_decision": "REVIEW_AND_CLASSIFY",
            })

        if not readme_path and not results_path and not conclusion_path and not summary:
            unresolved.append({
                "folder": folder.name,
                "reason": "NO_RECOGNIZED_EVIDENCE_ARTIFACTS",
            })

        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            try:
                file_rows.append({
                    "experiment_id": experiment_id,
                    "folder": folder.name,
                    "relative_path": path.relative_to(repo).as_posix(),
                    "file_name": path.name,
                    "suffix": "".join(path.suffixes).lower(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path) if path.stat().st_size <= 16 * 1024 * 1024 else "",
                })
            except Exception as exc:
                unresolved.append({"folder": folder.name, "reason": f"FILE_ERROR:{path}:{exc}"})

    write_csv(output / "experiment_evidence_registry.csv", evidence_rows)
    write_csv(output / "experiment_file_inventory.csv", file_rows)
    write_csv(output / "unindexed_experiments.csv", unresolved, ["folder","reason"])
    write_csv(output / "negative_results_registry_seed.csv", negative_seed)
    manifest = {
        "created_at_utc": utc_now(),
        "experiments_root": str(experiments),
        "experiment_folder_count": len(folders),
        "evidence_rows": len(evidence_rows),
        "file_rows": len(file_rows),
        "unresolved_count": len(unresolved),
        "negative_seed_count": len(negative_seed),
    }
    write_json(output / "experiment_index_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
