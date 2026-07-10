#!/usr/bin/env python3
"""Manage the sequential HKG T+24 research state and champion ledger."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import load_json, sha256_file, utc_now, write_csv, write_json

TARGET_MAE = 0.45


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init")
    sub.add_parser("status")

    begin = sub.add_parser("begin")
    begin.add_argument("--spec", required=True, type=Path)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--experiment-folder", required=True, type=Path)

    stop = sub.add_parser("stop")
    stop.add_argument("--reason", required=True)

    unblock = sub.add_parser("resume")
    unblock.add_argument("--reason", required=True)

    return p


def state_path(repo: Path) -> Path:
    return repo / ".hkg_t24_research" / "research_state.json"


def initial_state(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project_root": str(repo),
        "experiments_root": str(repo / "experiments"),
        "target": "Hong Kong Observatory daily Tmax at canonical T-24 cutoff",
        "target_mae_c": TARGET_MAE,
        "phase": "DISCOVERY",
        "iteration": 0,
        "confirmation": {
            "locked": True,
            "opened_once": False,
            "authorized_by_owner": False,
            "period_start": "2024-01-01",
        },
        "champions": {},
        "open_experiment": None,
        "completed_experiments": [],
        "candidate_queue": [],
        "stop_reason": None,
        "updated_at_utc": utc_now(),
    }


def read_state(repo: Path) -> dict:
    path = state_path(repo)
    if not path.is_file():
        raise FileNotFoundError(
            f"Research state not initialized: {path}. Run research_state.py --repo-root ... init"
        )
    return load_json(path)


def save_state(repo: Path, state: dict) -> None:
    state["updated_at_utc"] = utc_now()
    write_json(state_path(repo), state)


def ensure_registry_files(repo: Path) -> None:
    root = repo / ".hkg_t24_research"
    templates = {
        "champion_ledger.csv": [
            "recorded_at_utc","frame_id","experiment_id","folder","candidate_id",
            "candidate_mae_c","baseline_id","baseline_mae_c","mae_delta_c",
            "n_common","date_start","date_end","leakage_status","status","reason",
        ],
        "candidate_queue.csv": [
            "rank","candidate_id","title","mechanism","response","novelty_class",
            "expected_information_gain_0_5","expected_mae_lift_0_5",
            "physical_plausibility_0_5","prior_support_0_5","readiness_0_5",
            "sample_sufficiency_0_5","robustness_potential_0_5",
            "downstream_value_0_5","timestamp_risk_0_5","overfit_risk_0_5",
            "complexity_cost_0_5","priority_score","status","spec_path",
        ],
        "negative_results_registry.csv": [
            "experiment_id","date","hypothesis","mechanism","data_families",
            "stations","feature_family","response","baseline","frame","status",
            "observed_effect","fold_consistency","tail_effect","failure_taxonomy",
            "mechanism_falsified","inconclusive","blocker","retest_condition",
            "related_experiments","director_decision",
        ],
        "source_eligibility_matrix.csv": [
            "source_family","source_id","path","date_start","date_end","cadence",
            "station_coverage","attributes","valid_time_field","issue_time_field",
            "available_at_field","eligibility","blocker","evidence","research_role",
        ],
        "station_dossier.csv": [
            "station_id","aliases","station_name","latitude","longitude","elevation_m",
            "distance_to_hko_km","bearing_from_hko_deg","distance_to_coast_km",
            "role_labels","role_confidence","date_start","date_end","variables",
            "missingness","eligibility","metadata_sources","notes",
        ],
        "interaction_discovery_queue.csv": [
            "rank","interaction_id","feature_a","feature_b","response","regime",
            "hypothesized_mechanism","deployable_construction","prior_evidence",
            "minimum_support","status","linked_experiments",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, fields in templates.items():
        path = root / name
        if not path.exists():
            write_csv(path, [], fields)
    decision_log = root / "research_decision_log.md"
    if not decision_log.exists():
        decision_log.write_text(
            "# HKG T+24 Research Decision Log\n\n"
            "Append every sequential selection and result. Never delete prior entries.\n",
            encoding="utf-8",
        )
    blockers = root / "blockers.md"
    if not blockers.exists():
        blockers.write_text("# HKG T+24 Blockers\n\nNo global blocker recorded.\n", encoding="utf-8")


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_decision(repo: Path, title: str, body: str) -> None:
    path = repo / ".hkg_t24_research" / "research_decision_log.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {utc_now()} — {title}\n\n{body.strip()}\n")


def run_required_validator(command: list[str], label: str) -> dict[str, Any]:
    """Run a required validator and refuse state mutation on any failure."""
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"{label} failed; research state was not changed.\n{detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{label} returned non-JSON output; research state was not changed.\n"
            f"{completed.stdout.strip()}"
        ) from exc


def validate_spec_before_begin(repo: Path, spec_path: Path) -> dict[str, Any]:
    validator = Path(__file__).resolve().parent / "validate_experiment_spec.py"
    if not validator.is_file():
        raise FileNotFoundError(f"Required specification validator is missing: {validator}")
    return run_required_validator(
        [
            sys.executable, str(validator), str(spec_path),
            "--repo-root", str(repo), "--json",
        ],
        "Experiment specification validation",
    )


def validate_folder_before_ingest(folder: Path) -> dict[str, Any]:
    skills_root = Path(__file__).resolve().parents[2]
    validator = (
        skills_root
        / "hkg-t24-experiment-executor"
        / "scripts"
        / "validate_experiment_folder.py"
    )
    if not validator.is_file():
        raise FileNotFoundError(f"Required strict folder validator is missing: {validator}")
    return run_required_validator(
        [sys.executable, str(validator), str(folder), "--strict", "--json"],
        "Strict experiment-folder validation",
    )


def command_init(repo: Path) -> int:
    if state_path(repo).exists():
        print(f"Research state already exists: {state_path(repo)}")
        ensure_registry_files(repo)
        return 0
    state = initial_state(repo)
    ensure_registry_files(repo)
    save_state(repo, state)
    append_decision(repo, "Research state initialized", f"Repository: `{repo}`")
    print(json.dumps(state, indent=2))
    return 0


def command_begin(repo: Path, spec_path: Path) -> int:
    state = read_state(repo)
    if state["phase"] not in {"DISCOVERY"}:
        raise RuntimeError(f"Cannot begin from phase {state['phase']}")
    if state.get("open_experiment") is not None:
        raise RuntimeError("Another experiment is already open")
    validate_spec_before_begin(repo, spec_path)
    spec = load_json(spec_path)
    if spec.get("owner_authorized_confirmation") is not False:
        raise ValueError("Development specification must keep confirmation unauthorized")
    state["iteration"] += 1
    state["phase"] = "EXECUTING"
    state["open_experiment"] = {
        "iteration": state["iteration"],
        "title": spec.get("title"),
        "slug": spec.get("slug"),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_file(spec_path),
        "started_at_utc": utc_now(),
    }
    save_state(repo, state)
    append_decision(
        repo,
        f"Iteration {state['iteration']} opened",
        f"Specification: `{spec_path.resolve()}`\n\n"
        f"Hypothesis: {spec.get('hypothesis','')}",
    )
    print(json.dumps(state["open_experiment"], indent=2))
    return 0


def command_ingest(repo: Path, folder: Path) -> int:
    state = read_state(repo)
    if state["phase"] != "EXECUTING" or not state.get("open_experiment"):
        raise RuntimeError("No open experiment to ingest")
    folder = folder.resolve()
    validate_folder_before_ingest(folder)
    summary_path = folder / "summary.json"
    spec_path = folder / "experiment_spec.json"
    if not summary_path.is_file() or not spec_path.is_file():
        raise FileNotFoundError("Experiment folder lacks summary.json/spec")
    summary = load_json(summary_path)
    spec = load_json(spec_path)
    open_item = state["open_experiment"]
    if spec.get("slug") != open_item.get("slug"):
        raise ValueError("Completed experiment slug does not match open specification")
    if sha256_file(Path(open_item["spec_path"])) != open_item["spec_sha256"]:
        raise ValueError("Open specification changed after begin")
    # The scaffold adds ID/created_at, so compare scientific identity, not raw hash.
    status = summary.get("status")
    experiment_id = str(summary.get("experiment_id"))
    completion = {
        "iteration": state["iteration"],
        "experiment_id": experiment_id,
        "folder": str(folder),
        "slug": summary.get("slug"),
        "status": status,
        "frame_id": summary.get("frame_id"),
        "candidate_id": summary.get("candidate_id"),
        "candidate_mae_c": summary.get("candidate_mae_c"),
        "baseline_id": summary.get("baseline_id"),
        "baseline_mae_c": summary.get("baseline_mae_c"),
        "mae_delta_c": summary.get("mae_delta_c"),
        "n_common": summary.get("n_common"),
        "leakage_status": summary.get("leakage_status"),
        "confirmation_rows_used": summary.get("confirmation_rows_used"),
        "ingested_at_utc": utc_now(),
    }
    state["completed_experiments"].append(completion)
    state["open_experiment"] = None

    eligible_champion = (
        status == "COMPLETED_PROMOTION_CANDIDATE"
        and summary.get("leakage_status") == "PASS"
        and summary.get("confirmation_rows_used") == 0
        and isinstance(summary.get("candidate_mae_c"), (int, float))
        and isinstance(summary.get("n_common"), int)
        and summary.get("n_common") > 0
        and summary.get("baseline_n", summary.get("n_common")) == summary.get("n_common")
        and summary.get("candidate_n", summary.get("n_common")) == summary.get("n_common")
    )
    champion_changed = False
    reason = "Not eligible to change champion"
    frame_id = str(summary.get("frame_id") or "")
    if eligible_champion and frame_id:
        old = state["champions"].get(frame_id)
        candidate_mae = float(summary["candidate_mae_c"])
        if old is None or candidate_mae < float(old["candidate_mae_c"]):
            state["champions"][frame_id] = completion
            champion_changed = True
            reason = "Eligible lower-MAE candidate on the same declared frame"
        else:
            reason = "Eligible result did not beat existing frame champion"
    if eligible_champion and float(summary["candidate_mae_c"]) <= TARGET_MAE:
        # Strict folder validation already passed above. Record development, not confirmation.
        state["phase"] = "DEVELOPMENT_GATE_REACHED"
        state["stop_reason"] = (
            f"Development candidate {experiment_id} reached MAE "
            f"{summary['candidate_mae_c']} on frame {frame_id}; freeze and await authorization."
        )
    elif status in {
        "FAILED_RUNTIME","BLOCKED_MISSING_DATA"
    } and not state["champions"]:
        state["phase"] = "BLOCKED"
        state["stop_reason"] = f"Iteration blocked: {status}"
    else:
        state["phase"] = "DISCOVERY"
        state["stop_reason"] = None

    save_state(repo, state)
    ledger_fields = [
        "recorded_at_utc","frame_id","experiment_id","folder","candidate_id",
        "candidate_mae_c","baseline_id","baseline_mae_c","mae_delta_c",
        "n_common","date_start","date_end","leakage_status","status","reason",
    ]
    append_csv(
        repo / ".hkg_t24_research" / "champion_ledger.csv",
        {
            "recorded_at_utc": utc_now(),
            "frame_id": frame_id,
            "experiment_id": experiment_id,
            "folder": str(folder),
            "candidate_id": summary.get("candidate_id"),
            "candidate_mae_c": summary.get("candidate_mae_c"),
            "baseline_id": summary.get("baseline_id"),
            "baseline_mae_c": summary.get("baseline_mae_c"),
            "mae_delta_c": summary.get("mae_delta_c"),
            "n_common": summary.get("n_common"),
            "date_start": summary.get("date_start"),
            "date_end": summary.get("date_end"),
            "leakage_status": summary.get("leakage_status"),
            "status": status,
            "reason": reason,
        },
        ledger_fields,
    )
    append_decision(
        repo,
        f"Iteration {state['iteration']} ingested",
        f"Folder: `{folder}`\n\n"
        f"Status: `{status}`\n\n"
        f"Candidate MAE: `{summary.get('candidate_mae_c')}`\n\n"
        f"Champion changed: `{champion_changed}`\n\n"
        f"Next phase: `{state['phase']}`\n\n"
        f"Reason: {reason}",
    )
    print(json.dumps({
        "completion": completion,
        "champion_changed": champion_changed,
        "phase": state["phase"],
        "stop_reason": state.get("stop_reason"),
    }, indent=2))
    return 0


def main() -> int:
    args = parser().parse_args()
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    if args.command == "init":
        return command_init(repo)
    if args.command == "status":
        print(json.dumps(read_state(repo), indent=2))
        return 0
    if args.command == "begin":
        return command_begin(repo, args.spec.resolve())
    if args.command == "ingest":
        return command_ingest(repo, args.experiment_folder)
    if args.command == "stop":
        state = read_state(repo)
        state["phase"] = "STOPPED"
        state["stop_reason"] = args.reason
        state["open_experiment"] = None
        save_state(repo, state)
        append_decision(repo, "Research stopped", args.reason)
        return 0
    if args.command == "resume":
        state = read_state(repo)
        if state.get("open_experiment"):
            raise RuntimeError("Cannot resume discovery while an experiment is open")
        state["phase"] = "DISCOVERY"
        state["stop_reason"] = None
        save_state(repo, state)
        append_decision(repo, "Research resumed", args.reason)
        return 0
    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
