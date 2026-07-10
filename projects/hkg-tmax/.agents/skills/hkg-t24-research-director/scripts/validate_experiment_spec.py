#!/usr/bin/env python3
"""Validate an HKG T+24 experiment specification before execution."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ALLOWED_MODES = {
    "promotion_oriented","exploratory","diagnostic_only","data_quality",
    "timestamp_unlock","frame_harmonization",
}
ALLOWED_ELIGIBILITY = {
    "DEPLOYABLE_PROVEN","DEPLOYABLE_LAGGED_ONLY","DIAGNOSTIC_ONLY",
    "PROSPECTIVE_ONLY","BLOCKED","REJECTED",
}
PROMOTION_BLOCKED = {"DIAGNOSTIC_ONLY","PROSPECTIVE_ONLY","BLOCKED","REJECTED"}
REQUIRED_TOP = [
    "schema_version","title","slug","mode","hypothesis","rationale",
    "expected_sign_and_falsification","novelty","target","frame",
    "data_sources","stations","features","response","baseline","validation",
    "metrics","sample_rules","acceptance_gates","rejection_conditions",
    "required_outputs","owner_authorized_confirmation",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("spec", type=Path)
    p.add_argument("--repo-root", type=Path)
    p.add_argument("--json", action="store_true", dest="as_json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(spec, dict):
        print("ERROR: specification root must be an object", file=sys.stderr)
        return 1

    for key in REQUIRED_TOP:
        if key not in spec:
            errors.append(f"Missing required field: {key}")

    if spec.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if spec.get("mode") not in ALLOWED_MODES:
        errors.append(f"Invalid mode: {spec.get('mode')!r}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,100}", str(spec.get("slug",""))):
        errors.append("slug must be lowercase letters/numbers/_/-")
    for key, min_len in (
        ("title",10),("hypothesis",40),("rationale",80),
        ("expected_sign_and_falsification",40),
    ):
        if len(str(spec.get(key,"")).strip()) < min_len:
            errors.append(f"{key} is underspecified")

    target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
    if target.get("variable") != "tmax_c":
        errors.append("target.variable must be tmax_c")
    if target.get("horizon") != "T-24":
        errors.append("target.horizon must be T-24")
    if target.get("timezone") != "Asia/Hong_Kong":
        errors.append("target.timezone must be Asia/Hong_Kong")
    if not str(target.get("cutoff_contract_path") or "").strip():
        errors.append("target.cutoff_contract_path is required")
    elif args.repo_root:
        cutoff = Path(target["cutoff_contract_path"])
        if not cutoff.is_absolute():
            cutoff = args.repo_root / cutoff
        if not cutoff.exists():
            errors.append(f"Cutoff contract does not exist: {cutoff}")

    frame = spec.get("frame") if isinstance(spec.get("frame"), dict) else {}
    if not str(frame.get("frame_id") or "").strip():
        errors.append("frame.frame_id is required")
    if frame.get("development_end_exclusive") != "2024-01-01":
        errors.append("frame.development_end_exclusive must be 2024-01-01")
    if frame.get("confirmation_locked") is not True:
        errors.append("frame.confirmation_locked must be true")
    if spec.get("owner_authorized_confirmation") is not False:
        errors.append("owner_authorized_confirmation must be false during development")

    data_sources = spec.get("data_sources")
    if not isinstance(data_sources, list) or not data_sources:
        errors.append("data_sources must be a non-empty list")
        data_sources = []
    source_ids: set[str] = set()
    for idx, source in enumerate(data_sources):
        if not isinstance(source, dict):
            errors.append(f"data_sources[{idx}] must be an object")
            continue
        source_id = str(source.get("source_id") or "")
        if not source_id:
            errors.append(f"data_sources[{idx}] lacks source_id")
        elif source_id in source_ids:
            errors.append(f"Duplicate source_id: {source_id}")
        source_ids.add(source_id)
        if not source.get("paths"):
            errors.append(f"Source {source_id or idx} has no paths")
        if not source.get("attributes"):
            errors.append(f"Source {source_id or idx} has no attributes")
        eligibility = source.get("eligibility")
        if eligibility not in ALLOWED_ELIGIBILITY:
            errors.append(f"Source {source_id or idx} has invalid eligibility {eligibility!r}")
        if len(str(source.get("availability_proof") or "").strip()) < 10:
            errors.append(f"Source {source_id or idx} lacks meaningful availability proof")
        if args.repo_root:
            for raw_path in source.get("paths") or []:
                path_text = str(raw_path)
                if any(ch in path_text for ch in "*?[]"):
                    matches = list(args.repo_root.glob(path_text))
                    if not matches:
                        warnings.append(f"Source glob matches no files: {path_text}")
                else:
                    path = Path(path_text)
                    if not path.is_absolute():
                        path = args.repo_root / path
                    if not path.exists():
                        warnings.append(f"Source path does not currently exist: {path}")

    if spec.get("mode") == "promotion_oriented":
        blocked = [
            str(source.get("source_id"))
            for source in data_sources
            if source.get("eligibility") in PROMOTION_BLOCKED
        ]
        if blocked:
            errors.append(
                "Promotion experiment includes non-deployable sources: " + ", ".join(blocked)
            )

    stations = spec.get("stations") if isinstance(spec.get("stations"), dict) else {}
    if stations.get("metadata_required") is not True:
        errors.append("stations.metadata_required must be true")
    selection = stations.get("selection")
    if selection not in {"explicit","all_inventory","explicit_or_all_inventory","role_groups"}:
        errors.append("Invalid stations.selection")
    if selection == "explicit" and not stations.get("ids"):
        errors.append("Explicit station selection requires ids")
    if "unknown_identity_policy" not in stations:
        warnings.append("stations.unknown_identity_policy is not specified")

    features = spec.get("features")
    if not isinstance(features, list):
        errors.append("features must be a list")
        features = []
    feature_names: set[str] = set()
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            errors.append(f"features[{idx}] must be an object")
            continue
        name = str(feature.get("name") or "")
        if not name:
            errors.append(f"features[{idx}] lacks name")
        elif name in feature_names:
            errors.append(f"Duplicate feature name: {name}")
        feature_names.add(name)
        for key in ("formula","inputs","availability_rule","fit_scope"):
            if key not in feature or feature.get(key) in (None,"",[]):
                errors.append(f"Feature {name or idx} lacks {key}")
        formula = str(feature.get("formula") or "").lower()
        if re.search(r"\btarget(?:_date)?\s*t\b|actual_tmax", formula):
            warnings.append(
                f"Feature {name} formula mentions target/actual; manually verify no target-day input"
            )
        if feature.get("fit_scope") not in {
            "none","prior_rows_only","fold_training_only","static_predeclared"
        }:
            errors.append(f"Feature {name} has invalid fit_scope")

    response = spec.get("response") if isinstance(spec.get("response"), dict) else {}
    if not response.get("name") or not response.get("definition"):
        errors.append("response requires name and definition")
    baseline = spec.get("baseline") if isinstance(spec.get("baseline"), dict) else {}
    if not baseline.get("id"):
        errors.append("baseline.id is required")
    if baseline.get("identical_rows_required") is not True:
        errors.append("baseline.identical_rows_required must be true")

    validation = spec.get("validation") if isinstance(spec.get("validation"), dict) else {}
    if validation.get("method") not in {
        "expanding_walk_forward","rolling_walk_forward","prequential_online",
        "diagnostic_temporal_folds",
    }:
        errors.append("validation.method is invalid")
    folds = validation.get("outer_folds")
    if not isinstance(folds, list) or not folds:
        errors.append("validation.outer_folds must be a non-empty list")
    if not isinstance(validation.get("minimum_history_rows"), int) or validation.get("minimum_history_rows",0) < 1:
        errors.append("validation.minimum_history_rows must be a positive integer")
    if "inner_selection" not in validation:
        errors.append("validation.inner_selection is required")

    metrics = spec.get("metrics")
    required_metrics = {"mae_c","rmse_c","bias_c","median_ae_c","p90_ae_c","p95_ae_c"}
    if not isinstance(metrics, list):
        errors.append("metrics must be a list")
    else:
        missing_metrics = required_metrics - set(metrics)
        if missing_metrics:
            errors.append("Missing mandatory metrics: " + ", ".join(sorted(missing_metrics)))

    for key in ("sample_rules","acceptance_gates"):
        value = spec.get(key)
        if not isinstance(value, dict) or not value:
            errors.append(f"{key} must be a non-empty object")
    if not isinstance(spec.get("rejection_conditions"), list) or not spec.get("rejection_conditions"):
        errors.append("rejection_conditions must be non-empty")
    required_outputs = spec.get("required_outputs")
    if not isinstance(required_outputs, list) or len(required_outputs) < 3:
        errors.append("required_outputs must list at least three artifacts")
    else:
        for required in ("README.md","RESULTS.md","CONCLUSION.md"):
            if required not in required_outputs:
                errors.append(f"required_outputs must include {required}")

    novelty = spec.get("novelty") if isinstance(spec.get("novelty"), dict) else {}
    if len(str(novelty.get("difference") or "").strip()) < 30:
        errors.append("novelty.difference is underspecified")

    report = {
        "spec": str(args.spec.resolve()),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"VALID: {'YES' if not errors else 'NO'}")
        for error in errors:
            print(f"ERROR: {error}")
        for warning in warnings:
            print(f"WARNING: {warning}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
