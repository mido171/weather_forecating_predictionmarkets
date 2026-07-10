#!/usr/bin/env python3
"""Compare a proposed experiment with the complete existing corpus."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _common import normalize_tokens, read_text_if_exists, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--experiments-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--top", type=int, default=25)
    return p.parse_args()


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def feature_names(spec: dict) -> set[str]:
    return {
        str(item.get("name")).lower()
        for item in spec.get("features", [])
        if isinstance(item, dict) and item.get("name")
    }


def source_ids(spec: dict) -> set[str]:
    return {
        str(item.get("source_id")).lower()
        for item in spec.get("data_sources", [])
        if isinstance(item, dict) and item.get("source_id")
    }


def load_spec(folder: Path) -> dict:
    path = folder / "experiment_spec.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> int:
    args = parse_args()
    proposal = json.loads(args.spec.read_text(encoding="utf-8"))
    proposal_text = json.dumps(proposal, sort_keys=True)
    proposal_tokens = normalize_tokens(proposal_text)
    proposal_features = feature_names(proposal)
    proposal_sources = source_ids(proposal)
    proposal_response = str(proposal.get("response", {}).get("name") or "").lower()
    proposal_baseline = str(proposal.get("baseline", {}).get("id") or "").lower()

    rows: list[dict] = []
    for folder in sorted(p for p in args.experiments_root.iterdir() if p.is_dir()):
        prior = load_spec(folder)
        fallback = "\n".join(
            read_text_if_exists(folder / name, 500_000)
            for name in ("README.md","RESULTS.md","CONCLUSION.md")
        )
        prior_text = json.dumps(prior, sort_keys=True) if prior else fallback
        prior_tokens = normalize_tokens(prior_text)
        prior_features = feature_names(prior)
        prior_sources = source_ids(prior)
        prior_response = str(prior.get("response", {}).get("name") or "").lower()
        prior_baseline = str(prior.get("baseline", {}).get("id") or "").lower()

        lexical = jaccard(proposal_tokens, prior_tokens)
        feature_overlap = jaccard(proposal_features, prior_features)
        source_overlap = jaccard(proposal_sources, prior_sources)
        response_match = int(bool(proposal_response) and proposal_response == prior_response)
        baseline_match = int(bool(proposal_baseline) and proposal_baseline == prior_baseline)
        combined = (
            0.35 * lexical + 0.30 * feature_overlap + 0.15 * source_overlap
            + 0.10 * response_match + 0.10 * baseline_match
        )
        rows.append({
            "prior_folder": folder.name,
            "combined_similarity": combined,
            "lexical_jaccard": lexical,
            "feature_jaccard": feature_overlap,
            "source_jaccard": source_overlap,
            "response_match": response_match,
            "baseline_match": baseline_match,
            "shared_features": "|".join(sorted(proposal_features & prior_features)),
            "shared_sources": "|".join(sorted(proposal_sources & prior_sources)),
            "novelty_review": (
                "HIGH_DUPLICATION_RISK" if combined >= 0.75
                else "MATERIAL_OVERLAP_REVIEW" if combined >= 0.50
                else "LOWER_SIMILARITY"
            ),
        })
    rows.sort(key=lambda row: row["combined_similarity"], reverse=True)
    write_csv(args.output, rows[:args.top])
    for row in rows[: min(10, len(rows))]:
        print(
            f"{row['combined_similarity']:.3f} {row['prior_folder']} "
            f"{row['novelty_review']}"
        )
    return 2 if rows and rows[0]["combined_similarity"] >= 0.75 else 0


if __name__ == "__main__":
    raise SystemExit(main())
