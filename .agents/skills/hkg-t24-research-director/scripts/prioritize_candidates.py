#!/usr/bin/env python3
"""Rank a prefilled candidate queue with transparent weights.

Human/Director judgment supplies 0-5 component scores. This script makes the
aggregation reproducible; it does not invent scientific scores.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _common import write_csv

BENEFITS = {
    "expected_information_gain_0_5": 1.5,
    "expected_mae_lift_0_5": 1.5,
    "physical_plausibility_0_5": 1.0,
    "prior_support_0_5": 1.0,
    "novelty_0_5": 0.9,
    "readiness_0_5": 1.1,
    "sample_sufficiency_0_5": 1.0,
    "robustness_potential_0_5": 1.2,
    "downstream_value_0_5": 1.0,
    "backfill_durability_0_5": 0.8,
}
PENALTIES = {
    "timestamp_risk_0_5": 1.6,
    "overfit_risk_0_5": 1.4,
    "data_quality_risk_0_5": 1.0,
    "complexity_cost_0_5": 0.7,
    "single_station_or_year_risk_0_5": 1.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    return p.parse_args()


def score_value(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, 0))
    except ValueError:
        raise ValueError(f"Candidate {row.get('candidate_id')} has invalid {key}")
    if not 0 <= value <= 5:
        raise ValueError(f"{key} must be between 0 and 5")
    return value


def main() -> int:
    args = parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        benefit = sum(score_value(row, key) * weight for key, weight in BENEFITS.items())
        penalty = sum(score_value(row, key) * weight for key, weight in PENALTIES.items())
        row["priority_score"] = round(benefit - penalty, 6)
    rows.sort(key=lambda row: float(row["priority_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    write_csv(args.output, rows)
    for row in rows:
        print(f"{row['rank']:>2} {float(row['priority_score']):>8.3f} {row.get('candidate_id','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
