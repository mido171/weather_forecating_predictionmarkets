"""Leakage and governance audits for probability-only HKG Tmax experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FORBIDDEN_LIVE_FIELDS = {
    "market_price",
    "market_probability",
    "edge",
    "ev",
    "kelly",
    "pnl",
    "order_book",
    "bid",
    "ask",
    "yes_price",
    "no_price",
    "trade",
}

FORBIDDEN_PREDICTOR_FRAGMENTS = (
    "target_tmax",
    "bucket_key",
    "bucket_index",
    "raw_audit",
    "settled",
    "canonical",
)


def audit_modeling_table(modeling: pd.DataFrame, predictor_columns: list[str] | None = None) -> dict[str, Any]:
    issue_after_cutoff = modeling[pd.to_datetime(modeling["issue_at_utc"], utc=True) > pd.to_datetime(modeling["cutoff_at_utc"], utc=True)]
    duplicate_rows = modeling[modeling.duplicated(["target_date", "cutoff_profile"], keep=False)]
    predictor_columns = predictor_columns or []
    forbidden_predictors = [
        column for column in predictor_columns if any(fragment in column.lower() for fragment in FORBIDDEN_PREDICTOR_FRAGMENTS)
    ]
    sealed_before_2024 = modeling[(modeling["target_table"] == "sealed_confirmation") & (pd.to_datetime(modeling["target_date"]) < pd.Timestamp("2024-01-01"))]
    violations = {
        "post_cutoff_forecast_rows": int(len(issue_after_cutoff)),
        "duplicate_target_cutoff_rows": int(len(duplicate_rows)),
        "forbidden_predictor_columns": forbidden_predictors,
        "sealed_rows_before_2024": int(len(sealed_before_2024)),
    }
    total = int(violations["post_cutoff_forecast_rows"] + violations["duplicate_target_cutoff_rows"] + len(forbidden_predictors) + violations["sealed_rows_before_2024"])
    return {
        "status": "pass" if total == 0 else "fail",
        "total_violations": total,
        "violations": violations,
        "governance": {
            "probability_only": True,
            "market_inputs_used": False,
            "sealed_rows_tuning_allowed": False,
            "primary_cutoff": "T-1 23:59 HKT",
        },
    }


def audit_live_output(payload: dict[str, Any]) -> dict[str, Any]:
    flattened = {key.lower() for key in payload.keys()}
    forbidden = sorted(field for field in FORBIDDEN_LIVE_FIELDS if field in flattened)
    return {
        "status": "pass" if not forbidden else "fail",
        "forbidden_fields": forbidden,
        "probability_only": not forbidden,
    }


def write_leakage_audit(output_dir: Path, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
