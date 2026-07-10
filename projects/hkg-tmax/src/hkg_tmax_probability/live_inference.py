"""Probability-only live inference helpers for latest official forecast rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS
from hkg_tmax_probability.leakage_audit import audit_live_output


def write_live_inference_example(output_dir: Path, validation_predictions: pd.DataFrame, champion_method: str) -> dict[str, Any]:
    latest = validation_predictions[
        (validation_predictions["method"] == champion_method) & (validation_predictions["is_primary_cutoff"])
    ].sort_values("target_date").tail(1)
    if latest.empty:
        payload = {"status": "unavailable", "reason": "no champion primary predictions"}
    else:
        row = latest.iloc[0]
        payload = {
            "status": "ok",
            "target_date": str(pd.Timestamp(row["target_date"]).date()),
            "cutoff_profile": row["cutoff_profile"],
            "forecast_max_c": float(row["forecast_max_c"]),
            "forecast_min_c": float(row["forecast_min_c"]),
            "method": champion_method,
            "bucket_probabilities": {bucket: float(row[f"p_{bucket}"]) for bucket in BUCKET_KEYS},
            "scope": "weather_probability_only",
        }
    audit = audit_live_output(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "live_inference_example_output.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (output_dir / "live_inference_example_input.json").write_text(
        json.dumps(
            {
                "required_fields": [
                    "target_date",
                    "forecast_min_c",
                    "forecast_max_c",
                    "forecast_range_c",
                    "issue_at_utc",
                    "cutoff_profile",
                ],
                "scope": "weather_probability_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "live_inference_no_trading_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return payload
