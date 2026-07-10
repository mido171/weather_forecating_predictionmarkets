from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any

import pandas as pd


def make_row_hash(*parts: object) -> str:
    """Return a stable, unambiguous SHA-256 identity for an evidence row."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def score_predictions(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    """Calculate standard point-forecast errors for valid target/prediction rows."""

    required = {"target_date", "target_tmax_c", prediction_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction-score columns: {', '.join(missing)}")
    scored = frame.loc[:, ["target_date", "target_tmax_c", prediction_column]].copy()
    scored["target_date"] = pd.to_datetime(scored["target_date"], errors="coerce")
    scored["target_tmax_c"] = pd.to_numeric(scored["target_tmax_c"], errors="coerce")
    scored[prediction_column] = pd.to_numeric(scored[prediction_column], errors="coerce")
    scored = scored.dropna().sort_values("target_date")
    if scored.empty:
        raise ValueError("No valid rows are available for prediction scoring")
    error = scored[prediction_column] - scored["target_tmax_c"]
    squared_error = error.pow(2)
    return {
        "n": int(len(scored)),
        "mae": float(error.abs().mean()),
        "rmse": float(math.sqrt(squared_error.mean())),
        "bias": float(error.mean()),
        "first_date": scored["target_date"].min().date().isoformat(),
        "last_date": scored["target_date"].max().date().isoformat(),
    }


def compute_date_gaps(dates: Iterable[object], frame_id: str) -> pd.DataFrame:
    """Describe missing calendar ranges between unique valid observation dates."""

    values = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna().dt.normalize()
    ordered = sorted(values.unique())
    rows: list[dict[str, object]] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_ts = pd.Timestamp(previous)
        current_ts = pd.Timestamp(current)
        missing_days = int((current_ts - previous_ts).days - 1)
        if missing_days <= 0:
            continue
        rows.append(
            {
                "frame_id": frame_id,
                "gap_start": (previous_ts + pd.Timedelta(days=1)).date().isoformat(),
                "gap_end": (current_ts - pd.Timedelta(days=1)).date().isoformat(),
                "missing_days": missing_days,
            }
        )
    return pd.DataFrame(rows, columns=["frame_id", "gap_start", "gap_end", "missing_days"])
