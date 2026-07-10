from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


def generate_forecast_report(
    connection: Connection,
    *,
    run_id_text: str,
    artifact_root: Path,
) -> dict[str, Any]:
    run = connection.execute(
        text(
            """
            SELECT *
            FROM reports.forecast_evaluation_runs
            WHERE run_id_text = :run_id_text
            """
        ),
        {"run_id_text": run_id_text},
    ).mappings().first()
    if run is None:
        raise RuntimeError(f"unknown forecast evaluation run: {run_id_text}")
    rows = connection.execute(
        text(
            """
            SELECT *
            FROM reports.v_forecast_accuracy_daily_scores
            WHERE run_id_text = :run_id_text
            ORDER BY target_date, cutoff_id
            """
        ),
        {"run_id_text": run_id_text},
    ).mappings().all()
    report_dir = artifact_root / "reports" / "forecast_accuracy" / run_id_text
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "summary.json"
    daily_path = report_dir / "daily_scores.csv"
    markdown_path = report_dir / "report.md"
    summary = {
        "run_id_text": run_id_text,
        "prediction_kind": run["prediction_kind"],
        "status": run["status"],
        "start_date": run["start_date"].isoformat(),
        "end_date": run["end_date"].isoformat(),
        "cutoff_id": run["cutoff_id"],
        "metrics": run["metrics_json"],
        "daily_rows": len(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    with daily_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
        else:
            handle.write("")
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "daily_scores_path": str(daily_path),
        "markdown_path": str(markdown_path),
        "daily_rows": len(rows),
    }


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics") or {}
    lines = [
        "# KLGA Tmax Forecast Accuracy Report",
        "",
        f"- Run: `{summary['run_id_text']}`",
        f"- Prediction kind: `{summary['prediction_kind']}`",
        f"- Date range: `{summary['start_date']}` to `{summary['end_date']}`",
        f"- Cutoff: `{summary['cutoff_id']}`",
        f"- Daily rows: `{summary['daily_rows']}`",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics):
        lines.append(f"- `{key}`: `{metrics[key]}`")
    lines.append("")
    return "\n".join(lines)
