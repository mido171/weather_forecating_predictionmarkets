from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_yaml
from .experiment_registry import (
    ExperimentError,
    _is_reparse_path,
    _regular_child_directories,
    _write_text_atomic,
)


def _campaign_status_paths(experiments_root: Path) -> list[Path]:
    """Return only campaign-level experiment statuses, never shard/run internals."""

    campaigns_root = experiments_root / "campaigns"
    if not campaigns_root.is_dir():
        return []
    statuses: list[Path] = []
    for campaign in _regular_child_directories(campaigns_root, "Campaigns directory"):
        for experiment in _regular_child_directories(campaign, "Campaign directory"):
            status = experiment / "STATUS.yaml"
            if status.is_file():
                if _is_reparse_path(status):
                    raise ExperimentError(
                        f"Experiment status must not be a reparse point: {status}"
                    )
                statuses.append(status)
    return statuses


def _escape_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def generate_index(root: Path) -> Path:
    experiments_root = root / "experiments"
    rows: list[dict[str, object]] = []
    for status_path in _campaign_status_paths(experiments_root):
        directory = status_path.parent
        relative_directory = directory.relative_to(experiments_root)
        status = load_yaml(status_path)
        decision = status.get("decision") or {}
        gates = status.get("gates") or {}
        reproducibility = gates.get("reproducibility")
        if reproducibility in (None, ""):
            reproducibility = status.get("reproducible", "")
        rows.append(
            {
                "id": status.get("experiment_id") or status.get("id") or directory.name,
                "title": status.get("title", directory.name),
                "status": status.get("status") or status.get("state") or "UNKNOWN",
                "conclusion": decision.get("primary_conclusion")
                or status.get("primary_conclusion", ""),
                "delta": decision.get("oos_delta", ""),
                "leakage": gates.get("asof_leakage") or status.get("leakage", ""),
                "reproducibility": reproducibility,
                "directory": relative_directory.as_posix(),
            }
        )

    lines = [
        "# Experiment Index",
        "",
        "Generated from experiment `STATUS.yaml` files.",
        "",
        "| ID | Title | Status | Primary conclusion | OOS delta | Leakage | Reproducible |",
        "|---|---|---|---|---:|---|---|",
    ]
    if not rows:
        lines.append("| — | No experiments completed yet | — | Run G1 first | — | — | — |")
    else:
        for row in rows:
            link = f"[{_escape_cell(row['id'])}](experiments/{row['directory']}/README.md)"
            lines.append(
                "| "
                + " | ".join(
                    [
                        link,
                        _escape_cell(row["title"]),
                        _escape_cell(row["status"]),
                        _escape_cell(row["conclusion"]),
                        _escape_cell(row["delta"]),
                        _escape_cell(row["leakage"]),
                        _escape_cell(row["reproducibility"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python -m hkg_tmax experiments index",
            "```",
            "",
        ]
    )
    path = root / "EXPERIMENT_INDEX.md"
    _write_text_atomic(path, "\n".join(lines))
    return path


def experiment_statuses(root: Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    experiments_root = root / "experiments"
    for path in _campaign_status_paths(experiments_root):
        relative_directory = path.parent.relative_to(experiments_root)
        data = load_yaml(path)
        data["_directory"] = relative_directory.as_posix()
        metrics_path = path.parent / "results" / "metrics.json"
        if metrics_path.is_file():
            if _is_reparse_path(metrics_path):
                raise ExperimentError(
                    f"Experiment metrics must not be a reparse point: {metrics_path}"
                )
            try:
                data["_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ExperimentError(f"Invalid metrics JSON: {metrics_path}") from exc
        statuses.append(data)
    return statuses


__all__ = ["experiment_statuses", "generate_index"]
