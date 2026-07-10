from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import dump_yaml, load_yaml


class ExperimentError(RuntimeError):
    """Raised for experiment registry or folder errors."""


_SLUG_RE = re.compile(r"[^a-z0-9]+")
EXPERIMENT_CONTROL_DIR = "registry"
EXPERIMENT_TEMPLATE_DIR = "templates/standard"
DEFAULT_CAMPAIGN = "general"


def _campaign_status_paths(experiments_root: Path) -> list[Path]:
    """Return only campaign-level experiment statuses, never shard/run internals."""

    campaigns_root = experiments_root / "campaigns"
    if not campaigns_root.is_dir():
        return []
    statuses: list[Path] = []
    for campaign in sorted(path for path in campaigns_root.iterdir() if path.is_dir()):
        for experiment in sorted(path for path in campaign.iterdir() if path.is_dir()):
            status = experiment / "STATUS.yaml"
            if status.is_file():
                statuses.append(status)
    return statuses


def slugify(value: str, max_length: int = 64) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not slug:
        slug = "experiment"
    return slug[:max_length].rstrip("-")


@contextmanager
def _registry_lock(registry_path: Path, timeout_seconds: float = 15.0) -> Iterator[None]:
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ExperimentError(f"Timed out waiting for registry lock: {lock_path}") from None
            time.sleep(0.1)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _replace_placeholders(path: Path, values: dict[str, str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    path.write_text(text, encoding="utf-8")


def create_experiment(root: Path, title: str) -> Path:
    title = title.strip()
    if not title:
        raise ExperimentError("Experiment title cannot be empty")

    experiments_root = root / "experiments"
    control_root = experiments_root / EXPERIMENT_CONTROL_DIR
    registry_path = control_root / "registry.yaml"
    template = experiments_root / EXPERIMENT_TEMPLATE_DIR
    if not template.is_dir():
        raise ExperimentError(f"Missing experiment template: {template}")

    with _registry_lock(registry_path):
        registry = load_yaml(registry_path)
        next_id = registry.get("next_id")
        items = registry.get("experiments")
        if not isinstance(next_id, int) or next_id < 1:
            raise ExperimentError("Registry next_id is invalid")
        if not isinstance(items, list):
            raise ExperimentError("Registry experiments must be a list")

        experiment_id = f"EXP-{next_id:04d}"
        directory_name = f"{experiment_id}-{slugify(title)}"
        destination = experiments_root / "campaigns" / DEFAULT_CAMPAIGN / directory_name
        if destination.exists():
            raise ExperimentError(f"Experiment directory already exists: {destination}")

        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        shutil.copytree(template, destination)
        values = {
            "EXPERIMENT_ID": experiment_id,
            "TITLE": title,
            "CREATED_AT_UTC": created_at,
        }
        for path in destination.rglob("*"):
            if path.is_file():
                _replace_placeholders(path, values)

        items.append(
            {
                "id": experiment_id,
                "title": title,
                "directory": destination.relative_to(experiments_root).as_posix(),
                "created_at_utc": created_at,
                "status": "PLANNED",
            }
        )
        registry["next_id"] = next_id + 1
        registry["experiments"] = items
        dump_yaml(registry_path, registry)
    return destination


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
    path.write_text("\n".join(lines), encoding="utf-8")
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
            try:
                data["_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ExperimentError(f"Invalid metrics JSON: {metrics_path}") from exc
        statuses.append(data)
    return statuses
