from __future__ import annotations

import json
import os
import re
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import ConfigError, dump_yaml, load_yaml


class ExperimentError(RuntimeError):
    """Raised for experiment registry or folder errors."""


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_EXPERIMENT_ID_RE = re.compile(r"EXP-([0-9]{4,})")
_EXPERIMENT_DIRECTORY_RE = re.compile(r"EXP-[0-9]{4,}-[a-z0-9]+(?:-[a-z0-9]+)*")
EXPERIMENT_CONTROL_DIR = "registry"
EXPERIMENT_TEMPLATE_DIR = "templates/standard"
GENERAL_CAMPAIGN = "general"
CANONICAL_CAMPAIGNS = (
    "hkg-tmax",
    "hkg-t24",
    "residual-modeling",
    "probability",
    "market-edges",
    GENERAL_CAMPAIGN,
)
CAMPAIGN_TITLES = {
    "hkg-tmax": "HKG Tmax campaign",
    "hkg-t24": "HKG T-24 campaign",
    "residual-modeling": "Residual-modeling campaign",
    "probability": "Probability campaign",
    "market-edges": "Market-edge campaign",
    GENERAL_CAMPAIGN: "General experiment campaign",
}
REQUIRED_EXPERIMENT_FILES = (
    "README.md",
    "DATA_MANIFEST.yaml",
    "RUN_CONFIG.yaml",
    "STATUS.yaml",
    "results/metrics.json",
)


def slugify(value: str, max_length: int = 64) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not slug:
        slug = "experiment"
    return slug[:max_length].rstrip("-")


def require_campaign(value: object) -> str:
    """Return an allowlisted campaign name or fail before touching the tree."""

    if not isinstance(value, str):
        raise ExperimentError("Experiment campaign must be a string")
    campaign = value.strip()
    if campaign not in CANONICAL_CAMPAIGNS:
        allowed = ", ".join(CANONICAL_CAMPAIGNS)
        raise ExperimentError(f"Unknown experiment campaign {value!r}; choose one of: {allowed}")
    return campaign


def _require_title(value: object) -> str:
    if not isinstance(value, str):
        raise ExperimentError("Experiment title must be a string")
    title = value.strip()
    if not title:
        raise ExperimentError("Experiment title cannot be empty")
    if len(title) > 160:
        raise ExperimentError("Experiment title cannot exceed 160 characters")
    if not title.isprintable() or len(title.splitlines()) != 1:
        raise ExperimentError("Experiment title must be a single printable line")
    return title


def _require_resolved_within(path: Path, parent: Path, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExperimentError(f"{label} escapes its governed root: {path}") from exc


def _is_reparse_path(path: Path) -> bool:
    try:
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError as exc:
        raise ExperimentError(f"Could not inspect path without following links: {path}") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _regular_child_directories(parent: Path, label: str) -> list[Path]:
    if not parent.is_dir():
        return []
    if _is_reparse_path(parent):
        raise ExperimentError(f"{label} must not be a symlink or reparse point: {parent}")
    directories: list[Path] = []
    for path in parent.iterdir():
        if _is_reparse_path(path):
            raise ExperimentError(f"{label} contains a symlink or reparse point: {path}")
        if path.is_dir():
            directories.append(path)
    return sorted(directories)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        dump_yaml(temporary, data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_rendered_experiment(destination: Path) -> None:
    missing = [name for name in REQUIRED_EXPERIMENT_FILES if not (destination / name).is_file()]
    if missing:
        raise ExperimentError(f"Rendered experiment is missing required files: {missing}")
    try:
        for name in ("DATA_MANIFEST.yaml", "RUN_CONFIG.yaml", "STATUS.yaml"):
            load_yaml(destination / name)
        json.loads((destination / "results" / "metrics.json").read_text(encoding="utf-8"))
    except (ConfigError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Rendered experiment is invalid: {exc}") from exc


def validate_registry_state(
    root: Path,
    registry: Mapping[str, Any] | None = None,
) -> int:
    """Validate the governed allocation ledger and every registered scaffold."""

    experiments_root = root / "experiments"
    registry_path = experiments_root / EXPERIMENT_CONTROL_DIR / "registry.yaml"
    if registry is None:
        try:
            registry = load_yaml(registry_path)
        except ConfigError as exc:
            raise ExperimentError(str(exc)) from exc
    if registry.get("registry_version") != 2:
        raise ExperimentError("Experiment registry_version must be 2")
    next_id = registry.get("next_id")
    entries = registry.get("experiments")
    if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 1:
        raise ExperimentError("Experiment registry next_id must be a positive integer")
    if not isinstance(entries, list):
        raise ExperimentError("Experiment registry experiments must be a list")

    seen_ids: set[str] = set()
    seen_directories: set[str] = set()
    allocated_numbers: list[int] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ExperimentError(f"Experiment registry entry {index} must be a mapping")
        experiment_id = entry.get("id")
        title = entry.get("title")
        campaign_value = entry.get("campaign")
        directory_value = entry.get("directory")
        created_at = entry.get("created_at_utc")
        registry_status = entry.get("status")

        match = (
            _EXPERIMENT_ID_RE.fullmatch(experiment_id) if isinstance(experiment_id, str) else None
        )
        if match is None or int(match.group(1)) < 1:
            raise ExperimentError(f"Experiment registry entry {index} has invalid id")
        allocated_number = int(match.group(1))
        if experiment_id != f"EXP-{allocated_number:04d}":
            raise ExperimentError(f"Experiment registry entry {index} has noncanonical id")
        if experiment_id in seen_ids:
            raise ExperimentError(f"Duplicate experiment id in registry: {experiment_id}")
        seen_ids.add(experiment_id)
        allocated_numbers.append(allocated_number)

        try:
            canonical_title = _require_title(title)
        except ExperimentError as exc:
            raise ExperimentError(f"Experiment {experiment_id} has invalid title: {exc}") from exc
        if canonical_title != title:
            raise ExperimentError(f"Experiment {experiment_id} title has outer whitespace")
        campaign = require_campaign(campaign_value)
        if not isinstance(created_at, str) or not created_at.strip():
            raise ExperimentError(f"Experiment {experiment_id} has invalid created_at_utc")
        if not isinstance(registry_status, str) or not registry_status.strip():
            raise ExperimentError(f"Experiment {experiment_id} has invalid status")
        if not isinstance(directory_value, str):
            raise ExperimentError(f"Experiment {experiment_id} has no directory")
        if (
            directory_value != directory_value.strip()
            or "\\" in directory_value
            or any(ord(character) < 32 or ord(character) == 127 for character in directory_value)
        ):
            raise ExperimentError(
                f"Experiment {experiment_id} has unsafe directory: {directory_value}"
            )
        parts = directory_value.split("/")
        if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
            raise ExperimentError(
                f"Experiment {experiment_id} must use campaigns/<campaign>/<experiment>"
            )
        if parts[0] != "campaigns" or parts[1] != campaign:
            raise ExperimentError(
                f"Experiment {experiment_id} campaign field does not match its directory"
            )
        if not _EXPERIMENT_DIRECTORY_RE.fullmatch(parts[2]) or not parts[2].startswith(
            f"{experiment_id}-"
        ):
            raise ExperimentError(
                f"Experiment {experiment_id} directory name must be {experiment_id}-<slug>"
            )
        if directory_value in seen_directories:
            raise ExperimentError(f"Duplicate experiment directory in registry: {directory_value}")
        seen_directories.add(directory_value)

        experiment_root = experiments_root.joinpath(*parts)
        _require_resolved_within(experiment_root, experiments_root, f"Experiment {experiment_id}")
        if not experiment_root.is_dir() or _is_reparse_path(experiment_root):
            raise ExperimentError(f"Experiment {experiment_id} directory is unavailable")
        campaign_readme = experiment_root.parent / "README.md"
        if not campaign_readme.is_file() or _is_reparse_path(campaign_readme):
            raise ExperimentError(f"Campaign {campaign} is missing a regular README.md")
        missing = [
            name
            for name in REQUIRED_EXPERIMENT_FILES
            if not (experiment_root / name).is_file() or _is_reparse_path(experiment_root / name)
        ]
        if missing:
            raise ExperimentError(f"Experiment {experiment_id} missing files: {missing}")
        markdown = sorted(path.name for path in experiment_root.glob("*.md"))
        if markdown != ["README.md"]:
            raise ExperimentError(
                f"Experiment {experiment_id} must have one top-level README.md: {markdown}"
            )
        _validate_rendered_experiment(experiment_root)
        try:
            status = load_yaml(experiment_root / "STATUS.yaml")
        except ConfigError as exc:
            raise ExperimentError(str(exc)) from exc
        if status.get("experiment_id") != experiment_id:
            raise ExperimentError(f"Experiment {experiment_id} STATUS.yaml id does not match")
        if status.get("title") != title:
            raise ExperimentError(f"Experiment {experiment_id} STATUS.yaml title does not match")
        for contract_name in ("DATA_MANIFEST.yaml", "RUN_CONFIG.yaml"):
            try:
                contract = load_yaml(experiment_root / contract_name)
            except ConfigError as exc:
                raise ExperimentError(str(exc)) from exc
            if contract.get("experiment_id") != experiment_id:
                raise ExperimentError(
                    f"Experiment {experiment_id} {contract_name} id does not match"
                )

    if allocated_numbers and next_id <= max(allocated_numbers):
        raise ExperimentError("Experiment registry next_id must exceed every allocated id")
    campaigns_root = experiments_root / "campaigns"
    if campaigns_root.is_dir():
        for campaign_root in _regular_child_directories(campaigns_root, "Campaigns directory"):
            for candidate in _regular_child_directories(campaign_root, "Campaign directory"):
                if not _EXPERIMENT_DIRECTORY_RE.fullmatch(candidate.name):
                    continue
                relative = candidate.relative_to(experiments_root).as_posix()
                if relative not in seen_directories:
                    raise ExperimentError(f"Unregistered governed experiment directory: {relative}")
    return len(entries)


__all__ = [
    "CAMPAIGN_TITLES",
    "CANONICAL_CAMPAIGNS",
    "EXPERIMENT_CONTROL_DIR",
    "EXPERIMENT_TEMPLATE_DIR",
    "GENERAL_CAMPAIGN",
    "REQUIRED_EXPERIMENT_FILES",
    "ExperimentError",
    "require_campaign",
    "slugify",
    "validate_registry_state",
]
