from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .acquisition_contracts import (
    AcquisitionContractError,
    validate_gridded_policy,
    validate_historical_live_pairs,
)
from .config import ConfigError, SourceCatalog, load_yaml
from .experiments import (
    EXPERIMENT_TEMPLATE_DIR,
    REQUIRED_EXPERIMENT_FILES,
    ExperimentError,
    validate_registry_state,
)
from .settlement import load_bucket_set
from .timeutils import asof_eligible, require_aware


class ValidationError(RuntimeError):
    """Raised when repository or data contracts fail."""


class LeakageError(ValidationError):
    """Raised when a record violates an as-of cutoff."""


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[str, ...]
    warnings: tuple[str, ...]


REQUIRED_EXPERIMENT_README_HEADINGS = (
    "## Status",
    "## Question and hypothesis",
    "## As-of contract",
    "## Method",
    "## Results",
    "## Decision",
    "## Reproduce",
    "## Evidence map",
)


def validate_source_catalog(root: Path) -> list[str]:
    path = root / "config" / "sources" / "data_sources.yaml"
    catalog = SourceCatalog.from_path(path)
    checks = [f"source catalog: {len(catalog.sources)} unique sources"]
    allowed_roles = {
        "OPERATIONAL_POINT_IN_TIME",
        "PROXY_WITH_LIMITATIONS",
        "RETROSPECTIVE_ONLY",
        "TARGET_ONLY",
        "MARKET_ONLY",
        "METADATA",
        "POTENTIAL_POINT_IN_TIME_ARCHIVE",
        "STATIC_METADATA",
    }
    for source in catalog.sources:
        _ = source.url
        if source.point_in_time_status not in allowed_roles:
            raise ValidationError(
                f"Source {source.id!r} has unknown point_in_time_status "
                f"{source.point_in_time_status!r}"
            )
        if "availability_rule" not in source.raw:
            raise ValidationError(f"Source {source.id!r} missing availability_rule")
        if "revision_policy" not in source.raw:
            raise ValidationError(f"Source {source.id!r} missing revision_policy")
    return checks


def validate_configs(root: Path) -> tuple[list[str], list[str]]:
    required = (
        Path("project/project.yaml"),
        Path("project/target.yaml"),
        Path("project/asof.yaml"),
        Path("project/evaluation.yaml"),
        Path("sources/data_sources.yaml"),
        Path("sources/historical_live_pairs.yaml"),
        Path("acquisition/gridded_acquisition_policy.yaml"),
        Path("sources/stations_hko.yaml"),
        Path("project/example_market_buckets.yaml"),
    )
    checks: list[str] = []
    warnings: list[str] = []
    for relative_path in required:
        data = load_yaml(root / "config" / relative_path)
        checks.append(f"config/{relative_path.as_posix()}: valid YAML mapping")
        if not data:
            raise ValidationError(f"config/{relative_path.as_posix()} is empty")

    target = load_yaml(root / "config" / "project" / "target.yaml")
    target_status = target.get("target", {}).get("canonical_status")
    if target_status != "verified":
        warnings.append(
            "Target canonical_status is not verified; predictive modelling remains gated by G1."
        )

    asof = load_yaml(root / "config" / "project" / "asof.yaml")
    if asof.get("primary_horizon") is None:
        warnings.append("Primary horizon is not selected; G2 remains open.")

    checks.extend(validate_source_catalog(root))
    try:
        checks.extend(validate_historical_live_pairs(root))
        checks.extend(validate_gridded_policy(root))
    except AcquisitionContractError as exc:
        raise ValidationError(str(exc)) from exc
    return checks, warnings


def validate_experiment_template(root: Path) -> list[str]:
    template = root / "experiments" / EXPERIMENT_TEMPLATE_DIR
    missing = [name for name in REQUIRED_EXPERIMENT_FILES if not (template / name).exists()]
    if missing:
        raise ValidationError(f"Experiment template missing files: {missing}")
    readme = (template / "README.md").read_text(encoding="utf-8")
    missing_headings = [
        heading for heading in REQUIRED_EXPERIMENT_README_HEADINGS if heading not in readme
    ]
    if missing_headings:
        raise ValidationError(f"Experiment template README missing headings: {missing_headings}")
    return [
        "experiment template: "
        f"{len(REQUIRED_EXPERIMENT_FILES)} required files and "
        f"{len(REQUIRED_EXPERIMENT_README_HEADINGS)} README sections present"
    ]


def validate_experiment_registry(root: Path) -> list[str]:
    try:
        count = validate_registry_state(root)
    except ExperimentError as exc:
        raise ValidationError(str(exc)) from exc
    return [f"experiment registry: {count} governed entries valid"]


def validate_bucket_fixture(root: Path) -> list[str]:
    bucket_set = load_bucket_set(root / "config" / "project" / "example_market_buckets.yaml")
    return [f"bucket fixture: {len(bucket_set.buckets)} non-overlapping full-coverage buckets"]


def validate_repository(root: Path) -> ValidationReport:
    checks, warnings = validate_configs(root)
    checks.extend(validate_experiment_template(root))
    checks.extend(validate_experiment_registry(root))
    checks.extend(validate_bucket_fixture(root))
    return ValidationReport(tuple(checks), tuple(warnings))


def assert_records_asof(
    records: Iterable[Mapping[str, Any]],
    *,
    cutoff_at: datetime,
    available_field: str = "available_at",
    id_field: str | None = None,
) -> None:
    require_aware(cutoff_at, "cutoff_at")
    for index, record in enumerate(records):
        record_id = record.get(id_field) if id_field else index
        value = record.get(available_field)
        if not isinstance(value, datetime):
            raise LeakageError(f"Record {record_id!r} has missing/non-datetime {available_field}")
        require_aware(value, f"{available_field} for record {record_id!r}")
        if not asof_eligible(value, cutoff_at):
            raise LeakageError(
                f"Record {record_id!r} leaks future information: "
                f"{available_field}={value.isoformat()} > cutoff={cutoff_at.isoformat()}"
            )


def assert_split_disjoint(
    train_dates: Iterable[Any],
    validation_dates: Iterable[Any],
    test_dates: Iterable[Any],
) -> None:
    train = set(train_dates)
    validation = set(validation_dates)
    test = set(test_dates)
    intersections = {
        "train_validation": train & validation,
        "train_test": train & test,
        "validation_test": validation & test,
    }
    nonempty = {name: values for name, values in intersections.items() if values}
    if nonempty:
        preview = {name: sorted(map(str, values))[:5] for name, values in nonempty.items()}
        raise LeakageError(f"Temporal split overlap detected: {preview}")


def _bounded_yaml_paths(root: Path) -> tuple[list[Path], int]:
    paths = [path for path in root.glob("*.yaml") if path.is_file()]
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    pending: list[Path] = []
    skipped_paths = 0
    for relative in (".agents", "config", "db", "docs", "experiments", "planning"):
        path = root / relative
        if not path.exists():
            continue
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        if path.is_symlink() or bool(attributes & reparse_flag):
            skipped_paths += 1
            continue
        if path.is_dir():
            pending.append(path)
    excluded_names = {".venv", "__pycache__", "data", "var"}
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: entry.name.casefold(), reverse=True)
        except FileNotFoundError:
            skipped_paths += 1
            continue
        for entry in entries:
            try:
                attributes = getattr(
                    entry.stat(follow_symlinks=False),
                    "st_file_attributes",
                    0,
                )
                is_reparse = entry.is_symlink() or bool(attributes & reparse_flag)
                if is_reparse:
                    skipped_paths += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in excluded_names:
                        pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".yaml"):
                    paths.append(Path(entry.path))
            except FileNotFoundError:
                skipped_paths += 1
    return sorted(set(paths)), skipped_paths


def _yaml_io_path(path: Path) -> Path:
    raw = str(path.absolute())
    if os.name != "nt" or raw.startswith("\\\\?\\") or len(raw) < 248:
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
    return Path("\\\\?\\" + raw)


def validate_yaml_tree(root: Path) -> list[str]:
    paths, skipped_paths = _bounded_yaml_paths(root)
    for path in paths:
        try:
            load_yaml(_yaml_io_path(path))
        except ConfigError as exc:
            raise ValidationError(str(exc)) from exc
    return [
        f"YAML tree: {len(paths)} bounded non-reparse files valid; "
        f"{skipped_paths} reparse/unavailable paths skipped"
    ]
