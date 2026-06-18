from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ConfigError, SourceCatalog, load_yaml
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


REQUIRED_EXPERIMENT_FILES = (
    "README.md",
    "HYPOTHESIS.md",
    "PROTOCOL.md",
    "ASOF_CONTRACT.md",
    "DATA_MANIFEST.yaml",
    "RUN_CONFIG.yaml",
    "RESULTS.md",
    "CONCLUSION.md",
    "REPRODUCE.md",
    "STATUS.yaml",
    "results/metrics.json",
)


def validate_source_catalog(root: Path) -> list[str]:
    path = root / "config" / "data_sources.yaml"
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
        "project.yaml",
        "target.yaml",
        "asof.yaml",
        "evaluation.yaml",
        "data_sources.yaml",
        "stations_hko.yaml",
        "example_market_buckets.yaml",
    )
    checks: list[str] = []
    warnings: list[str] = []
    for name in required:
        data = load_yaml(root / "config" / name)
        checks.append(f"config/{name}: valid YAML mapping")
        if not data:
            raise ValidationError(f"config/{name} is empty")

    target = load_yaml(root / "config" / "target.yaml")
    target_status = target.get("target", {}).get("canonical_status")
    if target_status != "verified":
        warnings.append(
            "Target canonical_status is not verified; predictive modelling remains gated by G1."
        )

    asof = load_yaml(root / "config" / "asof.yaml")
    if asof.get("primary_horizon") is None:
        warnings.append("Primary horizon is not selected; G2 remains open.")

    checks.extend(validate_source_catalog(root))
    return checks, warnings


def validate_experiment_template(root: Path) -> list[str]:
    template = root / "experiments" / "_template"
    missing = [name for name in REQUIRED_EXPERIMENT_FILES if not (template / name).exists()]
    if missing:
        raise ValidationError(f"Experiment template missing files: {missing}")
    registry = load_yaml(root / "experiments" / "registry.yaml")
    if not isinstance(registry.get("next_id"), int) or registry["next_id"] < 1:
        raise ValidationError("experiments/registry.yaml next_id must be a positive integer")
    return [f"experiment template: {len(REQUIRED_EXPERIMENT_FILES)} required files present"]


def validate_bucket_fixture(root: Path) -> list[str]:
    bucket_set = load_bucket_set(root / "config" / "example_market_buckets.yaml")
    return [f"bucket fixture: {len(bucket_set.buckets)} non-overlapping full-coverage buckets"]


def validate_repository(root: Path) -> ValidationReport:
    checks, warnings = validate_configs(root)
    checks.extend(validate_experiment_template(root))
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
            raise LeakageError(
                f"Record {record_id!r} has missing/non-datetime {available_field}"
            )
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


def validate_yaml_tree(root: Path) -> list[str]:
    checks: list[str] = []
    for path in sorted(root.rglob("*.yaml")):
        if any(part in {".venv", "data"} for part in path.parts):
            continue
        try:
            load_yaml(path)
        except ConfigError as exc:
            raise ValidationError(str(exc)) from exc
        checks.append(f"{path.relative_to(root)}: valid YAML")
    return checks
