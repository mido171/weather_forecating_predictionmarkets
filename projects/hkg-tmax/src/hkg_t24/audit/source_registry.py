"""Final-patch source registry population and reporting."""

from __future__ import annotations

from dataclasses import astuple
from pathlib import Path
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import SOURCE_REGISTRY_ROWS, SourceRegistryRow

CSV_HEADERS = (
    "source_code",
    "source_family",
    "source_role",
    "feature_prefix",
    "strict_allowed",
    "proxy_allowed",
    "shadow_allowed",
    "blocked",
    "live_only",
    "support_only",
    "unit_semantics_verified",
    "availability_grade",
    "source_time_policy",
    "min_target_date_hkt",
    "max_target_date_hkt",
    "required_source_scope",
    "blocker_reason",
    "promotion_gate",
    "notes",
)


def source_registry_rows() -> tuple[SourceRegistryRow, ...]:
    return SOURCE_REGISTRY_ROWS


def populate_source_registry(connection: Any, writer: ReportWriter) -> Path:
    rows = [astuple(row) for row in SOURCE_REGISTRY_ROWS]
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO model_core.source_registry (
              source_code, source_family, source_role, feature_prefix,
              strict_allowed, proxy_allowed, shadow_allowed, blocked, live_only, support_only,
              unit_semantics_verified, availability_grade, source_time_policy,
              min_target_date_hkt, max_target_date_hkt, required_source_scope, blocker_reason,
              promotion_gate, notes
            )
            VALUES (
              %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::date,%s::date,%s,%s,%s,%s
            )
            ON CONFLICT (source_code) DO UPDATE SET
              source_family = EXCLUDED.source_family,
              source_role = EXCLUDED.source_role,
              feature_prefix = EXCLUDED.feature_prefix,
              strict_allowed = EXCLUDED.strict_allowed,
              proxy_allowed = EXCLUDED.proxy_allowed,
              shadow_allowed = EXCLUDED.shadow_allowed,
              blocked = EXCLUDED.blocked,
              live_only = EXCLUDED.live_only,
              support_only = EXCLUDED.support_only,
              unit_semantics_verified = EXCLUDED.unit_semantics_verified,
              availability_grade = EXCLUDED.availability_grade,
              source_time_policy = EXCLUDED.source_time_policy,
              min_target_date_hkt = EXCLUDED.min_target_date_hkt,
              max_target_date_hkt = EXCLUDED.max_target_date_hkt,
              required_source_scope = EXCLUDED.required_source_scope,
              blocker_reason = EXCLUDED.blocker_reason,
              promotion_gate = EXCLUDED.promotion_gate,
              notes = EXCLUDED.notes,
              updated_at_utc = now()
            """,
            rows,
        )
    connection.commit()
    return writer.write_csv("source_registry.csv", CSV_HEADERS, rows)


def validate_source_registry_contract(rows: tuple[SourceRegistryRow, ...] = SOURCE_REGISTRY_ROWS) -> None:
    feature_prefixes = [row.feature_prefix for row in rows]
    duplicates = sorted(
        prefix for prefix in set(feature_prefixes) if feature_prefixes.count(prefix) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate source-registry feature prefixes: {', '.join(duplicates)}")
    for row in rows:
        if row.blocked and row.availability_grade != "BLOCKED":
            raise ValueError(f"Blocked source must have BLOCKED grade: {row.source_code}")
        if row.source_code in {"nbmoc", "aigefssfc"} and not row.blocked:
            raise ValueError(f"{row.source_code} must remain blocked in Jira 001")
        if row.source_code == "aigfspres" and not row.support_only:
            raise ValueError("aigfspres must be support_only")
