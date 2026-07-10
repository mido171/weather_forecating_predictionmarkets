"""Feature-matrix assembly and persistence helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from hkg_t24.constants import (
    CUTOFF_ID,
    PROXY_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
    STRICT_SCHEMA_VERSION,
)
from hkg_t24.features.calendar import calendar_feature_map
from hkg_t24.features.feature_dictionary import ordered_feature_names, validate_feature_names
from hkg_t24.features.target_memory import assert_target_year_index_matches_calendar
from hkg_t24.timeutils import snapshot_id
from hkg_t24.utils.hashing import sha256_json

FeatureValue = float | int | str | bool | None


@dataclass(frozen=True)
class FeatureMatrixRow:
    target_date_hkt: date
    cutoff_id: str
    snapshot_id: str
    feature_scope: str
    schema_version: str
    features: dict[str, FeatureValue]
    target_tmax_c: float | None = None


def schema_version_for_scope(scope: str) -> str:
    if scope == "strict":
        return STRICT_SCHEMA_VERSION
    if scope == "proxy":
        return PROXY_SCHEMA_VERSION
    if scope == "live_shadow":
        return SHADOW_SCHEMA_VERSION
    raise ValueError(f"Unsupported feature scope: {scope}")


def _normalize_features(features: Mapping[str, FeatureValue]) -> dict[str, FeatureValue]:
    ordered = ordered_feature_names(features.keys())
    return {name: features[name] for name in ordered}


def build_strict_matrix_rows(
    *,
    target_dates: Sequence[date],
    target_memory_by_date: Mapping[date, Mapping[str, FeatureValue]],
    official_by_date: Mapping[date, Mapping[str, FeatureValue]],
    nwp_by_date: Mapping[date, Mapping[str, FeatureValue]],
    online_by_date: Mapping[date, Mapping[str, FeatureValue]],
    labels_by_date: Mapping[date, float | None] | None = None,
) -> list[FeatureMatrixRow]:
    rows: list[FeatureMatrixRow] = []
    for target_date_hkt in sorted(target_dates):
        calendar_features = calendar_feature_map(target_date_hkt)
        target_features = dict(target_memory_by_date.get(target_date_hkt, {}))
        if target_features:
            assert_target_year_index_matches_calendar(target_features, calendar_features)
        combined: dict[str, FeatureValue] = {}
        combined.update(calendar_features)
        combined.update(target_features)
        combined.update(official_by_date.get(target_date_hkt, {}))
        combined.update(online_by_date.get(target_date_hkt, {}))
        combined.update(nwp_by_date.get(target_date_hkt, {}))
        validate_feature_names("strict", tuple(combined))
        rows.append(
            FeatureMatrixRow(
                target_date_hkt=target_date_hkt,
                cutoff_id=CUTOFF_ID,
                snapshot_id=snapshot_id(target_date_hkt),
                feature_scope="strict",
                schema_version=STRICT_SCHEMA_VERSION,
                features=_normalize_features(combined),
                target_tmax_c=None if labels_by_date is None else labels_by_date.get(target_date_hkt),
            )
        )
    return rows


def build_scoped_matrix_rows(
    *,
    scope: str,
    target_dates: Sequence[date],
    feature_by_date: Mapping[date, Mapping[str, FeatureValue]],
    labels_by_date: Mapping[date, float | None] | None = None,
) -> list[FeatureMatrixRow]:
    schema_version = schema_version_for_scope(scope)
    rows: list[FeatureMatrixRow] = []
    for target_date_hkt in sorted(target_dates):
        features = dict(feature_by_date.get(target_date_hkt, {}))
        validate_feature_names(scope, tuple(features))
        rows.append(
            FeatureMatrixRow(
                target_date_hkt=target_date_hkt,
                cutoff_id=CUTOFF_ID,
                snapshot_id=snapshot_id(target_date_hkt),
                feature_scope=scope,
                schema_version=schema_version,
                features=_normalize_features(features),
                target_tmax_c=None if labels_by_date is None else labels_by_date.get(target_date_hkt),
            )
        )
    return rows


def persist_feature_matrix_rows(connection: Any, rows: Sequence[FeatureMatrixRow]) -> int:
    """Upsert feature-matrix rows into the final physical table."""
    with connection.cursor() as cursor:
        for row in rows:
            source_hash = sha256_json(
                {
                    "target_date_hkt": row.target_date_hkt.isoformat(),
                    "feature_scope": row.feature_scope,
                    "schema_version": row.schema_version,
                    "features": row.features,
                }
            )
            cursor.execute(
                """
                INSERT INTO model_features.feature_matrix (
                  target_date_hkt, cutoff_id, feature_scope, schema_version, snapshot_id,
                  features_jsonb, feature_count, source_hash, leakage_status, matrix_status,
                  target_tmax_c, feature_names_jsonb, availability_jsonb
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, 'passed', 'active', %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (target_date_hkt, cutoff_id, feature_scope, schema_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  features_jsonb = EXCLUDED.features_jsonb,
                  feature_count = EXCLUDED.feature_count,
                  generated_at_utc = now(),
                  source_hash = EXCLUDED.source_hash,
                  leakage_status = EXCLUDED.leakage_status,
                  matrix_status = EXCLUDED.matrix_status,
                  target_tmax_c = EXCLUDED.target_tmax_c,
                  feature_names_jsonb = EXCLUDED.feature_names_jsonb,
                  availability_jsonb = EXCLUDED.availability_jsonb
                """,
                (
                    row.target_date_hkt,
                    row.cutoff_id,
                    row.feature_scope,
                    row.schema_version,
                    row.snapshot_id,
                    json.dumps(row.features, sort_keys=True),
                    len(row.features),
                    source_hash,
                    row.target_tmax_c,
                    json.dumps(list(row.features), sort_keys=True),
                    json.dumps(
                        {
                            "non_null_feature_count": sum(
                                1 for value in row.features.values() if value is not None
                            ),
                            "null_feature_count": sum(
                                1 for value in row.features.values() if value is None
                            ),
                        },
                        sort_keys=True,
                    ),
                ),
            )
    return len(rows)


def matrix_row_to_model_record(row: FeatureMatrixRow) -> dict[str, FeatureValue]:
    """Flatten one matrix row with metadata columns first for model/export use."""
    record: dict[str, FeatureValue] = {
        "target_date_hkt": row.target_date_hkt.isoformat(),
        "cutoff_id": row.cutoff_id,
        "snapshot_id": row.snapshot_id,
        "feature_scope": row.feature_scope,
        "schema_version": row.schema_version,
    }
    record.update(row.features)
    return record
