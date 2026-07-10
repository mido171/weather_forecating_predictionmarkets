"""Database-backed expert factory orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import CUTOFF_ID, STRICT_SCHEMA_VERSION
from hkg_t24.features.matrix_builder import FeatureMatrixRow, FeatureValue
from hkg_t24.models.artifact_store import (
    persist_expert_predictions,
    persist_model_artifact,
    write_model_selection_artifact,
    write_prediction_csv,
)
from hkg_t24.models.experts import ExpertPrediction, generate_expert_oof_predictions
from hkg_t24.models.folds import FoldSpec, default_pre2024_folds, validate_folds
from hkg_t24.models.model_selection import CandidateMetric, select_model_candidate
from hkg_t24.models.oof import score_expert_predictions, write_oof_reports


@dataclass(frozen=True)
class ExpertFactorySummary:
    prediction_rows: int
    active_rows: int
    placeholder_rows: int
    artifact_rows: int


def _parse_features(raw: object) -> dict[str, FeatureValue]:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, str):
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return {str(key): value for key, value in loaded.items()}
    raise ValueError("features_jsonb was not a JSON object")


def load_feature_matrix_rows(
    connection: Any,
    *,
    scope: str,
    start_date: date,
    end_date: date,
) -> list[FeatureMatrixRow]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT target_date_hkt, cutoff_id, snapshot_id, feature_scope, schema_version,
                   features_jsonb, target_tmax_c::double precision
            FROM model_features.feature_matrix
            WHERE cutoff_id = 'H24N'
              AND feature_scope = %s
              AND target_date_hkt BETWEEN %s AND %s
              AND matrix_status = 'active'
            ORDER BY target_date_hkt
            """,
            (scope, start_date, end_date),
        )
        rows = cursor.fetchall()
    return [
        FeatureMatrixRow(
            target_date_hkt=row[0],
            cutoff_id=str(row[1]),
            snapshot_id=str(row[2]),
            feature_scope=str(row[3]),
            schema_version=str(row[4]),
            features=_parse_features(row[5]),
            target_tmax_c=None if row[6] is None else float(row[6]),
        )
        for row in rows
    ]


def labels_from_rows(rows: Sequence[FeatureMatrixRow]) -> dict[date, float]:
    return {
        row.target_date_hkt: float(row.target_tmax_c)
        for row in rows
        if row.target_tmax_c is not None
    }


def smoke_fold(start_date: date, end_date: date) -> list[FoldSpec]:
    train_end = min(start_date + timedelta(days=30), end_date - timedelta(days=7))
    test_start = train_end + timedelta(days=1)
    fold = FoldSpec("smoke_fold_0", start_date, train_end, test_start, end_date)
    validate_folds([fold])
    return [fold]


def folds_for_scope(*, start_date: date, end_date: date, smoke: bool) -> list[FoldSpec]:
    if smoke:
        return smoke_fold(start_date, end_date)
    folds = [
        fold
        for fold in default_pre2024_folds()
        if fold.test_end_date >= start_date and fold.test_start_date <= end_date
    ]
    return folds or smoke_fold(start_date, end_date)


def _score_by_expert(rows: Sequence[FeatureMatrixRow], predictions: Sequence[ExpertPrediction]) -> Mapping[str, float]:
    labels = labels_from_rows(rows)
    scores = score_expert_predictions(predictions, labels)
    return {score.expert_id: 999.0 if score.mae_c is None else score.mae_c for score in scores}


def run_expert_factory(
    connection: Any,
    writer: ReportWriter,
    *,
    scope: str,
    start_date: date,
    end_date: date,
    smoke: bool,
    persist_predictions: bool,
) -> ExpertFactorySummary:
    if scope != "strict-pre2024":
        raise ValueError("Jira 002 expert factory currently supports --scope strict-pre2024")
    rows = load_feature_matrix_rows(
        connection,
        scope="strict",
        start_date=start_date,
        end_date=end_date,
    )
    if not rows:
        raise ValueError("No strict feature_matrix rows found; run build-features --scope strict first")
    folds = folds_for_scope(start_date=start_date, end_date=end_date, smoke=smoke)
    predictions = generate_expert_oof_predictions(rows, folds)
    write_prediction_csv(writer, predictions)
    write_oof_reports(writer, predictions=predictions, labels_by_date=labels_from_rows(rows))
    scores = _score_by_expert(rows, predictions)
    baseline_mae = scores.get("E0_OFFICIAL_RAW_ANCHOR", 999.0)
    e1_mae = scores.get("E1_OFFICIAL_RESIDUAL", 999.0)
    selection = select_model_candidate(
        [
            CandidateMetric(
                candidate_id="E1_OFFICIAL_RESIDUAL_mean_residual_v1",
                expert_id="E1_OFFICIAL_RESIDUAL",
                validation_mae_c=e1_mae,
                baseline_mae_c=baseline_mae,
                row_count=len(rows),
                complexity_rank=1,
            )
        ],
        required_improvement_c=0.01,
        promoted_weight_cap=0.80,
        demoted_weight_cap=0.0,
    )
    artifact_path = write_model_selection_artifact(writer, selection)
    persist_model_artifact(
        connection,
        artifact_id="E1_OFFICIAL_RESIDUAL_selection_v1",
        expert_id="E1_OFFICIAL_RESIDUAL",
        artifact_kind="model_selection",
        artifact_payload=selection.to_json_dict(),
        feature_schema_version=STRICT_SCHEMA_VERSION,
    )
    if persist_predictions:
        persist_expert_predictions(connection, predictions)
    writer.write_root_report(
        "expert_factory_report.md",
        "HKG-T24-002 Expert Factory Report",
        (
            ("Status", "PASS"),
            ("Scope", f"`{scope}`"),
            ("Rows Loaded", str(len(rows))),
            ("Predictions", str(len(predictions))),
            ("Model Selection Artifact", str(artifact_path)),
            ("Cutoff", CUTOFF_ID),
        ),
    )
    return ExpertFactorySummary(
        prediction_rows=len(predictions),
        active_rows=sum(1 for row in predictions if row.prediction_status == "active"),
        placeholder_rows=sum(1 for row in predictions if row.prediction_status == "placeholder"),
        artifact_rows=1,
    )
