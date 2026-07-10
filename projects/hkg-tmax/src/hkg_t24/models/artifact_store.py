"""Artifact and database storage helpers for Jira 002 experts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.models.experts import ExpertPrediction
from hkg_t24.models.model_selection import ModelSelection
from hkg_t24.utils.hashing import sha256_json


def write_model_selection_artifact(
    writer: ReportWriter,
    selection: ModelSelection,
) -> Path:
    path = writer.paths.reports_dir / f"model_selection_{selection.expert_id}.json"
    path.write_text(json.dumps(selection.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    writer.write_root_report(
        "model_selection_report.md",
        "HKG-T24-002 Model Selection Report",
        (
            ("Selected Candidate", f"`{selection.selected_candidate_id}`"),
            ("Expert", f"`{selection.expert_id}`"),
            ("Promoted", str(selection.promoted)),
            ("Router Weight Cap", str(selection.router_weight_cap)),
            ("Tie Breaker", selection.tie_breaker),
        ),
    )
    return path


def write_prediction_csv(writer: ReportWriter, predictions: Sequence[ExpertPrediction]) -> Path:
    return writer.write_csv(
        "expert_predictions_oof.csv",
        (
            "target_date_hkt",
            "cutoff_id",
            "snapshot_id",
            "expert_id",
            "expert_scope",
            "fold_id",
            "prediction_tmax_c",
            "prediction_residual_c",
            "raw_anchor_tmax_c",
            "prediction_status",
            "placeholder_reason",
            "train_end_date",
            "test_start_date",
            "router_weight_cap",
            "feature_schema_version",
        ),
        [
            (
                prediction.target_date_hkt,
                prediction.cutoff_id,
                prediction.snapshot_id,
                prediction.expert_id,
                prediction.expert_scope,
                prediction.fold_id,
                prediction.prediction_tmax_c,
                prediction.prediction_residual_c,
                prediction.raw_anchor_tmax_c,
                prediction.prediction_status,
                prediction.placeholder_reason,
                prediction.train_end_date,
                prediction.test_start_date,
                prediction.router_weight_cap,
                prediction.feature_schema_version,
            )
            for prediction in predictions
        ],
    )


def persist_expert_predictions(connection: Any, predictions: Sequence[ExpertPrediction]) -> int:
    with connection.cursor() as cursor:
        for prediction in predictions:
            cursor.execute(
                """
                INSERT INTO model_oof.expert_prediction (
                  target_date_hkt, cutoff_id, snapshot_id, expert_id, expert_scope, fold_id,
                  prediction_tmax_c, prediction_residual_c, raw_anchor_tmax_c,
                  prediction_status, placeholder_reason, train_end_date, test_start_date,
                  router_weight_cap, feature_schema_version
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (target_date_hkt, cutoff_id, expert_id, fold_id) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  expert_scope = EXCLUDED.expert_scope,
                  prediction_tmax_c = EXCLUDED.prediction_tmax_c,
                  prediction_residual_c = EXCLUDED.prediction_residual_c,
                  raw_anchor_tmax_c = EXCLUDED.raw_anchor_tmax_c,
                  prediction_status = EXCLUDED.prediction_status,
                  placeholder_reason = EXCLUDED.placeholder_reason,
                  train_end_date = EXCLUDED.train_end_date,
                  test_start_date = EXCLUDED.test_start_date,
                  router_weight_cap = EXCLUDED.router_weight_cap,
                  feature_schema_version = EXCLUDED.feature_schema_version,
                  created_at_utc = now()
                """,
                (
                    prediction.target_date_hkt,
                    prediction.cutoff_id,
                    prediction.snapshot_id,
                    prediction.expert_id,
                    prediction.expert_scope,
                    prediction.fold_id,
                    prediction.prediction_tmax_c,
                    prediction.prediction_residual_c,
                    prediction.raw_anchor_tmax_c,
                    prediction.prediction_status,
                    prediction.placeholder_reason,
                    prediction.train_end_date,
                    prediction.test_start_date,
                    prediction.router_weight_cap,
                    prediction.feature_schema_version,
                ),
            )
    return len(predictions)


def persist_model_artifact(
    connection: Any,
    *,
    artifact_id: str,
    expert_id: str,
    artifact_kind: str,
    artifact_payload: dict[str, object],
    feature_schema_version: str,
) -> None:
    payload_hash = sha256_json(artifact_payload)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO model_oof.expert_artifact (
              artifact_id, expert_id, artifact_kind, artifact_jsonb, artifact_sha256,
              feature_schema_version
            )
            VALUES (%s,%s,%s,%s::jsonb,%s,%s)
            ON CONFLICT (artifact_id) DO UPDATE SET
              expert_id = EXCLUDED.expert_id,
              artifact_kind = EXCLUDED.artifact_kind,
              artifact_jsonb = EXCLUDED.artifact_jsonb,
              artifact_sha256 = EXCLUDED.artifact_sha256,
              feature_schema_version = EXCLUDED.feature_schema_version,
              created_at_utc = now()
            """,
            (
                artifact_id,
                expert_id,
                artifact_kind,
                json.dumps(artifact_payload, sort_keys=True),
                payload_hash,
                feature_schema_version,
            ),
        )


def prediction_to_json_dict(prediction: ExpertPrediction) -> dict[str, object]:
    return asdict(prediction)
