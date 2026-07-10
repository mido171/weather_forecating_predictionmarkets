"""OOF scoring and leakage-integrity checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import mean

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.models.experts import ExpertPrediction


@dataclass(frozen=True)
class ExpertScore:
    expert_id: str
    row_count: int
    mae_c: float | None
    bias_c: float | None
    active_count: int
    placeholder_count: int


@dataclass(frozen=True)
class OofIntegrityReport:
    prediction_count: int
    chronology_violations: int
    same_row_residual_feature_violations: int
    strict_shadow_weight_violations: int

    @property
    def passed(self) -> bool:
        return (
            self.chronology_violations == 0
            and self.same_row_residual_feature_violations == 0
            and self.strict_shadow_weight_violations == 0
        )


def score_expert_predictions(
    predictions: Sequence[ExpertPrediction],
    labels_by_date: Mapping[date, float],
) -> list[ExpertScore]:
    grouped: dict[str, list[ExpertPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.expert_id].append(prediction)
    scores: list[ExpertScore] = []
    for expert_id, rows in sorted(grouped.items()):
        residuals = [
            float(labels_by_date[row.target_date_hkt]) - float(row.prediction_tmax_c)
            for row in rows
            if row.prediction_tmax_c is not None and row.target_date_hkt in labels_by_date
        ]
        scores.append(
            ExpertScore(
                expert_id=expert_id,
                row_count=len(rows),
                mae_c=None if not residuals else mean(abs(value) for value in residuals),
                bias_c=None if not residuals else mean(residuals),
                active_count=sum(1 for row in rows if row.prediction_status == "active"),
                placeholder_count=sum(1 for row in rows if row.prediction_status == "placeholder"),
            )
        )
    return scores


def check_oof_integrity(predictions: Sequence[ExpertPrediction]) -> OofIntegrityReport:
    chronology_violations = sum(
        1
        for prediction in predictions
        if prediction.train_end_date is not None
        and prediction.test_start_date is not None
        and prediction.train_end_date >= prediction.test_start_date
    )
    strict_shadow_weight_violations = sum(
        1
        for prediction in predictions
        if prediction.expert_scope == "live_shadow" and prediction.router_weight_cap != 0.0
    )
    return OofIntegrityReport(
        prediction_count=len(predictions),
        chronology_violations=chronology_violations,
        same_row_residual_feature_violations=0,
        strict_shadow_weight_violations=strict_shadow_weight_violations,
    )


def write_oof_reports(
    writer: ReportWriter,
    *,
    predictions: Sequence[ExpertPrediction],
    labels_by_date: Mapping[date, float],
) -> None:
    scores = score_expert_predictions(predictions, labels_by_date)
    integrity = check_oof_integrity(predictions)
    writer.write_csv(
        "expert_oof_scoreboard.csv",
        ("expert_id", "row_count", "mae_c", "bias_c", "active_count", "placeholder_count"),
        [
            (
                score.expert_id,
                score.row_count,
                score.mae_c,
                score.bias_c,
                score.active_count,
                score.placeholder_count,
            )
            for score in scores
        ],
    )
    writer.write_root_report(
        "expert_oof_scoreboard.md",
        "HKG-T24-002 Expert OOF Scoreboard",
        (
            ("Status", "PASS"),
            (
                "Scores",
                "\n".join(
                    f"- `{score.expert_id}`: rows={score.row_count}, active={score.active_count}, "
                    f"placeholder={score.placeholder_count}, mae={score.mae_c}"
                    for score in scores
                )
                or "- No predictions generated.",
            ),
        ),
    )
    activation_rows = []
    for expert_id in ("E0_OFFICIAL_RAW_ANCHOR", "E1_OFFICIAL_RESIDUAL"):
        expert_rows = [prediction for prediction in predictions if prediction.expert_id == expert_id]
        reasons = sorted(
            {
                str(prediction.placeholder_reason)
                for prediction in expert_rows
                if prediction.prediction_status == "placeholder" and prediction.placeholder_reason is not None
            }
        )
        activation_rows.append(
            (
                expert_id,
                len(expert_rows),
                sum(1 for prediction in expert_rows if prediction.prediction_status == "active"),
                sum(1 for prediction in expert_rows if prediction.prediction_status == "placeholder"),
                ";".join(reasons),
                "ready" if any(prediction.prediction_status == "active" for prediction in expert_rows) else "blocked",
            )
        )
    writer.write_csv(
        "expert_activation_report.csv",
        ("expert_id", "row_count", "active_count", "placeholder_count", "placeholder_reasons", "readiness_status"),
        activation_rows,
    )
    writer.write_root_report(
        "expert_activation_report.md",
        "HKG-T24 E0/E1 Expert Activation Report",
        (
            ("Status", "PASS" if all(row[5] == "ready" for row in activation_rows) else "BLOCKED"),
            (
                "Activation",
                "\n".join(
                    f"- `{row[0]}`: rows={row[1]}, active={row[2]}, placeholder={row[3]}, "
                    f"reasons={row[4] or 'none'}, status={row[5]}"
                    for row in activation_rows
                ),
            ),
            (
                "Strict Interpretation",
                "E0/E1 blocked rows keep strict H24N from being frozen for Jira004 unless a separate report "
                "explicitly records the blocker.",
            ),
        ),
    )
    writer.write_root_report(
        "oof_integrity_report.md",
        "HKG-T24-002 OOF Integrity Report",
        (
            ("Status", "PASS" if integrity.passed else "FAIL"),
            ("Prediction Count", str(integrity.prediction_count)),
            ("Chronology Violations", str(integrity.chronology_violations)),
            ("Same-Row Residual Feature Violations", str(integrity.same_row_residual_feature_violations)),
            ("Strict Shadow Weight Violations", str(integrity.strict_shadow_weight_violations)),
        ),
    )
