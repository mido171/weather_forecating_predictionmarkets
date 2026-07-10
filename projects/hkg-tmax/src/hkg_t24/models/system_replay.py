"""End-to-end Jira003 strict-pre2024 replay orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.models.db_expert_factory import labels_from_rows, load_feature_matrix_rows
from hkg_t24.models.distribution import (
    persist_system_predictions,
    train_distribution_from_inputs,
    write_distribution_reports,
)
from hkg_t24.models.experts import ExpertPrediction
from hkg_t24.models.final_formula import SystemPrediction, assemble_pre_distribution_predictions
from hkg_t24.models.router import (
    RouterPrediction,
    RouterTrainingResult,
    load_expert_predictions,
    persist_router_results,
    synthetic_router_inputs,
    train_router_suite_from_inputs,
    write_router_reports,
)
from hkg_t24.models.specialists import (
    persist_specialist_results,
    train_specialists_from_inputs,
    write_specialist_reports,
)
from hkg_t24.validation.ablation import final_vs_pre_distribution_ablation
from hkg_t24.validation.metrics import forecast_metrics
from hkg_t24.validation.slices import monthly_system_metrics


def _load_inputs(
    connection: Any | None,
    *,
    start_date: date,
    end_date: date,
    smoke: bool,
) -> tuple[list[FeatureMatrixRow], list[ExpertPrediction]]:
    if smoke and connection is None:
        return synthetic_router_inputs()
    if connection is None:
        raise ValueError("connection is required for DB-backed Jira003 replay")
    rows = load_feature_matrix_rows(
        connection,
        scope="strict",
        start_date=start_date,
        end_date=end_date,
    )
    predictions = load_expert_predictions(connection, start_date=start_date, end_date=end_date)
    return rows, predictions


def _selected_router_predictions(
    router_results: Sequence[RouterTrainingResult],
) -> Sequence[RouterPrediction]:
    r0 = router_results[0]
    r1 = router_results[1]
    return r1.predictions if r1.promotion_status == "promoted" else r0.predictions


def _has_nwp_feature(row: FeatureMatrixRow) -> bool:
    return any(
        value is not None and name.startswith(("gfs__", "gefsens__", "gefsmean__"))
        for name, value in row.features.items()
    )


def _system_replay_coverage_rows(
    predictions: Sequence[SystemPrediction],
    rows: Sequence[FeatureMatrixRow],
) -> tuple[tuple[str, int, str], ...]:
    prediction_by_date = {prediction.target_date_hkt: prediction for prediction in predictions}
    strict_usable_forecasts = sum(
        1
        for prediction in predictions
        if prediction.final_pre_distribution_c is not None and prediction.leakage_status == "passed"
    )
    return (
        ("strict_h24n_matrix_rows", len(rows), "All strict feature-matrix rows requested for replay."),
        (
            "strict_h24n_usable_forecast_rows",
            strict_usable_forecasts,
            "Rows with a pre-distribution point forecast and passed leakage status.",
        ),
        (
            "strict_h24n_scoreable_final_rows",
            sum(1 for prediction in predictions if prediction.final_point_tmax_c is not None),
            "Rows with a rounded final point forecast after distribution handling.",
        ),
        (
            "official_anchor_available_rows",
            sum(1 for row in rows if row.features.get("official__forecast_max_c") is not None),
            "Rows with strict pre-freeze official forecast max available in the feature matrix.",
        ),
        (
            "official_anchor_unavailable_rows",
            sum(1 for row in rows if row.features.get("official__forecast_max_c") is None),
            "Rows where strict H24N official max is unavailable and E0 cannot be active.",
        ),
        (
            "target_memory_feature_available_rows",
            sum(
                1
                for row in rows
                if row.features.get("target__lag365_tmax_c") is not None
                or row.features.get("target__clim30_mean_c") is not None
            ),
            "Rows with at least one repaired long-history target-memory feature.",
        ),
        (
            "target_memory_fallback_rows",
            sum(
                1
                for prediction in predictions
                if prediction.component_jsonb.get("fallback_expert") == "E2_TARGET_MEMORY"
            ),
            "Rows where router fallback selected the strict target-memory expert.",
        ),
        (
            "nwp_backed_rows",
            sum(1 for row in rows if _has_nwp_feature(row)),
            "Rows with at least one strict GFS or GEFS feature present.",
        ),
        (
            "no_forecast_rows",
            sum(1 for prediction in predictions if prediction.final_pre_distribution_c is None),
            "Rows where no strict base forecast was available before distribution.",
        ),
        (
            "no_trade_rows",
            sum(1 for prediction in predictions if prediction.no_trade_flag),
            "Rows flagged no-trade after final formula and distribution confidence handling.",
        ),
        (
            "failed_closed_rows",
            sum(
                1
                for row in rows
                if (prediction := prediction_by_date.get(row.target_date_hkt)) is None
                or prediction.leakage_status != "passed"
            ),
            "Rows without a passed strict replay prediction.",
        ),
    )


def _write_system_reports(
    writer: ReportWriter,
    predictions: Sequence[SystemPrediction],
    rows: Sequence[FeatureMatrixRow],
) -> None:
    labels = labels_from_rows(rows)
    pairs = [
        (float(prediction.final_point_tmax_c), labels[prediction.target_date_hkt])
        for prediction in predictions
        if prediction.final_point_tmax_c is not None and prediction.target_date_hkt in labels
    ]
    metrics = forecast_metrics(pairs) if pairs else None
    writer.write_csv(
        "system_scoreboard_strict.csv",
        (
            "system_version",
            "row_count",
            "mae_c",
            "rmse_c",
            "bias_c",
            "p90_abs_error_c",
            "no_trade_count",
            "leakage_error_count",
        ),
        [
            (
                predictions[0].system_version if predictions else "system_v1_20260627",
                0 if metrics is None else metrics.row_count,
                None if metrics is None else metrics.mae_c,
                None if metrics is None else metrics.rmse_c,
                None if metrics is None else metrics.bias_c,
                None if metrics is None else metrics.p90_abs_error_c,
                sum(1 for prediction in predictions if prediction.no_trade_flag),
                sum(1 for prediction in predictions if prediction.leakage_status != "passed"),
            )
        ],
    )
    coverage_rows = _system_replay_coverage_rows(predictions, rows)
    writer.write_csv(
        "system_replay_coverage_report.csv",
        ("category", "row_count", "definition"),
        coverage_rows,
    )
    writer.write_csv(
        "system_scoreboard_proxy.csv",
        ("system_version", "status", "reason"),
        [("system_v1_20260627", "not_run", "proxy system is diagnostic-only and outside Jira003 strict replay")],
    )
    ablation = final_vs_pre_distribution_ablation(predictions, labels)
    writer.write_csv(
        "system_ablation_matrix.csv",
        (
            "candidate_id",
            "baseline_id",
            "candidate_mae_c",
            "baseline_mae_c",
            "delta_mae_c",
        ),
        []
        if ablation is None
        else [
            (
                ablation.candidate_id,
                ablation.baseline_id,
                ablation.metrics.mae_c,
                ablation.baseline_metrics.mae_c,
                ablation.delta_mae_c,
            )
        ],
    )
    slice_rows = monthly_system_metrics(predictions, labels)
    writer.write_csv(
        "system_slice_scoreboard.csv",
        ("month", "row_count", "mae_c", "rmse_c", "bias_c", "p90_abs_error_c"),
        [
            (
                month,
                summary.row_count,
                summary.mae_c,
                summary.rmse_c,
                summary.bias_c,
                summary.p90_abs_error_c,
            )
            for month, summary in sorted(slice_rows.items())
        ],
    )
    writer.write_root_report(
        "system_replay_report.md",
        "HKG-T24-003 Strict System Replay Report",
        (
            ("Status", "PASS"),
            ("Rows", str(len(predictions))),
            ("No-Trade Rows", str(sum(1 for prediction in predictions if prediction.no_trade_flag))),
            (
                "Replay Coverage",
                "\n".join(f"- `{category}`: {row_count}" for category, row_count, _ in coverage_rows),
            ),
            (
                "Metrics",
                "No scoreable rows."
                if metrics is None
                else f"MAE={metrics.mae_c:.6f}, RMSE={metrics.rmse_c:.6f}, bias={metrics.bias_c:.6f}, P90AE={metrics.p90_abs_error_c:.6f}.",
            ),
        ),
    )
    writer.write_root_report(
        "system_replay_coverage_report.md",
        "HKG-T24 Strict System Replay Coverage Report",
        tuple(
            (category, f"rows={row_count}; {definition}")
            for category, row_count, definition in coverage_rows
        ),
    )


def _write_jira004_readiness_report(
    writer: ReportWriter,
    *,
    rows: Sequence[FeatureMatrixRow],
    expert_predictions: Sequence[ExpertPrediction],
    router_results: Sequence[RouterTrainingResult],
    system_predictions: Sequence[SystemPrediction],
) -> None:
    e0_active = sum(
        1
        for prediction in expert_predictions
        if prediction.expert_id == "E0_OFFICIAL_RAW_ANCHOR" and prediction.prediction_status == "active"
    )
    e1_active = sum(
        1
        for prediction in expert_predictions
        if prediction.expert_id == "E1_OFFICIAL_RESIDUAL" and prediction.prediction_status == "active"
    )
    target_lag365_non_null = sum(1 for row in rows if row.features.get("target__lag365_tmax_c") is not None)
    official_non_null = sum(1 for row in rows if row.features.get("official__forecast_max_c") is not None)
    r0 = next((result for result in router_results if result.router_id == "R0_OFFICIAL_LONG_HISTORY"), None)
    r1 = next((result for result in router_results if result.router_id == "R1_CORE_GFS_GEFS"), None)
    blockers: list[str] = []
    if target_lag365_non_null == 0:
        blockers.append("TARGET_MEMORY_LONG_HISTORY_NOT_PROVEN")
    if official_non_null == 0:
        blockers.append("STRICT_OFFICIAL_FEATURES_ALL_NULL")
    if e0_active == 0:
        blockers.append("STRICT_E0_OFFICIAL_RAW_ANCHOR_INACTIVE")
    if r0 is None or r0.promotion_status != "promoted":
        blockers.append("R0_OFFICIAL_LONG_HISTORY_NOT_PROMOTED")
    status = "READY_FOR_JIRA004" if not blockers else "BLOCKED_BEFORE_JIRA004"
    no_trade = sum(1 for prediction in system_predictions if prediction.no_trade_flag)
    writer.write_csv(
        "jira004_readiness_blocker_report.csv",
        (
            "status",
            "blockers_json",
            "strict_rows",
            "target_lag365_non_null_rows",
            "official_forecast_max_non_null_rows",
            "e0_active_rows",
            "e1_active_rows",
            "r0_status",
            "r0_reason",
            "r1_status",
            "r1_reason",
            "system_rows",
            "system_no_trade_rows",
        ),
        [
            (
                status,
                json.dumps(blockers, sort_keys=True),
                len(rows),
                target_lag365_non_null,
                official_non_null,
                e0_active,
                e1_active,
                None if r0 is None else r0.promotion_status,
                None if r0 is None else r0.demotion_reason,
                None if r1 is None else r1.promotion_status,
                None if r1 is None else r1.demotion_reason,
                len(system_predictions),
                no_trade,
            )
        ],
    )
    writer.write_root_report(
        "jira004_readiness_blocker_report.md",
        "HKG-T24 Jira004 Readiness Blocker Report",
        (
            ("Status", status),
            ("Blockers", "\n".join(f"- `{blocker}`" for blocker in blockers) or "- None."),
            (
                "Strict Dataset Completeness",
                f"rows={len(rows)}; target_lag365_non_null={target_lag365_non_null}; "
                f"official_forecast_max_non_null={official_non_null}.",
            ),
            ("Expert Activation", f"E0 active={e0_active}; E1 active={e1_active}."),
            (
                "Router Status",
                f"R0={None if r0 is None else r0.promotion_status} "
                f"({None if r0 is None else r0.demotion_reason}); "
                f"R1={None if r1 is None else r1.promotion_status} "
                f"({None if r1 is None else r1.demotion_reason}).",
            ),
            (
                "Jira004 Decision",
                "Do not freeze or sealed-score this candidate until blockers are empty."
                if blockers
                else "The strict candidate has the prerequisite surfaces required for Jira004 orchestration.",
            ),
        ),
    )


def _persist_system_components(connection: Any, predictions: Sequence[SystemPrediction]) -> int:
    count = 0
    with connection.cursor() as cursor:
        if predictions:
            cursor.execute(
                """
                DELETE FROM model_eval.system_prediction_component
                WHERE run_mode = 'sealed_replay'
                  AND model_candidate_id = %s
                  AND target_date_hkt BETWEEN %s AND %s
                """,
                (
                    predictions[0].system_version,
                    min(prediction.target_date_hkt for prediction in predictions),
                    max(prediction.target_date_hkt for prediction in predictions),
                ),
            )
        for prediction in predictions:
            cursor.execute(
                """
                INSERT INTO model_eval.system_prediction_component (
                  target_date_hkt, cutoff_id, model_candidate_id, run_mode,
                  component_kind, component_name, component_value, component_weight,
                  component_status, details_jsonb
                )
                VALUES (%s,%s,%s,'sealed_replay',%s,%s,%s,%s,%s,%s::jsonb)
                """,
                (
                    prediction.target_date_hkt,
                    prediction.cutoff_id,
                    prediction.system_version,
                    "final_formula",
                    "component_jsonb",
                    prediction.final_point_tmax_c,
                    None,
                    "active" if prediction.leakage_status == "passed" else "failed_closed",
                    json.dumps(prediction.component_jsonb, sort_keys=True),
                ),
            )
            count += 1
    return count


def _persist_validation_scoreboard(
    connection: Any,
    predictions: Sequence[SystemPrediction],
    rows: Sequence[FeatureMatrixRow],
) -> int:
    labels = labels_from_rows(rows)
    scoreboard_windows = (
        ("system_v1_20260627__strict_pre2024_scoreable", "strict-pre2024_scoreable", None, None),
        (
            "system_v1_20260627__nwp_backed_2020_2023",
            "strict-pre2024_nwp_backed_2020-10-03_2023-12-31",
            date(2020, 10, 3),
            date(2023, 12, 31),
        ),
    )
    count = 0
    with connection.cursor() as cursor:
        for scoreboard_id, scope, start_date, end_date in scoreboard_windows:
            scored = [
                (prediction.target_date_hkt, float(prediction.final_point_tmax_c), labels[prediction.target_date_hkt])
                for prediction in predictions
                if prediction.final_point_tmax_c is not None
                and prediction.target_date_hkt in labels
                and (start_date is None or prediction.target_date_hkt >= start_date)
                and (end_date is None or prediction.target_date_hkt <= end_date)
            ]
            if not scored:
                continue
            metrics = forecast_metrics((prediction, label) for _, prediction, label in scored)
            first_date = min(target_date for target_date, _, _ in scored)
            last_date = max(target_date for target_date, _, _ in scored)
            no_trade_count = sum(
                1
                for prediction in predictions
                if prediction.target_date_hkt >= first_date
                and prediction.target_date_hkt <= last_date
                and prediction.no_trade_flag
            )
            leakage_error_count = sum(
                1
                for prediction in predictions
                if prediction.target_date_hkt >= first_date
                and prediction.target_date_hkt <= last_date
                and prediction.leakage_status != "passed"
            )
            cursor.execute(
                """
                INSERT INTO model_validation.scoreboard (
                  scoreboard_id, scoreboard_scope, candidate_id, baseline_id, row_count,
                  first_target_date_hkt, last_target_date_hkt, mae_c, rmse_c, bias_c,
                  median_abs_error_c, p75_abs_error_c, p90_abs_error_c, p95_abs_error_c,
                  large_error_ge_1c_rate, large_error_ge_2c_rate, delta_mae_vs_baseline_c,
                  slice_jsonb, pass_fail_status, run_id
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,NULL)
                ON CONFLICT (scoreboard_id) DO UPDATE SET
                  scoreboard_scope = EXCLUDED.scoreboard_scope,
                  candidate_id = EXCLUDED.candidate_id,
                  baseline_id = EXCLUDED.baseline_id,
                  row_count = EXCLUDED.row_count,
                  first_target_date_hkt = EXCLUDED.first_target_date_hkt,
                  last_target_date_hkt = EXCLUDED.last_target_date_hkt,
                  mae_c = EXCLUDED.mae_c,
                  rmse_c = EXCLUDED.rmse_c,
                  bias_c = EXCLUDED.bias_c,
                  median_abs_error_c = EXCLUDED.median_abs_error_c,
                  p75_abs_error_c = EXCLUDED.p75_abs_error_c,
                  p90_abs_error_c = EXCLUDED.p90_abs_error_c,
                  p95_abs_error_c = EXCLUDED.p95_abs_error_c,
                  large_error_ge_1c_rate = EXCLUDED.large_error_ge_1c_rate,
                  large_error_ge_2c_rate = EXCLUDED.large_error_ge_2c_rate,
                  delta_mae_vs_baseline_c = EXCLUDED.delta_mae_vs_baseline_c,
                  slice_jsonb = EXCLUDED.slice_jsonb,
                  pass_fail_status = EXCLUDED.pass_fail_status,
                  updated_at_utc = now()
                """,
                (
                    scoreboard_id,
                    scope,
                    predictions[0].system_version if predictions else "system_v1_20260627",
                    None,
                    metrics.row_count,
                    first_date,
                    last_date,
                    metrics.mae_c,
                    metrics.rmse_c,
                    metrics.bias_c,
                    metrics.median_abs_error_c,
                    metrics.p75_abs_error_c,
                    metrics.p90_abs_error_c,
                    metrics.p95_abs_error_c,
                    metrics.large_error_ge_1c_rate,
                    metrics.large_error_ge_2c_rate,
                    None,
                    json.dumps(
                        {
                            "no_trade_count": no_trade_count,
                            "leakage_error_count": leakage_error_count,
                            "source": "run-system-replay",
                        },
                        sort_keys=True,
                    ),
                    "pass",
                ),
            )
            count += 1
    return count


def run_jira003_replay(
    connection: Any | None,
    writer: ReportWriter,
    *,
    scope: str,
    start_date: date,
    end_date: date,
    smoke: bool,
    persist: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if scope != "strict-pre2024":
        raise ValueError("Jira003 system replay supports --scope strict-pre2024")
    rows, expert_predictions = _load_inputs(
        connection,
        start_date=start_date,
        end_date=end_date,
        smoke=smoke,
    )
    if not rows:
        raise ValueError("No strict feature_matrix rows available for Jira003 replay")
    if not expert_predictions:
        raise ValueError("No expert OOF predictions available for Jira003 replay")
    router_results = train_router_suite_from_inputs(rows, expert_predictions)
    write_router_reports(writer, router_results)
    selected_router_predictions = _selected_router_predictions(router_results)
    specialist_results = train_specialists_from_inputs(rows, selected_router_predictions)
    write_specialist_reports(writer, specialist_results)
    pre_distribution = assemble_pre_distribution_predictions(
        rows=rows,
        expert_predictions=expert_predictions,
        router_results=router_results,
        specialist_results=specialist_results,
    )
    distribution_result = train_distribution_from_inputs(pre_distribution, rows)
    write_distribution_reports(writer, distribution_result)
    _write_system_reports(writer, distribution_result.updated_predictions, rows)
    _write_jira004_readiness_report(
        writer,
        rows=rows,
        expert_predictions=expert_predictions,
        router_results=router_results,
        system_predictions=distribution_result.updated_predictions,
    )
    if persist and connection is not None:
        persist_router_results(connection, router_results)
        persist_specialist_results(connection, specialist_results)
        persist_system_predictions(connection, distribution_result.updated_predictions)
        _persist_system_components(connection, distribution_result.updated_predictions)
        _persist_validation_scoreboard(connection, distribution_result.updated_predictions, rows)
    return (
        "PASS",
        (),
        (
            f"router_rows={sum(len(result.predictions) for result in router_results)}",
            f"specialist_rows={sum(len(result.predictions) for result in specialist_results)}",
            f"system_rows={len(distribution_result.updated_predictions)}",
            f"distribution_status={distribution_result.distribution_status}",
        ),
    )
