from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.engine import Connection


FEATURE_BUILD_VERSION = "TMAX_THIN_V1"
FEATURE_FAMILY = "gribstream_tmax_thin"
DEFAULT_CUTOFF_ID = "T_1245UTC"
DEFAULT_LOOKBACK_DAYS = 45
DEFAULT_HALF_LIFE_DAYS = 15.0
DEFAULT_LABEL_LAG_DAYS = 2
DEFAULT_MIN_TEST_DAYS = 730


@dataclass(frozen=True)
class ForecastRow:
    model_id: str
    target_date: date
    cutoff_utc: datetime
    raw_tmax_f: float
    settled_wu_tmax_f: float
    label_available_at_utc: datetime
    label_source_record_id: str
    label_revision_number: int
    max_source_run_time_utc: datetime | None
    max_source_available_at_utc: datetime | None
    source_feature_names: tuple[str, ...]
    raw_method: str


@dataclass(frozen=True)
class ScoredForecastRow:
    model_id: str
    target_date: date
    cutoff_utc: datetime
    raw_tmax_f: float
    bias_estimate_f: float
    corrected_tmax_f: float
    settled_wu_tmax_f: float
    raw_error_f: float
    corrected_error_f: float
    prior_error_count: int
    oldest_prior_target_date: date | None
    newest_prior_target_date: date | None
    max_source_run_time_utc: datetime | None
    max_source_available_at_utc: datetime | None
    label_available_at_utc: datetime
    label_source_record_id: str
    label_revision_number: int
    source_feature_names: tuple[str, ...]
    raw_method: str


@dataclass(frozen=True)
class ModelCoverage:
    model_id: str
    first_target_date: date
    last_target_date: date
    target_days: int
    buffered_target_days: int
    feature_rows: int
    buffered_feature_rows: int
    included_by_coverage: bool


@dataclass(frozen=True)
class MetricSummary:
    model_id: str
    status: str
    first_scored_date: date | None
    last_scored_date: date | None
    scored_days: int
    baseline_mae_f: float | None
    corrected_mae_f: float | None
    mae_delta_f: float | None
    mae_pct_change: float | None
    baseline_rmse_f: float | None
    corrected_rmse_f: float | None
    baseline_bias_f: float | None
    corrected_bias_f: float | None
    baseline_within_1f: float | None
    corrected_within_1f: float | None
    baseline_within_2f: float | None
    corrected_within_2f: float | None
    mean_prior_error_count: float | None
    coverage_target_days: int
    buffered_target_days: int
    raw_method: str | None


@dataclass(frozen=True)
class BacktestResult:
    config: dict[str, Any]
    coverage: list[ModelCoverage]
    summaries: list[MetricSummary]
    scored_rows: list[ScoredForecastRow]
    output_dir: Path | None


def run_backtest(
    connection: Connection,
    *,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    label_lag_days: int = DEFAULT_LABEL_LAG_DAYS,
    min_test_days: int = DEFAULT_MIN_TEST_DAYS,
    output_dir: Path | None = None,
) -> BacktestResult:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if label_lag_days < 1:
        raise ValueError("label_lag_days must be at least 1")
    if min_test_days <= 0:
        raise ValueError("min_test_days must be positive")

    coverage = load_coverage(connection, cutoff_id=cutoff_id)
    raw_forecasts = load_buffered_forecasts(connection, cutoff_id=cutoff_id)
    scored_by_model = score_models(
        raw_forecasts,
        lookback_days=lookback_days,
        half_life_days=half_life_days,
        label_lag_days=label_lag_days,
    )
    summaries = summarize_models(
        coverage,
        scored_by_model,
        min_test_days=min_test_days,
    )
    scored_rows = [row for rows in scored_by_model.values() for row in rows]
    summaries = [*summaries, summarize_equal_weight_ensemble(scored_rows)]
    config = {
        "source_table": "gold.feature_values",
        "feature_build_version": FEATURE_BUILD_VERSION,
        "feature_family": FEATURE_FAMILY,
        "cutoff_id": cutoff_id,
        "model_run_buffer_hours": 4,
        "lookback_days": lookback_days,
        "half_life_days": half_life_days,
        "label_lag_days": label_lag_days,
        "min_test_days": min_test_days,
        "source_of_truth": "public.wunderground_daily_tmax accepted/manual_confirmed KLGA tmax_f",
        "leakage_policy": (
            "feature max_source_available_at_utc <= cutoff_utc; "
            "source_latest_run_time_utc <= cutoff_utc - 4h; "
            "bias history uses only prior target dates at least label_lag_days old "
            "and labels available by the current cutoff"
        ),
    }
    result = BacktestResult(
        config=config,
        coverage=coverage,
        summaries=summaries,
        scored_rows=scored_rows,
        output_dir=output_dir,
    )
    if output_dir is not None:
        write_outputs(result, output_dir)
    return result


def load_coverage(connection: Connection, *, cutoff_id: str) -> list[ModelCoverage]:
    rows = connection.execute(
        text(
            """
            WITH model_day AS (
                SELECT
                    fv.source_trace_json->>'model_id' AS model_id,
                    ti.target_date,
                    COUNT(*) AS feature_rows,
                    COUNT(*) FILTER (
                        WHERE fv.source_latest_run_time_utc <= ti.cutoff_utc - INTERVAL '4 hours'
                          AND (fv.max_source_available_at_utc IS NULL OR fv.max_source_available_at_utc <= ti.cutoff_utc)
                    ) AS buffered_feature_rows
                FROM gold.feature_values fv
                JOIN gold.target_instances ti
                  ON ti.target_instance_id = fv.target_instance_id
                WHERE fv.feature_build_version = :feature_build_version
                  AND fv.feature_family = :feature_family
                  AND ti.cutoff_id = :cutoff_id
                  AND fv.source_trace_json ? 'model_id'
                GROUP BY 1, 2
            )
            SELECT
                model_id,
                MIN(target_date) AS first_target_date,
                MAX(target_date) AS last_target_date,
                COUNT(*) AS target_days,
                COUNT(*) FILTER (WHERE buffered_feature_rows > 0) AS buffered_target_days,
                SUM(feature_rows) AS feature_rows,
                SUM(buffered_feature_rows) AS buffered_feature_rows
            FROM model_day
            GROUP BY model_id
            ORDER BY first_target_date, model_id
            """
        ),
        {
            "feature_build_version": FEATURE_BUILD_VERSION,
            "feature_family": FEATURE_FAMILY,
            "cutoff_id": cutoff_id,
        },
    ).mappings()
    return [
        ModelCoverage(
            model_id=str(row["model_id"]),
            first_target_date=row["first_target_date"],
            last_target_date=row["last_target_date"],
            target_days=int(row["target_days"]),
            buffered_target_days=int(row["buffered_target_days"]),
            feature_rows=int(row["feature_rows"]),
            buffered_feature_rows=int(row["buffered_feature_rows"]),
            included_by_coverage=int(row["buffered_target_days"]) >= DEFAULT_MIN_TEST_DAYS,
        )
        for row in rows
    ]


def load_buffered_forecasts(connection: Connection, *, cutoff_id: str) -> list[ForecastRow]:
    feature_rows = connection.execute(
        text(
            """
            SELECT
                fv.source_trace_json->>'model_id' AS model_id,
                ti.target_date,
                ti.cutoff_utc,
                fv.feature_name,
                fv.feature_value,
                fv.source_latest_run_time_utc,
                fv.max_source_available_at_utc,
                actual.tmax_f AS settled_wu_tmax_f,
                actual.settlement_available_at_utc AS label_available_at_utc,
                NULL::uuid AS label_source_record_id,
                1 AS label_revision_number
            FROM gold.feature_values fv
            JOIN gold.target_instances ti
              ON ti.target_instance_id = fv.target_instance_id
            JOIN public.wunderground_daily_tmax actual
              ON actual.local_date = ti.target_date
             AND actual.station_id = 'KLGA'
             AND actual.validation_status IN ('accepted','manual_confirmed')
             AND actual.tmax_f IS NOT NULL
            WHERE fv.feature_build_version = :feature_build_version
              AND fv.feature_family = :feature_family
              AND ti.cutoff_id = :cutoff_id
              AND fv.source_trace_json ? 'model_id'
              AND fv.feature_available = true
              AND fv.feature_value IS NOT NULL
              AND (
                    fv.feature_name LIKE '%_member_0_tmax_proxy_f'
                 OR fv.feature_name LIKE '%_tmax_proxy_mean_f'
                 OR fv.feature_name LIKE '%_mean_proxy_f'
                 OR fv.feature_name LIKE '%_tmax_2m_f'
                 OR fv.feature_name LIKE '%_tmp_2m_peak_window_max_f'
                 OR fv.feature_name LIKE '%_current_tmp_2m_f'
                 OR fv.feature_name LIKE '%_valid_18z_tmp_2m_mean_f'
                 OR fv.feature_name LIKE '%_valid_00z_nextday_tmp_2m_mean_f'
                 OR fv.feature_name LIKE '%_valid_18z_tmp_2m_f'
                 OR fv.feature_name LIKE '%_valid_00z_nextday_tmp_2m_f'
              )
              AND fv.source_latest_run_time_utc <= ti.cutoff_utc - INTERVAL '4 hours'
              AND (fv.max_source_available_at_utc IS NULL OR fv.max_source_available_at_utc <= ti.cutoff_utc)
            ORDER BY model_id, ti.target_date, fv.feature_name
            """
        ),
        {
            "feature_build_version": FEATURE_BUILD_VERSION,
            "feature_family": FEATURE_FAMILY,
            "cutoff_id": cutoff_id,
        },
    ).mappings()

    grouped: dict[tuple[str, date], dict[str, Any]] = {}
    for row in feature_rows:
        key = (str(row["model_id"]), row["target_date"])
        group = grouped.setdefault(
            key,
            {
                "model_id": str(row["model_id"]),
                "target_date": row["target_date"],
                "cutoff_utc": row["cutoff_utc"],
                "settled_wu_tmax_f": float(row["settled_wu_tmax_f"]),
                "label_available_at_utc": row["label_available_at_utc"],
                "label_source_record_id": str(row["label_source_record_id"]),
                "label_revision_number": int(row["label_revision_number"]),
                "features": {},
                "run_times": [],
                "available_times": [],
            },
        )
        group["features"][str(row["feature_name"])] = float(row["feature_value"])
        if row["source_latest_run_time_utc"] is not None:
            group["run_times"].append(row["source_latest_run_time_utc"])
        if row["max_source_available_at_utc"] is not None:
            group["available_times"].append(row["max_source_available_at_utc"])

    forecasts: list[ForecastRow] = []
    for group in grouped.values():
        selection = select_raw_tmax(
            model_id=group["model_id"],
            features=group["features"],
        )
        if selection is None:
            continue
        value, method, source_feature_names = selection
        forecasts.append(
            ForecastRow(
                model_id=group["model_id"],
                target_date=group["target_date"],
                cutoff_utc=group["cutoff_utc"],
                raw_tmax_f=value,
                settled_wu_tmax_f=group["settled_wu_tmax_f"],
                label_available_at_utc=group["label_available_at_utc"],
                label_source_record_id=group["label_source_record_id"],
                label_revision_number=group["label_revision_number"],
                max_source_run_time_utc=max(group["run_times"]) if group["run_times"] else None,
                max_source_available_at_utc=max(group["available_times"]) if group["available_times"] else None,
                source_feature_names=tuple(source_feature_names),
                raw_method=method,
            )
        )
    return sorted(forecasts, key=lambda row: (row.model_id, row.target_date))


def select_raw_tmax(
    *,
    model_id: str,
    features: dict[str, float],
) -> tuple[float, str, tuple[str, ...]] | None:
    exact_candidates = (
        f"grib_{model_id}_klga_core_member_0_tmax_proxy_f",
        f"grib_{model_id}_klga_core_tmax_proxy_mean_f",
        f"grib_{model_id}_klga_core_mean_proxy_f",
        f"grib_{model_id}_klga_core_tmax_2m_f",
        f"grib_{model_id}_klga_core_tmp_2m_peak_window_max_f",
        f"grib_{model_id}_klga_core_current_tmp_2m_f",
    )
    for feature_name in exact_candidates:
        if feature_name in features:
            return (
                float(features[feature_name]),
                f"direct:{feature_name}",
                (feature_name,),
            )

    synoptic_candidates = (
        f"grib_{model_id}_klga_core_valid_18z_tmp_2m_mean_f",
        f"grib_{model_id}_klga_core_valid_00z_nextday_tmp_2m_mean_f",
        f"grib_{model_id}_klga_core_valid_18z_tmp_2m_f",
        f"grib_{model_id}_klga_core_valid_00z_nextday_tmp_2m_f",
    )
    selected = [
        (feature_name, float(features[feature_name]))
        for feature_name in synoptic_candidates
        if feature_name in features
    ]
    if selected:
        value = max(value for _, value in selected)
        return value, "synoptic:max_18z_00z_temperature_proxy", tuple(name for name, _ in selected)
    return None


def score_models(
    forecasts: Iterable[ForecastRow],
    *,
    lookback_days: int,
    half_life_days: float,
    label_lag_days: int,
) -> dict[str, list[ScoredForecastRow]]:
    by_model: dict[str, list[ForecastRow]] = defaultdict(list)
    for row in forecasts:
        by_model[row.model_id].append(row)

    scored: dict[str, list[ScoredForecastRow]] = {}
    for model_id, rows in by_model.items():
        rows = sorted(rows, key=lambda row: row.target_date)
        history: list[ForecastRow] = []
        model_scores: list[ScoredForecastRow] = []
        for row in rows:
            prior_rows = [
                prior
                for prior in history
                if label_lag_days <= (row.target_date - prior.target_date).days <= lookback_days
                and prior.label_available_at_utc <= row.cutoff_utc
            ]
            if prior_rows:
                bias = half_life_weighted_bias(
                    current_target_date=row.target_date,
                    prior_rows=prior_rows,
                    half_life_days=half_life_days,
                )
                corrected = row.raw_tmax_f - bias
                prior_dates = [prior.target_date for prior in prior_rows]
                model_scores.append(
                    ScoredForecastRow(
                        model_id=row.model_id,
                        target_date=row.target_date,
                        cutoff_utc=row.cutoff_utc,
                        raw_tmax_f=row.raw_tmax_f,
                        bias_estimate_f=bias,
                        corrected_tmax_f=corrected,
                        settled_wu_tmax_f=row.settled_wu_tmax_f,
                        raw_error_f=row.raw_tmax_f - row.settled_wu_tmax_f,
                        corrected_error_f=corrected - row.settled_wu_tmax_f,
                        prior_error_count=len(prior_rows),
                        oldest_prior_target_date=min(prior_dates),
                        newest_prior_target_date=max(prior_dates),
                        max_source_run_time_utc=row.max_source_run_time_utc,
                        max_source_available_at_utc=row.max_source_available_at_utc,
                        label_available_at_utc=row.label_available_at_utc,
                        label_source_record_id=row.label_source_record_id,
                        label_revision_number=row.label_revision_number,
                        source_feature_names=row.source_feature_names,
                        raw_method=row.raw_method,
                    )
                )
            history.append(row)
        scored[model_id] = model_scores
    return scored


def half_life_weighted_bias(
    *,
    current_target_date: date,
    prior_rows: Iterable[ForecastRow],
    half_life_days: float,
) -> float:
    weighted_error_sum = 0.0
    weight_sum = 0.0
    for row in prior_rows:
        age_days = (current_target_date - row.target_date).days
        weight = 0.5 ** (age_days / half_life_days)
        weighted_error_sum += weight * (row.raw_tmax_f - row.settled_wu_tmax_f)
        weight_sum += weight
    if weight_sum == 0.0:
        raise ValueError("cannot calculate bias with zero total weight")
    return weighted_error_sum / weight_sum


def summarize_models(
    coverage: list[ModelCoverage],
    scored_by_model: dict[str, list[ScoredForecastRow]],
    *,
    min_test_days: int,
) -> list[MetricSummary]:
    summaries: list[MetricSummary] = []
    for model_coverage in coverage:
        rows = scored_by_model.get(model_coverage.model_id, [])
        if model_coverage.buffered_target_days < min_test_days:
            status = "excluded_lt_2y_buffered_history"
        elif len(rows) < min_test_days:
            status = "excluded_lt_2y_scored_history"
        else:
            status = "included"
        summaries.append(
            _metric_summary(
                model_id=model_coverage.model_id,
                rows=rows if status == "included" else [],
                status=status,
                coverage=model_coverage,
            )
        )
    return summaries


def summarize_equal_weight_ensemble(rows: list[ScoredForecastRow]) -> MetricSummary:
    by_date: dict[date, list[ScoredForecastRow]] = defaultdict(list)
    for row in rows:
        by_date[row.target_date].append(row)

    ensemble_rows: list[ScoredForecastRow] = []
    for target_date, date_rows in sorted(by_date.items()):
        if not date_rows:
            continue
        raw = sum(row.raw_tmax_f for row in date_rows) / len(date_rows)
        corrected = sum(row.corrected_tmax_f for row in date_rows) / len(date_rows)
        actual = date_rows[0].settled_wu_tmax_f
        ensemble_rows.append(
            ScoredForecastRow(
                model_id="equal_weight_all_available",
                target_date=target_date,
                cutoff_utc=date_rows[0].cutoff_utc,
                raw_tmax_f=raw,
                bias_estimate_f=raw - corrected,
                corrected_tmax_f=corrected,
                settled_wu_tmax_f=actual,
                raw_error_f=raw - actual,
                corrected_error_f=corrected - actual,
                prior_error_count=sum(row.prior_error_count for row in date_rows) // len(date_rows),
                oldest_prior_target_date=min(
                    row.oldest_prior_target_date for row in date_rows if row.oldest_prior_target_date is not None
                ),
                newest_prior_target_date=max(
                    row.newest_prior_target_date for row in date_rows if row.newest_prior_target_date is not None
                ),
                max_source_run_time_utc=max(
                    row.max_source_run_time_utc for row in date_rows if row.max_source_run_time_utc is not None
                ),
                max_source_available_at_utc=max(
                    row.max_source_available_at_utc for row in date_rows if row.max_source_available_at_utc is not None
                ),
                label_available_at_utc=date_rows[0].label_available_at_utc,
                label_source_record_id=date_rows[0].label_source_record_id,
                label_revision_number=date_rows[0].label_revision_number,
                source_feature_names=("equal_weight_all_available",),
                raw_method="ensemble:equal_weight_scored_models_available_each_date",
            )
        )
    return _metric_summary(
        model_id="equal_weight_all_available",
        rows=ensemble_rows,
        status="diagnostic",
        coverage=ModelCoverage(
            model_id="equal_weight_all_available",
            first_target_date=min((row.target_date for row in ensemble_rows), default=None),
            last_target_date=max((row.target_date for row in ensemble_rows), default=None),
            target_days=len(ensemble_rows),
            buffered_target_days=len(ensemble_rows),
            feature_rows=0,
            buffered_feature_rows=0,
            included_by_coverage=True,
        ),
    )


def _metric_summary(
    *,
    model_id: str,
    rows: list[ScoredForecastRow],
    status: str,
    coverage: ModelCoverage,
) -> MetricSummary:
    if not rows:
        return MetricSummary(
            model_id=model_id,
            status=status,
            first_scored_date=None,
            last_scored_date=None,
            scored_days=0,
            baseline_mae_f=None,
            corrected_mae_f=None,
            mae_delta_f=None,
            mae_pct_change=None,
            baseline_rmse_f=None,
            corrected_rmse_f=None,
            baseline_bias_f=None,
            corrected_bias_f=None,
            baseline_within_1f=None,
            corrected_within_1f=None,
            baseline_within_2f=None,
            corrected_within_2f=None,
            mean_prior_error_count=None,
            coverage_target_days=coverage.target_days,
            buffered_target_days=coverage.buffered_target_days,
            raw_method=None,
        )

    raw_errors = [row.raw_error_f for row in rows]
    corrected_errors = [row.corrected_error_f for row in rows]
    baseline_mae = _mean(abs(error) for error in raw_errors)
    corrected_mae = _mean(abs(error) for error in corrected_errors)
    mae_delta = corrected_mae - baseline_mae
    return MetricSummary(
        model_id=model_id,
        status=status,
        first_scored_date=min(row.target_date for row in rows),
        last_scored_date=max(row.target_date for row in rows),
        scored_days=len(rows),
        baseline_mae_f=baseline_mae,
        corrected_mae_f=corrected_mae,
        mae_delta_f=mae_delta,
        mae_pct_change=(mae_delta / baseline_mae) if baseline_mae else None,
        baseline_rmse_f=math.sqrt(_mean(error * error for error in raw_errors)),
        corrected_rmse_f=math.sqrt(_mean(error * error for error in corrected_errors)),
        baseline_bias_f=_mean(raw_errors),
        corrected_bias_f=_mean(corrected_errors),
        baseline_within_1f=_mean(1.0 if abs(error) <= 1.0 else 0.0 for error in raw_errors),
        corrected_within_1f=_mean(1.0 if abs(error) <= 1.0 else 0.0 for error in corrected_errors),
        baseline_within_2f=_mean(1.0 if abs(error) <= 2.0 else 0.0 for error in raw_errors),
        corrected_within_2f=_mean(1.0 if abs(error) <= 2.0 else 0.0 for error in corrected_errors),
        mean_prior_error_count=_mean(float(row.prior_error_count) for row in rows),
        coverage_target_days=coverage.target_days,
        buffered_target_days=coverage.buffered_target_days,
        raw_method=rows[0].raw_method,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def write_outputs(result: BacktestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": result.config,
                "coverage": [_json_ready(asdict(row)) for row in result.coverage],
                "summaries": [_json_ready(asdict(row)) for row in result.summaries],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "model_summary.csv", [asdict(row) for row in result.summaries])
    _write_csv(output_dir / "coverage.csv", [asdict(row) for row in result.coverage])
    _write_csv(output_dir / "daily_scored_forecasts.csv", [asdict(row) for row in result.scored_rows])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_ready(row))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def format_markdown_table(summaries: list[MetricSummary]) -> str:
    columns = [
        "model",
        "status",
        "coverage_days",
        "buffered_days",
        "scored_days",
        "baseline_mae",
        "corrected_mae",
        "delta",
        "raw_bias",
        "corrected_bias",
        "scored_range",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.model_id,
                    summary.status,
                    str(summary.coverage_target_days),
                    str(summary.buffered_target_days),
                    str(summary.scored_days),
                    _fmt_float(summary.baseline_mae_f),
                    _fmt_float(summary.corrected_mae_f),
                    _fmt_float(summary.mae_delta_f),
                    _fmt_float(summary.baseline_bias_f),
                    _fmt_float(summary.corrected_bias_f),
                    _fmt_range(summary.first_scored_date, summary.last_scored_date),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _fmt_range(first: date | None, last: date | None) -> str:
    if first is None or last is None:
        return ""
    return f"{first.isoformat()}..{last.isoformat()}"
