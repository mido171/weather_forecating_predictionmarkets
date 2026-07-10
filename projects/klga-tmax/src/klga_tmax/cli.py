from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
import json
from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterator

from alembic import command
from alembic.config import Config
import requests
from sqlalchemy import text
import typer

from klga_tmax.config import ConfigError, Settings, load_settings
from klga_tmax.constants import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_CONTRACT_ERROR,
    EXIT_EVALUATION_ERROR,
    EXIT_MIGRATION_ERROR,
    EXIT_PREDICTION_ERROR,
    EXIT_REPORT_ERROR,
    EXIT_TARGET_GRID_ERROR,
    EXIT_TRAINING_ERROR,
    EXIT_VALIDATION_ERROR,
    FEATURE_VERSION,
    PROJECT_ROOT,
)
from klga_tmax.db.audit import finish_pipeline_run, start_pipeline_run
from klga_tmax.db.engine import make_engine
from klga_tmax.db.migrations_check import ContractInspection, inspect_contract
from klga_tmax.db.normalize_acquisition import AcquisitionContractError, normalize_acquisition
from klga_tmax.evaluation.accuracy import evaluate_accuracy
from klga_tmax.features.materialize import materialize_features
from klga_tmax.models.calibration import calibrate_predictions
from klga_tmax.models.forecasting import predict_range, train_expert_registry
from klga_tmax.providers.gribstream import catalog as gribstream_catalog
from klga_tmax.providers.gribstream import client as gribstream_client
from klga_tmax.providers.gribstream.backfill import (
    parse_model_ids,
    parse_tmax_thin_model_ids,
    prepare_gribstream_plan,
    prepare_tmax_thin_plan,
    run_gribstream_backfill,
    run_tmax_thin_backfill,
)
from klga_tmax.providers.gribstream.client import OneThreadRateLimiter
from klga_tmax.providers.gribstream.catalog import spec_summary_rows
from klga_tmax.providers.gribstream.config import (
    load_gribstream_settings,
    redacted_settings_payload as redacted_gribstream_settings_payload,
)
from klga_tmax.providers.gribstream.persistence import job_status, model_status
from klga_tmax.providers.gribstream.plan import (
    DEFAULT_CUTOFF_ID,
    DEFAULT_END_DATE,
    TMAX_THIN_JOB_ID,
    tmax_thin_spec_summary_rows,
    supported_cutoff_ids,
)
from klga_tmax.providers.polymarket.cutoff_analysis import run_cutoff_analysis
from klga_tmax.providers.wunderground.config import (
    load_wunderground_settings,
    redacted_settings_payload,
)
from klga_tmax.providers.wunderground.java_truth import run_java_wu_truth
from klga_tmax.registry.materialize_targets import materialize_target_instances
from klga_tmax.registry.seed import seed_all
from klga_tmax.reports.forecast_report import generate_forecast_report
from klga_tmax.validation.foundation import validate_foundation
from klga_tmax.validation.gribstream import validate_gribstream
from klga_tmax.validation.station_universe import validate_station_universe
from klga_tmax.validation.wunderground import validate_wunderground

app = typer.Typer(no_args_is_help=True, help="KLGA Tmax strategy foundation CLI.")
db_app = typer.Typer(no_args_is_help=True, help="Database migration and inspection commands.")
registry_app = typer.Typer(no_args_is_help=True, help="Registry seed and target materialization commands.")
validate_app = typer.Typer(no_args_is_help=True, help="Validation commands.")
wunderground_app = typer.Typer(
    no_args_is_help=True,
    help="Wunderground/Weather.com actuals acquisition commands.",
)
wu_app = typer.Typer(
    no_args_is_help=True,
    help="Plain Wunderground settled Tmax truth-table commands.",
)
gribstream_app = typer.Typer(
    no_args_is_help=True,
    help="GribStream NWP single-cutoff acquisition commands.",
)
polymarket_app = typer.Typer(
    no_args_is_help=True,
    help="Polymarket market-timing and cutoff optimization commands.",
)
features_app = typer.Typer(no_args_is_help=True, help="Leakage-safe strategy feature commands.")
train_app = typer.Typer(no_args_is_help=True, help="Forecast model registration/training commands.")
predict_app = typer.Typer(no_args_is_help=True, help="Forecast prediction commands.")
forecast_app = typer.Typer(no_args_is_help=True, help="Single-target forecast commands.")
evaluate_app = typer.Typer(no_args_is_help=True, help="Forecast accuracy evaluation commands.")
report_app = typer.Typer(no_args_is_help=True, help="Forecast report generation commands.")
settlement_app = typer.Typer(no_args_is_help=True, help="Settlement-label normalization commands.")

DEFAULT_LIVE_BUDGET = 1
MAX_WUNDERGROUND_DATE_SPAN_DAYS = 31
MAX_PROVIDER_REQUESTS = 100
MAX_WUNDERGROUND_WORKERS = 2
MIN_GRIBSTREAM_SPACING_SECONDS = 12.0

app.add_typer(db_app, name="db")
app.add_typer(registry_app, name="registry")
app.add_typer(validate_app, name="validate")
app.add_typer(wunderground_app, name="wunderground")
app.add_typer(wu_app, name="wu")
app.add_typer(gribstream_app, name="gribstream")
app.add_typer(polymarket_app, name="polymarket")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")
app.add_typer(forecast_app, name="forecast")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(report_app, name="report")
app.add_typer(settlement_app, name="settlement")


def _settings_or_exit() -> Settings:
    try:
        return load_settings(require_db=True)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc


def _offline_settings_or_exit() -> Settings:
    """Load bounded runtime paths/defaults without requiring a database DSN."""

    try:
        return load_settings(require_db=False)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc


def _wunderground_settings_or_exit(*, require_api_key: bool):
    try:
        return load_wunderground_settings(require_api_key=require_api_key)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc


def _gribstream_settings_or_exit(*, require_api_token: bool):
    try:
        return load_gribstream_settings(require_api_token=require_api_token)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc


def _alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _print_inspection(inspection: ContractInspection) -> None:
    payload = {
        "ok": inspection.ok,
        "details": inspection.details,
        "failures": inspection.failures,
        "warnings": inspection.warnings,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _parse_iso_date(value: str, option_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be YYYY-MM-DD") from exc


def _parse_date_range(start_date: str, end_date: str) -> tuple[date, date]:
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")
    if start > end:
        raise typer.BadParameter("--start-date must be on or before --end-date")
    return start, end


def _require_execute(enabled: bool, operation: str) -> None:
    if not enabled:
        raise typer.BadParameter(
            f"{operation} can contact a provider or mutate the database; "
            "re-run with --execute after reviewing the explicit scope and budgets"
        )


def _parse_explicit_models(
    raw_models: str,
    *,
    parser: Callable[[str | None], tuple[str, ...] | None],
    max_models: int,
) -> tuple[str, ...]:
    try:
        models = parser(raw_models)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--models") from exc
    if models is None:
        raise typer.BadParameter(
            "--models must list explicit model IDs; all and * are not allowed for live commands",
            param_hint="--models",
        )
    if len(models) > max_models:
        raise typer.BadParameter(
            f"--models selects {len(models)} models but --max-models is {max_models}",
            param_hint="--models",
        )
    return models


def _validate_gribstream_chunk_scope(
    *,
    start_date: date,
    end_date: date,
    max_chunks: int,
) -> None:
    target_days = (end_date - start_date).days + 1
    if target_days > max_chunks:
        raise typer.BadParameter(
            f"date scope contains {target_days} target days but --max-chunks is {max_chunks}; "
            "increase the explicit chunk budget or narrow the dates",
            param_hint="--max-chunks",
        )


def _parse_explicit_stations(
    raw_stations: str,
    *,
    option_name: str = "--stations",
) -> tuple[str, ...]:
    if raw_stations.strip().lower() in {"", "all", "*"}:
        raise typer.BadParameter(
            f"{option_name} must list explicit station IDs; all and * are not allowed for live commands",
            param_hint=option_name,
        )
    stations = tuple(
        dict.fromkeys(
            station.strip().upper()
            for station in raw_stations.split(",")
            if station.strip()
        )
    )
    if not stations:
        raise typer.BadParameter(f"{option_name} must contain at least one station ID")
    return stations


def _parse_single_station(raw_station: str, option_name: str) -> str:
    stations = _parse_explicit_stations(raw_station, option_name=option_name)
    if len(stations) != 1:
        raise typer.BadParameter(
            f"{option_name} must contain exactly one station ID",
            param_hint=option_name,
        )
    return stations[0]


def _validate_wunderground_scope(
    *,
    start_date: date,
    end_date: date,
    stations: tuple[str, ...],
    chunk_days: int,
    max_requests: int,
    workers: int,
    allow_long_date_range: bool,
    resume: bool,
) -> int:
    span_days = (end_date - start_date).days + 1
    if span_days > MAX_WUNDERGROUND_DATE_SPAN_DAYS and not allow_long_date_range:
        raise typer.BadParameter(
            f"Wunderground date scope is {span_days} days; the default maximum is "
            f"{MAX_WUNDERGROUND_DATE_SPAN_DAYS}. Use --allow-long-date-range together "
            "with an explicit --max-requests budget to override it.",
            param_hint="--end-date",
        )
    if workers > MAX_WUNDERGROUND_WORKERS:
        raise typer.BadParameter(
            f"--workers must be at most {MAX_WUNDERGROUND_WORKERS}",
            param_hint="--workers",
        )
    # Resume can fragment missing dates into one-day windows, so budget its safe
    # worst case instead of assuming contiguous chunks.
    windows_per_station = span_days if resume else ceil(span_days / chunk_days)
    planned_requests = len(stations) * windows_per_station
    if planned_requests > max_requests:
        raise typer.BadParameter(
            f"scope requires {planned_requests} provider request windows but --max-requests "
            f"is {max_requests}; narrow the scope or explicitly increase the budget",
            param_hint="--max-requests",
        )
    return planned_requests


def _print_provider_dry_run(
    *,
    operation: str,
    scope: dict[str, Any],
    budgets: dict[str, Any],
) -> None:
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "dry_run": True,
                "operation": operation,
                "provider_calls": 0,
                "database_calls": 0,
                "scope": scope,
                "budgets": budgets,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


@contextmanager
def _bounded_http_requests(max_requests: int) -> Iterator[None]:
    """Enforce a process-local HTTP-attempt budget for the Polymarket CLI run."""
    original_request = requests.sessions.Session.request
    request_count = 0

    def bounded_request(session, method, url, **kwargs):
        nonlocal request_count
        if request_count >= max_requests:
            raise RuntimeError(
                f"provider request budget exhausted after {max_requests} HTTP attempts"
            )
        request_count += 1
        return original_request(session, method, url, **kwargs)

    requests.sessions.Session.request = bounded_request
    try:
        yield
    finally:
        requests.sessions.Session.request = original_request


@contextmanager
def _spaced_gribstream_requests(spacing_seconds: float) -> Iterator[None]:
    """Apply one >=12s limiter across catalog and authenticated data requests."""
    limiter = OneThreadRateLimiter(spacing_seconds=spacing_seconds)
    original_catalog_urlopen = gribstream_catalog.urlopen
    original_client_urlopen = gribstream_client.urlopen

    def spaced_catalog_urlopen(*args, **kwargs):
        limiter.acquire()
        return original_catalog_urlopen(*args, **kwargs)

    def spaced_client_urlopen(*args, **kwargs):
        limiter.acquire()
        return original_client_urlopen(*args, **kwargs)

    gribstream_catalog.urlopen = spaced_catalog_urlopen
    gribstream_client.urlopen = spaced_client_urlopen
    try:
        yield
    finally:
        gribstream_catalog.urlopen = original_catalog_urlopen
        gribstream_client.urlopen = original_client_urlopen


def _job_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_gribstream_cutoff_id(value: str) -> str:
    supported = set(supported_cutoff_ids())
    if value not in supported:
        raise typer.BadParameter(f"--cutoff-id must be one of: {', '.join(sorted(supported))}")
    return value


def _parse_prediction_kind(value: str) -> str:
    allowed = {"oof", "holdout", "forecast", "replay"}
    if value not in allowed:
        raise typer.BadParameter(f"prediction kind must be one of: {', '.join(sorted(allowed))}")
    return value


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _run_audited(
    *,
    command_name: str,
    command_args: dict[str, Any],
    failure_exit_code: int,
    operation: Callable[[Any], dict[str, Any]],
    exception_exit_codes: dict[type[BaseException], int] | None = None,
) -> None:
    settings = _settings_or_exit()
    engine = make_engine(settings.database_url)
    run_id = None
    try:
        with engine.begin() as connection:
            run_id = start_pipeline_run(
                connection,
                command_name=command_name,
                command_args=command_args,
            )
        row_counts = operation(engine)
        with engine.begin() as connection:
            finish_pipeline_run(
                connection,
                pipeline_run_id=run_id,
                status="success",
                exit_code=0,
                row_counts=row_counts,
            )
    except typer.Exit:
        raise
    except Exception as exc:
        exit_code = failure_exit_code
        for exception_type, mapped_exit_code in (exception_exit_codes or {}).items():
            if isinstance(exc, exception_type):
                exit_code = mapped_exit_code
                break
        if run_id is not None:
            try:
                with engine.begin() as connection:
                    finish_pipeline_run(
                        connection,
                        pipeline_run_id=run_id,
                        status="failed",
                        exit_code=exit_code,
                        error_message=str(exc),
                    )
            except Exception:
                pass
        typer.echo(str(exc), err=True)
        raise typer.Exit(exit_code) from exc


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply Alembic migrations and seed foundation registry rows."""
    settings = _settings_or_exit()
    engine = make_engine(settings.database_url)
    run_id = None
    try:
        with engine.begin() as connection:
            run_id = start_pipeline_run(
                connection,
                command_name="db migrate",
                command_args={},
            )
        command.upgrade(_alembic_config(settings.database_url or ""), "head")
        with engine.begin() as connection:
            row_counts = seed_all(connection)
            finish_pipeline_run(
                connection,
                pipeline_run_id=run_id,
                status="success",
                exit_code=0,
                row_counts=row_counts,
            )
        typer.echo(json.dumps({"ok": True, "row_counts": row_counts}, sort_keys=True))
    except typer.Exit:
        raise
    except Exception as exc:
        if run_id is not None:
            try:
                with engine.begin() as connection:
                    finish_pipeline_run(
                        connection,
                        pipeline_run_id=run_id,
                        status="failed",
                        exit_code=EXIT_MIGRATION_ERROR,
                        error_message=str(exc),
                    )
            except Exception:
                pass
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_MIGRATION_ERROR) from exc


@db_app.command("inspect-contract")
def db_inspect_contract() -> None:
    """Verify schemas, tables, columns, indexes, seeds, and pgcrypto."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            inspection = inspect_contract(connection)
            _print_inspection(inspection)
            if not inspection.ok:
                raise RuntimeError("; ".join(inspection.failures))
            return dict(inspection.details)

    _run_audited(
        command_name="db inspect-contract",
        command_args={},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@registry_app.command("seed")
def registry_seed() -> None:
    """Insert or refresh canonical cutoffs, stations, and default feature version."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = seed_all(connection)
        typer.echo(json.dumps({"ok": True, "row_counts": row_counts}, sort_keys=True))
        return row_counts

    _run_audited(
        command_name="registry seed",
        command_args={},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@registry_app.command("materialize-targets")
def registry_materialize_targets(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    replace: bool = typer.Option(
        False,
        "--replace/--no-replace",
        help="Delete and reinsert selected target-date/cutoff rows.",
    ),
) -> None:
    """Materialize target/cutoff rows in gold.target_instances."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            inserted = materialize_target_instances(
                connection,
                start_date=start,
                end_date=end,
                replace=replace,
            )
        payload = {
            "ok": True,
            "target_instances_inserted": inserted,
            "replace": replace,
        }
        typer.echo(json.dumps(payload, sort_keys=True))
        return {"gold.target_instances_inserted": inserted}

    _run_audited(
        command_name="registry materialize-targets",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "replace": replace,
        },
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@validate_app.command("foundation")
def validate_foundation_command() -> None:
    """Run task-00 foundation validations."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            inspection = validate_foundation(connection)
            _print_inspection(inspection)
            if not inspection.ok:
                raise RuntimeError("; ".join(inspection.failures))
            return dict(inspection.details)

    _run_audited(
        command_name="validate foundation",
        command_args={},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@validate_app.command("station-universe")
def validate_station_universe_command() -> None:
    """Run task-01 station universe and coordinate registry validations."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            inspection = validate_station_universe(connection)
            _print_inspection(inspection)
            if not inspection.ok:
                raise RuntimeError("; ".join(inspection.failures))
            return dict(inspection.details)

    _run_audited(
        command_name="validate station-universe",
        command_args={},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@validate_app.command("wunderground")
def validate_wunderground_command() -> None:
    """Run task-02 Wunderground settlement actuals validations."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            inspection = validate_wunderground(connection)
            _print_inspection(inspection)
            if not inspection.ok:
                raise RuntimeError("; ".join(inspection.failures))
            return dict(inspection.details)

    _run_audited(
        command_name="validate wunderground",
        command_args={},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@validate_app.command("gribstream")
def validate_gribstream_command() -> None:
    """Run GribStream single-cutoff schema and lineage validations."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            inspection = validate_gribstream(connection)
            _print_inspection(inspection)
            if not inspection.ok:
                raise RuntimeError("; ".join(inspection.failures))
            return dict(inspection.details)

    _run_audited(
        command_name="validate gribstream",
        command_args={},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@features_app.command("materialize")
def features_materialize(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    cutoff_id: str | None = typer.Option(None, "--cutoff-id", help="Optional registered cutoff ID."),
    feature_version: str = typer.Option(FEATURE_VERSION, "--feature-version", help="Registry feature version to materialize."),
    replace: bool = typer.Option(False, "--replace/--no-replace", help="Replace existing strategy feature rows."),
) -> None:
    """Build leakage-safe strategy feature rows and matrices."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = materialize_features(
                connection,
                start_date=start,
                end_date=end,
                cutoff_id=cutoff_id,
                feature_version=feature_version,
                replace=replace,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="features materialize",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "cutoff_id": cutoff_id,
            "feature_version": feature_version,
            "replace": replace,
        },
        failure_exit_code=EXIT_DATA_CONTRACT_ERROR,
        operation=operation,
        exception_exit_codes={AcquisitionContractError: EXIT_VALIDATION_ERROR},
    )


@train_app.command("experts")
def train_experts_command(
    start_date: str = typer.Option(..., "--start-date", help="First training date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last training date, inclusive."),
    cutoff_id: str | None = typer.Option(None, "--cutoff-id", help="Accepted for CLI symmetry; model registry is cutoff-agnostic."),
    fold_policy: str = typer.Option("annual_walk_forward", "--fold-policy", help="Training fold policy label."),
) -> None:
    """Register deterministic PMF experts and the combiner model version."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = train_expert_registry(connection, start_date=start, end_date=end)
        payload = {"ok": True, "fold_policy": fold_policy, "cutoff_id": cutoff_id, **row_counts}
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return _json_safe(payload)

    _run_audited(
        command_name="train experts",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "cutoff_id": cutoff_id,
            "fold_policy": fold_policy,
        },
        failure_exit_code=EXIT_TRAINING_ERROR,
        operation=operation,
    )


@train_app.command("combiner")
def train_combiner_command(
    start_date: str = typer.Option(..., "--start-date", help="First training date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last training date, inclusive."),
    cutoff_id: str | None = typer.Option(None, "--cutoff-id", help="Accepted for CLI symmetry; combiner registry is cutoff-agnostic."),
) -> None:
    """Register the static leakage-safe PMF combiner model version."""
    train_experts_command(
        start_date=start_date,
        end_date=end_date,
        cutoff_id=cutoff_id,
        fold_policy="annual_walk_forward",
    )


@predict_app.command("oof")
def predict_oof_command(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    cutoff_id: str | None = typer.Option(None, "--cutoff-id", help="Optional registered cutoff ID."),
) -> None:
    """Generate OOF expert and final Tmax PMFs for settled historical rows."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = predict_range(
                connection,
                start_date=start,
                end_date=end,
                cutoff_id=cutoff_id,
                prediction_kind="oof",
                require_labels=True,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="predict oof",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "cutoff_id": cutoff_id,
        },
        failure_exit_code=EXIT_PREDICTION_ERROR,
        operation=operation,
    )


@app.command("calibrate")
def calibrate_command(
    start_date: str = typer.Option(..., "--start-date", help="First settled prediction date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last settled prediction date, inclusive."),
    cutoff_id: str | None = typer.Option(None, "--cutoff-id", help="Optional registered cutoff ID."),
    prediction_kind: str = typer.Option("oof", "--prediction-kind", help="Prediction kind to calibrate."),
) -> None:
    """Calibrate final PMFs against settled Wunderground KLGA Tmax."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")
    kind = _parse_prediction_kind(prediction_kind)

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = calibrate_predictions(
                connection,
                start_date=start,
                end_date=end,
                cutoff_id=cutoff_id,
                prediction_kind=kind,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="calibrate",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "cutoff_id": cutoff_id,
            "prediction_kind": kind,
        },
        failure_exit_code=EXIT_TARGET_GRID_ERROR,
        operation=operation,
    )


@forecast_app.command("run")
def forecast_run_command(
    target_date: str = typer.Option(..., "--target-date", help="Target date to forecast."),
    cutoff_id: str = typer.Option(..., "--cutoff-id", help="Registered cutoff ID."),
) -> None:
    """Generate a current/replay forecast PMF for one target date and cutoff."""
    parsed_date = _parse_iso_date(target_date, "--target-date")

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = predict_range(
                connection,
                start_date=parsed_date,
                end_date=parsed_date,
                cutoff_id=cutoff_id,
                prediction_kind="forecast",
                require_labels=False,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="forecast run",
        command_args={"target_date": parsed_date.isoformat(), "cutoff_id": cutoff_id},
        failure_exit_code=EXIT_PREDICTION_ERROR,
        operation=operation,
    )


@evaluate_app.command("accuracy")
def evaluate_accuracy_command(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    cutoff_id: str | None = typer.Option(None, "--cutoff-id", help="Optional registered cutoff ID."),
    prediction_kind: str = typer.Option("oof", "--prediction-kind", help="Prediction kind to evaluate."),
) -> None:
    """Evaluate Tmax forecast accuracy against settled Wunderground KLGA labels."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")
    kind = _parse_prediction_kind(prediction_kind)

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = evaluate_accuracy(
                connection,
                start_date=start,
                end_date=end,
                cutoff_id=cutoff_id,
                prediction_kind=kind,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="evaluate accuracy",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "cutoff_id": cutoff_id,
            "prediction_kind": kind,
        },
        failure_exit_code=EXIT_EVALUATION_ERROR,
        operation=operation,
    )


@evaluate_app.command("day")
def evaluate_day_command(
    target_date: str = typer.Option(..., "--target-date", help="Settled target date to evaluate."),
    cutoff_id: str = typer.Option(..., "--cutoff-id", help="Registered cutoff ID."),
    prediction_kind: str = typer.Option("forecast", "--prediction-kind", help="Prediction kind to evaluate."),
) -> None:
    """Evaluate one target-date/cutoff forecast against settled Wunderground KLGA Tmax."""
    evaluate_accuracy_command(
        start_date=target_date,
        end_date=target_date,
        cutoff_id=cutoff_id,
        prediction_kind=prediction_kind,
    )


@report_app.command("generate")
def report_generate_command(
    run_id: str = typer.Option(..., "--run-id", help="reports.forecast_evaluation_runs.run_id_text."),
) -> None:
    """Generate local forecast accuracy report artifacts for an evaluation run."""

    def operation(engine) -> dict[str, Any]:
        settings = load_settings(require_db=True)
        with engine.begin() as connection:
            report = generate_forecast_report(
                connection,
                run_id_text=run_id,
                artifact_root=settings.artifact_root,
            )
        typer.echo(json.dumps({"ok": True, **report}, indent=2, sort_keys=True, default=str))
        return _json_safe(report)

    _run_audited(
        command_name="report generate",
        command_args={"run_id": run_id},
        failure_exit_code=EXIT_REPORT_ERROR,
        operation=operation,
    )


@settlement_app.command("update")
def settlement_update_command(
    start_date: str = typer.Option(..., "--start-date", help="First settled date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last settled date, inclusive."),
) -> None:
    """Normalize settled Wunderground actuals into canonical label tables."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            row_counts = normalize_acquisition(
                connection,
                start_date=start,
                end_date=end,
                cutoff_id=None,
            )
            target_rows = materialize_target_instances(
                connection,
                start_date=start,
                end_date=end,
                replace=False,
            )
            row_counts["gold.target_instances_inserted"] = target_rows
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="settlement update",
        command_args={"start_date": start.isoformat(), "end_date": end.isoformat()},
        failure_exit_code=EXIT_DATA_CONTRACT_ERROR,
        operation=operation,
        exception_exit_codes={AcquisitionContractError: EXIT_VALIDATION_ERROR},
    )


@wunderground_app.command("inspect-config")
def wunderground_inspect_config() -> None:
    """Print redacted Wunderground/Weather.com provider configuration."""
    settings = _wunderground_settings_or_exit(require_api_key=False)
    typer.echo(json.dumps(redacted_settings_payload(settings), indent=2, sort_keys=True))


@gribstream_app.command("inspect-config")
def gribstream_inspect_config() -> None:
    """Print redacted GribStream provider configuration and model action plan."""
    settings = _gribstream_settings_or_exit(require_api_token=False)
    payload = {
        "settings": redacted_gribstream_settings_payload(settings),
        "default_end_date": DEFAULT_END_DATE.isoformat(),
        "default_cutoff_id": DEFAULT_CUTOFF_ID,
        "supported_cutoff_ids": list(supported_cutoff_ids()),
        "models_by_cutoff": {
            cutoff_id: spec_summary_rows(end_date=DEFAULT_END_DATE, cutoff_id=cutoff_id)
            for cutoff_id in supported_cutoff_ids()
        },
        "tmax_thin_profile": {
            "default_job_id": TMAX_THIN_JOB_ID,
            "default_spacing_seconds": MIN_GRIBSTREAM_SPACING_SECONDS,
            "models": tmax_thin_spec_summary_rows(end_date=DEFAULT_END_DATE),
        },
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@gribstream_app.command("plan")
def gribstream_plan(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    models: str = typer.Option(..., "--models", help="Explicit comma-separated model IDs."),
    cutoff_id: str = typer.Option(
        ...,
        "--cutoff-id",
        help="GribStream cutoff profile to plan.",
    ),
    coordinate_tier: str = typer.Option(..., "--coordinate-tier", help="Coordinate tier to fetch."),
    job_id: str | None = typer.Option(None, "--job-id", help="Stable job ID; defaults to timestamped ID."),
    chunk_days: int = typer.Option(
        1,
        "--chunk-days",
        min=1,
        help="Days per chunk for no-asOf retrospective models. asOf-backed models are forced to one day.",
    ),
    max_chunks: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-chunks",
        min=1,
        help="Maximum planned target-day chunks; defaults to one.",
    ),
    max_models: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-models",
        min=1,
        help="Maximum explicitly selected models; defaults to one.",
    ),
    spacing_seconds: float = typer.Option(
        MIN_GRIBSTREAM_SPACING_SECONDS,
        "--spacing-seconds",
        min=MIN_GRIBSTREAM_SPACING_SECONDS,
        help="Minimum delay between all GribStream HTTP attempts.",
    ),
    persist: bool = typer.Option(
        False,
        "--persist/--dry-run",
        help="Persist a live catalog-resolved plan; default is a zero-call local dry-run.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider and database side effects.",
    ),
) -> None:
    """Resolve live catalog selectors and create the resumable GribStream action plan."""
    parsed_start, parsed_end = _parse_date_range(start_date, end_date)
    parsed_models = _parse_explicit_models(
        models,
        parser=parse_model_ids,
        max_models=max_models,
    )
    _validate_gribstream_chunk_scope(
        start_date=parsed_start,
        end_date=parsed_end,
        max_chunks=max_chunks,
    )
    parsed_cutoff_id = _parse_gribstream_cutoff_id(cutoff_id)
    effective_job_id = job_id or _job_id("gribstream_plan")
    if not persist:
        _print_provider_dry_run(
            operation="gribstream plan",
            scope={
                "start_date": parsed_start.isoformat(),
                "end_date": parsed_end.isoformat(),
                "models": list(parsed_models),
                "cutoff_id": parsed_cutoff_id,
                "coordinate_tier": coordinate_tier.upper(),
            },
            budgets={
                "max_chunks": max_chunks,
                "max_models": max_models,
                "spacing_seconds": spacing_seconds,
            },
        )
        return

    _require_execute(execute, "gribstream plan")
    settings = replace(
        _gribstream_settings_or_exit(require_api_token=False),
        spacing_seconds=spacing_seconds,
    )

    def operation(engine) -> dict[str, Any]:
        with _spaced_gribstream_requests(spacing_seconds):
            row_counts = prepare_gribstream_plan(
                engine=engine,
                settings=settings,
                job_id=effective_job_id,
                end_date=parsed_end,
                coordinate_tier=coordinate_tier,
                model_ids=parsed_models,
                start_date_override=parsed_start,
                chunk_days_override=chunk_days,
                cutoff_id=parsed_cutoff_id,
                persist=persist,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="gribstream plan",
        command_args={
            "end_date": parsed_end.isoformat(),
            "models": models,
            "cutoff_id": parsed_cutoff_id,
            "coordinate_tier": coordinate_tier.upper(),
            "start_date": parsed_start.isoformat(),
            "job_id": effective_job_id,
            "chunk_days": chunk_days,
            "max_chunks": max_chunks,
            "max_models": max_models,
            "spacing_seconds": spacing_seconds,
            "persist": persist,
            "execute": execute,
        },
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@gribstream_app.command("tmax-thin-plan")
def gribstream_tmax_thin_plan(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    models: str = typer.Option(..., "--models", help="Explicit comma-separated thin model IDs."),
    job_id: str = typer.Option(TMAX_THIN_JOB_ID, "--job-id", help="Stable thin backfill job ID."),
    chunk_days: int = typer.Option(
        1,
        "--chunk-days",
        min=1,
        help="Override days per HKG-style /runs chunk.",
    ),
    max_chunks: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-chunks",
        min=1,
        help="Maximum planned target-day chunks; defaults to one.",
    ),
    max_models: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-models",
        min=1,
        help="Maximum explicitly selected models; defaults to one.",
    ),
    spacing_seconds: float = typer.Option(
        MIN_GRIBSTREAM_SPACING_SECONDS,
        "--spacing-seconds",
        min=MIN_GRIBSTREAM_SPACING_SECONDS,
        help="Minimum delay between all GribStream HTTP attempts.",
    ),
    persist: bool = typer.Option(
        False,
        "--persist/--dry-run",
        help="Persist a live catalog-resolved plan; default is a zero-call local dry-run.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider and database side effects.",
    ),
) -> None:
    """Resolve live selectors and create the Tmax-thin /runs action plan."""
    parsed_start, parsed_end = _parse_date_range(start_date, end_date)
    parsed_models = _parse_explicit_models(
        models,
        parser=parse_tmax_thin_model_ids,
        max_models=max_models,
    )
    _validate_gribstream_chunk_scope(
        start_date=parsed_start,
        end_date=parsed_end,
        max_chunks=max_chunks,
    )
    if not persist:
        _print_provider_dry_run(
            operation="gribstream tmax-thin-plan",
            scope={
                "start_date": parsed_start.isoformat(),
                "end_date": parsed_end.isoformat(),
                "models": list(parsed_models),
            },
            budgets={
                "max_chunks": max_chunks,
                "max_models": max_models,
                "spacing_seconds": spacing_seconds,
            },
        )
        return

    _require_execute(execute, "gribstream tmax-thin-plan")
    settings = replace(
        _gribstream_settings_or_exit(require_api_token=False),
        spacing_seconds=spacing_seconds,
    )

    def operation(engine) -> dict[str, Any]:
        with _spaced_gribstream_requests(spacing_seconds):
            row_counts = prepare_tmax_thin_plan(
                engine=engine,
                settings=settings,
                job_id=job_id,
                end_date=parsed_end,
                model_ids=parsed_models,
                start_date_override=parsed_start,
                chunk_days_override=chunk_days,
                persist=persist,
            )
        typer.echo(json.dumps({"ok": True, **row_counts}, indent=2, sort_keys=True, default=str))
        return _json_safe(row_counts)

    _run_audited(
        command_name="gribstream tmax-thin-plan",
        command_args={
            "end_date": parsed_end.isoformat(),
            "models": models,
            "start_date": parsed_start.isoformat(),
            "job_id": job_id,
            "chunk_days": chunk_days,
            "max_chunks": max_chunks,
            "max_models": max_models,
            "spacing_seconds": spacing_seconds,
            "persist": persist,
            "execute": execute,
        },
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@gribstream_app.command("run")
def gribstream_run(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    models: str = typer.Option(..., "--models", help="Explicit comma-separated model IDs."),
    cutoff_id: str = typer.Option(
        ...,
        "--cutoff-id",
        help="GribStream cutoff profile to fetch.",
    ),
    coordinate_tier: str = typer.Option(..., "--coordinate-tier", help="Coordinate tier to fetch."),
    job_id: str | None = typer.Option(None, "--job-id", help="Stable job ID; defaults to timestamped ID."),
    chunk_days: int = typer.Option(
        1,
        "--chunk-days",
        min=1,
        help="Days per chunk for no-asOf retrospective models. asOf-backed models are forced to one day.",
    ),
    max_chunks: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-chunks",
        min=1,
        help="Hard chunk limit; defaults to one.",
    ),
    max_models: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-models",
        min=1,
        help="Maximum explicitly selected models; defaults to one.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip request_sha256 chunks already completed.",
    ),
    spacing_seconds: float = typer.Option(
        MIN_GRIBSTREAM_SPACING_SECONDS,
        "--spacing-seconds",
        min=MIN_GRIBSTREAM_SPACING_SECONDS,
        help="Minimum delay between authenticated GribStream calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider and database side effects.",
    ),
) -> None:
    """Fetch/persist GribStream chunks with one worker and 12s authenticated-call spacing."""
    parsed_start, parsed_end = _parse_date_range(start_date, end_date)
    parsed_models = _parse_explicit_models(
        models,
        parser=parse_model_ids,
        max_models=max_models,
    )
    _validate_gribstream_chunk_scope(
        start_date=parsed_start,
        end_date=parsed_end,
        max_chunks=max_chunks,
    )
    parsed_cutoff_id = _parse_gribstream_cutoff_id(cutoff_id)
    effective_job_id = job_id or _job_id("gribstream_backfill")
    _require_execute(execute, "gribstream run")
    settings = replace(
        _gribstream_settings_or_exit(require_api_token=True),
        spacing_seconds=spacing_seconds,
    )

    def operation(engine) -> dict[str, Any]:
        with _spaced_gribstream_requests(spacing_seconds):
            row_counts = run_gribstream_backfill(
                engine=engine,
                settings=settings,
                job_id=effective_job_id,
                end_date=parsed_end,
                coordinate_tier=coordinate_tier,
                model_ids=parsed_models,
                start_date_override=parsed_start,
                chunk_days_override=chunk_days,
                cutoff_id=parsed_cutoff_id,
                max_chunks=max_chunks,
                resume=resume,
            )
        typer.echo(json.dumps({"ok": row_counts.get("stopped_reason") is None, **row_counts}, indent=2, sort_keys=True, default=str))
        if row_counts.get("stopped_reason") is not None:
            raise RuntimeError(str(row_counts["stopped_reason"]))
        return _json_safe(row_counts)

    _run_audited(
        command_name="gribstream run",
        command_args={
            "end_date": parsed_end.isoformat(),
            "models": models,
            "cutoff_id": parsed_cutoff_id,
            "coordinate_tier": coordinate_tier.upper(),
            "start_date": parsed_start.isoformat(),
            "job_id": effective_job_id,
            "chunk_days": chunk_days,
            "max_chunks": max_chunks,
            "max_models": max_models,
            "resume": resume,
            "spacing_seconds": spacing_seconds,
            "execute": execute,
        },
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@gribstream_app.command("tmax-thin-run")
def gribstream_tmax_thin_run(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    models: str = typer.Option(..., "--models", help="Explicit comma-separated thin model IDs."),
    job_id: str = typer.Option(TMAX_THIN_JOB_ID, "--job-id", help="Stable thin backfill job ID."),
    chunk_days: int = typer.Option(
        1,
        "--chunk-days",
        min=1,
        help="Override days per HKG-style /runs chunk.",
    ),
    max_chunks: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-chunks",
        min=1,
        help="Hard chunk limit; defaults to one.",
    ),
    max_models: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-models",
        min=1,
        help="Maximum explicitly selected models; defaults to one.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip request_sha256 chunks already completed.",
    ),
    spacing_seconds: float = typer.Option(
        MIN_GRIBSTREAM_SPACING_SECONDS,
        "--spacing-seconds",
        min=MIN_GRIBSTREAM_SPACING_SECONDS,
        help="Minimum delay between authenticated GribStream calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider and database side effects.",
    ),
) -> None:
    """Fetch/persist the Tmax-thin T_1245UTC GribStream /runs backfill."""
    parsed_start, parsed_end = _parse_date_range(start_date, end_date)
    parsed_models = _parse_explicit_models(
        models,
        parser=parse_tmax_thin_model_ids,
        max_models=max_models,
    )
    _validate_gribstream_chunk_scope(
        start_date=parsed_start,
        end_date=parsed_end,
        max_chunks=max_chunks,
    )
    _require_execute(execute, "gribstream tmax-thin-run")
    settings = replace(
        _gribstream_settings_or_exit(require_api_token=True),
        spacing_seconds=spacing_seconds,
    )

    def operation(engine) -> dict[str, Any]:
        with _spaced_gribstream_requests(spacing_seconds):
            row_counts = run_tmax_thin_backfill(
                engine=engine,
                settings=settings,
                job_id=job_id,
                end_date=parsed_end,
                model_ids=parsed_models,
                start_date_override=parsed_start,
                chunk_days_override=chunk_days,
                max_chunks=max_chunks,
                resume=resume,
            )
        typer.echo(json.dumps({"ok": row_counts.get("stopped_reason") is None, **row_counts}, indent=2, sort_keys=True, default=str))
        if row_counts.get("stopped_reason") is not None:
            raise RuntimeError(str(row_counts["stopped_reason"]))
        return _json_safe(row_counts)

    _run_audited(
        command_name="gribstream tmax-thin-run",
        command_args={
            "end_date": parsed_end.isoformat(),
            "models": models,
            "start_date": parsed_start.isoformat(),
            "job_id": job_id,
            "chunk_days": chunk_days,
            "max_chunks": max_chunks,
            "max_models": max_models,
            "resume": resume,
            "spacing_seconds": spacing_seconds,
            "execute": execute,
        },
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@gribstream_app.command("tmax-thin-smoke")
def gribstream_tmax_thin_smoke(
    target_date: str = typer.Option(..., "--target-date", help="One target date to fetch."),
    models: str = typer.Option(..., "--models", help="Explicit comma-separated thin model IDs."),
    max_chunks: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-chunks",
        min=1,
        help="Hard chunk limit; defaults to one.",
    ),
    max_models: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-models",
        min=1,
        help="Maximum explicitly selected models; defaults to one.",
    ),
    spacing_seconds: float = typer.Option(
        MIN_GRIBSTREAM_SPACING_SECONDS,
        "--spacing-seconds",
        min=MIN_GRIBSTREAM_SPACING_SECONDS,
        help="Minimum delay between authenticated GribStream calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider and database side effects.",
    ),
) -> None:
    """Fetch one target date through the real Tmax-thin gold-only path."""
    parsed_date = _parse_iso_date(target_date, "--target-date")
    gribstream_tmax_thin_run(
        end_date=parsed_date.isoformat(),
        models=models,
        start_date=parsed_date.isoformat(),
        job_id=_job_id("gribstream_tmax_thin_smoke"),
        chunk_days=1,
        max_chunks=max_chunks,
        max_models=max_models,
        resume=False,
        spacing_seconds=spacing_seconds,
        execute=execute,
    )


@gribstream_app.command("smoke")
def gribstream_smoke(
    model: str = typer.Option(..., "--model", help="Single explicit model to smoke fetch."),
    target_date: str = typer.Option(..., "--target-date", help="One target date to fetch."),
    cutoff_id: str = typer.Option(
        ...,
        "--cutoff-id",
        help="GribStream cutoff profile to smoke fetch.",
    ),
    coordinate_tier: str = typer.Option(..., "--coordinate-tier", help="Coordinate tier to fetch."),
    spacing_seconds: float = typer.Option(
        MIN_GRIBSTREAM_SPACING_SECONDS,
        "--spacing-seconds",
        min=MIN_GRIBSTREAM_SPACING_SECONDS,
        help="Minimum delay between authenticated GribStream calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider and database side effects.",
    ),
) -> None:
    """Fetch exactly one GribStream chunk for a live smoke test."""
    parsed_date = _parse_iso_date(target_date, "--target-date")
    parsed_cutoff_id = _parse_gribstream_cutoff_id(cutoff_id)
    gribstream_run(
        end_date=parsed_date.isoformat(),
        models=model,
        cutoff_id=parsed_cutoff_id,
        coordinate_tier=coordinate_tier,
        start_date=parsed_date.isoformat(),
        job_id=_job_id("gribstream_smoke"),
        chunk_days=1,
        max_chunks=1,
        max_models=1,
        resume=False,
        spacing_seconds=spacing_seconds,
        execute=execute,
    )


@gribstream_app.command("status")
def gribstream_status(
    job_id: str | None = typer.Option(None, "--job-id", help="Optional job ID."),
) -> None:
    """Report GribStream job and per-model complete/remaining counts."""

    def operation(engine) -> dict[str, Any]:
        with engine.begin() as connection:
            jobs = job_status(connection, job_id)
            models_payload = model_status(connection, job_id) if job_id else []
        payload = {"ok": True, "jobs": jobs, "model_status": models_payload}
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return {"jobs": len(jobs), "model_status_rows": len(models_payload)}

    _run_audited(
        command_name="gribstream status",
        command_args={"job_id": job_id},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@polymarket_app.command("cutoff-analysis")
def polymarket_cutoff_analysis(
    start_date: str = typer.Option(..., "--start-date", help="First target date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last target date, inclusive."),
    artifact_root: Path | None = typer.Option(
        None,
        "--artifact-root",
        help=(
            "Directory where raw, processed, report, and manifest artifacts are written; "
            "defaults under KLGA_ARTIFACT_ROOT."
        ),
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh/--use-cache",
        help="Ignore cached raw API payloads and refetch public Polymarket data.",
    ),
    sleep_seconds: float = typer.Option(
        0.20,
        "--sleep-seconds",
        min=0.0,
        help="Delay between uncached public API calls.",
    ),
    max_requests: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-requests",
        min=1,
        max=MAX_PROVIDER_REQUESTS,
        help="Hard limit on outbound HTTP attempts; defaults to one.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge public provider calls and artifact writes.",
    ),
) -> None:
    """Download NYC Tmax Polymarket histories and optimize forecast cutoff timing."""
    parsed_start, parsed_end = _parse_date_range(start_date, end_date)
    _require_execute(execute, "polymarket cutoff-analysis")
    resolved_artifact_root = artifact_root
    if resolved_artifact_root is None:
        resolved_artifact_root = (
            _offline_settings_or_exit().artifact_root / "polymarket_cutoff_analysis"
        )
    with _bounded_http_requests(max_requests):
        summary = run_cutoff_analysis(
            start_date=parsed_start,
            end_date=parsed_end,
            artifact_root=resolved_artifact_root,
            use_cache=not refresh,
            sleep_seconds=sleep_seconds,
        )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True, default=str))


def _run_java_wu_command(
    *,
    command_name: str,
    java_command: str,
    java_args: dict[str, object],
    require_api_key: bool,
    execute: bool,
) -> None:
    _require_execute(execute, command_name)
    wu_settings = _wunderground_settings_or_exit(require_api_key=require_api_key)

    def operation(engine) -> dict[str, Any]:
        settings = _settings_or_exit()
        result = run_java_wu_truth(
            settings=settings,
            wu_settings=wu_settings,
            command=java_command,
            args=java_args,
        )
        typer.echo(json.dumps(result.payload, indent=2, sort_keys=True, default=str))
        if not result.payload.get("ok", False):
            raise RuntimeError(f"Java WU truth command reported failure: {result.payload}")
        return _json_safe(
            {
                key: value
                for key, value in result.payload.items()
                if isinstance(value, (int, float, bool))
            }
        )

    _run_audited(
        command_name=command_name,
        command_args={key: value for key, value in java_args.items() if key not in {"api_key"}},
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


@wu_app.command("rebuild")
def wu_rebuild(
    start_date: str = typer.Option(..., "--start-date", help="First local date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last local date, inclusive."),
    stations: str = typer.Option(..., "--stations", help="Explicit comma-separated station IDs."),
    workers: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--workers",
        min=1,
        max=MAX_WUNDERGROUND_WORKERS,
        help="Java fetch worker count; defaults to one and is capped at two.",
    ),
    chunk_days: int = typer.Option(
        MAX_WUNDERGROUND_DATE_SPAN_DAYS,
        "--chunk-days",
        min=1,
        max=MAX_WUNDERGROUND_DATE_SPAN_DAYS,
        help="Days per Weather.com request, capped at 31.",
    ),
    rate_limit_per_minute: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--rate-limit-per-minute",
        min=1,
        help="Shared Weather.com request limit across all Java workers.",
    ),
    max_requests: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-requests",
        min=1,
        max=MAX_PROVIDER_REQUESTS,
        help="Hard request-window budget; defaults to one.",
    ),
    allow_long_date_range: bool = typer.Option(
        False,
        "--allow-long-date-range",
        help="Allow more than 31 days only when --max-requests still bounds the request windows.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Skip completed days; disabled by default because fragmented gaps expand request windows.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider, subprocess, and database side effects.",
    ),
) -> None:
    """Fetch Weather.com hourly observations into public.wunderground_daily_tmax."""
    start, end = _parse_date_range(start_date, end_date)
    parsed_stations = _parse_explicit_stations(stations)
    _validate_wunderground_scope(
        start_date=start,
        end_date=end,
        stations=parsed_stations,
        chunk_days=chunk_days,
        max_requests=max_requests,
        workers=workers,
        allow_long_date_range=allow_long_date_range,
        resume=resume,
    )
    _run_java_wu_command(
        command_name="wu rebuild",
        java_command="rebuild",
        java_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "stations": ",".join(parsed_stations),
            "workers": workers,
            "chunk_days": chunk_days,
            "rate_limit_per_minute": rate_limit_per_minute,
            "max_retries": 0,
            "resume": resume,
        },
        require_api_key=True,
        execute=execute,
    )


@wu_app.command("audit-day")
def wu_audit_day(
    station: str = typer.Option(..., "--station", help="Explicit canonical station ID."),
    local_date: str = typer.Option(..., "--date", help="New York local date to fetch/audit."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider, subprocess, and database side effects.",
    ),
) -> None:
    """Fetch one station-day and print the saved public truth-table row."""
    parsed_station = _parse_single_station(station, "--station")
    parsed_date = _parse_iso_date(local_date, "--date")
    _run_java_wu_command(
        command_name="wu audit-day",
        java_command="audit-day",
        java_args={
            "station": parsed_station,
            "date": parsed_date.isoformat(),
            "workers": 1,
            "max_retries": 0,
        },
        require_api_key=True,
        execute=execute,
    )


@wu_app.command("validate-sample")
def wu_validate_sample(
    sample_size: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--sample-size",
        min=1,
        help="Deterministic DB row sample size; defaults to one.",
    ),
    seed: int = typer.Option(1729, "--seed", help="Deterministic sample seed."),
    max_requests: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-requests",
        min=1,
        max=MAX_PROVIDER_REQUESTS,
        help="Maximum sampled page checks; defaults to one.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge subprocess, database, and possible page-fetch side effects.",
    ),
) -> None:
    """Validate sampled truth-table rows against their stored hourly observations and WU page URL."""
    if sample_size > max_requests:
        raise typer.BadParameter(
            f"--sample-size is {sample_size} but --max-requests is {max_requests}",
            param_hint="--max-requests",
        )
    _run_java_wu_command(
        command_name="wu validate-sample",
        java_command="validate-sample",
        java_args={"sample_size": sample_size, "seed": seed, "max_retries": 0},
        require_api_key=False,
        execute=execute,
    )


@wunderground_app.command("smoke")
def wunderground_smoke(
    station_id: str = typer.Option(..., "--station-id", help="Explicit canonical station ID."),
    local_date: str = typer.Option(..., "--local-date", help="New York local date to fetch."),
    persist: bool = typer.Option(
        False,
        "--persist/--dry-run",
        help="Persist through the Java runner; default dry-run performs zero provider calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider, subprocess, and database side effects.",
    ),
) -> None:
    """Fetch one station-day as a live provider smoke test."""
    parsed_station = _parse_single_station(station_id, "--station-id")
    parsed_date = _parse_iso_date(local_date, "--local-date")
    if not persist:
        _print_provider_dry_run(
            operation="wunderground smoke",
            scope={
                "station_id": parsed_station,
                "local_date": parsed_date.isoformat(),
            },
            budgets={"max_requests": 1, "workers": 1},
        )
        return

    _run_java_wu_command(
        command_name="wunderground smoke",
        java_command="audit-day",
        java_args={
            "station": parsed_station,
            "date": parsed_date.isoformat(),
            "workers": 1,
            "max_retries": 0,
        },
        require_api_key=True,
        execute=execute,
    )


@wunderground_app.command("fetch-day")
def wunderground_fetch_day(
    station_id: str = typer.Option(..., "--station-id", help="Explicit canonical station ID."),
    local_date: str = typer.Option(..., "--local-date", help="New York local date to fetch."),
    persist: bool = typer.Option(
        False,
        "--persist/--dry-run",
        help="Persist rows to Postgres; default dry-run performs zero provider calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider, subprocess, and database side effects.",
    ),
) -> None:
    """Fetch one station-day, optionally persisting to public.wunderground_daily_tmax."""
    parsed_date = _parse_iso_date(local_date, "--local-date")
    wunderground_smoke(
        station_id=station_id,
        local_date=parsed_date.isoformat(),
        persist=persist,
        execute=execute,
    )


@wunderground_app.command("backfill")
def wunderground_backfill(
    start_date: str = typer.Option(..., "--start-date", help="First local date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last local date, inclusive."),
    stations: str = typer.Option(..., "--stations", help="Explicit comma-separated station IDs."),
    chunk_days: int = typer.Option(
        MAX_WUNDERGROUND_DATE_SPAN_DAYS,
        "--chunk-days",
        min=1,
        max=MAX_WUNDERGROUND_DATE_SPAN_DAYS,
        help="Days per provider request window, capped at 31.",
    ),
    workers: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--workers",
        min=1,
        max=MAX_WUNDERGROUND_WORKERS,
        help="Worker count; defaults to one and is capped at two.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Skip completed days; disabled by default because fragmented gaps expand request windows.",
    ),
    rate_limit_per_minute: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--rate-limit-per-minute",
        min=1,
        help="Shared Weather.com request limit across all Java workers.",
    ),
    max_requests: int = typer.Option(
        DEFAULT_LIVE_BUDGET,
        "--max-requests",
        min=1,
        max=MAX_PROVIDER_REQUESTS,
        help="Hard request-window budget; defaults to one.",
    ),
    allow_long_date_range: bool = typer.Option(
        False,
        "--allow-long-date-range",
        help="Allow more than 31 days only when --max-requests still bounds the request windows.",
    ),
    persist: bool = typer.Option(
        False,
        "--persist/--dry-run",
        help="Persist through the Java runner; default dry-run performs zero provider calls.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Acknowledge provider, subprocess, and database side effects.",
    ),
) -> None:
    """Backfill Wunderground actuals with resumable coverage tracking."""
    start, end = _parse_date_range(start_date, end_date)
    parsed_stations = _parse_explicit_stations(stations)
    planned_requests = _validate_wunderground_scope(
        start_date=start,
        end_date=end,
        stations=parsed_stations,
        chunk_days=chunk_days,
        max_requests=max_requests,
        workers=workers,
        allow_long_date_range=allow_long_date_range,
        resume=resume,
    )
    if not persist:
        _print_provider_dry_run(
            operation="wunderground backfill",
            scope={
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "stations": list(parsed_stations),
                "resume": resume,
            },
            budgets={
                "chunk_days": chunk_days,
                "max_requests": max_requests,
                "planned_requests": planned_requests,
                "workers": workers,
                "rate_limit_per_minute": rate_limit_per_minute,
            },
        )
        return

    job_id = _job_id("wu_backfill")

    _run_java_wu_command(
        command_name="wunderground backfill",
        java_command="rebuild",
        java_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "stations": ",".join(parsed_stations),
            "workers": workers,
            "chunk_days": chunk_days,
            "rate_limit_per_minute": rate_limit_per_minute,
            "max_retries": 0,
            "resume": resume,
            "job_id": job_id,
        },
        require_api_key=True,
        execute=execute,
    )


@wunderground_app.command("coverage")
def wunderground_coverage(
    start_date: str = typer.Option(..., "--start-date", help="First local date, inclusive."),
    end_date: str = typer.Option(..., "--end-date", help="Last local date, inclusive."),
    station_id: str | None = typer.Option(None, "--station-id", help="Optional canonical station ID."),
) -> None:
    """Summarize station-date Wunderground coverage states."""
    start = _parse_iso_date(start_date, "--start-date")
    end = _parse_iso_date(end_date, "--end-date")

    def operation(engine) -> dict[str, Any]:
        query = text(
            """
            SELECT
                station_id,
                validation_status AS status,
                count(*)::integer AS row_count,
                min(local_date)::text AS min_date,
                max(local_date)::text AS max_date,
                count(*) FILTER (WHERE tmax_f IS NOT NULL)::integer AS rows_with_tmax
            FROM public.wunderground_daily_tmax
            WHERE local_date BETWEEN :start_date AND :end_date
              AND (CAST(:station_id AS text) IS NULL OR station_id = CAST(:station_id AS text))
            GROUP BY station_id, validation_status
            ORDER BY station_id, validation_status
            """
        )
        with engine.begin() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    query,
                    {
                        "start_date": start,
                        "end_date": end,
                        "station_id": station_id.upper() if station_id else None,
                    },
                ).mappings()
            ]
        typer.echo(json.dumps({"ok": True, "rows": rows}, indent=2, sort_keys=True))
        return {"coverage_summary_rows": len(rows)}

    _run_audited(
        command_name="wunderground coverage",
        command_args={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "station_id": station_id.upper() if station_id else None,
        },
        failure_exit_code=EXIT_VALIDATION_ERROR,
        operation=operation,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
