"""Phase-0 source contract inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.audit.schema_contracts import (
    DiscoveredTable,
    SourceCheck,
    TableRef,
    count_rows,
    discover_table,
    table_columns,
    table_exists,
)
from hkg_t24.constants import ARWF_WARNING, CWA_WRF_WARNING, LIGHTGBM_ERROR


@dataclass(frozen=True)
class SourceContractResult:
    target_labels: DiscoveredTable | None
    official_forecasts: DiscoveredTable | None
    checks: tuple[SourceCheck, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def verify_lightgbm_required() -> None:
    try:
        import lightgbm  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(LIGHTGBM_ERROR) from exc


def _has_required_columns(
    connection: Any,
    table_ref: TableRef,
    required: tuple[str, ...],
) -> SourceCheck:
    if not table_exists(connection, table_ref):
        return SourceCheck(table_ref.qualified, "FAIL", "Required table is absent.")
    columns = table_columns(connection, table_ref)
    missing = sorted(set(required) - columns)
    if missing:
        return SourceCheck(
            table_ref.qualified,
            "FAIL",
            "Missing required columns: " + ", ".join(missing),
            count_rows(connection, table_ref),
        )
    return SourceCheck(table_ref.qualified, "PASS", "Required columns present.", count_rows(connection, table_ref))


def _scalar_int(connection: Any, sql: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute(sql)
        row = cursor.fetchone()
    return 0 if row is None else int(row[0])


def run_source_contract_checks(connection: Any) -> SourceContractResult:
    checks: list[SourceCheck] = []
    warnings: list[str] = []
    failures: list[str] = []

    target, target_checks = discover_table(
        connection,
        logical_name="hko_target_labels",
        primary=TableRef("public", "hko_daily_tmax_target_labels"),
        fallbacks=(
            TableRef("label_core", "hko_daily_tmax"),
            TableRef("feature_safe", "hko_target_history_pre2024"),
        ),
        date_columns=("target_date_hkt", "local_date", "target_date"),
        value_columns=("target_tmax_c", "tmax_c", "value"),
        ordered_fallbacks=True,
    )
    checks.extend(target_checks)
    if target is None:
        failures.append("Target-label source discovery failed.")
    elif target.date_column is None or target.value_column is None:
        failures.append(f"Target-label date/value column mapping is ambiguous for {target.table_ref.qualified}.")

    official, official_checks = discover_table(
        connection,
        logical_name="hko_official_forecasts",
        primary=TableRef("public", "hko_historical_forecasts_2000_2026"),
        fallbacks=(),
        date_columns=("target_date",),
        value_columns=("forecast_max_c",),
    )
    checks.extend(official_checks)
    if official is None:
        failures.append("Official forecast archive is absent.")
    else:
        official_required = (
            "target_date",
            "issue_at_utc",
            "issue_at_hkt",
            "product_type",
            "row_quality_status",
            "forecast_min_c",
            "forecast_max_c",
        )
        official_required_check = _has_required_columns(connection, official.table_ref, official_required)
        checks.append(official_required_check)
        if official_required_check.status == "FAIL":
            failures.append("Official forecast archive column contract failed.")
        usable_rows = _scalar_int(
            connection,
            """
            SELECT count(*)
            FROM public.hko_historical_forecasts_2000_2026
            WHERE row_quality_status = 'usable_local_minmax'
              AND target_date IS NOT NULL
              AND issue_at_utc IS NOT NULL
            """,
        )
        checks.append(
            SourceCheck(
                official.table_ref.qualified,
                "PASS" if usable_rows >= 100_000 else "FAIL",
                f"Usable official local min/max rows = {usable_rows}.",
                usable_rows,
            )
        )
        if usable_rows < 100_000:
            failures.append("Official clean subset has fewer than 100000 usable local rows.")

    forecast_wide = TableRef("nwp_tactical", "forecast_wide")
    raw_object = TableRef("nwp_tactical", "raw_response_object")
    forecast_check = _has_required_columns(
        connection,
        forecast_wide,
        (
            "target_date_hkt",
            "cutoff_id",
            "dataset_code",
            "run_time_utc",
            "valid_time_utc",
            "source_response_object_id",
        ),
    )
    raw_check = _has_required_columns(
        connection,
        raw_object,
        ("response_object_id", "object_uri", "row_count", "byte_size"),
    )
    checks.extend([forecast_check, raw_check])
    if forecast_check.status == "FAIL" or raw_check.status == "FAIL":
        failures.append("NWP tactical table contract failed.")
    if table_exists(connection, raw_object):
        raw_columns = table_columns(connection, raw_object)
        if "response_sha256" not in raw_columns and "sha256" not in raw_columns:
            warnings.append("raw_response_object has no response_sha256/sha256 column; hash proof is warning-only.")

    if table_exists(connection, forecast_wide) and table_exists(connection, raw_object):
        full_scope_rows = _scalar_int(
            connection,
            """
            SELECT count(*)
            FROM nwp_tactical.forecast_wide fw
            JOIN nwp_tactical.raw_response_object r
              ON r.response_object_id = fw.source_response_object_id
            WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
            """,
        )
        checks.append(
            SourceCheck(
                "nwp_tactical.full_tactical_backfill_ok_tmax",
                "PASS" if full_scope_rows >= 1_900_000 else "FAIL",
                f"Full tactical scoped rows = {full_scope_rows}.",
                full_scope_rows,
            )
        )
        if full_scope_rows < 1_900_000:
            failures.append("GribStream full tactical filter has fewer than 1900000 rows.")
        cwawrf15_rows = _scalar_int(
            connection,
            """
            SELECT count(*)
            FROM nwp_tactical.forecast_wide fw
            JOIN nwp_tactical.raw_response_object r
              ON r.response_object_id = fw.source_response_object_id
            WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
              AND fw.dataset_code = 'cwawrf15'
            """,
        )
        if cwawrf15_rows < 100:
            warnings.append(CWA_WRF_WARNING)

    arwf_primary = TableRef("public", "hko_arwf_station_daily_forecasts")
    if not table_exists(connection, arwf_primary):
        warnings.append(ARWF_WARNING)

    return SourceContractResult(
        target_labels=target,
        official_forecasts=official,
        checks=tuple(checks),
        warnings=tuple(dict.fromkeys(warnings)),
        failures=tuple(dict.fromkeys(failures)),
    )


def write_source_contract_reports(writer: ReportWriter, result: SourceContractResult) -> None:
    inventory_rows = [
        (
            check.object_name,
            check.status,
            check.row_count,
            check.message,
        )
        for check in result.checks
    ]
    writer.write_root_report(
        "source_inventory_report.md",
        "HKG-T24-001 Source Inventory Report",
        (
            ("Status", "PASS" if result.passed else "FAIL"),
            (
                "Checks",
                "\n".join(
                    f"- `{check.object_name}`: {check.status}; {check.message}"
                    for check in result.checks
                ),
            ),
            ("Warnings", "\n".join(f"- {warning}" for warning in result.warnings) or "- None."),
            ("Failures", "\n".join(f"- {failure}" for failure in result.failures) or "- None."),
        ),
    )
    writer.write_root_report(
        "schema_contract_report.md",
        "HKG-T24-001 Schema Contract Report",
        (
            ("Status", "PASS" if result.passed else "FAIL"),
            (
                "Discovered Target Labels",
                "Not discovered."
                if result.target_labels is None
                else (
                    f"`{result.target_labels.table_ref.qualified}` with date column "
                    f"`{result.target_labels.date_column}` and value column "
                    f"`{result.target_labels.value_column}`."
                ),
            ),
            (
                "Discovered Official Forecasts",
                "Not discovered."
                if result.official_forecasts is None
                else f"`{result.official_forecasts.table_ref.qualified}`.",
            ),
            (
                "Detailed Checks",
                "\n".join(
                    f"- `{name}`: {status}; rows={rows}; {message}"
                    for name, status, rows, message in inventory_rows
                ),
            ),
        ),
    )
