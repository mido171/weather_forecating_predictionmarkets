"""GribStream H24N safe-row ledger and reports."""

from __future__ import annotations

from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.audit.leakage_events import leakage_error_count
from hkg_t24.audit.schema_contracts import TableRef, table_exists


def gribstream_tables_available(connection: Any) -> bool:
    return table_exists(connection, TableRef("nwp_tactical", "forecast_wide")) and table_exists(
        connection, TableRef("nwp_tactical", "raw_response_object")
    )


def refresh_nwp_safe_row_ledger(connection: Any, writer: ReportWriter) -> None:
    if not gribstream_tables_available(connection):
        writer.write_root_report(
            "gribstream_source_scope_audit.md",
            "HKG-T24-001 GribStream Source Scope Audit",
            (("Status", "FAIL"), ("Reason", "Required nwp_tactical tables are absent.")),
        )
        return
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM model_features.nwp_safe_row_ledger
            WHERE source_scope = 'full_tactical_backfill_ok_tmax'
            """
        )
        cursor.execute(
            """
            INSERT INTO model_features.nwp_safe_row_ledger (
              target_date_hkt, cutoff_id, dataset_code, run_time_utc, valid_time_utc,
              source_response_object_id, object_uri, row_is_safe_h24n, exclusion_reason,
              source_scope, publication_buffer_hours
            )
            SELECT
              fw.target_date_hkt::date,
              fw.cutoff_id,
              fw.dataset_code,
              fw.run_time_utc,
              fw.valid_time_utc,
              fw.source_response_object_id,
              r.object_uri,
              (
                fw.cutoff_id = 'H24N'
                AND fw.dataset_code NOT IN ('nbmoc','aigfspres','aigefssfc')
                AND fw.run_time_utc + interval '6 hours'
                    <= ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
              ) AS row_is_safe_h24n,
              CASE
                WHEN fw.cutoff_id <> 'H24N' THEN 'NON_H24N_CUTOFF'
                WHEN fw.dataset_code IN ('nbmoc','aigfspres','aigefssfc') THEN 'BLOCKED_SOURCE'
                WHEN fw.run_time_utc + interval '6 hours'
                    > ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
                  THEN 'AFTER_H24N_CUTOFF_WITH_BUFFER'
                ELSE NULL
              END AS exclusion_reason,
              'full_tactical_backfill_ok_tmax' AS source_scope,
              6 AS publication_buffer_hours
            FROM nwp_tactical.forecast_wide fw
            JOIN nwp_tactical.raw_response_object r
              ON r.response_object_id = fw.source_response_object_id
            WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
            """
        )
    connection.commit()
    write_gribstream_reports(writer, connection)


def write_gribstream_reports(writer: ReportWriter, connection: Any) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
              dataset_code,
              count(*) AS scoped_rows,
              sum(row_is_safe_h24n::int) AS safe_rows,
              count(*) - sum(row_is_safe_h24n::int) AS excluded_rows,
              count(DISTINCT target_date_hkt) AS target_days
            FROM model_features.nwp_safe_row_ledger
            WHERE source_scope = 'full_tactical_backfill_ok_tmax'
            GROUP BY dataset_code
            ORDER BY dataset_code
            """
        )
        rows = cursor.fetchall()
    writer.write_csv(
        "gribstream_source_scope_audit.csv",
        ("dataset_code", "scoped_rows", "safe_rows", "excluded_rows", "target_days"),
        rows,
    )
    writer.write_root_report(
        "gribstream_source_scope_audit.md",
        "HKG-T24-001 GribStream Source Scope Audit",
        (
            ("Status", "PASS"),
            (
                "Required Filter",
                "`forecast_wide` rows are joined to `raw_response_object`, filtered to "
                "`full_tactical_backfill_ok_tmax`, constrained to `H24N`, guarded by "
                "`run_time_utc + interval '6 hours' <= formal_cutoff_utc`, and daily Tmax "
                "blocked datasets are excluded from safe rows.",
            ),
            (
                "Dataset Counts",
                "\n".join(
                    f"- `{row[0]}`: scoped={row[1]}, safe={row[2]}, excluded={row[3]}, days={row[4]}"
                    for row in rows
                )
                or "- No scoped rows found.",
            ),
            (
                "Publication Buffer",
                "The 6-hour buffer is a conservative project guardrail, not a confirmed "
                "GribStream provider availability SLA.",
            ),
        ),
    )
    errors = leakage_error_count(connection)
    writer.write_root_report(
        "leakage_audit_report.md",
        "HKG-T24-001 Leakage Audit Report",
        (
            ("Status", "PASS" if errors == 0 else "FAIL"),
            ("Error Events", str(errors)),
            (
                "Scope",
                "Jira 001 verifies the data-contract foundation and safe-row filters; it does not "
                "train models or promote proxy/shadow sources into strict features.",
            ),
        ),
    )
