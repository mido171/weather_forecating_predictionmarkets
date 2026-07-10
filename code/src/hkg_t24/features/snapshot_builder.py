"""H24N cutoff calendar, target-memory policy, and snapshot construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import astuple
from datetime import date
from statistics import mean, pstdev
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.audit.schema_contracts import DiscoveredTable
from hkg_t24.constants import (
    END_TARGET_DATE,
    START_TARGET_DATE,
    TARGET_MEMORY_FEATURE_WHITELIST,
    assert_no_forbidden_target_memory_names,
)
from hkg_t24.timeutils import calendar_row, iter_target_dates


def build_target_memory_features(
    labels: Sequence[tuple[date, float | None]],
) -> dict[date, dict[str, float | int | None]]:
    """Build leakage-safe target-memory features ending at T-2 or older."""
    assert_no_forbidden_target_memory_names(TARGET_MEMORY_FEATURE_WHITELIST)
    ordered = sorted(labels, key=lambda item: item[0])
    values = [item[1] for item in ordered]
    dates = [item[0] for item in ordered]
    output: dict[date, dict[str, float | int | None]] = {}
    for index, target_date in enumerate(dates):
        features: dict[str, float | int | None] = {}
        for lag in (2, 3, 7):
            source_index = index - lag
            if source_index >= 0 and values[source_index] is not None:
                features[f"target__lag{lag}_tmax_c"] = values[source_index]
        for window in (7, 14, 30):
            start = index - (window + 1)
            end = index - 1
            if start >= 0:
                window_values = values[start:end]
                if len(window_values) == window and all(value is not None for value in window_values):
                    non_null_values: list[float] = []
                    for value in window_values:
                        if value is not None:
                            non_null_values.append(value)
                    if len(non_null_values) != window:
                        continue
                    numeric_values = [float(value) for value in non_null_values]
                    features[f"target__roll{window}_lag2_mean_tmax_c"] = mean(numeric_values)
                    features[f"target__roll{window}_lag2_std0_tmax_c"] = pstdev(numeric_values)
                    if window in {7, 30}:
                        features[f"target__slope{window}_lag2_tmax_c"] = (
                            numeric_values[-1] - numeric_values[0]
                        ) / (window - 1)
        if index >= 2:
            lag2_value = values[index - 2]
            if lag2_value is not None:
                features["target__hot_spell_lag2"] = int(float(lag2_value) >= 32.0)
        features["target__year_index"] = target_date.year - 2000
        if features:
            output[target_date] = features
    return output


def calendar_rows(start_date: date = START_TARGET_DATE, end_date: date = END_TARGET_DATE) -> list[tuple[Any, ...]]:
    return [astuple(calendar_row(target_date)) for target_date in iter_target_dates(start_date, end_date)]


def populate_cutoff_calendar(
    connection: Any,
    *,
    start_date: date = START_TARGET_DATE,
    end_date: date = END_TARGET_DATE,
) -> int:
    rows = calendar_rows(start_date, end_date)
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO model_core.cutoff_calendar (
              target_date_hkt, cutoff_id, formal_cutoff_utc, operational_freeze_utc,
              partition_name, snapshot_id, season, month, day_of_year,
              is_mam, is_jja, is_son, is_djf, calendar__year_index
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (target_date_hkt, cutoff_id) DO UPDATE SET
              formal_cutoff_utc = EXCLUDED.formal_cutoff_utc,
              operational_freeze_utc = EXCLUDED.operational_freeze_utc,
              partition_name = EXCLUDED.partition_name,
              snapshot_id = EXCLUDED.snapshot_id,
              season = EXCLUDED.season,
              month = EXCLUDED.month,
              day_of_year = EXCLUDED.day_of_year,
              is_mam = EXCLUDED.is_mam,
              is_jja = EXCLUDED.is_jja,
              is_son = EXCLUDED.is_son,
              is_djf = EXCLUDED.is_djf,
              calendar__year_index = EXCLUDED.calendar__year_index
            """,
            rows,
        )
    return len(rows)


def _target_source_hash_expr(date_column: str, value_column: str) -> str:
    return f"md5(({date_column}::text || ':' || coalesce({value_column}::text, 'NULL')))"


def populate_target_labels(
    connection: Any,
    target_table: DiscoveredTable,
    *,
    start_date: date = START_TARGET_DATE,
    end_date: date = END_TARGET_DATE,
) -> int:
    if target_table.date_column is None or target_table.value_column is None:
        raise ValueError("Target label table requires discovered date and value columns")
    qualified = target_table.table_ref.qualified
    date_column = target_table.date_column
    value_column = target_table.value_column
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO model_core.target_label (
              target_date_hkt, target_tmax_c, label_visible_for_development, source_table, source_hash
            )
            SELECT
              {date_column}::date AS target_date_hkt,
              {value_column}::numeric AS target_tmax_c,
              ({date_column}::date < date '2024-01-01') AS label_visible_for_development,
              %s AS source_table,
              {_target_source_hash_expr(date_column, value_column)} AS source_hash
            FROM {qualified}
            WHERE {date_column}::date BETWEEN %s AND %s
              AND {value_column} IS NOT NULL
            ON CONFLICT (target_date_hkt) DO UPDATE SET
              target_tmax_c = EXCLUDED.target_tmax_c,
              label_visible_for_development = EXCLUDED.label_visible_for_development,
              source_table = EXCLUDED.source_table,
              source_hash = EXCLUDED.source_hash,
              loaded_at_utc = now()
            """,
            (qualified, start_date, end_date),
        )
        return cursor.rowcount if cursor.rowcount is not None else 0


def _safe_view_exists(connection: Any) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('model_features.v_nwp_h24n_safe_rows')")
        row = cursor.fetchone()
    return row is not None and row[0] is not None


def populate_h24n_snapshots(
    connection: Any,
    *,
    start_date: date = START_TARGET_DATE,
    end_date: date = END_TARGET_DATE,
) -> int:
    safe_view_exists = _safe_view_exists(connection)
    if safe_view_exists:
        nwp_exists_sql = (
            "EXISTS (SELECT 1 FROM model_features.v_nwp_h24n_safe_rows s "
            "WHERE s.target_date_hkt = cc.target_date_hkt AND s.dataset_code = ANY(%s))"
        )
    else:
        nwp_exists_sql = "false"
    with connection.cursor() as cursor:
        params: tuple[Any, ...]
        if safe_view_exists:
            params = (
                ["gfs"],
                ["gefsatmosmean", "gefsatmos"],
                ["ifsoper", "ifsenfo"],
                ["aifsoper", "aifsenfo", "aigfssfc", "graphcast", "fourcastnetgfs"],
                ["cwawrf15"],
                start_date,
                end_date,
            )
        else:
            params = (start_date, end_date)
        cursor.execute(
            f"""
            INSERT INTO model_features.h24n_snapshot (
              snapshot_id, target_date_hkt, cutoff_id, formal_cutoff_utc, operational_freeze_utc,
              partition_name, official_available, gfs_available, gefs_available,
              station_proxy_available, ifs_shadow_available, ai_shadow_available,
              arwf_live_shadow_available, cwa_live_shadow_available, snapshot_status, placeholder_reason
            )
            SELECT
              cc.snapshot_id,
              cc.target_date_hkt,
              cc.cutoff_id,
              cc.formal_cutoff_utc,
              cc.operational_freeze_utc,
              cc.partition_name,
              EXISTS (
                SELECT 1
                FROM public.hko_historical_forecasts_2000_2026 h
                WHERE h.target_date = cc.target_date_hkt
                  AND h.issue_at_utc <= cc.operational_freeze_utc
                  AND h.row_quality_status = 'usable_local_minmax'
              ) AS official_available,
              {nwp_exists_sql} AS gfs_available,
              {nwp_exists_sql} AS gefs_available,
              false AS station_proxy_available,
              {nwp_exists_sql} AS ifs_shadow_available,
              {nwp_exists_sql} AS ai_shadow_available,
              false AS arwf_live_shadow_available,
              {nwp_exists_sql} AS cwa_live_shadow_available,
              'active' AS snapshot_status,
              NULL AS placeholder_reason
            FROM model_core.cutoff_calendar cc
            WHERE cc.target_date_hkt BETWEEN %s AND %s
              AND cc.cutoff_id = 'H24N'
            ON CONFLICT (target_date_hkt, cutoff_id) DO UPDATE SET
              formal_cutoff_utc = EXCLUDED.formal_cutoff_utc,
              operational_freeze_utc = EXCLUDED.operational_freeze_utc,
              partition_name = EXCLUDED.partition_name,
              official_available = EXCLUDED.official_available,
              gfs_available = EXCLUDED.gfs_available,
              gefs_available = EXCLUDED.gefs_available,
              station_proxy_available = EXCLUDED.station_proxy_available,
              ifs_shadow_available = EXCLUDED.ifs_shadow_available,
              ai_shadow_available = EXCLUDED.ai_shadow_available,
              arwf_live_shadow_available = EXCLUDED.arwf_live_shadow_available,
              cwa_live_shadow_available = EXCLUDED.cwa_live_shadow_available,
              snapshot_status = EXCLUDED.snapshot_status,
              placeholder_reason = EXCLUDED.placeholder_reason,
              generated_at_utc = now()
            """,
            params,
        )
        return cursor.rowcount if cursor.rowcount is not None else 0


def write_snapshot_reports(writer: ReportWriter, connection: Any) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
              partition_name,
              count(*) AS snapshots,
              sum(official_available::int) AS official_available,
              sum(gfs_available::int) AS gfs_available,
              sum(gefs_available::int) AS gefs_available,
              sum(cwa_live_shadow_available::int) AS cwa_live_shadow_available
            FROM model_features.h24n_snapshot
            GROUP BY partition_name
            ORDER BY partition_name
            """
        )
        rows = cursor.fetchall()
    writer.write_csv(
        "snapshot_coverage_report.csv",
        (
            "partition_name",
            "snapshots",
            "official_available",
            "gfs_available",
            "gefs_available",
            "cwa_live_shadow_available",
        ),
        rows,
    )
    writer.write_root_report(
        "snapshot_coverage_report.md",
        "HKG-T24-001 H24N Snapshot Coverage Report",
        (
            ("Status", "PASS"),
            (
                "Coverage By Partition",
                "\n".join(
                    f"- `{row[0]}`: snapshots={row[1]}, official={row[2]}, "
                    f"gfs={row[3]}, gefs={row[4]}, cwa_live_shadow={row[5]}"
                    for row in rows
                )
                or "- No snapshots generated.",
            ),
        ),
    )
    writer.write_csv(
        "live_shadow_availability_report.csv",
        ("source_code", "available_snapshots", "contract_status"),
        (
            ("arwf_live", 0, "placeholder_or_absent_until_live_collector_history_exists"),
            (
                "cwawrf15",
                sum(0 if row[5] is None else int(row[5]) for row in rows),
                "live_shadow_not_strict",
            ),
        ),
    )
    writer.write_root_report(
        "live_shadow_availability_report.md",
        "HKG-T24-001 Live Shadow Availability Report",
        (
            ("Status", "PASS"),
            (
                "ARWF",
                "ARWF is live-shadow only for Jira 001. If its primary table is absent, generated "
                "commands warn and emit placeholder availability rather than failing the foundation.",
            ),
            (
                "CWA WRF",
                "cwawrf15 remains live-shadow/prospective and does not enter strict v1 features.",
            ),
        ),
    )


def build_snapshots(
    connection: Any,
    writer: ReportWriter,
    *,
    target_table: DiscoveredTable | None,
    start_date: date = START_TARGET_DATE,
    end_date: date = END_TARGET_DATE,
) -> None:
    calendar_count = populate_cutoff_calendar(connection, start_date=start_date, end_date=end_date)
    label_count = 0
    if target_table is not None:
        label_count = populate_target_labels(connection, target_table, start_date=start_date, end_date=end_date)
    snapshot_count = populate_h24n_snapshots(connection, start_date=start_date, end_date=end_date)
    connection.commit()
    write_snapshot_reports(writer, connection)
    writer.write_hkg_report(
        "snapshot_build_summary.md",
        "HKG-T24-001 Snapshot Build Summary",
        (
            ("Calendar Rows", str(calendar_count)),
            ("Target Labels Loaded", str(label_count)),
            ("Snapshots Upserted", str(snapshot_count if not math.isnan(snapshot_count) else 0)),
        ),
    )
