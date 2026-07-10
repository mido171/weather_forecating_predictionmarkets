from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
DEFAULT_OUTPUT_NAME = "full_tactical_backfill_ok_tmax"
EXPERIMENT_ROOT = (
    REPO / "experiments" / "campaigns" / "hkg-t24" / "0214_tactical_h24n_gribstream_backfill"
)

SOURCE_CASE = """
CASE
  WHEN r.object_uri LIKE '%%full_tactical_backfill_ok_tmax%%' THEN 'full'
  WHEN r.object_uri LIKE '%%batch_smoke_10w%%' THEN 'batch_10w'
  WHEN r.object_uri LIKE '%%first_week%%' THEN 'first_week'
  WHEN r.object_uri LIKE '%%smoke%%' THEN 'smoke'
  WHEN r.object_uri IS NULL THEN 'null_source'
  ELSE 'other'
END
"""
FULL_WHERE = "r.object_uri LIKE '%%full_tactical_backfill_ok_tmax%%'"

VALUE_COLUMNS = {
    "temperature_2m_k": (250.0, 335.0, "surface temperature K"),
    "interval_tmax_2m_k": (250.0, 340.0, "interval Tmax K"),
    "dewpoint_2m_k": (230.0, 330.0, "2m dewpoint K"),
    "relative_humidity_2m_pct": (0.0, 100.0, "2m RH pct"),
    "u_wind_10m_mps": (-80.0, 80.0, "10m U wind m/s"),
    "v_wind_10m_mps": (-80.0, 80.0, "10m V wind m/s"),
    "mslp_pa": (85000.0, 110000.0, "MSLP Pa"),
    "low_cloud_pct": (0.0, 100.0, "low cloud pct"),
    "accumulated_precip_kg_m2": (0.0, 500.0, "APCP kg/m2"),
    "downward_shortwave_w_m2": (0.0, 1500.0, "DSWRF W/m2"),
    "net_shortwave_w_m2": (-300.0, 1500.0, "NSWRF W/m2"),
    "total_precip_m": (0.0, 1.0, "total precipitation m"),
    "shortwave_down_j_m2": (0.0, 50000000.0, "surface solar down J/m2"),
    "total_column_water_vapour_kg_m2": (0.0, 100.0, "TCWV kg/m2"),
    "pwat_kg_m2": (0.0, 100.0, "PWAT kg/m2"),
    "temperature_925_k": (220.0, 330.0, "925 hPa temp K"),
    "temperature_850_k": (210.0, 325.0, "850 hPa temp K"),
    "relative_humidity_700_pct": (0.0, 100.0, "700 hPa RH pct"),
    "geopotential_height_500_m": (4500.0, 6500.0, "500 hPa height m"),
}

EXPECTED_MEMBERS = {
    "gefsatmos": set(range(31)),
    "ifsenfo": set(range(51)),
    "aifsenfo": set(range(51)),
    "aigefssfc": set(range(31)),
}
DETERMINISTIC_12 = {
    "gfs",
    "gefsatmosmean",
    "ifsoper",
    "cwawrf15",
    "aifsoper",
    "aigfssfc",
    "aigfspres",
    "graphcast",
    "fourcastnetgfs",
}
HKO_ONLY = {"gefsatmos", "ifsenfo", "aifsenfo", "aigefssfc"}


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def local_path_for_check(uri: str) -> Path:
    if os.name != "nt":
        return Path(uri)

    path = str(Path(uri))
    if path.startswith("\\\\?\\"):
        return Path(path)
    if path.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + path.lstrip("\\"))
    return Path("\\\\?\\" + path)


def fetch_all(cursor: Any, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cursor.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def fetch_one(cursor: Any, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    cursor.execute(sql, params)
    row = cursor.fetchone()
    return dict(row) if row else {}


def stage(message: str) -> None:
    print(f"[audit] {datetime.now(timezone.utc).isoformat()} {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def range_compress(dates: list[date]) -> list[str]:
    if not dates:
        return []
    dates = sorted(dates)
    ranges: list[str] = []
    start = prev = dates[0]
    for current in dates[1:]:
        if current == prev + timedelta(days=1):
            prev = current
            continue
        ranges.append(start.isoformat() if start == prev else f"{start.isoformat()}..{prev.isoformat()}")
        start = prev = current
    ranges.append(start.isoformat() if start == prev else f"{start.isoformat()}..{prev.isoformat()}")
    return ranges


def read_csv_summary(results_csv: Path) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    non_completed: list[dict[str, Any]] = []
    status_counts = Counter()
    source_counts = Counter()

    with results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            status_counts[row["status"]] += 1
            source_counts[row["source"]] += 1
            entry = by_dataset.setdefault(
                dataset,
                {"chunks": 0, "rows": 0, "credits": 0, "expected_rows": 0, "statuses": Counter()},
            )
            entry["chunks"] += 1
            entry["rows"] += int(float(row["row_count"] or 0))
            entry["credits"] += int(float(row["estimated_credits_consumed"] or 0))
            entry["expected_rows"] += int(float(row["expected_rows"] or 0))
            entry["statuses"][row["status"]] += 1
            if row["status"] != "completed":
                non_completed.append(
                    {
                        key: row.get(key)
                        for key in [
                            "chunk_index",
                            "dataset",
                            "status",
                            "http_status",
                            "first_run_time",
                            "last_run_time",
                            "row_count",
                            "expected_rows",
                            "sanity_issues",
                            "error_class",
                            "error_message",
                        ]
                    }
                )

    for entry in by_dataset.values():
        entry["statuses"] = dict(entry["statuses"])
    return {
        "by_dataset": dict(sorted(by_dataset.items())),
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "non_completed_chunks": non_completed,
    }


def read_api_log(api_log: Path) -> dict[str, Any]:
    if not api_log.exists():
        return {"exists": False}

    counts = Counter()
    http_statuses = Counter()
    retry_after = Counter()
    elapsed_ms: list[float] = []
    wait_seconds: list[float] = []
    first_event = None
    last_event = None
    line_count = 0

    with api_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            event = json.loads(line)
            first_event = first_event or event
            last_event = event
            counts[event.get("event", "unknown")] += 1
            if "http_status" in event:
                http_statuses[str(event["http_status"])] += 1
            if event.get("retry_after"):
                retry_after[str(event["retry_after"])] += 1
            if "elapsed_ms" in event:
                elapsed_ms.append(float(event["elapsed_ms"]))
            if "rate_wait_seconds" in event:
                wait_seconds.append(float(event["rate_wait_seconds"]))

    def stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}

    return {
        "exists": True,
        "line_count": line_count,
        "events": dict(counts),
        "http_statuses": dict(http_statuses),
        "http_error_count": sum(count for status, count in http_statuses.items() if int(status) >= 400),
        "retry_after_values": dict(retry_after),
        "elapsed_ms": stats(elapsed_ms),
        "rate_wait_seconds": stats(wait_seconds),
        "first_event_utc": first_event.get("updated_at_utc") if first_event else None,
        "last_event_utc": last_event.get("updated_at_utc") if last_event else None,
    }


def audit_db(database_url: str, skip_file_hash: bool) -> dict[str, Any]:
    report: dict[str, Any] = {}
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            stage("connected to postgres")
            cursor.execute("SET TIME ZONE 'UTC'")
            stage("materializing full-run scalar temp table")
            cursor.execute(
                f"""
                CREATE TEMP TABLE audit_full_forecast ON COMMIT DROP AS
                SELECT
                    fw.dataset_code,
                    fw.acquisition_version,
                    fw.target_date_hkt,
                    fw.cutoff_id,
                    fw.run_time_utc,
                    fw.valid_time_utc,
                    fw.lead_hours,
                    fw.location_code,
                    fw.requested_latitude,
                    fw.requested_longitude,
                    fw.returned_latitude,
                    fw.returned_longitude,
                    fw.returned_grid_distance_km,
                    fw.member_number,
                    fw.temperature_2m_k,
                    fw.interval_tmax_2m_k,
                    fw.dewpoint_2m_k,
                    fw.relative_humidity_2m_pct,
                    fw.u_wind_10m_mps,
                    fw.v_wind_10m_mps,
                    fw.mslp_pa,
                    fw.low_cloud_pct,
                    fw.accumulated_precip_kg_m2,
                    fw.downward_shortwave_w_m2,
                    fw.net_shortwave_w_m2,
                    fw.total_precip_m,
                    fw.shortwave_down_j_m2,
                    fw.total_column_water_vapour_kg_m2,
                    fw.pwat_kg_m2,
                    fw.temperature_925_k,
                    fw.temperature_850_k,
                    fw.relative_humidity_700_pct,
                    fw.geopotential_height_500_m,
                    fw.source_response_object_id,
                    fw.quality_status
                FROM nwp_tactical.forecast_wide fw
                JOIN nwp_tactical.raw_response_object r
                  ON r.response_object_id = fw.source_response_object_id
                WHERE {FULL_WHERE}
                """
            )
            cursor.execute("CREATE INDEX ON audit_full_forecast(dataset_code)")
            cursor.execute("CREATE INDEX ON audit_full_forecast(source_response_object_id)")
            cursor.execute("CREATE INDEX ON audit_full_forecast(target_date_hkt)")
            cursor.execute("ANALYZE audit_full_forecast")
            stage("temp table ready")
            report["table_counts"] = {
                "forecast_wide": fetch_one(cursor, "SELECT count(*) AS rows FROM nwp_tactical.forecast_wide"),
                "acquisition_chunk": fetch_one(cursor, "SELECT count(*) AS rows FROM nwp_tactical.acquisition_chunk"),
                "raw_response_object": fetch_one(
                    cursor,
                    """
                    SELECT count(*) AS rows,
                           coalesce(sum(row_count),0) AS row_count_sum,
                           coalesce(sum(byte_size),0) AS byte_size_sum
                    FROM nwp_tactical.raw_response_object
                    """,
                ),
                "validation_issue": fetch_one(cursor, "SELECT count(*) AS rows FROM nwp_tactical.validation_issue"),
            }
            stage("table counts collected")
            report["forecast_counts_by_source_dataset"] = fetch_all(
                cursor,
                f"""
                SELECT {SOURCE_CASE} AS source_scope, fw.dataset_code, count(*) AS rows
                FROM nwp_tactical.forecast_wide fw
                LEFT JOIN nwp_tactical.raw_response_object r
                  ON r.response_object_id = fw.source_response_object_id
                GROUP BY 1,2 ORDER BY 1,2
                """,
            )
            stage("source scope counts collected")
            report["forecast_non_full_rows"] = fetch_all(
                cursor,
                f"""
                SELECT {SOURCE_CASE} AS source_scope, fw.dataset_code, count(*) AS rows,
                       min(fw.run_time_utc) AS min_run_time_utc,
                       max(fw.run_time_utc) AS max_run_time_utc,
                       min(fw.target_date_hkt) AS min_target_date_hkt,
                       max(fw.target_date_hkt) AS max_target_date_hkt
                FROM nwp_tactical.forecast_wide fw
                LEFT JOIN nwp_tactical.raw_response_object r
                  ON r.response_object_id = fw.source_response_object_id
                WHERE NOT ({FULL_WHERE})
                GROUP BY 1,2 ORDER BY 1,2
                """,
            )
            stage("non-full row scan collected")
            report["full_dataset_summary"] = fetch_all(
                cursor,
                f"""
                SELECT fw.dataset_code,
                       count(*) AS rows,
                       min(fw.run_time_utc) AS min_run_time_utc,
                       max(fw.run_time_utc) AS max_run_time_utc,
                       min(fw.valid_time_utc) AS min_valid_time_utc,
                       max(fw.valid_time_utc) AS max_valid_time_utc,
                       min(fw.target_date_hkt) AS min_target_date_hkt,
                       max(fw.target_date_hkt) AS max_target_date_hkt,
                       count(DISTINCT fw.run_time_utc) AS distinct_run_times,
                       count(DISTINCT fw.valid_time_utc) AS distinct_valid_times,
                       count(DISTINCT fw.target_date_hkt) AS distinct_target_dates,
                       count(DISTINCT fw.location_code) AS distinct_locations,
                       count(DISTINCT fw.member_number) AS distinct_members,
                       min(fw.lead_hours) AS min_lead_hours,
                       max(fw.lead_hours) AS max_lead_hours
                FROM audit_full_forecast fw
                GROUP BY fw.dataset_code ORDER BY fw.dataset_code
                """,
            )
            stage("dataset coverage collected")
            report["model_plan_vs_full_rows"] = fetch_all(
                cursor,
                f"""
                WITH full_rows AS (
                  SELECT fw.dataset_code,
                         count(*) AS rows,
                         min(fw.run_time_utc) AS min_run,
                         max(fw.run_time_utc) AS max_run
                  FROM audit_full_forecast fw
                  GROUP BY fw.dataset_code
                )
                SELECT mp.dataset_code, mp.priority, mp.stage, mp.location_policy, mp.member_policy,
                       mp.archive_run_start_utc, mp.archive_run_end_utc, mp.expected_wide_rows,
                       mp.approximate_credits, coalesce(fr.rows,0) AS full_rows,
                       fr.min_run AS full_min_run_time_utc,
                       fr.max_run AS full_max_run_time_utc,
                       coalesce(fr.rows,0) - coalesce(mp.expected_wide_rows,0) AS full_rows_minus_plan
                FROM nwp_tactical.model_plan mp
                LEFT JOIN full_rows fr ON fr.dataset_code = mp.dataset_code
                ORDER BY mp.dataset_code
                """,
            )
            stage("model plan comparison collected")
            report["chunk_status_by_dataset"] = fetch_all(
                cursor,
                """
                SELECT dataset_code, status, count(*) AS chunks,
                       coalesce(sum(row_count),0) AS row_count_sum,
                       coalesce(sum(expected_rows),0) AS expected_rows_sum,
                       count(*) FILTER (WHERE http_status IS NOT NULL AND http_status <> 200) AS non_200_chunks
                FROM nwp_tactical.acquisition_chunk
                GROUP BY dataset_code, status ORDER BY dataset_code, status
                """,
            )
            stage("chunk ledger collected")
            report["chunk_policy_violations"] = fetch_one(
                cursor,
                """
                SELECT count(*) FILTER (WHERE endpoint <> 'runs') AS non_runs_endpoint,
                       count(*) FILTER (WHERE time_selector <> 'timesList') AS non_timeslist_selector,
                       count(*) FILTER (WHERE request_json ? 'forecastedFrom') AS has_forecasted_from,
                       count(*) FILTER (WHERE request_json ? 'forecastedUntil') AS has_forecasted_until,
                       count(*) FILTER (WHERE NOT (request_json ? 'timesList')) AS missing_timeslist,
                       count(*) FILTER (WHERE max_lead_hours < min_lead_hours) AS invalid_lead_bounds,
                       count(*) FILTER (WHERE http_status IS NOT NULL AND http_status <> 200) AS non_200_http
                FROM nwp_tactical.acquisition_chunk
                """,
            )
            report["failed_or_empty_chunks"] = fetch_all(
                cursor,
                """
                SELECT chunk_id, dataset_code, status, http_status, row_count, expected_rows,
                       error_class, error_message, raw_object_uri,
                       run_times_utc[1] AS first_run_time_utc,
                       run_times_utc[array_length(run_times_utc, 1)] AS last_run_time_utc
                FROM nwp_tactical.acquisition_chunk
                WHERE status <> 'completed'
                ORDER BY dataset_code, first_run_time_utc
                """,
            )
            report["lineage_checks"] = {
                "forecast_rows_null_or_missing_source": fetch_one(
                    cursor,
                    """
                    SELECT count(*) FILTER (WHERE fw.source_response_object_id IS NULL) AS null_source_response_object_id,
                           count(*) FILTER (
                               WHERE fw.source_response_object_id IS NOT NULL
                                 AND r.response_object_id IS NULL
                           ) AS missing_raw_response_object
                    FROM nwp_tactical.forecast_wide fw
                    LEFT JOIN nwp_tactical.raw_response_object r
                      ON r.response_object_id = fw.source_response_object_id
                    """,
                ),
                "raw_object_vs_sourced_rows_mismatches": fetch_all(
                    cursor,
                    f"""
                    SELECT r.response_object_id, r.chunk_id, r.object_uri,
                           r.row_count AS raw_row_count,
                           count(fw.*) AS sourced_forecast_rows
                    FROM nwp_tactical.raw_response_object r
                    LEFT JOIN audit_full_forecast fw
                      ON fw.source_response_object_id = r.response_object_id
                    WHERE r.object_uri LIKE '%%full_tactical_backfill_ok_tmax%%'
                    GROUP BY r.response_object_id, r.chunk_id, r.object_uri, r.row_count
                    HAVING r.row_count <> count(fw.*)
                    ORDER BY abs(r.row_count - count(fw.*)) DESC, r.response_object_id
                    LIMIT 50
                    """,
                ),
            }
            stage("lineage checks collected")
            report["structural_consistency"] = {
                "lead_mismatch": fetch_one(
                    cursor,
                    f"""
                    SELECT count(*) AS rows
                    FROM audit_full_forecast fw
                    WHERE abs(fw.lead_hours - extract(epoch FROM (fw.valid_time_utc - fw.run_time_utc))/3600.0) > 0.01
                    """,
                ),
                "target_date_mismatch": fetch_one(
                    cursor,
                    f"""
                    SELECT count(*) AS rows
                    FROM audit_full_forecast fw
                    WHERE fw.target_date_hkt <> (fw.valid_time_utc AT TIME ZONE 'Asia/Hong_Kong')::date
                    """,
                ),
                "non_h24n_or_non_raw_valid": fetch_one(
                    cursor,
                    f"""
                    SELECT count(*) FILTER (WHERE cutoff_id <> 'H24N') AS non_h24n,
                           count(*) FILTER (WHERE acquisition_version <> 'tactical_h24n_v1') AS non_tactical_version,
                           count(*) FILTER (WHERE quality_status <> 'raw_valid') AS non_raw_valid
                    FROM audit_full_forecast fw
                    """,
                ),
                "requested_coord_mismatch_vs_stencil": fetch_one(
                    cursor,
                    f"""
                    SELECT count(*) AS rows
                    FROM audit_full_forecast fw
                    JOIN nwp_tactical.location_stencil ls ON ls.location_code = fw.location_code
                    WHERE (
                        abs(fw.requested_latitude - ls.latitude) > 1e-6
                        OR abs(fw.requested_longitude - ls.longitude) > 1e-6
                      )
                    """,
                ),
                "raw_values_empty": fetch_one(
                    cursor,
                    f"""
                    SELECT count(*) AS rows
                    FROM nwp_tactical.forecast_wide fw
                    JOIN nwp_tactical.raw_response_object r
                      ON r.response_object_id = fw.source_response_object_id
                    WHERE {FULL_WHERE} AND fw.raw_values_jsonb = '{{}}'::jsonb
                    """,
                ),
            }
            stage("structural checks collected")
            report["location_policy_anomalies"] = {
                "deterministic_groups_not_12_locations": fetch_all(
                    cursor,
                    f"""
                    SELECT dataset_code, count(*) AS affected_groups
                    FROM (
                      SELECT fw.dataset_code, fw.run_time_utc, fw.valid_time_utc,
                             fw.member_number, count(DISTINCT fw.location_code) AS location_count
                      FROM audit_full_forecast fw
                      WHERE fw.dataset_code = ANY(%s)
                      GROUP BY fw.dataset_code, fw.run_time_utc, fw.valid_time_utc, fw.member_number
                    ) grouped
                    WHERE location_count <> 12
                    GROUP BY dataset_code ORDER BY dataset_code
                    """,
                    (list(DETERMINISTIC_12),),
                ),
                "hko_only_groups_not_one_location": fetch_all(
                    cursor,
                    f"""
                    SELECT dataset_code, count(*) AS affected_groups
                    FROM (
                      SELECT fw.dataset_code, fw.run_time_utc, fw.valid_time_utc,
                             fw.member_number, count(DISTINCT fw.location_code) AS location_count,
                             bool_and(fw.location_code = 'hko_center') AS all_hko_center
                      FROM audit_full_forecast fw
                      WHERE fw.dataset_code = ANY(%s)
                      GROUP BY fw.dataset_code, fw.run_time_utc, fw.valid_time_utc, fw.member_number
                    ) grouped
                    WHERE location_count <> 1 OR NOT all_hko_center
                    GROUP BY dataset_code ORDER BY dataset_code
                    """,
                    (list(HKO_ONLY),),
                ),
            }
            stage("location policy checks collected")
            report["member_counts_by_dataset"] = fetch_all(
                cursor,
                f"""
                SELECT fw.dataset_code, count(DISTINCT fw.member_number) AS distinct_members,
                       min(fw.member_number) AS min_member, max(fw.member_number) AS max_member
                FROM audit_full_forecast fw
                GROUP BY fw.dataset_code ORDER BY fw.dataset_code
                """,
            )
            member_groups = fetch_all(
                cursor,
                f"""
                SELECT fw.dataset_code, fw.run_time_utc, fw.valid_time_utc,
                       array_agg(DISTINCT fw.member_number ORDER BY fw.member_number) AS members
                FROM audit_full_forecast fw
                WHERE fw.dataset_code = ANY(%s)
                GROUP BY fw.dataset_code, fw.run_time_utc, fw.valid_time_utc
                ORDER BY fw.dataset_code, fw.run_time_utc, fw.valid_time_utc
                """,
                (list(EXPECTED_MEMBERS),),
            )
            report["member_coverage_anomalies"] = summarize_member_anomalies(member_groups)
            stage("member coverage checks collected")
            report["target_date_gap_scan"] = scan_target_date_gaps(cursor)
            stage("target date gap scan collected")
            report["value_coverage_and_ranges"] = value_coverage(cursor)
            stage("value coverage collected")
            report["cross_field_anomalies"] = {
                "dewpoint_above_temperature_by_dataset": fetch_all(
                    cursor,
                    f"""
                    SELECT dataset_code, count(*) AS rows
                    FROM audit_full_forecast fw
                    WHERE fw.temperature_2m_k IS NOT NULL
                      AND fw.dewpoint_2m_k IS NOT NULL
                      AND fw.dewpoint_2m_k > fw.temperature_2m_k + 0.5
                    GROUP BY dataset_code ORDER BY dataset_code
                    """,
                ),
                "interval_tmax_below_temperature_by_dataset": fetch_all(
                    cursor,
                    f"""
                    SELECT dataset_code, count(*) AS rows
                    FROM audit_full_forecast fw
                    WHERE fw.temperature_2m_k IS NOT NULL
                      AND fw.interval_tmax_2m_k IS NOT NULL
                      AND fw.interval_tmax_2m_k + 0.01 < fw.temperature_2m_k
                    GROUP BY dataset_code ORDER BY dataset_code
                    """,
                ),
            }
            stage("cross-field anomaly checks collected")
            report["h24n_leakage_filter_6h"] = fetch_all(
                cursor,
                f"""
                SELECT fw.dataset_code,
                       count(*) AS rows,
                       count(*) FILTER (
                           WHERE fw.run_time_utc + interval '6 hours'
                              <= (((fw.target_date_hkt - 1)::timestamp + interval '15 hours') AT TIME ZONE 'Asia/Hong_Kong')
                       ) AS safe_rows,
                       count(*) FILTER (
                           WHERE NOT (
                               fw.run_time_utc + interval '6 hours'
                                  <= (((fw.target_date_hkt - 1)::timestamp + interval '15 hours') AT TIME ZONE 'Asia/Hong_Kong')
                           )
                       ) AS unsafe_rows
                FROM audit_full_forecast fw
                GROUP BY fw.dataset_code ORDER BY fw.dataset_code
                """,
            )
            stage("h24n leakage filter counts collected")
            report["tmax_derivability_safe_6h"] = fetch_all(
                cursor,
                f"""
                WITH safe AS (
                  SELECT fw.*, coalesce(fw.interval_tmax_2m_k, fw.temperature_2m_k) AS tmax_candidate_k
                  FROM audit_full_forecast fw
                  WHERE fw.run_time_utc + interval '6 hours'
                       <= (((fw.target_date_hkt - 1)::timestamp + interval '15 hours') AT TIME ZONE 'Asia/Hong_Kong')
                ),
                daily AS (
                  SELECT dataset_code, target_date_hkt,
                         max(tmax_candidate_k) AS network_or_member_max_k,
                         max(tmax_candidate_k) FILTER (WHERE location_code = 'hko_center') AS hko_center_max_k,
                         count(*) FILTER (WHERE tmax_candidate_k IS NOT NULL) AS usable_rows,
                         count(DISTINCT location_code) FILTER (WHERE tmax_candidate_k IS NOT NULL) AS usable_locations,
                         count(DISTINCT member_number) FILTER (WHERE tmax_candidate_k IS NOT NULL) AS usable_members
                  FROM safe
                  GROUP BY dataset_code, target_date_hkt
                )
                SELECT dataset_code,
                       count(*) AS target_days_seen,
                       count(*) FILTER (WHERE network_or_member_max_k IS NOT NULL) AS target_days_with_any_tmax_candidate,
                       count(*) FILTER (WHERE hko_center_max_k IS NOT NULL) AS target_days_with_hko_center_tmax_candidate,
                       min(target_date_hkt) AS min_target_date_hkt,
                       max(target_date_hkt) AS max_target_date_hkt,
                       min(network_or_member_max_k - 273.15) AS min_daily_candidate_c,
                       max(network_or_member_max_k - 273.15) AS max_daily_candidate_c,
                       min(usable_rows) AS min_usable_rows_per_day,
                       max(usable_rows) AS max_usable_rows_per_day,
                       min(usable_locations) AS min_usable_locations_per_day,
                       max(usable_locations) AS max_usable_locations_per_day,
                       min(usable_members) AS min_usable_members_per_day,
                       max(usable_members) AS max_usable_members_per_day
                FROM daily
                GROUP BY dataset_code ORDER BY dataset_code
                """,
            )
            stage("tmax derivability summary collected")
            report["tmax_missing_safe_target_dates"] = fetch_all(
                cursor,
                f"""
                WITH safe AS (
                  SELECT fw.dataset_code, fw.target_date_hkt,
                         coalesce(fw.interval_tmax_2m_k, fw.temperature_2m_k) AS tmax_candidate_k
                  FROM audit_full_forecast fw
                  WHERE fw.run_time_utc + interval '6 hours'
                       <= (((fw.target_date_hkt - 1)::timestamp + interval '15 hours') AT TIME ZONE 'Asia/Hong_Kong')
                )
                SELECT dataset_code, target_date_hkt, count(*) AS rows_seen
                FROM safe
                GROUP BY dataset_code, target_date_hkt
                HAVING count(tmax_candidate_k) = 0
                ORDER BY dataset_code, target_date_hkt
                LIMIT 300
                """,
            )
            stage("missing tmax target dates collected")
            report["sample_daily_tmax_examples"] = fetch_all(
                cursor,
                f"""
                WITH safe AS (
                  SELECT fw.*, coalesce(fw.interval_tmax_2m_k, fw.temperature_2m_k) AS tmax_candidate_k
                  FROM audit_full_forecast fw
                  WHERE fw.run_time_utc + interval '6 hours'
                       <= (((fw.target_date_hkt - 1)::timestamp + interval '15 hours') AT TIME ZONE 'Asia/Hong_Kong')
                    AND coalesce(fw.interval_tmax_2m_k, fw.temperature_2m_k) IS NOT NULL
                ),
                sample_days AS (
                  SELECT dataset_code, min(target_date_hkt) AS target_date_hkt
                  FROM safe
                  GROUP BY dataset_code
                  UNION
                  SELECT dataset_code, max(target_date_hkt) AS target_date_hkt
                  FROM safe
                  GROUP BY dataset_code
                ),
                ranked AS (
                  SELECT safe.*,
                         row_number() OVER (
                           PARTITION BY safe.dataset_code, safe.target_date_hkt
                           ORDER BY safe.tmax_candidate_k DESC NULLS LAST
                         ) AS rn
                  FROM safe
                  JOIN sample_days
                    ON sample_days.dataset_code = safe.dataset_code
                   AND sample_days.target_date_hkt = safe.target_date_hkt
                )
                SELECT dataset_code, target_date_hkt, location_code, member_number,
                       run_time_utc, valid_time_utc, lead_hours,
                       round((tmax_candidate_k - 273.15)::numeric, 3) AS max_candidate_c,
                       CASE
                         WHEN interval_tmax_2m_k IS NOT NULL THEN 'interval_tmax_2m_k'
                         ELSE 'temperature_2m_k'
                       END AS source_column
                FROM ranked
                WHERE rn = 1
                ORDER BY dataset_code, target_date_hkt
                """,
            )
            stage("leakage and tmax derivability collected")
            full_raws = fetch_all(
                cursor,
                """
                SELECT response_object_id, chunk_id, object_uri, byte_size, sha256, row_count
                FROM nwp_tactical.raw_response_object
                WHERE object_uri LIKE '%%full_tactical_backfill_ok_tmax%%'
                ORDER BY response_object_id
                """,
            )

    stage("validating raw files")
    report["full_raw_file_checks"] = validate_raw_files(full_raws, skip_file_hash)
    stage("raw file validation collected")
    return report


def summarize_member_anomalies(groups: list[dict[str, Any]]) -> dict[str, Any]:
    anomalies: dict[str, dict[str, Any]] = {}
    for group in groups:
        dataset = group["dataset_code"]
        expected = EXPECTED_MEMBERS[dataset]
        got = set(group["members"] or [])
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if not missing and not extra:
            continue
        entry = anomalies.setdefault(
            dataset,
            {"affected_groups": 0, "examples": [], "missing_member_counts": Counter(), "extra_member_counts": Counter()},
        )
        entry["affected_groups"] += 1
        for member in missing:
            entry["missing_member_counts"][str(member)] += 1
        for member in extra:
            entry["extra_member_counts"][str(member)] += 1
        if len(entry["examples"]) < 20:
            entry["examples"].append(
                {
                    "run_time_utc": group["run_time_utc"],
                    "valid_time_utc": group["valid_time_utc"],
                    "member_count": len(got),
                    "missing": missing,
                    "extra": extra,
                }
            )
    for entry in anomalies.values():
        entry["missing_member_counts"] = dict(entry["missing_member_counts"])
        entry["extra_member_counts"] = dict(entry["extra_member_counts"])
    return anomalies


def scan_target_date_gaps(cursor: Any) -> dict[str, Any]:
    rows = fetch_all(
        cursor,
        """
        SELECT fw.dataset_code, fw.target_date_hkt
        FROM audit_full_forecast fw
        GROUP BY fw.dataset_code, fw.target_date_hkt
        ORDER BY fw.dataset_code, fw.target_date_hkt
        """,
    )
    dates_by_dataset: dict[str, set[date]] = defaultdict(set)
    for row in rows:
        dates_by_dataset[row["dataset_code"]].add(row["target_date_hkt"])

    gaps: dict[str, Any] = {}
    for dataset, present_dates in sorted(dates_by_dataset.items()):
        start = min(present_dates)
        end = max(present_dates)
        all_dates = {start + timedelta(days=offset) for offset in range((end - start).days + 1)}
        missing = sorted(all_dates - present_dates)
        gaps[dataset] = {
            "min_target_date_hkt": start,
            "max_target_date_hkt": end,
            "target_dates_present": len(present_dates),
            "missing_count_between_min_max": len(missing),
            "missing_ranges_first_20": range_compress(missing)[:20],
        }
    return gaps


def value_coverage(cursor: Any) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for column, (low, high, label) in VALUE_COLUMNS.items():
        coverage.extend(
            fetch_all(
                cursor,
                f"""
                SELECT dataset_code, %s AS column_name, %s AS label,
                       count(*) AS rows,
                       count({column}) AS non_null,
                       round((100.0 * count({column}) / nullif(count(*),0))::numeric, 4) AS non_null_pct,
                       min({column}) AS min_value,
                       max({column}) AS max_value,
                       avg({column}) AS avg_value,
                       count({column}) FILTER (WHERE {column} < %s OR {column} > %s) AS physical_range_anomaly_rows
                FROM audit_full_forecast fw
                GROUP BY dataset_code ORDER BY dataset_code
                """,
                (column, label, low, high),
            )
        )
    return coverage


def validate_raw_files(full_raws: list[dict[str, Any]], skip_file_hash: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "full_raw_objects": len(full_raws),
        "missing_files": [],
        "size_mismatch": [],
        "sha256_mismatch": [],
        "checked_sha256_count": 0,
        "total_byte_size_checked": 0,
        "skip_file_hash": skip_file_hash,
    }
    for raw in full_raws:
        recorded_uri = str(raw["object_uri"])
        path = local_path_for_check(recorded_uri)
        if not path.exists():
            checks["missing_files"].append({"response_object_id": raw["response_object_id"], "path": recorded_uri})
            continue
        file_size = path.stat().st_size
        if int(raw["byte_size"]) != file_size:
            checks["size_mismatch"].append(
                {
                    "response_object_id": raw["response_object_id"],
                    "path": recorded_uri,
                    "db_size": int(raw["byte_size"]),
                    "file_size": file_size,
                }
            )
        checks["total_byte_size_checked"] += file_size
        if skip_file_hash:
            continue
        file_sha256 = sha256_file(path)
        checks["checked_sha256_count"] += 1
        if file_sha256 != raw["sha256"]:
            checks["sha256_mismatch"].append(
                {
                    "response_object_id": raw["response_object_id"],
                    "path": recorded_uri,
                    "db_sha256": raw["sha256"],
                    "file_sha256": file_sha256,
                }
            )
    return checks


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Deep Sanity Audit - Full Tactical GribStream Backfill - 2026-06-25",
        "",
        "## Executive Result",
        "",
    ]
    table_counts = report["db"]["table_counts"]
    total_rows = table_counts["forecast_wide"]["rows"]
    full_rows = sum(row["rows"] for row in report["db"]["forecast_counts_by_source_dataset"] if row["source_scope"] == "full")
    non_full = total_rows - full_rows
    file_checks = report["db"]["full_raw_file_checks"]
    lines.extend(
        [
            f"- Current `nwp_tactical.forecast_wide` rows: {total_rows:,}.",
            f"- Rows sourced from the full backfill raw root: {full_rows:,}.",
            f"- Non-full rows still present in `forecast_wide`: {non_full:,}.",
            f"- API log HTTP error count: {report['api_log'].get('http_error_count')}.",
            (
                f"- Full raw objects checked: {file_checks['full_raw_objects']:,}; "
                f"missing files: {len(file_checks['missing_files'])}; "
                f"size mismatches: {len(file_checks['size_mismatch'])}; "
                f"sha256 files rehashed: {file_checks['checked_sha256_count']:,}; "
                f"sha256 mismatches: {len(file_checks['sha256_mismatch'])}."
            ),
            "",
        ]
    )
    if non_full:
        lines.extend(
            [
                "Critical table-scope issue: the live table is not pure full-run data. Filter by joining `source_response_object_id` to `nwp_tactical.raw_response_object` and requiring `object_uri LIKE '%full_tactical_backfill_ok_tmax%'` until old smoke rows are purged or moved.",
                "",
            ]
        )
    append_source_scope(lines, report)
    append_dataset_coverage(lines, report)
    append_non_completed_chunks(lines, report)
    append_structural_checks(lines, report)
    append_member_coverage(lines, report)
    append_tmax(lines, report)
    append_target_gaps(lines, report)
    append_physical_anomalies(lines, report)
    lines.extend(
        [
            "## Known Blockers / Do-Not-Model Warnings",
            "",
            "- The live DB table has 933 older `batch_smoke_10w` `gefsatmos` rows mixed into `forecast_wide`.",
            "- `ifsenfo` has recent missing-member-0 chunks, although HTTP status was 200 and other members persisted.",
            "- `fourcastnetgfs` full-run rows end before the requested tail; the final tail request returned empty.",
            "- `nbmoc` returned zero rows and is not a usable HKO Tmax source from this pull.",
            "- `aigfspres` is upper-air support only, not a surface Tmax source.",
            "- `aigefssfc` has very poor usable 2m temperature coverage and should remain blocked as a Tmax source unless a selector/provider probe fixes it.",
            "- Raw rows are not feature-safe unless the H24N cutoff filter is applied in downstream feature extraction.",
            "",
            "Full machine-readable detail is in `deep_sanity_audit_20260625.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def append_source_scope(lines: list[str], report: dict[str, Any]) -> None:
    lines.extend(["## Source Scope Counts", "", "| Source scope | Dataset | Rows |", "| --- | --- | ---: |"])
    for row in report["db"]["forecast_counts_by_source_dataset"]:
        lines.append(f"| `{row['source_scope']}` | `{row['dataset_code']}` | {row['rows']:,} |")
    lines.append("")


def append_dataset_coverage(lines: list[str], report: dict[str, Any]) -> None:
    lines.extend(
        [
            "## Full-Run Dataset Coverage",
            "",
            "| Dataset | Rows | Run time UTC | Target dates HKT | Runs | Locations | Members | Lead h |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["db"]["full_dataset_summary"]:
        lines.append(
            f"| `{row['dataset_code']}` | {row['rows']:,} | {row['min_run_time_utc']} to {row['max_run_time_utc']} | "
            f"{row['min_target_date_hkt']} to {row['max_target_date_hkt']} ({row['distinct_target_dates']:,}) | "
            f"{row['distinct_run_times']:,} | {row['distinct_locations']} | {row['distinct_members']} | "
            f"{row['min_lead_hours']}..{row['max_lead_hours']} |"
        )
    lines.append("")


def append_non_completed_chunks(lines: list[str], report: dict[str, Any]) -> None:
    lines.extend(
        [
            "## Non-Clean Chunks",
            "",
            "| Chunk | Dataset | Status | HTTP | Rows | Expected | Window UTC | Issue |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["csv"]["non_completed_chunks"]:
        issue = row.get("sanity_issues") or row.get("error_message") or ""
        lines.append(
            f"| {row.get('chunk_index')} | `{row.get('dataset')}` | `{row.get('status')}` | {row.get('http_status')} | "
            f"{row.get('row_count')} | {row.get('expected_rows')} | {row.get('first_run_time')} to {row.get('last_run_time')} | {issue} |"
        )
    lines.append("")


def append_structural_checks(lines: list[str], report: dict[str, Any]) -> None:
    structural = report["db"]["structural_consistency"]
    policy = report["db"]["chunk_policy_violations"]
    lines.extend(
        [
            "## Structural Checks",
            "",
            f"- Chunk policy violations: {policy}.",
            f"- Lead-hour mismatches: {structural['lead_mismatch']['rows']}.",
            f"- Target-date mismatches vs valid time HKT date: {structural['target_date_mismatch']['rows']}.",
            f"- Non-H24N rows: {structural['non_h24n_or_non_raw_valid']['non_h24n']}; non tactical version rows: {structural['non_h24n_or_non_raw_valid']['non_tactical_version']}; non raw-valid rows: {structural['non_h24n_or_non_raw_valid']['non_raw_valid']}.",
            f"- Requested coordinate mismatches vs stencil: {structural['requested_coord_mismatch_vs_stencil']['rows']}.",
            f"- Rows with empty raw_values_jsonb: {structural['raw_values_empty']['rows']}.",
            "",
        ]
    )


def append_member_coverage(lines: list[str], report: dict[str, Any]) -> None:
    lines.extend(["## Member Coverage Anomalies", ""])
    anomalies = report["db"]["member_coverage_anomalies"]
    if not anomalies:
        lines.extend(["- None detected for configured ensemble member sets.", ""])
        return
    for dataset, entry in anomalies.items():
        lines.append(
            f"- `{dataset}`: {entry['affected_groups']:,} run/valid groups affected; "
            f"missing counts {entry['missing_member_counts']}; extra counts {entry['extra_member_counts']}."
        )
    lines.append("")


def append_tmax(lines: list[str], report: dict[str, Any]) -> None:
    lines.extend(
        [
            "## H24N Leakage Filter and Tmax Derivability",
            "",
            "| Dataset | Rows | Safe rows | Unsafe rows | Tmax days any | Tmax days hko_center | Daily C range | Usable rows/day | Locations/day | Members/day |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    leak_by_dataset = {row["dataset_code"]: row for row in report["db"]["h24n_leakage_filter_6h"]}
    tmax_by_dataset = {row["dataset_code"]: row for row in report["db"]["tmax_derivability_safe_6h"]}
    for dataset in sorted(set(leak_by_dataset) | set(tmax_by_dataset)):
        leak = leak_by_dataset.get(dataset, {})
        tmax = tmax_by_dataset.get(dataset, {})
        c_range = ""
        if tmax.get("min_daily_candidate_c") is not None:
            c_range = f"{float(tmax['min_daily_candidate_c']):.2f}..{float(tmax['max_daily_candidate_c']):.2f}"
        lines.append(
            f"| `{dataset}` | {leak.get('rows', 0):,} | {leak.get('safe_rows', 0):,} | {leak.get('unsafe_rows', 0):,} | "
            f"{tmax.get('target_days_with_any_tmax_candidate', 0):,} | {tmax.get('target_days_with_hko_center_tmax_candidate', 0):,} | "
            f"{c_range} | {tmax.get('min_usable_rows_per_day')}..{tmax.get('max_usable_rows_per_day')} | "
            f"{tmax.get('min_usable_locations_per_day')}..{tmax.get('max_usable_locations_per_day')} | "
            f"{tmax.get('min_usable_members_per_day')}..{tmax.get('max_usable_members_per_day')} |"
        )
    lines.append("")


def append_target_gaps(lines: list[str], report: dict[str, Any]) -> None:
    lines.extend(
        [
            "## Target-Date Gap Scan",
            "",
            "| Dataset | Dates present | Missing between min/max | First missing ranges |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for dataset, gap in report["db"]["target_date_gap_scan"].items():
        ranges = ", ".join(gap["missing_ranges_first_20"])
        lines.append(
            f"| `{dataset}` | {gap['target_dates_present']:,} | {gap['missing_count_between_min_max']:,} | {ranges} |"
        )
    lines.append("")


def append_physical_anomalies(lines: list[str], report: dict[str, Any]) -> None:
    anomalies = [
        row
        for row in report["db"]["value_coverage_and_ranges"]
        if int(row["physical_range_anomaly_rows"] or 0) > 0
    ]
    lines.extend(["## Physical Range Anomalies", ""])
    if not anomalies:
        lines.extend(["No values breached the broad physical sanity ranges configured in this audit.", ""])
        return
    lines.extend(["| Dataset | Column | Non-null | Min | Max | Anomaly rows |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in anomalies:
        lines.append(
            f"| `{row['dataset_code']}` | `{row['column_name']}` | {row['non_null']:,} | "
            f"{row['min_value']} | {row['max_value']} | {row['physical_range_anomaly_rows']:,} |"
        )
    lines.append("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep sanity audit for tactical GribStream backfill outputs.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = EXPERIMENT_ROOT / args.output_name
    results_csv = output_dir / "batch_results.csv"
    progress_json = output_dir / "progress.json"
    api_log = EXPERIMENT_ROOT / "logs" / f"gribstream_{args.output_name}_api_events.jsonl"
    report_json = output_dir / "deep_sanity_audit_20260625.json"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "scope": f"{args.output_name} plus live nwp_tactical table state",
        "progress_json": json.loads(progress_json.read_text(encoding="utf-8")),
        "csv": read_csv_summary(results_csv),
        "api_log": read_api_log(api_log),
        "db": audit_db(args.database_url, args.skip_file_hash),
    }
    report_json.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")

    total_rows = report["db"]["table_counts"]["forecast_wide"]["rows"]
    full_rows = sum(row["rows"] for row in report["db"]["forecast_counts_by_source_dataset"] if row["source_scope"] == "full")
    print(
        json.dumps(
            {
                "report_json": str(report_json),
                "forecast_wide_rows": total_rows,
                "full_scope_rows": full_rows,
                "non_full_rows": total_rows - full_rows,
                "http_error_count": report["api_log"].get("http_error_count"),
                "missing_raw_files": len(report["db"]["full_raw_file_checks"]["missing_files"]),
                "raw_size_mismatches": len(report["db"]["full_raw_file_checks"]["size_mismatch"]),
                "raw_sha256_mismatches": len(report["db"]["full_raw_file_checks"]["sha256_mismatch"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
