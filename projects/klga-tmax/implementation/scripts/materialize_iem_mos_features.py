"""Materialize KLGA IEM MOS cutoff-safe gold features from persisted rows."""

from __future__ import annotations

import argparse
import os
import time

import psycopg


DEFAULT_DB_URL = "postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="klga_iem_mos_full_backfill_v1")
    parser.add_argument("--cutoff-id", default="T_1245UTC")
    parser.add_argument("--start-date", default="2003-12-16")
    parser.add_argument("--through-date", default="2026-06-28")
    parser.add_argument("--db-url", default=os.environ.get("KLGA_DB_URL", DEFAULT_DB_URL))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def ensure_target_instances(cur: psycopg.Cursor, cutoff_id: str, start_date: str, through_date: str) -> int:
    cur.execute(
        """
        WITH days AS (
          SELECT generate_series(%s::date, %s::date, interval '1 day')::date AS target_date
        )
        INSERT INTO gold.target_instances (
          target_date, cutoff_id, cutoff_utc, local_day_start_utc, local_day_end_utc
        )
        SELECT
          target_date,
          %s,
          (target_date::timestamp + time '12:45:00') AT TIME ZONE 'UTC',
          target_date::timestamp AT TIME ZONE 'America/New_York',
          (target_date + 1)::timestamp AT TIME ZONE 'America/New_York'
        FROM days
        ON CONFLICT (target_date, cutoff_id) DO NOTHING
        """,
        (start_date, through_date, cutoff_id),
    )
    return cur.rowcount


def reset_gold(cur: psycopg.Cursor, cutoff_id: str, job_id: str) -> tuple[int, int]:
    cur.execute("DELETE FROM gold.iem_mos_feature_matrix_v1 WHERE cutoff_id = %s", (cutoff_id,))
    matrix_deleted = cur.rowcount
    cur.execute(
        """
        DELETE FROM gold.iem_mos_daily_features
        WHERE cutoff_id = %s
          AND feature_build_version = 'iem_mos_daily_features_v1'
        """,
        (cutoff_id,),
    )
    daily_deleted = cur.rowcount
    cur.execute(
        """
        UPDATE audit.iem_mos_backfill_chunks
        SET feature_rows_upserted = 0,
            updated_at = now()
        WHERE job_id = %s
        """,
        (job_id,),
    )
    return daily_deleted, matrix_deleted


def load_chunks(
    cur: psycopg.Cursor,
    job_id: str,
    missing_only: bool,
) -> list[tuple[str, str, str, str, str, str, str, str]]:
    cur.execute(
        """
        SELECT
          chunk_id,
          station_id,
          source_product,
          cutoff_id,
          window_start_utc::date AS start_date,
          (window_end_utc::date - 1) AS end_date,
          request_sha256,
          job_id
        FROM audit.iem_mos_backfill_chunks
        WHERE job_id = %s
          AND status = 'completed'
          AND (%s = false OR feature_rows_upserted = 0)
        ORDER BY source_product, station_id, window_start_utc
        """,
        (job_id, missing_only),
    )
    return list(cur.fetchall())


def materialize_chunk(cur: psycopg.Cursor, chunk: tuple[str, str, str, str, str, str, str, str]) -> int:
    chunk_id, station_id, source_product, cutoff_id, start_date, end_date, request_sha256, job_id = chunk
    cur.execute(
        """
        WITH target_days AS (
          SELECT
            ti.target_instance_id,
            ti.target_date,
            ti.cutoff_utc,
            ti.local_day_start_utc,
            ti.local_day_end_utc
          FROM gold.target_instances ti
          WHERE ti.cutoff_id = %(cutoff_id)s
            AND ti.target_date BETWEEN CAST(%(start_date)s AS date) AND CAST(%(end_date)s AS date)
        ),
        bounds AS (
          SELECT
            min(local_day_start_utc) AS min_valid_utc,
            max(local_day_end_utc) AS max_valid_utc
          FROM target_days
        ),
        latest_runtime AS (
          SELECT
            td.target_date,
            max(r.run_time_utc) AS chosen_run_time_utc
          FROM target_days td
          JOIN bounds b ON true
          JOIN silver.iem_mos_forecast_rows r
            ON r.station_id = %(station_id)s
           AND r.source_product = %(source_product)s
           AND r.forecast_valid_time_utc >= b.min_valid_utc
           AND r.forecast_valid_time_utc < b.max_valid_utc
           AND r.effective_available_at_utc <= td.cutoff_utc
           AND r.forecast_valid_time_utc >= td.local_day_start_utc
           AND r.forecast_valid_time_utc < td.local_day_end_utc
          GROUP BY td.target_date
        ),
        rows_for_latest AS (
          SELECT td.*, r.*
          FROM target_days td
          JOIN latest_runtime lr
            ON lr.target_date = td.target_date
          JOIN silver.iem_mos_forecast_rows r
            ON r.station_id = %(station_id)s
           AND r.source_product = %(source_product)s
           AND r.run_time_utc = lr.chosen_run_time_utc
           AND r.forecast_valid_time_utc >= td.local_day_start_utc
           AND r.forecast_valid_time_utc < td.local_day_end_utc
        ),
        aggregated AS (
          SELECT
            target_instance_id,
            target_date,
            max(run_time_utc) AS chosen_run_time_utc,
            max(forecast_valid_time_utc) AS latest_valid_time_utc,
            max(effective_available_at_utc) AS max_source_available_at_utc,
            max(availability_method) AS availability_method,
            max(n_x_f) AS tmax_f,
            max(tmp_f) FILTER (
              WHERE EXTRACT(HOUR FROM forecast_valid_time_utc AT TIME ZONE 'America/New_York')
                    BETWEEN 12 AND 21
            ) AS tmp_peak_window_max_f,
            avg(tmp_f) FILTER (
              WHERE EXTRACT(HOUR FROM forecast_valid_time_utc AT TIME ZONE 'America/New_York')
                    BETWEEN 12 AND 21
            ) AS tmp_peak_window_mean_f,
            avg(dpt_f) FILTER (
              WHERE EXTRACT(HOUR FROM forecast_valid_time_utc AT TIME ZONE 'America/New_York')
                    BETWEEN 12 AND 21
            ) AS dpt_peak_window_mean_f,
            avg(wsp_kt) FILTER (
              WHERE EXTRACT(HOUR FROM forecast_valid_time_utc AT TIME ZONE 'America/New_York')
                    BETWEEN 12 AND 21
            ) AS wind_speed_peak_window_mean_kt,
            max(pop) AS pop_max,
            max(qpf) AS qpf_max,
            max(tstm_prob) AS tstm_prob_max,
            count(*) AS source_row_count
          FROM rows_for_latest
          GROUP BY target_instance_id, target_date
        )
        INSERT INTO gold.iem_mos_daily_features (
          target_date, cutoff_id, target_instance_id, station_id, source_product,
          chosen_run_time_utc, latest_valid_time_utc, max_source_available_at_utc,
          availability_method, tmax_f, tmp_peak_window_max_f, tmp_peak_window_mean_f,
          dpt_peak_window_mean_f, wind_speed_peak_window_mean_kt, pop_max, qpf_max,
          tstm_prob_max, source_row_count, source_trace_json, feature_build_version
        )
        SELECT
          target_date, %(cutoff_id)s, target_instance_id, %(station_id)s, %(source_product)s,
          chosen_run_time_utc, latest_valid_time_utc, max_source_available_at_utc,
          availability_method, tmax_f, tmp_peak_window_max_f, tmp_peak_window_mean_f,
          dpt_peak_window_mean_f, wind_speed_peak_window_mean_kt, pop_max, qpf_max,
          tstm_prob_max, source_row_count,
          jsonb_build_object(
            'job_id', CAST(%(job_id)s AS text),
            'chunk_id', CAST(%(chunk_id)s AS text),
            'request_sha256', CAST(%(request_sha256)s AS text),
            'availability_rule', 'runtime_plus_2h',
            'materializer', 'materialize_iem_mos_features.py'
          ),
          'iem_mos_daily_features_v1'
        FROM aggregated
        ON CONFLICT (target_date, cutoff_id, station_id, source_product, feature_build_version)
        DO UPDATE SET
          target_instance_id = EXCLUDED.target_instance_id,
          chosen_run_time_utc = EXCLUDED.chosen_run_time_utc,
          latest_valid_time_utc = EXCLUDED.latest_valid_time_utc,
          max_source_available_at_utc = EXCLUDED.max_source_available_at_utc,
          availability_method = EXCLUDED.availability_method,
          tmax_f = EXCLUDED.tmax_f,
          tmp_peak_window_max_f = EXCLUDED.tmp_peak_window_max_f,
          tmp_peak_window_mean_f = EXCLUDED.tmp_peak_window_mean_f,
          dpt_peak_window_mean_f = EXCLUDED.dpt_peak_window_mean_f,
          wind_speed_peak_window_mean_kt = EXCLUDED.wind_speed_peak_window_mean_kt,
          pop_max = EXCLUDED.pop_max,
          qpf_max = EXCLUDED.qpf_max,
          tstm_prob_max = EXCLUDED.tstm_prob_max,
          source_row_count = EXCLUDED.source_row_count,
          source_trace_json = EXCLUDED.source_trace_json,
          updated_at = now()
        """,
        {
            "chunk_id": chunk_id,
            "station_id": station_id,
            "source_product": source_product,
            "cutoff_id": cutoff_id,
            "start_date": start_date,
            "end_date": end_date,
            "request_sha256": request_sha256,
            "job_id": job_id,
        },
    )
    rowcount = max(cur.rowcount, 0)
    cur.execute(
        """
        UPDATE audit.iem_mos_backfill_chunks
        SET feature_rows_upserted = %s,
            updated_at = now()
        WHERE chunk_id = %s
        """,
        (rowcount, chunk_id),
    )
    return rowcount


def rebuild_matrix(cur: psycopg.Cursor, cutoff_id: str, start_date: str, through_date: str) -> int:
    cur.execute(
        """
        WITH kv AS (
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_tmax_f') AS feature_name,
                 to_jsonb(tmax_f) AS feature_value,
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = %s
            AND target_date BETWEEN %s::date AND %s::date
            AND tmax_f IS NOT NULL
          UNION ALL
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_tmp_peak_window_max_f'),
                 to_jsonb(tmp_peak_window_max_f),
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = %s
            AND target_date BETWEEN %s::date AND %s::date
            AND tmp_peak_window_max_f IS NOT NULL
          UNION ALL
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_dpt_peak_window_mean_f'),
                 to_jsonb(dpt_peak_window_mean_f),
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = %s
            AND target_date BETWEEN %s::date AND %s::date
            AND dpt_peak_window_mean_f IS NOT NULL
          UNION ALL
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_wind_speed_peak_window_mean_kt'),
                 to_jsonb(wind_speed_peak_window_mean_kt),
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = %s
            AND target_date BETWEEN %s::date AND %s::date
            AND wind_speed_peak_window_mean_kt IS NOT NULL
        ),
        aggregated AS (
          SELECT
            target_instance_id,
            target_date,
            cutoff_id,
            jsonb_object_agg(feature_name, feature_value ORDER BY feature_name) AS feature_vector_json,
            jsonb_object_agg(feature_name, source_trace_json ORDER BY feature_name) AS feature_trace_json,
            count(*) AS source_feature_count
          FROM kv
          GROUP BY target_instance_id, target_date, cutoff_id
        )
        INSERT INTO gold.iem_mos_feature_matrix_v1 (
          target_instance_id, target_date, cutoff_id, feature_vector_json,
          feature_trace_json, source_feature_count
        )
        SELECT target_instance_id, target_date, cutoff_id, feature_vector_json,
               feature_trace_json, source_feature_count
        FROM aggregated
        ON CONFLICT (target_instance_id) DO UPDATE SET
          feature_vector_json = EXCLUDED.feature_vector_json,
          feature_trace_json = EXCLUDED.feature_trace_json,
          source_feature_count = EXCLUDED.source_feature_count,
          updated_at = now()
        """,
        (
            cutoff_id, start_date, through_date,
            cutoff_id, start_date, through_date,
            cutoff_id, start_date, through_date,
            cutoff_id, start_date, through_date,
        ),
    )
    return max(cur.rowcount, 0)


def refresh_job_summary(cur: psycopg.Cursor, job_id: str) -> None:
    cur.execute(
        """
        WITH stats AS (
          SELECT
            count(*)::integer AS planned_chunks,
            count(*) FILTER (WHERE status = 'completed')::integer AS completed_chunks,
            count(*) FILTER (WHERE status = 'completed_empty')::integer AS completed_empty_chunks,
            count(*) FILTER (WHERE status = 'failed')::integer AS failed_chunks,
            COALESCE(sum(rows_upserted), 0)::bigint AS rows_upserted,
            COALESCE(sum(feature_rows_upserted), 0)::bigint AS feature_rows_upserted,
            COALESCE(sum(response_size_bytes), 0)::bigint AS bytes_fetched,
            count(*) FILTER (WHERE status IN ('planned','running','rate_limited'))::integer AS remaining_chunks
          FROM audit.iem_mos_backfill_chunks
          WHERE job_id = %s
        )
        UPDATE audit.iem_mos_backfill_jobs j
        SET planned_chunks = stats.planned_chunks,
            completed_chunks = stats.completed_chunks,
            completed_empty_chunks = stats.completed_empty_chunks,
            failed_chunks = stats.failed_chunks,
            rows_upserted = stats.rows_upserted,
            feature_rows_upserted = stats.feature_rows_upserted,
            bytes_fetched = stats.bytes_fetched,
            status = CASE
              WHEN stats.remaining_chunks = 0 AND stats.failed_chunks = 0 THEN 'completed'
              WHEN stats.failed_chunks > 0 THEN 'failed'
              ELSE j.status
            END,
            finished_at_utc = now(),
            updated_at = now()
        FROM stats
        WHERE j.job_id = %s
        """,
        (job_id, job_id),
    )


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    with psycopg.connect(args.db_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            target_rows = ensure_target_instances(cur, args.cutoff_id, args.start_date, args.through_date)
            conn.commit()
            print(f"target_instances_inserted={target_rows}")
            if args.reset:
                daily_deleted, matrix_deleted = reset_gold(cur, args.cutoff_id, args.job_id)
                conn.commit()
                print(f"reset_daily_features={daily_deleted} reset_feature_matrix={matrix_deleted}")
            chunks = load_chunks(cur, args.job_id, args.missing_only)
            print(f"chunks_to_materialize={len(chunks)}")
            total = 0
            for index, chunk in enumerate(chunks, start=1):
                rows = materialize_chunk(cur, chunk)
                total += rows
                conn.commit()
                if index == 1 or index % args.progress_every == 0 or index == len(chunks):
                    elapsed = time.monotonic() - started
                    print(
                        f"daily_feature_progress={index}/{len(chunks)} "
                        f"last_rows={rows} total_rows={total} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
            matrix_rows = rebuild_matrix(cur, args.cutoff_id, args.start_date, args.through_date)
            refresh_job_summary(cur, args.job_id)
            conn.commit()
            print(f"matrix_rows_upserted={matrix_rows}")
            print(f"daily_feature_rows_upserted={total}")
            print(f"elapsed_s={time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
