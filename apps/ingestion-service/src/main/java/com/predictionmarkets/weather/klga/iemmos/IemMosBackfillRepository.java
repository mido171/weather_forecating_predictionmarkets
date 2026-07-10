package com.predictionmarkets.weather.klga.iemmos;

import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

@Repository
public class IemMosBackfillRepository {
  private static final int BATCH_SIZE = 1000;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public IemMosBackfillRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public List<IemMosStation> loadMosStations() {
    String sql = """
        SELECT station_id, mos_station_id
        FROM registry.station_registry
        WHERE mos_station_id IS NOT NULL
          AND mos_station_id <> ''
          AND role <> 'gridded_pseudo_point'
          AND station_registry_version = (
            SELECT max(station_registry_version) FROM registry.station_registry
          )
        ORDER BY
          CASE station_id
            WHEN 'KLGA' THEN 1
            WHEN 'KNYC' THEN 2
            WHEN 'KJFK' THEN 3
            WHEN 'KEWR' THEN 4
            WHEN 'KTEB' THEN 5
            ELSE 100
          END,
          station_id
        """;
    return jdbcTemplate.query(sql, Map.of(), (rs, rowNum) ->
        new IemMosStation(rs.getString("station_id"), rs.getString("mos_station_id")));
  }

  public boolean cutoffExists(String cutoffId) {
    Integer count = jdbcTemplate.queryForObject(
        "SELECT count(*) FROM registry.cutoffs WHERE cutoff_id = :cutoffId",
        Map.of("cutoffId", cutoffId),
        Integer.class);
    return count != null && count == 1;
  }

  @Transactional
  public void initializeJob(IemMosBackfillProperties properties, List<IemMosChunk> chunks) {
    if (chunks.isEmpty()) {
      throw new IllegalArgumentException("Cannot initialize IEM MOS job with no chunks");
    }
    LocalDate startDate = chunks.stream()
        .map(IemMosChunk::startDate)
        .min(LocalDate::compareTo)
        .orElseThrow();
    LocalDate endDate = chunks.stream()
        .map(IemMosChunk::endDateInclusive)
        .max(LocalDate::compareTo)
        .orElseThrow();
    String configJson = """
        {"source":"iem_mos","endpoint":"/cgi-bin/request/mos.py","threads":%d,"requestSpacingMs":%d,"resume":%s,"mode":"%s"}
        """.formatted(
            properties.getThreads(),
            properties.getRequestSpacingMs(),
            properties.isResume(),
            safe(properties.getMode()));
    jdbcTemplate.update(
        """
        INSERT INTO audit.iem_mos_backfill_jobs (
          job_id, cutoff_id, start_date, end_date, status, planned_chunks, config_json
        ) VALUES (
          :jobId, :cutoffId, :startDate, :endDate, 'planned', :plannedChunks, CAST(:configJson AS jsonb)
        )
        ON CONFLICT (job_id) DO UPDATE SET
          cutoff_id = EXCLUDED.cutoff_id,
          start_date = EXCLUDED.start_date,
          end_date = EXCLUDED.end_date,
          planned_chunks = EXCLUDED.planned_chunks,
          config_json = EXCLUDED.config_json,
          updated_at = now()
        """,
        new MapSqlParameterSource()
            .addValue("jobId", properties.getJobId())
            .addValue("cutoffId", properties.getCutoffId())
            .addValue("startDate", startDate)
            .addValue("endDate", endDate)
            .addValue("plannedChunks", chunks.size())
            .addValue("configJson", configJson));
    upsertPlannedChunks(chunks);
    if (properties.isResume()) {
      resetRunningChunks(properties.getJobId());
    }
    refreshJobSummary(properties.getJobId(), false);
  }

  public int upsertPlannedChunks(List<IemMosChunk> chunks) {
    if (chunks.isEmpty()) {
      return 0;
    }
    int updated = 0;
    for (int start = 0; start < chunks.size(); start += BATCH_SIZE) {
      List<IemMosChunk> slice = chunks.subList(start, Math.min(start + BATCH_SIZE, chunks.size()));
      SqlParameterSource[] params = slice.stream()
          .map(this::chunkParams)
          .toArray(SqlParameterSource[]::new);
      updated += Arrays.stream(jdbcTemplate.batchUpdate(
          """
          INSERT INTO audit.iem_mos_backfill_chunks (
            chunk_id, job_id, station_id, mos_station_id, source_product, endpoint_model,
            cutoff_id, window_start_utc, window_end_utc, request_sha256, request_json, status
          ) VALUES (
            :chunkId, :jobId, :stationId, :mosStationId, :sourceProduct, :endpointModel,
            :cutoffId, :windowStartUtc, :windowEndUtc, :requestSha256, CAST(:requestJson AS jsonb), 'planned'
          )
          ON CONFLICT (chunk_id) DO NOTHING
          """,
          params)).sum();
    }
    return updated;
  }

  public void resetRunningChunks(String jobId) {
    jdbcTemplate.update(
        """
        UPDATE audit.iem_mos_backfill_chunks
        SET status = 'planned',
            error_type = 'STALE_RUNNING_RESET',
            error_message = 'Reset from running during resume',
            updated_at = now()
        WHERE job_id = :jobId
          AND status = 'running'
        """,
        Map.of("jobId", jobId));
  }

  public List<IemMosChunk> chunksToRun(String jobId) {
    String sql = """
        SELECT chunk_id, job_id, station_id, mos_station_id, source_product, endpoint_model,
               cutoff_id, window_start_utc, window_end_utc, request_sha256, request_json::text
        FROM audit.iem_mos_backfill_chunks
        WHERE job_id = :jobId
          AND status IN ('planned','failed')
        ORDER BY
          CASE source_product
            WHEN 'MAV' THEN 1
            WHEN 'MET' THEN 2
            WHEN 'MEX' THEN 3
            WHEN 'LAV' THEN 4
            WHEN 'NBS' THEN 5
            WHEN 'NBE' THEN 6
            ELSE 99
          END,
          station_id,
          window_start_utc
        """;
    return jdbcTemplate.query(sql, Map.of("jobId", jobId), (rs, rowNum) -> {
      IemMosStation station = new IemMosStation(
          rs.getString("station_id"),
          rs.getString("mos_station_id"));
      IemMosProduct product = IemMosProduct.valueOf(rs.getString("source_product"));
      return new IemMosChunk(
          rs.getString("chunk_id"),
          rs.getString("job_id"),
          station,
          product,
          rs.getString("cutoff_id"),
          rs.getTimestamp("window_start_utc").toInstant(),
          rs.getTimestamp("window_end_utc").toInstant(),
          rs.getString("request_sha256"),
          rs.getString("request_json"));
    });
  }

  public void markJobRunning(String jobId) {
    jdbcTemplate.update(
        """
        UPDATE audit.iem_mos_backfill_jobs
        SET status = 'running',
            started_at_utc = COALESCE(started_at_utc, now()),
            updated_at = now()
        WHERE job_id = :jobId
        """,
        Map.of("jobId", jobId));
  }

  public void markChunkRunning(IemMosChunk chunk) {
    jdbcTemplate.update(
        """
        UPDATE audit.iem_mos_backfill_chunks
        SET status = 'running',
            attempts = attempts + 1,
            started_at_utc = now(),
            error_type = NULL,
            error_message = NULL,
            updated_at = now()
        WHERE chunk_id = :chunkId
        """,
        Map.of("chunkId", chunk.chunkId()));
  }

  @Transactional
  public IemMosStoredRequest persistSourceArtifacts(IemMosChunk chunk,
                                                    IemMosFetchResult result,
                                                    String rawStorageUri,
                                                    String responseSha256) {
    String sourceRequestId = "iem_mos_" + chunk.requestSha256();
    jdbcTemplate.update(
        """
        INSERT INTO bronze.source_requests (
          source_request_id, source_name, source_endpoint, request_method, request_params_json,
          request_headers_redacted, retrieved_at_utc, http_status, response_content_type,
          response_body_sha256, response_size_bytes, raw_storage_uri, parser_version
        ) VALUES (
          :sourceRequestId, 'iem_mos', '/cgi-bin/request/mos.py', 'GET', CAST(:requestJson AS jsonb),
          '{}'::jsonb, :retrievedAtUtc, :httpStatus, :contentType,
          :responseSha256, :responseSizeBytes, :rawStorageUri, :parserVersion
        )
        ON CONFLICT (source_request_id) DO UPDATE SET
          retrieved_at_utc = EXCLUDED.retrieved_at_utc,
          http_status = EXCLUDED.http_status,
          response_content_type = EXCLUDED.response_content_type,
          response_body_sha256 = EXCLUDED.response_body_sha256,
          response_size_bytes = EXCLUDED.response_size_bytes,
          raw_storage_uri = EXCLUDED.raw_storage_uri,
          parser_version = EXCLUDED.parser_version
        """,
        new MapSqlParameterSource()
            .addValue("sourceRequestId", sourceRequestId)
            .addValue("requestJson", chunk.requestJson())
            .addValue("retrievedAtUtc", Timestamp.from(result.retrievedAtUtc()))
            .addValue("httpStatus", result.httpStatus())
            .addValue("contentType", result.contentType())
            .addValue("responseSha256", responseSha256)
            .addValue("responseSizeBytes", result.body().length)
            .addValue("rawStorageUri", rawStorageUri)
            .addValue("parserVersion", IemMosParser.PARSER_VERSION));
    UUID sourceRecordId = jdbcTemplate.queryForObject(
        """
        INSERT INTO bronze.source_records (
          source_request_id, source_name, provider_name, endpoint_name, provider_record_key,
          request_hash, payload_hash, payload_format, payload_uri, acquired_at_utc, revision_number, is_current
        ) VALUES (
          :sourceRequestId, 'iem_mos', 'Iowa Environmental Mesonet', 'request/mos.py',
          :providerRecordKey, :requestSha256, :payloadHash, 'binary_uri', :payloadUri,
          :acquiredAtUtc, 1, true
        )
        ON CONFLICT (source_name, provider_name, endpoint_name, provider_record_key, revision_number)
        DO UPDATE SET
          source_request_id = EXCLUDED.source_request_id,
          request_hash = EXCLUDED.request_hash,
          payload_hash = EXCLUDED.payload_hash,
          payload_uri = EXCLUDED.payload_uri,
          acquired_at_utc = EXCLUDED.acquired_at_utc,
          is_current = true
        RETURNING source_record_id
        """,
        new MapSqlParameterSource()
            .addValue("sourceRequestId", sourceRequestId)
            .addValue("providerRecordKey", chunk.chunkId())
            .addValue("requestSha256", chunk.requestSha256())
            .addValue("payloadHash", responseSha256)
            .addValue("payloadUri", rawStorageUri)
            .addValue("acquiredAtUtc", Timestamp.from(result.retrievedAtUtc())),
        UUID.class);
    return new IemMosStoredRequest(
        sourceRequestId,
        sourceRecordId,
        rawStorageUri,
        responseSha256,
        result.body().length);
  }

  public int upsertForecastRows(List<IemMosForecastRow> rows) {
    if (rows.isEmpty()) {
      return 0;
    }
    Map<String, IemMosForecastRow> uniqueRowsByHash = new LinkedHashMap<>();
    for (IemMosForecastRow row : rows) {
      uniqueRowsByHash.put(row.rawRowHash(), row);
    }
    List<IemMosForecastRow> rowsToWrite = new ArrayList<>(uniqueRowsByHash.values());
    for (int start = 0; start < rowsToWrite.size(); start += BATCH_SIZE) {
      List<IemMosForecastRow> slice = rowsToWrite.subList(
          start,
          Math.min(start + BATCH_SIZE, rowsToWrite.size()));
      SqlParameterSource[] params = slice.stream()
          .map(this::forecastRowParams)
          .toArray(SqlParameterSource[]::new);
      jdbcTemplate.batchUpdate(
          """
          INSERT INTO silver.iem_mos_forecast_rows (
            station_id, mos_station_id, source_product, endpoint_model, cutoff_id,
            run_time_utc, forecast_valid_time_utc, forecast_hour, period_type,
            n_x_f, tmp_f, dpt_f, wdr, wsp_kt, gst_kt, sky_or_cloud, pop, qpf, tstm_prob,
            raw_values_jsonb, raw_payload_hash, provider_available_at_utc, effective_available_at_utc,
            availability_method, source_request_id, source_record_id, request_sha256,
            raw_row_hash, parser_version, quality_flag, quality_note
          ) VALUES (
            :stationId, :mosStationId, :sourceProduct, :endpointModel, :cutoffId,
            :runTimeUtc, :forecastValidTimeUtc, :forecastHour, :periodType,
            :nxF, :tmpF, :dptF, :wdr, :wspKt, :gstKt, :skyOrCloud, :pop, :qpf, :tstmProb,
            CAST(:rawValuesJson AS jsonb), :rawPayloadHash, :providerAvailableAtUtc, :effectiveAvailableAtUtc,
            :availabilityMethod, :sourceRequestId, :sourceRecordId, :requestSha256,
            :rawRowHash, :parserVersion, :qualityFlag, :qualityNote
          )
          ON CONFLICT (raw_row_hash) DO UPDATE SET
            n_x_f = EXCLUDED.n_x_f,
            tmp_f = EXCLUDED.tmp_f,
            dpt_f = EXCLUDED.dpt_f,
            wdr = EXCLUDED.wdr,
            wsp_kt = EXCLUDED.wsp_kt,
            gst_kt = EXCLUDED.gst_kt,
            sky_or_cloud = EXCLUDED.sky_or_cloud,
            pop = EXCLUDED.pop,
            qpf = EXCLUDED.qpf,
            tstm_prob = EXCLUDED.tstm_prob,
            source_request_id = EXCLUDED.source_request_id,
            source_record_id = EXCLUDED.source_record_id,
            updated_at = now()
          """,
          params);
    }
    return rowsToWrite.size();
  }

  public int materializeTargetInstances(String cutoffId, LocalDate startDate, LocalDate endDate) {
    return jdbcTemplate.update(
        """
        WITH days AS (
          SELECT generate_series(:startDate::date, :endDate::date, interval '1 day')::date AS target_date
        )
        INSERT INTO gold.target_instances (
          target_date, cutoff_id, cutoff_utc, local_day_start_utc, local_day_end_utc
        )
        SELECT
          target_date,
          :cutoffId,
          (target_date::timestamp + time '12:45:00') AT TIME ZONE 'UTC',
          target_date::timestamp AT TIME ZONE 'America/New_York',
          (target_date + 1)::timestamp AT TIME ZONE 'America/New_York'
        FROM days
        ON CONFLICT (target_date, cutoff_id) DO NOTHING
        """,
        new MapSqlParameterSource()
            .addValue("cutoffId", cutoffId)
            .addValue("startDate", startDate)
            .addValue("endDate", endDate));
  }

  public int rebuildDailyFeatures(IemMosChunk chunk) {
    return jdbcTemplate.update(
        """
        WITH target_days AS (
          SELECT
            ti.target_instance_id,
            ti.target_date,
            ti.cutoff_utc,
            ti.local_day_start_utc,
            ti.local_day_end_utc
          FROM gold.target_instances ti
          WHERE ti.cutoff_id = :cutoffId
            AND ti.target_date BETWEEN :startDate AND :endDate
        ),
        latest_runtime AS (
          SELECT
            td.target_date,
            max(r.run_time_utc) AS chosen_run_time_utc
          FROM target_days td
          JOIN silver.iem_mos_forecast_rows r
            ON r.station_id = :stationId
           AND r.source_product = :sourceProduct
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
            ON r.station_id = :stationId
           AND r.source_product = :sourceProduct
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
          target_date, :cutoffId, target_instance_id, :stationId, :sourceProduct,
          chosen_run_time_utc, latest_valid_time_utc, max_source_available_at_utc,
          availability_method, tmax_f, tmp_peak_window_max_f, tmp_peak_window_mean_f,
          dpt_peak_window_mean_f, wind_speed_peak_window_mean_kt, pop_max, qpf_max,
          tstm_prob_max, source_row_count,
          jsonb_build_object(
            'job_id', :jobId,
            'chunk_id', :chunkId,
            'request_sha256', :requestSha256,
            'availability_rule', 'runtime_plus_2h'
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
        new MapSqlParameterSource()
            .addValue("cutoffId", chunk.cutoffId())
            .addValue("startDate", chunk.startDate())
            .addValue("endDate", chunk.endDateInclusive())
            .addValue("stationId", chunk.station().stationId())
            .addValue("sourceProduct", chunk.product().productCode())
            .addValue("jobId", chunk.jobId())
            .addValue("chunkId", chunk.chunkId())
            .addValue("requestSha256", chunk.requestSha256()));
  }

  public int rebuildFeatureMatrix(String cutoffId, LocalDate startDate, LocalDate endDate) {
    return jdbcTemplate.update(
        """
        WITH kv AS (
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_tmax_f') AS feature_name,
                 to_jsonb(tmax_f) AS feature_value,
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = :cutoffId
            AND target_date BETWEEN :startDate AND :endDate
            AND tmax_f IS NOT NULL
          UNION ALL
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_tmp_peak_window_max_f'),
                 to_jsonb(tmp_peak_window_max_f),
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = :cutoffId
            AND target_date BETWEEN :startDate AND :endDate
            AND tmp_peak_window_max_f IS NOT NULL
          UNION ALL
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_dpt_peak_window_mean_f'),
                 to_jsonb(dpt_peak_window_mean_f),
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = :cutoffId
            AND target_date BETWEEN :startDate AND :endDate
            AND dpt_peak_window_mean_f IS NOT NULL
          UNION ALL
          SELECT target_instance_id, target_date, cutoff_id,
                 lower('mos_' || source_product || '_' || station_id || '_wind_speed_peak_window_mean_kt'),
                 to_jsonb(wind_speed_peak_window_mean_kt),
                 source_trace_json
          FROM gold.iem_mos_daily_features
          WHERE cutoff_id = :cutoffId
            AND target_date BETWEEN :startDate AND :endDate
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
        new MapSqlParameterSource()
            .addValue("cutoffId", cutoffId)
            .addValue("startDate", startDate)
            .addValue("endDate", endDate));
  }

  public void markChunkCompleted(IemMosChunk chunk,
                                 IemMosStoredRequest storedRequest,
                                 int rowsUpserted,
                                 int featureRowsUpserted) {
    jdbcTemplate.update(
        """
        UPDATE audit.iem_mos_backfill_chunks
        SET status = CASE WHEN :rowsUpserted = 0 THEN 'completed_empty' ELSE 'completed' END,
            http_status = 200,
            source_request_id = :sourceRequestId,
            source_record_id = :sourceRecordId,
            raw_storage_uri = :rawStorageUri,
            rows_upserted = :rowsUpserted,
            feature_rows_upserted = :featureRowsUpserted,
            response_size_bytes = :responseSizeBytes,
            finished_at_utc = now(),
            updated_at = now()
        WHERE chunk_id = :chunkId
        """,
        new MapSqlParameterSource()
            .addValue("chunkId", chunk.chunkId())
            .addValue("sourceRequestId", storedRequest.sourceRequestId())
            .addValue("sourceRecordId", storedRequest.sourceRecordId())
            .addValue("rawStorageUri", storedRequest.rawStorageUri())
            .addValue("rowsUpserted", rowsUpserted)
            .addValue("featureRowsUpserted", featureRowsUpserted)
            .addValue("responseSizeBytes", storedRequest.responseSizeBytes()));
  }

  public void markChunkFailed(IemMosChunk chunk,
                              String status,
                              Integer httpStatus,
                              String errorType,
                              String errorMessage) {
    jdbcTemplate.update(
        """
        UPDATE audit.iem_mos_backfill_chunks
        SET status = :status,
            http_status = :httpStatus,
            error_type = :errorType,
            error_message = :errorMessage,
            finished_at_utc = now(),
            updated_at = now()
        WHERE chunk_id = :chunkId
        """,
        new MapSqlParameterSource()
            .addValue("chunkId", chunk.chunkId())
            .addValue("status", status)
            .addValue("httpStatus", httpStatus)
            .addValue("errorType", errorType)
            .addValue("errorMessage", truncate(errorMessage, 4000)));
  }

  public int recordGap(IemMosChunk chunk,
                       String gapType,
                       String gapReason,
                       String evidenceJson) {
    return jdbcTemplate.update(
        """
        INSERT INTO audit.iem_mos_source_gaps (
          job_id, chunk_id, station_id, mos_station_id, source_product, endpoint_model,
          window_start_utc, window_end_utc, cutoff_id, gap_type, gap_reason, evidence_json
        ) VALUES (
          :jobId, :chunkId, :stationId, :mosStationId, :sourceProduct, :endpointModel,
          :windowStartUtc, :windowEndUtc, :cutoffId, :gapType, :gapReason, CAST(:evidenceJson AS jsonb)
        )
        ON CONFLICT DO NOTHING
        """,
        new MapSqlParameterSource()
            .addValue("jobId", chunk.jobId())
            .addValue("chunkId", chunk.chunkId())
            .addValue("stationId", chunk.station().stationId())
            .addValue("mosStationId", chunk.station().mosStationId())
            .addValue("sourceProduct", chunk.product().productCode())
            .addValue("endpointModel", chunk.product().endpointModel().name())
            .addValue("windowStartUtc", Timestamp.from(chunk.windowStartUtc()))
            .addValue("windowEndUtc", Timestamp.from(chunk.windowEndUtc()))
            .addValue("cutoffId", chunk.cutoffId())
            .addValue("gapType", gapType)
            .addValue("gapReason", gapReason)
            .addValue("evidenceJson", evidenceJson == null ? "{}" : evidenceJson));
  }

  public void refreshJobSummary(String jobId, boolean terminal) {
    jdbcTemplate.update(
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
          WHERE job_id = :jobId
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
              WHEN :terminal AND stats.remaining_chunks = 0 AND stats.failed_chunks = 0 THEN 'completed'
              WHEN :terminal AND stats.failed_chunks > 0 THEN 'failed'
              ELSE j.status
            END,
            finished_at_utc = CASE
              WHEN :terminal THEN now()
              ELSE j.finished_at_utc
            END,
            updated_at = now()
        FROM stats
        WHERE j.job_id = :jobId
        """,
        new MapSqlParameterSource()
            .addValue("jobId", jobId)
            .addValue("terminal", terminal));
  }

  public IemMosProgress progress(String jobId) {
    return jdbcTemplate.queryForObject(
        """
        SELECT
          count(*)::integer AS chunks_total,
          count(*) FILTER (WHERE status = 'completed')::integer AS completed,
          count(*) FILTER (WHERE status = 'completed_empty')::integer AS completed_empty,
          count(*) FILTER (WHERE status = 'failed')::integer AS failed,
          count(*) FILTER (WHERE status IN ('planned','running','rate_limited'))::integer AS remaining,
          COALESCE(sum(rows_upserted), 0)::bigint AS rows_upserted,
          COALESCE(sum(feature_rows_upserted), 0)::bigint AS feature_rows_upserted,
          COALESCE(sum(response_size_bytes), 0)::bigint AS bytes_fetched
        FROM audit.iem_mos_backfill_chunks
        WHERE job_id = :jobId
        """,
        Map.of("jobId", jobId),
        (rs, rowNum) -> new IemMosProgress(
            rs.getInt("chunks_total"),
            rs.getInt("completed"),
            rs.getInt("completed_empty"),
            rs.getInt("failed"),
            rs.getInt("remaining"),
            rs.getLong("rows_upserted"),
            rs.getLong("feature_rows_upserted"),
            rs.getLong("bytes_fetched")));
  }

  private SqlParameterSource chunkParams(IemMosChunk chunk) {
    return new MapSqlParameterSource()
        .addValue("chunkId", chunk.chunkId())
        .addValue("jobId", chunk.jobId())
        .addValue("stationId", chunk.station().stationId())
        .addValue("mosStationId", chunk.station().mosStationId())
        .addValue("sourceProduct", chunk.product().productCode())
        .addValue("endpointModel", chunk.product().endpointModel().name())
        .addValue("cutoffId", chunk.cutoffId())
        .addValue("windowStartUtc", Timestamp.from(chunk.windowStartUtc()))
        .addValue("windowEndUtc", Timestamp.from(chunk.windowEndUtc()))
        .addValue("requestSha256", chunk.requestSha256())
        .addValue("requestJson", chunk.requestJson());
  }

  private SqlParameterSource forecastRowParams(IemMosForecastRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId())
        .addValue("mosStationId", row.mosStationId())
        .addValue("sourceProduct", row.sourceProduct())
        .addValue("endpointModel", row.endpointModel())
        .addValue("cutoffId", row.cutoffId())
        .addValue("runTimeUtc", Timestamp.from(row.runTimeUtc()))
        .addValue("forecastValidTimeUtc", Timestamp.from(row.forecastValidTimeUtc()))
        .addValue("forecastHour", row.forecastHour())
        .addValue("periodType", row.periodType())
        .addValue("nxF", row.nxF())
        .addValue("tmpF", row.tmpF())
        .addValue("dptF", row.dptF())
        .addValue("wdr", row.wdr())
        .addValue("wspKt", row.wspKt())
        .addValue("gstKt", row.gstKt())
        .addValue("skyOrCloud", row.skyOrCloud())
        .addValue("pop", row.pop())
        .addValue("qpf", row.qpf())
        .addValue("tstmProb", row.tstmProb())
        .addValue("rawValuesJson", row.rawValuesJson())
        .addValue("rawPayloadHash", row.rawPayloadHash())
        .addValue("providerAvailableAtUtc", Timestamp.from(row.providerAvailableAtUtc()))
        .addValue("effectiveAvailableAtUtc", Timestamp.from(row.effectiveAvailableAtUtc()))
        .addValue("availabilityMethod", row.availabilityMethod())
        .addValue("sourceRequestId", row.sourceRequestId())
        .addValue("sourceRecordId", row.sourceRecordId())
        .addValue("requestSha256", row.requestSha256())
        .addValue("rawRowHash", row.rawRowHash())
        .addValue("parserVersion", row.parserVersion())
        .addValue("qualityFlag", row.qualityFlag())
        .addValue("qualityNote", row.qualityNote());
  }

  private static String safe(String value) {
    return value == null ? "" : value.replace("\"", "");
  }

  private static String truncate(String value, int maxLength) {
    if (value == null || value.length() <= maxLength) {
      return value;
    }
    return value.substring(0, maxLength);
  }
}
