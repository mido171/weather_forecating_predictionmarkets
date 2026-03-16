package com.predictionmarkets.weather.pilot.catalog;

import com.predictionmarkets.weather.pilot.metrics.JobMetricsAccumulator;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class JobRunService {
  private static final Logger logger = LoggerFactory.getLogger(JobRunService.class);

  private final SqliteCatalogService catalogService;

  public JobRunService(SqliteCatalogService catalogService) {
    this.catalogService = catalogService;
  }

  public String startRun(String jobId, String stationKey) {
    String runId = UUID.randomUUID().toString();
    catalogService.execute("""
        INSERT INTO ingest_job_run (
          run_id, job_id, station_key, started_at_utc, status
        ) VALUES (?, ?, ?, ?, ?)
        """,
        runId,
        jobId,
        stationKey,
        Instant.now().toString(),
        "RUNNING");
    logEvent(Map.of(
        "timestamp", Instant.now().toString(),
        "level", "INFO",
        "job_id", jobId,
        "run_id", runId,
        "station_key", stationKey,
        "action", "job_start",
        "status", "RUNNING"));
    return runId;
  }

  public void completeRun(String runId,
                          String jobId,
                          String stationKey,
                          String status,
                          JobMetricsAccumulator metricsAccumulator) {
    catalogService.execute("""
        UPDATE ingest_job_run
        SET finished_at_utc = ?,
            status = ?,
            total_requests = ?,
            succeeded = ?,
            failed = ?,
            skipped = ?,
            deduped = ?,
            total_bytes = ?,
            total_rows = ?,
            min_request_duration_ms = ?,
            mean_request_duration_ms = ?,
            p95_request_duration_ms = ?,
            throughput_mb_s = ?,
            parser_failures_json = ?,
            missing_counts_json = ?,
            summary_json = ?
        WHERE run_id = ?
        """,
        Instant.now().toString(),
        status,
        metricsAccumulator.totalRequests(),
        metricsAccumulator.succeeded(),
        metricsAccumulator.failed(),
        metricsAccumulator.skipped(),
        metricsAccumulator.deduped(),
        metricsAccumulator.totalBytes(),
        metricsAccumulator.totalRows(),
        metricsAccumulator.minRequestDurationMs(),
        metricsAccumulator.meanRequestDurationMs(),
        metricsAccumulator.p95RequestDurationMs(),
        metricsAccumulator.throughputMbPerSec(),
        catalogService.toJson(metricsAccumulator.parserFailures()),
        catalogService.toJson(metricsAccumulator.missingCounts()),
        catalogService.toJson(metricsAccumulator.toSummaryMap()),
        runId);
    logEvent(Map.of(
        "timestamp", Instant.now().toString(),
        "level", "INFO",
        "job_id", jobId,
        "run_id", runId,
        "station_key", stationKey,
        "action", "job_complete",
        "status", status,
        "metrics", metricsAccumulator.toSummaryMap()));
  }

  public void logEvent(Map<String, ?> payload) {
    logger.info(catalogService.toJson(payload));
  }

  public void logStructuredEvent(String jobId,
                                 String runId,
                                 String stationKey,
                                 String sourceName,
                                 String action,
                                 String status,
                                 Map<String, Object> extras) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("timestamp", Instant.now().toString());
    payload.put("level", "INFO");
    payload.put("job_id", jobId);
    payload.put("run_id", runId);
    payload.put("station_key", stationKey);
    payload.put("source_name", sourceName);
    payload.put("action", action);
    payload.put("status", status);
    if (extras != null) {
      payload.putAll(extras);
    }
    logEvent(payload);
  }
}
