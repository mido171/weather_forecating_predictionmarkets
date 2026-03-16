package com.predictionmarkets.weather.pilot.metrics;

import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class MetricsService {
  private final SqliteCatalogService catalogService;

  public MetricsService(SqliteCatalogService catalogService) {
    this.catalogService = catalogService;
  }

  public JobMetricsAccumulator newAccumulator() {
    return new JobMetricsAccumulator();
  }

  public List<Map<String, Object>> recentJobRuns() {
    return catalogService.query("""
        SELECT run_id, job_id, station_key, started_at_utc, finished_at_utc, status,
               total_requests, succeeded, failed, skipped, deduped, total_bytes, total_rows,
               throughput_mb_s
        FROM ingest_job_run
        ORDER BY started_at_utc DESC
        LIMIT 100
        """);
  }

  public Map<String, Object> jobRun(String runId) {
    return catalogService.querySingle("""
        SELECT *
        FROM ingest_job_run
        WHERE run_id = ?
        """, runId);
  }
}
