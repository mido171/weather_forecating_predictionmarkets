package com.predictionmarkets.weather.pilot.api;

import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ingest")
public class IngestAdminController {
  private final MetricsService metricsService;
  private final SqliteCatalogService catalogService;

  public IngestAdminController(MetricsService metricsService, SqliteCatalogService catalogService) {
    this.metricsService = metricsService;
    this.catalogService = catalogService;
  }

  @GetMapping("/metrics")
  public Map<String, Object> metrics() {
    List<Map<String, Object>> jobs = metricsService.recentJobRuns();
    return Map.of(
        "recentJobs", jobs,
        "databasePath", catalogService.withConnection(connection -> connection.getMetaData().getURL()));
  }

  @GetMapping("/gaps")
  public List<Map<String, Object>> gaps() {
    return catalogService.query("""
        SELECT *
        FROM source_gap_audit
        ORDER BY created_at_utc DESC
        LIMIT 200
        """);
  }
}
