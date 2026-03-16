package com.predictionmarkets.weather.pilot.metrics;

import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class GapAuditService {
  private final SqliteCatalogService catalogService;

  public GapAuditService(SqliteCatalogService catalogService) {
    this.catalogService = catalogService;
  }

  public void recordGap(String runId,
                        String stationKey,
                        String sourceName,
                        String gapStartUtc,
                        String gapEndUtc,
                        Integer expectedCount,
                        Integer actualCount,
                        String status,
                        Map<String, Object> details) {
    catalogService.execute("""
        INSERT INTO source_gap_audit (
          gap_id, run_id, station_key, source_name, gap_start_utc, gap_end_utc,
          expected_count, actual_count, status, details_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        UUID.randomUUID().toString(),
        runId,
        stationKey,
        sourceName,
        gapStartUtc,
        gapEndUtc,
        expectedCount,
        actualCount,
        status,
        catalogService.toJson(details),
        Instant.now().toString());
  }
}
