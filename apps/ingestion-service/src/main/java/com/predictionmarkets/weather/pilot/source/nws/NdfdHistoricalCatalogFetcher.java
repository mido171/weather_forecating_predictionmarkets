package com.predictionmarkets.weather.pilot.source.nws;

import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.catalog.SourceInventoryRecord;
import com.predictionmarkets.weather.pilot.manifest.ManifestService;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class NdfdHistoricalCatalogFetcher {
  private final ManifestService manifestService;
  private final JobRunService jobRunService;

  public NdfdHistoricalCatalogFetcher(ManifestService manifestService, JobRunService jobRunService) {
    this.manifestService = manifestService;
    this.jobRunService = jobRunService;
  }

  public int markNotImplemented(String jobId, String runId, String stationKey, String targetDateLocal) {
    manifestService.upsertSourceInventory(new SourceInventoryRecord(
        UUID.randomUUID().toString(),
        stationKey,
        "ndfd_historical",
        "ncei",
        "target_day",
        targetDateLocal,
        null,
        null,
        "NOT_IMPLEMENTED",
        "{}",
        Instant.now().toString(),
        Instant.now().toString()));
    jobRunService.logStructuredEvent(jobId, runId, stationKey, "ndfd_historical",
        "historical_catalog_pending", "NOT_IMPLEMENTED", Map.of("targetDateLocal", targetDateLocal));
    return 0;
  }
}
