package com.predictionmarkets.weather.pilot.jobs;

import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.config.PilotConfigLoader;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class BackfillHeavyModelSourcesJob {
  private final PilotConfigLoader configLoader;
  private final MetricsService metricsService;
  private final JobRunService jobRunService;

  public BackfillHeavyModelSourcesJob(PilotConfigLoader configLoader,
                                      MetricsService metricsService,
                                      JobRunService jobRunService) {
    this.configLoader = configLoader;
    this.metricsService = metricsService;
    this.jobRunService = jobRunService;
  }

  public String run() {
    StationConfig station = configLoader.requireDefaultStation();
    String runId = jobRunService.startRun("backfillHeavyModelSourcesJob", station.getStationKey());
    jobRunService.logStructuredEvent("backfillHeavyModelSourcesJob", runId, station.getStationKey(),
        "heavy_model_workers", "heavy_model_backfill_pending", "NOT_IMPLEMENTED",
        Map.of("message", "Python worker orchestration scaffold added; extraction implementation still pending"));
    jobRunService.completeRun(runId, "backfillHeavyModelSourcesJob", station.getStationKey(),
        "NOT_IMPLEMENTED", metricsService.newAccumulator());
    return runId;
  }
}
