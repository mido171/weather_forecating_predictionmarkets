package com.predictionmarkets.weather.pilot.jobs;

import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.config.PilotConfigLoader;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class LivePollingJob {
  private final PilotConfigLoader configLoader;
  private final MetricsService metricsService;
  private final JobRunService jobRunService;

  public LivePollingJob(PilotConfigLoader configLoader,
                        MetricsService metricsService,
                        JobRunService jobRunService) {
    this.configLoader = configLoader;
    this.metricsService = metricsService;
    this.jobRunService = jobRunService;
  }

  public String run() {
    StationConfig station = configLoader.requireDefaultStation();
    String runId = jobRunService.startRun("livePollingJob", station.getStationKey());
    jobRunService.logStructuredEvent("livePollingJob", runId, station.getStationKey(),
        "livePollingJob", "live_polling_pending", "NOT_IMPLEMENTED",
        Map.of("message", "Live polling cadence definitions are staged, but the scheduler wiring is not finished yet."));
    jobRunService.completeRun(runId, "livePollingJob", station.getStationKey(),
        "NOT_IMPLEMENTED", metricsService.newAccumulator());
    return runId;
  }
}
