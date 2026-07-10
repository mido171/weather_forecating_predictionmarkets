package com.predictionmarkets.weather.weathercom.service;

import com.predictionmarkets.weather.models.WeatherComIngestionRun;
import com.predictionmarkets.weather.models.WeatherComIngestionStatus;
import com.predictionmarkets.weather.weathercom.config.WeatherComBackfillRunnerProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(prefix = "weathercom.backfill-runner", name = "enabled", havingValue = "true")
public class WeatherComBackfillCommandLineRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(WeatherComBackfillCommandLineRunner.class);

  private final WeatherComBackfillRunnerProperties properties;
  private final WeatherComIngestionService ingestionService;

  public WeatherComBackfillCommandLineRunner(WeatherComBackfillRunnerProperties properties,
                                             WeatherComIngestionService ingestionService) {
    this.properties = properties;
    this.ingestionService = ingestionService;
  }

  @Override
  public void run(String... args) throws Exception {
    if (!properties.isEnabled()) {
      return;
    }
    if (properties.getStartDate() == null || properties.getEndDate() == null) {
      throw new IllegalArgumentException("weathercom.backfill-runner start-date and end-date are required");
    }

    WeatherComIngestionRun started = ingestionService.triggerIngestion(
        properties.getLocationIds(),
        properties.getStartDate(),
        properties.getEndDate(),
        properties.getUnits(),
        properties.getRequestedBy());
    Long runId = started.getId();
    int pollIntervalSeconds = Math.max(1, properties.getPollIntervalSeconds());

    logger.info(
        "weathercom backfill runner started runId={} locations={} startDate={} endDate={} units={}",
        runId,
        properties.getLocationIds() == null ? 0 : properties.getLocationIds().size(),
        properties.getStartDate(),
        properties.getEndDate(),
        properties.getUnits());

    while (true) {
      WeatherComIngestionRun run = ingestionService.getRun(runId);
      logger.info(
          "weathercom backfill runner poll runId={} status={} succeeded={} failed={} total={}",
          runId,
          run.getStatus(),
          run.getSucceededTasks(),
          run.getFailedTasks(),
          run.getTotalTasks());
      if (run.getStatus() != WeatherComIngestionStatus.RUNNING) {
        if (properties.isFailOnNonSucceeded()
            && run.getStatus() != WeatherComIngestionStatus.SUCCEEDED) {
          throw new IllegalStateException(
              "WeatherCom backfill finished with non-success status " + run.getStatus()
                  + " (runId=" + runId + ")");
        }
        return;
      }
      Thread.sleep(pollIntervalSeconds * 1000L);
    }
  }
}
