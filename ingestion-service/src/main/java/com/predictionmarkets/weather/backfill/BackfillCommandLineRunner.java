package com.predictionmarkets.weather.backfill;

import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class BackfillCommandLineRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(BackfillCommandLineRunner.class);

  private final BackfillOrchestrator orchestrator;
  private final BackfillProperties properties;

  public BackfillCommandLineRunner(BackfillOrchestrator orchestrator,
                                   BackfillProperties properties) {
    this.orchestrator = orchestrator;
    this.properties = properties;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    BackfillJobType jobType = BackfillJobType.fromJobName(properties.getJob());
    List<String> seriesTickers = splitSeriesTickers(properties.getSeriesTicker());
    BackfillRequest request = new BackfillRequest(
        jobType,
        seriesTickers,
        properties.getDateStartLocal(),
        properties.getDateEndLocal(),
        properties.getAsofPolicyId(),
        properties.getModels(),
        properties.getMosWindowDays());
    logger.info("Starting backfill job {}", jobType.jobName());
    orchestrator.run(request);
  }

  private List<String> splitSeriesTickers(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      return List.of();
    }
    String[] parts = seriesTicker.split(",");
    List<String> tickers = new ArrayList<>();
    for (String part : parts) {
      String trimmed = part.trim();
      if (!trimmed.isEmpty()) {
        tickers.add(trimmed);
      }
    }
    return List.copyOf(tickers);
  }
}
