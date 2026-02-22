package com.predictionmarkets.weather.experiments;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

@Component
@Order(1)
public class ModelExperimentIngestCommandLineRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(ModelExperimentIngestCommandLineRunner.class);

  private final ModelExperimentIngestService ingestService;
  private final ModelExperimentIngestProperties properties;

  public ModelExperimentIngestCommandLineRunner(ModelExperimentIngestService ingestService,
                                                ModelExperimentIngestProperties properties) {
    this.ingestService = ingestService;
    this.properties = properties;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    ModelExperimentIngestReport report = ingestService.ingestAll();
    logger.info("Experiment ingest complete: {}", report);
  }
}
