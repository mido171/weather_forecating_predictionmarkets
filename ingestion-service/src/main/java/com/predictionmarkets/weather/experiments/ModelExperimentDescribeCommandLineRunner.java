package com.predictionmarkets.weather.experiments;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

@Component
@Order(3)
public class ModelExperimentDescribeCommandLineRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(ModelExperimentDescribeCommandLineRunner.class);

  private final ModelExperimentDescriptionService descriptionService;
  private final ModelExperimentDescribeProperties properties;

  public ModelExperimentDescribeCommandLineRunner(ModelExperimentDescriptionService descriptionService,
                                                  ModelExperimentDescribeProperties properties) {
    this.descriptionService = descriptionService;
    this.properties = properties;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }

    ModelExperimentDescriptionService.DescriptionSnapshot snapshot =
        descriptionService.writeDescriptionsSnapshot(properties.getOutputDir(), properties.getLimit());
    logger.info("Description snapshot written to {} (markdown {})",
        snapshot.ndjsonPath(), snapshot.markdownPath());

    if (!properties.isApplyToDatabase()) {
      logger.info("applyToDatabase=false; skipping DB update");
      return;
    }

    int updated = descriptionService.applyDescriptionsSnapshot(snapshot.ndjsonPath());
    logger.info("Description refresh complete. Updated {} rows", updated);
  }
}
