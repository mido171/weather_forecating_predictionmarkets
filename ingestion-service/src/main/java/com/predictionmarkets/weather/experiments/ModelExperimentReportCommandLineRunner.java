package com.predictionmarkets.weather.experiments;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

@Component
@Order(2)
public class ModelExperimentReportCommandLineRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(ModelExperimentReportCommandLineRunner.class);

  private final ModelExperimentReportService reportService;
  private final ModelExperimentReportProperties properties;

  public ModelExperimentReportCommandLineRunner(ModelExperimentReportService reportService,
                                                ModelExperimentReportProperties properties) {
    this.reportService = reportService;
    this.properties = properties;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    reportService.writeAggregatedReport(properties);
    logger.info("Experiment report generation complete");
  }
}
