package com.predictionmarkets.weather.experiments;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "experiments.report")
public class ModelExperimentReportProperties {
  private boolean enabled;
  private String outputDir = "artifacts/experiment_reports";
  private List<String> stationFilter = List.of("KMIA");
  private int limit = 0;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getOutputDir() {
    return outputDir;
  }

  public void setOutputDir(String outputDir) {
    this.outputDir = outputDir;
  }

  public List<String> getStationFilter() {
    return stationFilter;
  }

  public void setStationFilter(List<String> stationFilter) {
    this.stationFilter = stationFilter;
  }

  public int getLimit() {
    return limit;
  }

  public void setLimit(int limit) {
    this.limit = limit;
  }
}
