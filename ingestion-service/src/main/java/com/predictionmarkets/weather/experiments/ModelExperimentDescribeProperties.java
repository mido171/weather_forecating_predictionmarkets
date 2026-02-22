package com.predictionmarkets.weather.experiments;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "experiments.describe")
public class ModelExperimentDescribeProperties {
  private boolean enabled;
  private int limit;
  private String outputDir = "artifacts/experiment_descriptions";
  private boolean applyToDatabase = true;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public int getLimit() {
    return limit;
  }

  public void setLimit(int limit) {
    this.limit = limit;
  }

  public String getOutputDir() {
    return outputDir;
  }

  public void setOutputDir(String outputDir) {
    this.outputDir = outputDir;
  }

  public boolean isApplyToDatabase() {
    return applyToDatabase;
  }

  public void setApplyToDatabase(boolean applyToDatabase) {
    this.applyToDatabase = applyToDatabase;
  }
}
