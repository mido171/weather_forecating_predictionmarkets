package com.predictionmarkets.weather.executors;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream.training-data")
public class GribstreamTrainingDataProperties {
  private String outputPath =
      "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets"
          + "\\weather_forecating_predictionmarkets\\ingestion-service\\src\\main\\resources"
          + "\\trainingdata_output\\gribstream_training_data.csv";
  private int pageSize = 5000;
  private boolean append = true;
  private Long mosAsofPolicyId = 2L;

  public String getOutputPath() {
    return outputPath;
  }

  public void setOutputPath(String outputPath) {
    this.outputPath = outputPath;
  }

  public int getPageSize() {
    return pageSize;
  }

  public void setPageSize(int pageSize) {
    this.pageSize = pageSize;
  }

  public boolean isAppend() {
    return append;
  }

  public void setAppend(boolean append) {
    this.append = append;
  }

  public Long getMosAsofPolicyId() {
    return mosAsofPolicyId;
  }

  public void setMosAsofPolicyId(Long mosAsofPolicyId) {
    this.mosAsofPolicyId = mosAsofPolicyId;
  }
}
