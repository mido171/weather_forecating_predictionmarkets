package com.predictionmarkets.weather.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream.cli-export")
public class GribstreamCliExportProperties {
  private String outputPath =
      "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets"
          + "\\weather_forecating_predictionmarkets\\ml\\data\\gribstream"
          + "\\gribstream_cli_training_data.csv";
  private int pageSize = 5000;
  private boolean append = false;
  private String stationId = "KMIA";

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

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }
}
