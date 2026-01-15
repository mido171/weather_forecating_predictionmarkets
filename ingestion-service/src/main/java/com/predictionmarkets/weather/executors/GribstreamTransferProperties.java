package com.predictionmarkets.weather.executors;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream.transfer")
public class GribstreamTransferProperties {
  private Export export = new Export();
  private Import importConfig = new Import();

  public Export getExport() {
    return export;
  }

  public void setExport(Export export) {
    this.export = export;
  }

  public Import getImport() {
    return importConfig;
  }

  public void setImport(Import importConfig) {
    this.importConfig = importConfig;
  }

  public static class Export {
    private String outputPath =
        "D:\\Ahmed\\git\\weather\\weather_forecating_predictionmarkets"
            + "\\ingestion-service\\src\\main\\resources\\gribstream_daily_feature.csv";
    private int pageSize = 5000;
    private boolean includeHeader = true;

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

    public boolean isIncludeHeader() {
      return includeHeader;
    }

    public void setIncludeHeader(boolean includeHeader) {
      this.includeHeader = includeHeader;
    }
  }

  public static class Import {
    private String inputPath =
        "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets"
            + "\\weather_forecating_predictionmarkets\\ingestion-service\\src\\main\\resources"
            + "\\gribstream_daily_feature.csv";
    private int batchSize = 1000;
    private boolean hasHeader = true;

    public String getInputPath() {
      return inputPath;
    }

    public void setInputPath(String inputPath) {
      this.inputPath = inputPath;
    }

    public int getBatchSize() {
      return batchSize;
    }

    public void setBatchSize(int batchSize) {
      this.batchSize = batchSize;
    }

    public boolean isHasHeader() {
      return hasHeader;
    }

    public void setHasHeader(boolean hasHeader) {
      this.hasHeader = hasHeader;
    }
  }
}
