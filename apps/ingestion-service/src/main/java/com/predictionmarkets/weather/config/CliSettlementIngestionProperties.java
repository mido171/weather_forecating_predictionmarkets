package com.predictionmarkets.weather.config;

import java.time.LocalDate;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "cli-settlement")
public class CliSettlementIngestionProperties {
  private List<String> stationIds;
  private LocalDate startDateLocal;
  private LocalDate endDateLocal;
  private boolean ingestEnabled = true;
  private int sourceFetchThreads = 8;
  private Export export = new Export();

  public List<String> getStationIds() {
    return stationIds;
  }

  public void setStationIds(List<String> stationIds) {
    this.stationIds = stationIds;
  }

  public LocalDate getStartDateLocal() {
    return startDateLocal;
  }

  public void setStartDateLocal(LocalDate startDateLocal) {
    this.startDateLocal = startDateLocal;
  }

  public LocalDate getEndDateLocal() {
    return endDateLocal;
  }

  public void setEndDateLocal(LocalDate endDateLocal) {
    this.endDateLocal = endDateLocal;
  }

  public boolean isIngestEnabled() {
    return ingestEnabled;
  }

  public void setIngestEnabled(boolean ingestEnabled) {
    this.ingestEnabled = ingestEnabled;
  }

  public int getSourceFetchThreads() {
    return sourceFetchThreads;
  }

  public void setSourceFetchThreads(int sourceFetchThreads) {
    this.sourceFetchThreads = sourceFetchThreads;
  }

  public Export getExport() {
    return export;
  }

  public void setExport(Export export) {
    this.export = export;
  }

  public static class Export {
    private boolean enabled = false;
    private String outputDir = "artifacts/cli_settlement_exports";
    private boolean includeHeader = true;

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

    public boolean isIncludeHeader() {
      return includeHeader;
    }

    public void setIncludeHeader(boolean includeHeader) {
      this.includeHeader = includeHeader;
    }
  }
}
