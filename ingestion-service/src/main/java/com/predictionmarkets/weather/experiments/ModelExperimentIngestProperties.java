package com.predictionmarkets.weather.experiments;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "experiments.ingest")
public class ModelExperimentIngestProperties {
  private boolean enabled;
  private String repoRoot;
  private List<String> scanRoots = List.of("artifacts", "ml");
  private List<String> stationFilter = List.of("KMIA");
  private long maxMetadataBytes = 1_000_000L;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getRepoRoot() {
    return repoRoot;
  }

  public void setRepoRoot(String repoRoot) {
    this.repoRoot = repoRoot;
  }

  public List<String> getScanRoots() {
    return scanRoots;
  }

  public void setScanRoots(List<String> scanRoots) {
    this.scanRoots = scanRoots;
  }

  public List<String> getStationFilter() {
    return stationFilter;
  }

  public void setStationFilter(List<String> stationFilter) {
    this.stationFilter = stationFilter;
  }

  public long getMaxMetadataBytes() {
    return maxMetadataBytes;
  }

  public void setMaxMetadataBytes(long maxMetadataBytes) {
    this.maxMetadataBytes = maxMetadataBytes;
  }
}
