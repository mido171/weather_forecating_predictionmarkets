package com.predictionmarkets.weather.klga.iemmos;

import java.nio.file.Path;
import java.time.LocalDate;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "iem-mos")
public class IemMosBackfillProperties {
  private boolean enabled;
  private String jobId = "klga_iem_mos_full_backfill_v1";
  private String cutoffId = "T_1245UTC";
  private LocalDate start;
  private LocalDate through = LocalDate.of(2026, 6, 28);
  private int threads = 10;
  private long requestSpacingMs = 1100L;
  private boolean resume = true;
  private int maxAttempts = 5;
  private long retryBackoffMs = 2000L;
  private String mode = "full";
  private boolean buildDailyFeatures = true;
  private boolean buildFeatureMatrix = false;
  private List<String> stationIds = List.of();
  private List<String> products = List.of();
  private Path rawOutputRoot = Path.of(
      "bootstrap/klga_tmax/implementation/artifacts/klga_tmax/iem_mos/raw");

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getJobId() {
    return jobId;
  }

  public void setJobId(String jobId) {
    this.jobId = jobId;
  }

  public String getCutoffId() {
    return cutoffId;
  }

  public void setCutoffId(String cutoffId) {
    this.cutoffId = cutoffId;
  }

  public LocalDate getStart() {
    return start;
  }

  public void setStart(LocalDate start) {
    this.start = start;
  }

  public LocalDate getThrough() {
    return through;
  }

  public void setThrough(LocalDate through) {
    this.through = through;
  }

  public int getThreads() {
    return threads;
  }

  public void setThreads(int threads) {
    this.threads = threads;
  }

  public long getRequestSpacingMs() {
    return requestSpacingMs;
  }

  public void setRequestSpacingMs(long requestSpacingMs) {
    this.requestSpacingMs = requestSpacingMs;
  }

  public boolean isResume() {
    return resume;
  }

  public void setResume(boolean resume) {
    this.resume = resume;
  }

  public int getMaxAttempts() {
    return maxAttempts;
  }

  public void setMaxAttempts(int maxAttempts) {
    this.maxAttempts = maxAttempts;
  }

  public long getRetryBackoffMs() {
    return retryBackoffMs;
  }

  public void setRetryBackoffMs(long retryBackoffMs) {
    this.retryBackoffMs = retryBackoffMs;
  }

  public String getMode() {
    return mode;
  }

  public void setMode(String mode) {
    this.mode = mode;
  }

  public boolean isBuildDailyFeatures() {
    return buildDailyFeatures;
  }

  public void setBuildDailyFeatures(boolean buildDailyFeatures) {
    this.buildDailyFeatures = buildDailyFeatures;
  }

  public boolean isBuildFeatureMatrix() {
    return buildFeatureMatrix;
  }

  public void setBuildFeatureMatrix(boolean buildFeatureMatrix) {
    this.buildFeatureMatrix = buildFeatureMatrix;
  }

  public List<String> getStationIds() {
    return stationIds;
  }

  public void setStationIds(List<String> stationIds) {
    this.stationIds = stationIds == null ? List.of() : List.copyOf(stationIds);
  }

  public List<String> getProducts() {
    return products;
  }

  public void setProducts(List<String> products) {
    this.products = products == null ? List.of() : List.copyOf(products);
  }

  public Path getRawOutputRoot() {
    return rawOutputRoot;
  }

  public void setRawOutputRoot(Path rawOutputRoot) {
    this.rawOutputRoot = rawOutputRoot;
  }
}
