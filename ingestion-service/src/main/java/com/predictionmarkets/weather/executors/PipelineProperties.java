package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "pipeline")
public class PipelineProperties {
  private List<String> seriesTickers;
  private LocalDate dateStartLocal;
  private LocalDate dateEndLocal;
  private List<MosModel> models;
  private int mosWindowDays;
  private int threadCount;
  private Long asofPolicyId;
  private String asofPolicyName;
  private LocalTime asofLocalTime;
  private int defaultRangeDays;
  private boolean resetCheckpoints;

  public List<String> getSeriesTickers() {
    return seriesTickers;
  }

  public void setSeriesTickers(List<String> seriesTickers) {
    this.seriesTickers = seriesTickers;
  }

  public LocalDate getDateStartLocal() {
    return dateStartLocal;
  }

  public void setDateStartLocal(LocalDate dateStartLocal) {
    this.dateStartLocal = dateStartLocal;
  }

  public LocalDate getDateEndLocal() {
    return dateEndLocal;
  }

  public void setDateEndLocal(LocalDate dateEndLocal) {
    this.dateEndLocal = dateEndLocal;
  }

  public List<MosModel> getModels() {
    return models;
  }

  public void setModels(List<MosModel> models) {
    this.models = models;
  }

  public int getMosWindowDays() {
    return mosWindowDays;
  }

  public void setMosWindowDays(int mosWindowDays) {
    this.mosWindowDays = mosWindowDays;
  }

  public int getThreadCount() {
    return threadCount;
  }

  public void setThreadCount(int threadCount) {
    this.threadCount = threadCount;
  }

  public Long getAsofPolicyId() {
    return asofPolicyId;
  }

  public void setAsofPolicyId(Long asofPolicyId) {
    this.asofPolicyId = asofPolicyId;
  }

  public String getAsofPolicyName() {
    return asofPolicyName;
  }

  public void setAsofPolicyName(String asofPolicyName) {
    this.asofPolicyName = asofPolicyName;
  }

  public LocalTime getAsofLocalTime() {
    return asofLocalTime;
  }

  public void setAsofLocalTime(LocalTime asofLocalTime) {
    this.asofLocalTime = asofLocalTime;
  }

  public int getDefaultRangeDays() {
    return defaultRangeDays;
  }

  public void setDefaultRangeDays(int defaultRangeDays) {
    this.defaultRangeDays = defaultRangeDays;
  }

  public boolean isResetCheckpoints() {
    return resetCheckpoints;
  }

  public void setResetCheckpoints(boolean resetCheckpoints) {
    this.resetCheckpoints = resetCheckpoints;
  }
}
