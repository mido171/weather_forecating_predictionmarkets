package com.predictionmarkets.weather.audit;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "audit")
public class AuditReportProperties {
  private boolean enabled;
  private String seriesTicker;
  private LocalDate dateStartLocal;
  private LocalDate dateEndLocal;
  private Long asofPolicyId;
  private List<MosModel> models = List.of(
      MosModel.GFS,
      MosModel.MEX,
      MosModel.NAM,
      MosModel.NBS,
      MosModel.NBE);
  private String outputDir = "reports";
  private int maxForecastDays = 10;
  private int sampleLimit = 10;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getSeriesTicker() {
    return seriesTicker;
  }

  public void setSeriesTicker(String seriesTicker) {
    this.seriesTicker = seriesTicker;
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

  public Long getAsofPolicyId() {
    return asofPolicyId;
  }

  public void setAsofPolicyId(Long asofPolicyId) {
    this.asofPolicyId = asofPolicyId;
  }

  public List<MosModel> getModels() {
    return models;
  }

  public void setModels(List<MosModel> models) {
    this.models = models;
  }

  public String getOutputDir() {
    return outputDir;
  }

  public void setOutputDir(String outputDir) {
    this.outputDir = outputDir;
  }

  public int getMaxForecastDays() {
    return maxForecastDays;
  }

  public void setMaxForecastDays(int maxForecastDays) {
    this.maxForecastDays = maxForecastDays;
  }

  public int getSampleLimit() {
    return sampleLimit;
  }

  public void setSampleLimit(int sampleLimit) {
    this.sampleLimit = sampleLimit;
  }
}
