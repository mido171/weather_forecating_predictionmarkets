package com.predictionmarkets.weather.backfill;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "backfill")
public class BackfillProperties {
  private boolean enabled;
  private String job;
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
  private int mosWindowDays = 1;

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getJob() {
    return job;
  }

  public void setJob(String job) {
    this.job = job;
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

  public int getMosWindowDays() {
    return mosWindowDays;
  }

  public void setMosWindowDays(int mosWindowDays) {
    this.mosWindowDays = mosWindowDays;
  }
}
