package com.predictionmarkets.weather.models;

import java.io.Serializable;
import java.time.LocalDate;
import java.util.Objects;
import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;

@Embeddable
public class MosAsofFeatureId implements Serializable {
  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Column(name = "target_date_local", nullable = false)
  private LocalDate targetDateLocal;

  @Column(name = "asof_policy_id", nullable = false)
  private Long asofPolicyId;

  @Enumerated(EnumType.STRING)
  @Column(name = "model", nullable = false, length = 16)
  private MosModel model;

  public MosAsofFeatureId() {
  }

  public MosAsofFeatureId(String stationId, LocalDate targetDateLocal, Long asofPolicyId, MosModel model) {
    this.stationId = stationId;
    this.targetDateLocal = targetDateLocal;
    this.asofPolicyId = asofPolicyId;
    this.model = model;
  }

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public LocalDate getTargetDateLocal() {
    return targetDateLocal;
  }

  public void setTargetDateLocal(LocalDate targetDateLocal) {
    this.targetDateLocal = targetDateLocal;
  }

  public Long getAsofPolicyId() {
    return asofPolicyId;
  }

  public void setAsofPolicyId(Long asofPolicyId) {
    this.asofPolicyId = asofPolicyId;
  }

  public MosModel getModel() {
    return model;
  }

  public void setModel(MosModel model) {
    this.model = model;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    MosAsofFeatureId that = (MosAsofFeatureId) o;
    return Objects.equals(stationId, that.stationId)
        && Objects.equals(targetDateLocal, that.targetDateLocal)
        && Objects.equals(asofPolicyId, that.asofPolicyId)
        && model == that.model;
  }

  @Override
  public int hashCode() {
    return Objects.hash(stationId, targetDateLocal, asofPolicyId, model);
  }
}
