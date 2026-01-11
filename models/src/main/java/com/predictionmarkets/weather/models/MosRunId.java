package com.predictionmarkets.weather.models;

import java.io.Serializable;
import java.time.Instant;
import java.util.Objects;
import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;

@Embeddable
public class MosRunId implements Serializable {
  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Enumerated(EnumType.STRING)
  @Column(name = "model", nullable = false, length = 16)
  private MosModel model;

  @Column(name = "runtime_utc", nullable = false)
  private Instant runtimeUtc;

  public MosRunId() {
  }

  public MosRunId(String stationId, MosModel model, Instant runtimeUtc) {
    this.stationId = stationId;
    this.model = model;
    this.runtimeUtc = runtimeUtc;
  }

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public MosModel getModel() {
    return model;
  }

  public void setModel(MosModel model) {
    this.model = model;
  }

  public Instant getRuntimeUtc() {
    return runtimeUtc;
  }

  public void setRuntimeUtc(Instant runtimeUtc) {
    this.runtimeUtc = runtimeUtc;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    MosRunId mosRunId = (MosRunId) o;
    return Objects.equals(stationId, mosRunId.stationId)
        && model == mosRunId.model
        && Objects.equals(runtimeUtc, mosRunId.runtimeUtc);
  }

  @Override
  public int hashCode() {
    return Objects.hash(stationId, model, runtimeUtc);
  }
}
