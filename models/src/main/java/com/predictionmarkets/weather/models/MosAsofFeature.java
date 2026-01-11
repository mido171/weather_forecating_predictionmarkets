package com.predictionmarkets.weather.models;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "mos_asof_feature")
public class MosAsofFeature {
  @EmbeddedId
  private MosAsofFeatureId id;

  @Column(name = "asof_utc", nullable = false)
  private Instant asofUtc;

  @Column(name = "asof_local", nullable = false)
  private LocalDateTime asofLocal;

  @Column(name = "station_zoneid", nullable = false, length = 64)
  private String stationZoneid;

  @Column(name = "chosen_runtime_utc")
  private Instant chosenRuntimeUtc;

  @Column(name = "tmax_f", precision = 5, scale = 2)
  private BigDecimal tmaxF;

  @Column(name = "missing_reason", length = 128)
  private String missingReason;

  @Column(name = "raw_payload_hash_ref", length = 64)
  private String rawPayloadHashRef;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

  public MosAsofFeatureId getId() {
    return id;
  }

  public void setId(MosAsofFeatureId id) {
    this.id = id;
  }

  public Instant getAsofUtc() {
    return asofUtc;
  }

  public void setAsofUtc(Instant asofUtc) {
    this.asofUtc = asofUtc;
  }

  public LocalDateTime getAsofLocal() {
    return asofLocal;
  }

  public void setAsofLocal(LocalDateTime asofLocal) {
    this.asofLocal = asofLocal;
  }

  public String getStationZoneid() {
    return stationZoneid;
  }

  public void setStationZoneid(String stationZoneid) {
    this.stationZoneid = stationZoneid;
  }

  public Instant getChosenRuntimeUtc() {
    return chosenRuntimeUtc;
  }

  public void setChosenRuntimeUtc(Instant chosenRuntimeUtc) {
    this.chosenRuntimeUtc = chosenRuntimeUtc;
  }

  public BigDecimal getTmaxF() {
    return tmaxF;
  }

  public void setTmaxF(BigDecimal tmaxF) {
    this.tmaxF = tmaxF;
  }

  public String getMissingReason() {
    return missingReason;
  }

  public void setMissingReason(String missingReason) {
    this.missingReason = missingReason;
  }

  public String getRawPayloadHashRef() {
    return rawPayloadHashRef;
  }

  public void setRawPayloadHashRef(String rawPayloadHashRef) {
    this.rawPayloadHashRef = rawPayloadHashRef;
  }

  public Instant getRetrievedAtUtc() {
    return retrievedAtUtc;
  }

  public void setRetrievedAtUtc(Instant retrievedAtUtc) {
    this.retrievedAtUtc = retrievedAtUtc;
  }
}
