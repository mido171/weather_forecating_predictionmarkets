package com.predictionmarkets.weather.models;

import java.time.Instant;
import java.time.LocalDate;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;

@Entity
@Table(
    name = "gribstream_daily_feature",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {
            "station_id", "target_date_local", "asof_utc", "model_code", "metric"
        })
    }
)
public class GribstreamDailyFeatureEntity {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "station_id", nullable = false, length = 32)
  private String stationId;

  @Column(name = "zone_id", nullable = false, length = 64)
  private String zoneId;

  @Column(name = "target_date_local", nullable = false)
  private LocalDate targetDateLocal;

  @Column(name = "asof_utc", nullable = false)
  private Instant asofUtc;

  @Column(name = "model_code", nullable = false, length = 32)
  private String modelCode;

  @Enumerated(EnumType.STRING)
  @Column(name = "metric", nullable = false, length = 64)
  private GribstreamMetric metric;

  @Column(name = "value_f")
  private Double valueF;

  @Column(name = "value_k")
  private Double valueK;

  @Column(name = "source_forecasted_at_utc")
  private Instant sourceForecastedAtUtc;

  @Column(name = "window_start_utc", nullable = false)
  private Instant windowStartUtc;

  @Column(name = "window_end_utc", nullable = false)
  private Instant windowEndUtc;

  @Column(name = "min_horizon_hours", nullable = false)
  private int minHorizonHours;

  @Column(name = "max_horizon_hours", nullable = false)
  private int maxHorizonHours;

  @Lob
  @Column(name = "request_json", nullable = false)
  private String requestJson;

  @Column(name = "request_sha256", nullable = false, length = 64)
  private String requestSha256;

  @Column(name = "response_sha256", nullable = false, length = 64)
  private String responseSha256;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

  @Column(name = "notes", length = 512)
  private String notes;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public String getZoneId() {
    return zoneId;
  }

  public void setZoneId(String zoneId) {
    this.zoneId = zoneId;
  }

  public LocalDate getTargetDateLocal() {
    return targetDateLocal;
  }

  public void setTargetDateLocal(LocalDate targetDateLocal) {
    this.targetDateLocal = targetDateLocal;
  }

  public Instant getAsofUtc() {
    return asofUtc;
  }

  public void setAsofUtc(Instant asofUtc) {
    this.asofUtc = asofUtc;
  }

  public String getModelCode() {
    return modelCode;
  }

  public void setModelCode(String modelCode) {
    this.modelCode = modelCode;
  }

  public GribstreamMetric getMetric() {
    return metric;
  }

  public void setMetric(GribstreamMetric metric) {
    this.metric = metric;
  }

  public Double getValueF() {
    return valueF;
  }

  public void setValueF(Double valueF) {
    this.valueF = valueF;
  }

  public Double getValueK() {
    return valueK;
  }

  public void setValueK(Double valueK) {
    this.valueK = valueK;
  }

  public Instant getSourceForecastedAtUtc() {
    return sourceForecastedAtUtc;
  }

  public void setSourceForecastedAtUtc(Instant sourceForecastedAtUtc) {
    this.sourceForecastedAtUtc = sourceForecastedAtUtc;
  }

  public Instant getWindowStartUtc() {
    return windowStartUtc;
  }

  public void setWindowStartUtc(Instant windowStartUtc) {
    this.windowStartUtc = windowStartUtc;
  }

  public Instant getWindowEndUtc() {
    return windowEndUtc;
  }

  public void setWindowEndUtc(Instant windowEndUtc) {
    this.windowEndUtc = windowEndUtc;
  }

  public int getMinHorizonHours() {
    return minHorizonHours;
  }

  public void setMinHorizonHours(int minHorizonHours) {
    this.minHorizonHours = minHorizonHours;
  }

  public int getMaxHorizonHours() {
    return maxHorizonHours;
  }

  public void setMaxHorizonHours(int maxHorizonHours) {
    this.maxHorizonHours = maxHorizonHours;
  }

  public String getRequestJson() {
    return requestJson;
  }

  public void setRequestJson(String requestJson) {
    this.requestJson = requestJson;
  }

  public String getRequestSha256() {
    return requestSha256;
  }

  public void setRequestSha256(String requestSha256) {
    this.requestSha256 = requestSha256;
  }

  public String getResponseSha256() {
    return responseSha256;
  }

  public void setResponseSha256(String responseSha256) {
    this.responseSha256 = responseSha256;
  }

  public Instant getRetrievedAtUtc() {
    return retrievedAtUtc;
  }

  public void setRetrievedAtUtc(Instant retrievedAtUtc) {
    this.retrievedAtUtc = retrievedAtUtc;
  }

  public String getNotes() {
    return notes;
  }

  public void setNotes(String notes) {
    this.notes = notes;
  }
}
