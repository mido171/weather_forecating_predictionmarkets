package com.predictionmarkets.weather.models;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;

@Entity
@Table(
    name = "mos_daily_value",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {
            "station_id",
            "model",
            "runtime_utc",
            "target_date_local",
            "variable_code"
        })
    }
)
public class MosDailyValue {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Column(name = "station_zoneid", nullable = false, length = 64)
  private String stationZoneid;

  @Column(name = "model", nullable = false, length = 16)
  private String model;

  @Column(name = "asof_utc")
  private Instant asofUtc;

  @Column(name = "runtime_utc", nullable = false)
  private Instant runtimeUtc;

  @Column(name = "target_date_local", nullable = false)
  private LocalDate targetDateLocal;

  @Column(name = "variable_code", nullable = false, length = 16)
  private String variableCode;

  @Column(name = "value_min", precision = 10, scale = 4)
  private BigDecimal valueMin;

  @Column(name = "value_max", precision = 10, scale = 4)
  private BigDecimal valueMax;

  @Column(name = "value_mean", precision = 10, scale = 4)
  private BigDecimal valueMean;

  @Column(name = "value_median", precision = 10, scale = 4)
  private BigDecimal valueMedian;

  @Column(name = "sample_count", nullable = false)
  private Integer sampleCount;

  @Column(name = "first_forecast_time_utc")
  private Instant firstForecastTimeUtc;

  @Column(name = "last_forecast_time_utc")
  private Instant lastForecastTimeUtc;

  @Column(name = "raw_payload_hash_ref", nullable = false, length = 64)
  private String rawPayloadHashRef;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

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

  public String getStationZoneid() {
    return stationZoneid;
  }

  public void setStationZoneid(String stationZoneid) {
    this.stationZoneid = stationZoneid;
  }

  public String getModel() {
    return model;
  }

  public void setModel(String model) {
    this.model = model;
  }

  public Instant getAsofUtc() {
    return asofUtc;
  }

  public void setAsofUtc(Instant asofUtc) {
    this.asofUtc = asofUtc;
  }

  public Instant getRuntimeUtc() {
    return runtimeUtc;
  }

  public void setRuntimeUtc(Instant runtimeUtc) {
    this.runtimeUtc = runtimeUtc;
  }

  public LocalDate getTargetDateLocal() {
    return targetDateLocal;
  }

  public void setTargetDateLocal(LocalDate targetDateLocal) {
    this.targetDateLocal = targetDateLocal;
  }

  public String getVariableCode() {
    return variableCode;
  }

  public void setVariableCode(String variableCode) {
    this.variableCode = variableCode;
  }

  public BigDecimal getValueMin() {
    return valueMin;
  }

  public void setValueMin(BigDecimal valueMin) {
    this.valueMin = valueMin;
  }

  public BigDecimal getValueMax() {
    return valueMax;
  }

  public void setValueMax(BigDecimal valueMax) {
    this.valueMax = valueMax;
  }

  public BigDecimal getValueMean() {
    return valueMean;
  }

  public void setValueMean(BigDecimal valueMean) {
    this.valueMean = valueMean;
  }

  public BigDecimal getValueMedian() {
    return valueMedian;
  }

  public void setValueMedian(BigDecimal valueMedian) {
    this.valueMedian = valueMedian;
  }

  public Integer getSampleCount() {
    return sampleCount;
  }

  public void setSampleCount(Integer sampleCount) {
    this.sampleCount = sampleCount;
  }

  public Instant getFirstForecastTimeUtc() {
    return firstForecastTimeUtc;
  }

  public void setFirstForecastTimeUtc(Instant firstForecastTimeUtc) {
    this.firstForecastTimeUtc = firstForecastTimeUtc;
  }

  public Instant getLastForecastTimeUtc() {
    return lastForecastTimeUtc;
  }

  public void setLastForecastTimeUtc(Instant lastForecastTimeUtc) {
    this.lastForecastTimeUtc = lastForecastTimeUtc;
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
