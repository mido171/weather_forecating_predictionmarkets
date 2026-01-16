package com.predictionmarkets.weather.models;

import java.math.BigDecimal;
import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;

@Entity
@Table(
    name = "mos_forecast_value",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {
            "station_id",
            "model",
            "runtime_utc",
            "forecast_time_utc",
            "variable_code"
        })
    }
)
public class MosForecastValue {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Column(name = "model", nullable = false, length = 16)
  private String model;

  @Column(name = "runtime_utc", nullable = false)
  private Instant runtimeUtc;

  @Column(name = "forecast_time_utc", nullable = false)
  private Instant forecastTimeUtc;

  @Column(name = "variable_code", nullable = false, length = 16)
  private String variableCode;

  @Column(name = "value_num", precision = 10, scale = 4)
  private BigDecimal valueNum;

  @Column(name = "value_text", length = 64)
  private String valueText;

  @Column(name = "value_raw", nullable = false, length = 128)
  private String valueRaw;

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

  public String getModel() {
    return model;
  }

  public void setModel(String model) {
    this.model = model;
  }

  public Instant getRuntimeUtc() {
    return runtimeUtc;
  }

  public void setRuntimeUtc(Instant runtimeUtc) {
    this.runtimeUtc = runtimeUtc;
  }

  public Instant getForecastTimeUtc() {
    return forecastTimeUtc;
  }

  public void setForecastTimeUtc(Instant forecastTimeUtc) {
    this.forecastTimeUtc = forecastTimeUtc;
  }

  public String getVariableCode() {
    return variableCode;
  }

  public void setVariableCode(String variableCode) {
    this.variableCode = variableCode;
  }

  public BigDecimal getValueNum() {
    return valueNum;
  }

  public void setValueNum(BigDecimal valueNum) {
    this.valueNum = valueNum;
  }

  public String getValueText() {
    return valueText;
  }

  public void setValueText(String valueText) {
    this.valueText = valueText;
  }

  public String getValueRaw() {
    return valueRaw;
  }

  public void setValueRaw(String valueRaw) {
    this.valueRaw = valueRaw;
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
