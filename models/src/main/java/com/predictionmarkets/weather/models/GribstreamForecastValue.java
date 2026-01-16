package com.predictionmarkets.weather.models;

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
    name = "gribstream_forecast_value",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {
            "station_id",
            "model_code",
            "asof_utc",
            "forecasted_at_utc",
            "forecasted_time_utc",
            "variable_alias",
            "member"
        })
    }
)
public class GribstreamForecastValue {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "station_id", nullable = false, length = 32)
  private String stationId;

  @Column(name = "zone_id", nullable = false, length = 64)
  private String zoneId;

  @Column(name = "model_code", nullable = false, length = 32)
  private String modelCode;

  @Column(name = "asof_utc", nullable = false)
  private Instant asofUtc;

  @Column(name = "forecasted_at_utc", nullable = false)
  private Instant forecastedAtUtc;

  @Column(name = "forecasted_time_utc", nullable = false)
  private Instant forecastedTimeUtc;

  @Column(name = "member")
  private Integer member;

  @Column(name = "variable_name", nullable = false, length = 32)
  private String variableName;

  @Column(name = "variable_level", nullable = false, length = 64)
  private String variableLevel;

  @Column(name = "variable_info", nullable = false, length = 64)
  private String variableInfo;

  @Column(name = "variable_alias", nullable = false, length = 64)
  private String variableAlias;

  @Column(name = "value_num")
  private Double valueNum;

  @Column(name = "value_text", length = 128)
  private String valueText;

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

  public String getModelCode() {
    return modelCode;
  }

  public void setModelCode(String modelCode) {
    this.modelCode = modelCode;
  }

  public Instant getAsofUtc() {
    return asofUtc;
  }

  public void setAsofUtc(Instant asofUtc) {
    this.asofUtc = asofUtc;
  }

  public Instant getForecastedAtUtc() {
    return forecastedAtUtc;
  }

  public void setForecastedAtUtc(Instant forecastedAtUtc) {
    this.forecastedAtUtc = forecastedAtUtc;
  }

  public Instant getForecastedTimeUtc() {
    return forecastedTimeUtc;
  }

  public void setForecastedTimeUtc(Instant forecastedTimeUtc) {
    this.forecastedTimeUtc = forecastedTimeUtc;
  }

  public Integer getMember() {
    return member;
  }

  public void setMember(Integer member) {
    this.member = member;
  }

  public String getVariableName() {
    return variableName;
  }

  public void setVariableName(String variableName) {
    this.variableName = variableName;
  }

  public String getVariableLevel() {
    return variableLevel;
  }

  public void setVariableLevel(String variableLevel) {
    this.variableLevel = variableLevel;
  }

  public String getVariableInfo() {
    return variableInfo;
  }

  public void setVariableInfo(String variableInfo) {
    this.variableInfo = variableInfo;
  }

  public String getVariableAlias() {
    return variableAlias;
  }

  public void setVariableAlias(String variableAlias) {
    this.variableAlias = variableAlias;
  }

  public Double getValueNum() {
    return valueNum;
  }

  public void setValueNum(Double valueNum) {
    this.valueNum = valueNum;
  }

  public String getValueText() {
    return valueText;
  }

  public void setValueText(String valueText) {
    this.valueText = valueText;
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
