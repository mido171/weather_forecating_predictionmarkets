package com.predictionmarkets.weather.models;

import java.time.Instant;
import java.time.LocalDate;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "weathercom_api_call")
public class WeatherComApiCall {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @JoinColumn(name = "ingestion_run_id", nullable = false)
  private WeatherComIngestionRun ingestionRun;

  @Column(name = "request_location_id", nullable = false, length = 64)
  private String requestLocationId;

  @Column(name = "units", nullable = false, length = 1)
  private String units;

  @Column(name = "start_date", nullable = false)
  private LocalDate startDate;

  @Column(name = "end_date", nullable = false)
  private LocalDate endDate;

  @Column(name = "response_location_id", length = 64)
  private String responseLocationId;

  @Column(name = "response_units", length = 1)
  private String responseUnits;

  @Column(name = "response_language", length = 32)
  private String responseLanguage;

  @Column(name = "transaction_id", length = 128)
  private String transactionId;

  @Column(name = "api_version", length = 32)
  private String apiVersion;

  @Column(name = "expire_time_gmt")
  private Long expireTimeGmt;

  @Column(name = "http_status", nullable = false)
  private int httpStatus;

  @Column(name = "fetched_at_utc", nullable = false)
  private Instant fetchedAtUtc;

  @Column(name = "duration_ms")
  private Integer durationMs;

  @Column(name = "error_type", length = 64)
  private String errorType;

  @Lob
  @Column(name = "error_message")
  private String errorMessage;

  @Lob
  @Column(name = "response_body_json")
  private String responseBodyJson;

  @Column(name = "response_body_hash", length = 64)
  private String responseBodyHash;

  @Column(name = "created_at_utc", nullable = false)
  private Instant createdAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public WeatherComIngestionRun getIngestionRun() {
    return ingestionRun;
  }

  public void setIngestionRun(WeatherComIngestionRun ingestionRun) {
    this.ingestionRun = ingestionRun;
  }

  public String getRequestLocationId() {
    return requestLocationId;
  }

  public void setRequestLocationId(String requestLocationId) {
    this.requestLocationId = requestLocationId;
  }

  public String getUnits() {
    return units;
  }

  public void setUnits(String units) {
    this.units = units;
  }

  public LocalDate getStartDate() {
    return startDate;
  }

  public void setStartDate(LocalDate startDate) {
    this.startDate = startDate;
  }

  public LocalDate getEndDate() {
    return endDate;
  }

  public void setEndDate(LocalDate endDate) {
    this.endDate = endDate;
  }

  public String getResponseLocationId() {
    return responseLocationId;
  }

  public void setResponseLocationId(String responseLocationId) {
    this.responseLocationId = responseLocationId;
  }

  public String getResponseUnits() {
    return responseUnits;
  }

  public void setResponseUnits(String responseUnits) {
    this.responseUnits = responseUnits;
  }

  public String getResponseLanguage() {
    return responseLanguage;
  }

  public void setResponseLanguage(String responseLanguage) {
    this.responseLanguage = responseLanguage;
  }

  public String getTransactionId() {
    return transactionId;
  }

  public void setTransactionId(String transactionId) {
    this.transactionId = transactionId;
  }

  public String getApiVersion() {
    return apiVersion;
  }

  public void setApiVersion(String apiVersion) {
    this.apiVersion = apiVersion;
  }

  public Long getExpireTimeGmt() {
    return expireTimeGmt;
  }

  public void setExpireTimeGmt(Long expireTimeGmt) {
    this.expireTimeGmt = expireTimeGmt;
  }

  public int getHttpStatus() {
    return httpStatus;
  }

  public void setHttpStatus(int httpStatus) {
    this.httpStatus = httpStatus;
  }

  public Instant getFetchedAtUtc() {
    return fetchedAtUtc;
  }

  public void setFetchedAtUtc(Instant fetchedAtUtc) {
    this.fetchedAtUtc = fetchedAtUtc;
  }

  public Integer getDurationMs() {
    return durationMs;
  }

  public void setDurationMs(Integer durationMs) {
    this.durationMs = durationMs;
  }

  public String getErrorType() {
    return errorType;
  }

  public void setErrorType(String errorType) {
    this.errorType = errorType;
  }

  public String getErrorMessage() {
    return errorMessage;
  }

  public void setErrorMessage(String errorMessage) {
    this.errorMessage = errorMessage;
  }

  public String getResponseBodyJson() {
    return responseBodyJson;
  }

  public void setResponseBodyJson(String responseBodyJson) {
    this.responseBodyJson = responseBodyJson;
  }

  public String getResponseBodyHash() {
    return responseBodyHash;
  }

  public void setResponseBodyHash(String responseBodyHash) {
    this.responseBodyHash = responseBodyHash;
  }

  public Instant getCreatedAtUtc() {
    return createdAtUtc;
  }

  public void setCreatedAtUtc(Instant createdAtUtc) {
    this.createdAtUtc = createdAtUtc;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }

  public void setUpdatedAtUtc(Instant updatedAtUtc) {
    this.updatedAtUtc = updatedAtUtc;
  }
}

