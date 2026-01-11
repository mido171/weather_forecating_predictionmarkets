package com.predictionmarkets.weather.models;

import java.math.BigDecimal;
import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "cli_daily")
public class CliDaily {
  @EmbeddedId
  private CliDailyId id;

  @Column(name = "tmax_f", precision = 5, scale = 2)
  private BigDecimal tmaxF;

  @Column(name = "tmin_f", precision = 5, scale = 2)
  private BigDecimal tminF;

  @Column(name = "report_issued_at_utc")
  private Instant reportIssuedAtUtc;

  @Column(name = "raw_payload_hash", nullable = false, length = 64)
  private String rawPayloadHash;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public CliDailyId getId() {
    return id;
  }

  public void setId(CliDailyId id) {
    this.id = id;
  }

  public BigDecimal getTmaxF() {
    return tmaxF;
  }

  public void setTmaxF(BigDecimal tmaxF) {
    this.tmaxF = tmaxF;
  }

  public BigDecimal getTminF() {
    return tminF;
  }

  public void setTminF(BigDecimal tminF) {
    this.tminF = tminF;
  }

  public Instant getReportIssuedAtUtc() {
    return reportIssuedAtUtc;
  }

  public void setReportIssuedAtUtc(Instant reportIssuedAtUtc) {
    this.reportIssuedAtUtc = reportIssuedAtUtc;
  }

  public String getRawPayloadHash() {
    return rawPayloadHash;
  }

  public void setRawPayloadHash(String rawPayloadHash) {
    this.rawPayloadHash = rawPayloadHash;
  }

  public Instant getRetrievedAtUtc() {
    return retrievedAtUtc;
  }

  public void setRetrievedAtUtc(Instant retrievedAtUtc) {
    this.retrievedAtUtc = retrievedAtUtc;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }

  public void setUpdatedAtUtc(Instant updatedAtUtc) {
    this.updatedAtUtc = updatedAtUtc;
  }
}
