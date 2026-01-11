package com.predictionmarkets.weather.models;

import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;

@Entity
@Table(name = "kalshi_series")
public class KalshiSeries {
  @Id
  @Column(name = "series_ticker", nullable = false, length = 32)
  private String seriesTicker;

  @Column(name = "title", nullable = false, length = 256)
  private String title;

  @Column(name = "category", nullable = false, length = 64)
  private String category;

  @Column(name = "settlement_source_name", nullable = false, length = 128)
  private String settlementSourceName;

  @Column(name = "settlement_source_url", nullable = false, length = 1024)
  private String settlementSourceUrl;

  @Column(name = "contract_terms_url", length = 1024)
  private String contractTermsUrl;

  @Column(name = "contract_url", length = 1024)
  private String contractUrl;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

  @Column(name = "raw_payload_hash", nullable = false, length = 64)
  private String rawPayloadHash;

  @Lob
  @Column(name = "raw_json")
  private String rawJson;

  public String getSeriesTicker() {
    return seriesTicker;
  }

  public void setSeriesTicker(String seriesTicker) {
    this.seriesTicker = seriesTicker;
  }

  public String getTitle() {
    return title;
  }

  public void setTitle(String title) {
    this.title = title;
  }

  public String getCategory() {
    return category;
  }

  public void setCategory(String category) {
    this.category = category;
  }

  public String getSettlementSourceName() {
    return settlementSourceName;
  }

  public void setSettlementSourceName(String settlementSourceName) {
    this.settlementSourceName = settlementSourceName;
  }

  public String getSettlementSourceUrl() {
    return settlementSourceUrl;
  }

  public void setSettlementSourceUrl(String settlementSourceUrl) {
    this.settlementSourceUrl = settlementSourceUrl;
  }

  public String getContractTermsUrl() {
    return contractTermsUrl;
  }

  public void setContractTermsUrl(String contractTermsUrl) {
    this.contractTermsUrl = contractTermsUrl;
  }

  public String getContractUrl() {
    return contractUrl;
  }

  public void setContractUrl(String contractUrl) {
    this.contractUrl = contractUrl;
  }

  public Instant getRetrievedAtUtc() {
    return retrievedAtUtc;
  }

  public void setRetrievedAtUtc(Instant retrievedAtUtc) {
    this.retrievedAtUtc = retrievedAtUtc;
  }

  public String getRawPayloadHash() {
    return rawPayloadHash;
  }

  public void setRawPayloadHash(String rawPayloadHash) {
    this.rawPayloadHash = rawPayloadHash;
  }

  public String getRawJson() {
    return rawJson;
  }

  public void setRawJson(String rawJson) {
    this.rawJson = rawJson;
  }
}
