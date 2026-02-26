package com.predictionmarkets.weather.weathercom.client.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public class WeatherComResponseMetadata {
  @JsonProperty("id")
  private String locationId;

  @JsonProperty("key")
  private String locationKey;

  @JsonProperty("units")
  private String units;

  @JsonProperty("language")
  private String language;

  @JsonProperty("transaction_id")
  private String transactionId;

  @JsonProperty("version")
  private String version;

  @JsonProperty("expire_time_gmt")
  private Long expireTimeGmt;

  public String getLocationId() {
    return locationId;
  }

  public void setLocationId(String locationId) {
    this.locationId = locationId;
  }

  public String getLocationKey() {
    return locationKey;
  }

  public void setLocationKey(String locationKey) {
    this.locationKey = locationKey;
  }

  public String getUnits() {
    return units;
  }

  public void setUnits(String units) {
    this.units = units;
  }

  public String getLanguage() {
    return language;
  }

  public void setLanguage(String language) {
    this.language = language;
  }

  public String getTransactionId() {
    return transactionId;
  }

  public void setTransactionId(String transactionId) {
    this.transactionId = transactionId;
  }

  public String getVersion() {
    return version;
  }

  public void setVersion(String version) {
    this.version = version;
  }

  public Long getExpireTimeGmt() {
    return expireTimeGmt;
  }

  public void setExpireTimeGmt(Long expireTimeGmt) {
    this.expireTimeGmt = expireTimeGmt;
  }
}

