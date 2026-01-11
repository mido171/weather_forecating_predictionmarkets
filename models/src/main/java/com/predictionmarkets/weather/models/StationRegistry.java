package com.predictionmarkets.weather.models;

import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "station_registry")
public class StationRegistry {
  @Id
  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Column(name = "issuedby", nullable = false, length = 8)
  private String issuedby;

  @Column(name = "wfo_site", nullable = false, length = 8)
  private String wfoSite;

  @Column(name = "series_ticker", nullable = false, length = 32)
  private String seriesTicker;

  @Column(name = "zone_id", nullable = false, length = 64)
  private String zoneId;

  @Column(name = "standard_offset_minutes", nullable = false)
  private Integer standardOffsetMinutes;

  @Enumerated(EnumType.STRING)
  @Column(name = "mapping_status", nullable = false, length = 32)
  private StationMappingStatus mappingStatus;

  @Column(name = "created_at_utc", nullable = false)
  private Instant createdAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public String getIssuedby() {
    return issuedby;
  }

  public void setIssuedby(String issuedby) {
    this.issuedby = issuedby;
  }

  public String getWfoSite() {
    return wfoSite;
  }

  public void setWfoSite(String wfoSite) {
    this.wfoSite = wfoSite;
  }

  public String getSeriesTicker() {
    return seriesTicker;
  }

  public void setSeriesTicker(String seriesTicker) {
    this.seriesTicker = seriesTicker;
  }

  public String getZoneId() {
    return zoneId;
  }

  public void setZoneId(String zoneId) {
    this.zoneId = zoneId;
  }

  public Integer getStandardOffsetMinutes() {
    return standardOffsetMinutes;
  }

  public void setStandardOffsetMinutes(Integer standardOffsetMinutes) {
    this.standardOffsetMinutes = standardOffsetMinutes;
  }

  public StationMappingStatus getMappingStatus() {
    return mappingStatus;
  }

  public void setMappingStatus(StationMappingStatus mappingStatus) {
    this.mappingStatus = mappingStatus;
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
