package com.predictionmarkets.weather.models;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "station_override")
public class StationOverride {
  @Id
  @Column(name = "issuedby", nullable = false, length = 8)
  private String issuedby;

  @Column(name = "station_id_override", nullable = false, length = 8)
  private String stationIdOverride;

  @Column(name = "zone_id_override", nullable = false, length = 64)
  private String zoneIdOverride;

  @Column(name = "notes", length = 512)
  private String notes;

  public String getIssuedby() {
    return issuedby;
  }

  public void setIssuedby(String issuedby) {
    this.issuedby = issuedby;
  }

  public String getStationIdOverride() {
    return stationIdOverride;
  }

  public void setStationIdOverride(String stationIdOverride) {
    this.stationIdOverride = stationIdOverride;
  }

  public String getZoneIdOverride() {
    return zoneIdOverride;
  }

  public void setZoneIdOverride(String zoneIdOverride) {
    this.zoneIdOverride = zoneIdOverride;
  }

  public String getNotes() {
    return notes;
  }

  public void setNotes(String notes) {
    this.notes = notes;
  }
}
