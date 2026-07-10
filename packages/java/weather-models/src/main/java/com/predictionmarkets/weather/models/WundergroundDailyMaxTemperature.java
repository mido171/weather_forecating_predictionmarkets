package com.predictionmarkets.weather.models;

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
    schema = "wunderground_ml",
    name = "wunderground_station_daily_max_temperature",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {"request_location_id", "obs_id", "target_date_local"})
    }
)
public class WundergroundDailyMaxTemperature {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "request_location_id", nullable = false, length = 64)
  private String requestLocationId;

  @Column(name = "obs_id", nullable = false, length = 32)
  private String obsId;

  @Column(name = "station_zoneid", nullable = false, length = 64)
  private String stationZoneid;

  @Column(name = "target_date_local", nullable = false)
  private LocalDate targetDateLocal;

  @Column(name = "max_temp_f", nullable = false)
  private Integer maxTempF;

  @Column(name = "source_valid_time_gmt")
  private Long sourceValidTimeGmt;

  @Column(name = "source_api_call_id")
  private Long sourceApiCallId;

  @Column(name = "source_type", nullable = false, length = 32)
  private String sourceType;

  @Column(name = "observation_count", nullable = false)
  private Integer observationCount;

  @Column(name = "created_at_utc", nullable = false)
  private Instant createdAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public Long getId() {
    return id;
  }

  public String getRequestLocationId() {
    return requestLocationId;
  }

  public String getObsId() {
    return obsId;
  }

  public String getStationZoneid() {
    return stationZoneid;
  }

  public LocalDate getTargetDateLocal() {
    return targetDateLocal;
  }

  public Integer getMaxTempF() {
    return maxTempF;
  }

  public Long getSourceValidTimeGmt() {
    return sourceValidTimeGmt;
  }

  public Long getSourceApiCallId() {
    return sourceApiCallId;
  }

  public String getSourceType() {
    return sourceType;
  }

  public Integer getObservationCount() {
    return observationCount;
  }

  public Instant getCreatedAtUtc() {
    return createdAtUtc;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }
}
