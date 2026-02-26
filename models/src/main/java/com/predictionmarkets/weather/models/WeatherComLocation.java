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
    name = "weathercom_location",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {"location_id"})
    }
)
public class WeatherComLocation {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "location_id", nullable = false, length = 64)
  private String locationId;

  @Column(name = "display_name", length = 255)
  private String displayName;

  @Column(name = "active", nullable = false)
  private boolean active = true;

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

  public String getLocationId() {
    return locationId;
  }

  public void setLocationId(String locationId) {
    this.locationId = locationId;
  }

  public String getDisplayName() {
    return displayName;
  }

  public void setDisplayName(String displayName) {
    this.displayName = displayName;
  }

  public boolean isActive() {
    return active;
  }

  public void setActive(boolean active) {
    this.active = active;
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

