package com.predictionmarkets.weather.models;

import java.time.LocalTime;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "asof_policy")
public class AsofPolicy {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "name", nullable = false, length = 64)
  private String name;

  @Column(name = "asof_local_time", nullable = false)
  private LocalTime asofLocalTime;

  @Enumerated(EnumType.STRING)
  @Column(name = "asof_time_zone", nullable = false, length = 16)
  private AsofTimeZone asofTimeZone = AsofTimeZone.LOCAL;

  @Column(name = "enabled", nullable = false)
  private Boolean enabled;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public LocalTime getAsofLocalTime() {
    return asofLocalTime;
  }

  public void setAsofLocalTime(LocalTime asofLocalTime) {
    this.asofLocalTime = asofLocalTime;
  }

  public AsofTimeZone getAsofTimeZone() {
    return asofTimeZone;
  }

  public void setAsofTimeZone(AsofTimeZone asofTimeZone) {
    this.asofTimeZone = asofTimeZone;
  }

  public Boolean getEnabled() {
    return enabled;
  }

  public void setEnabled(Boolean enabled) {
    this.enabled = enabled;
  }
}
