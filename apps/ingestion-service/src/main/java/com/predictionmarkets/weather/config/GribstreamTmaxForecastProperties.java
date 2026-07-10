package com.predictionmarkets.weather.config;

import java.time.Instant;
import java.time.LocalDate;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream.tmax-forecast")
public class GribstreamTmaxForecastProperties {
  private String stationId;
  private LocalDate startDateLocal = LocalDate.of(2021, 3, 23);
  private LocalDate endDateLocal = LocalDate.of(2025, 12, 31);
  private Instant smokeTestForecastedAtUtc = Instant.parse("2025-01-15T12:00:00Z");
  private int minHorizonHours = 12;
  private int maxHorizonHours = 48;
  private int rapMaxHorizonHours = 51;
  private int hrrrMaxHorizonHours = 48;
  private int minPointsPerDay = 8;
  private int rapMinPointsPerDay = 5;
  private int gefsMinPointsPerDay = 4;
  private int runRangeDays = 7;

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public LocalDate getStartDateLocal() {
    return startDateLocal;
  }

  public void setStartDateLocal(LocalDate startDateLocal) {
    this.startDateLocal = startDateLocal;
  }

  public LocalDate getEndDateLocal() {
    return endDateLocal;
  }

  public void setEndDateLocal(LocalDate endDateLocal) {
    this.endDateLocal = endDateLocal;
  }

  public Instant getSmokeTestForecastedAtUtc() {
    return smokeTestForecastedAtUtc;
  }

  public void setSmokeTestForecastedAtUtc(Instant smokeTestForecastedAtUtc) {
    this.smokeTestForecastedAtUtc = smokeTestForecastedAtUtc;
  }

  public int getMinHorizonHours() {
    return minHorizonHours;
  }

  public void setMinHorizonHours(int minHorizonHours) {
    this.minHorizonHours = minHorizonHours;
  }

  public int getMaxHorizonHours() {
    return maxHorizonHours;
  }

  public void setMaxHorizonHours(int maxHorizonHours) {
    this.maxHorizonHours = maxHorizonHours;
  }

  public int getRapMaxHorizonHours() {
    return rapMaxHorizonHours;
  }

  public void setRapMaxHorizonHours(int rapMaxHorizonHours) {
    this.rapMaxHorizonHours = rapMaxHorizonHours;
  }

  public int getHrrrMaxHorizonHours() {
    return hrrrMaxHorizonHours;
  }

  public void setHrrrMaxHorizonHours(int hrrrMaxHorizonHours) {
    this.hrrrMaxHorizonHours = hrrrMaxHorizonHours;
  }

  public int getMinPointsPerDay() {
    return minPointsPerDay;
  }

  public void setMinPointsPerDay(int minPointsPerDay) {
    this.minPointsPerDay = minPointsPerDay;
  }

  public int getRapMinPointsPerDay() {
    return rapMinPointsPerDay;
  }

  public void setRapMinPointsPerDay(int rapMinPointsPerDay) {
    this.rapMinPointsPerDay = rapMinPointsPerDay;
  }

  public int getGefsMinPointsPerDay() {
    return gefsMinPointsPerDay;
  }

  public void setGefsMinPointsPerDay(int gefsMinPointsPerDay) {
    this.gefsMinPointsPerDay = gefsMinPointsPerDay;
  }

  public int getRunRangeDays() {
    return runRangeDays;
  }

  public void setRunRangeDays(int runRangeDays) {
    this.runRangeDays = runRangeDays;
  }
}
