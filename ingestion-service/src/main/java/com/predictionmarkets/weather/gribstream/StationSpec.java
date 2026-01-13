package com.predictionmarkets.weather.gribstream;

import java.util.Locale;

public record StationSpec(
    String stationId,
    String zoneId,
    double lat,
    double lon,
    String name) {

  public StationSpec {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    if (zoneId == null || zoneId.isBlank()) {
      throw new IllegalArgumentException("zoneId is required");
    }
    if (name == null || name.isBlank()) {
      name = stationId;
    }
    stationId = stationId.trim().toUpperCase(Locale.ROOT);
    zoneId = zoneId.trim();
    name = name.trim();
  }
}
