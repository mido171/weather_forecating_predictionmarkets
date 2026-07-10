package com.predictionmarkets.weather.weathercom.service;

import java.util.regex.Pattern;

final class WeatherComValidation {
  private static final Pattern LOCATION_ID_PATTERN = Pattern.compile("^[^:]+(:[^:]+){2,}$");

  private WeatherComValidation() {
  }

  static String normalizeLocationId(String locationId) {
    if (locationId == null || locationId.isBlank()) {
      throw new IllegalArgumentException("locationId is required");
    }
    String normalized = locationId.trim();
    if (!LOCATION_ID_PATTERN.matcher(normalized).matches()) {
      throw new IllegalArgumentException(
          "locationId must contain at least three colon-delimited segments");
    }
    return normalized;
  }
}

