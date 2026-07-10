package com.predictionmarkets.weather.weathercom.web.dto;

import java.util.Locale;

public enum WeatherComApiCallStatusFilter {
  ALL,
  FAILED,
  SUCCEEDED;

  public static WeatherComApiCallStatusFilter parse(String value) {
    if (value == null || value.isBlank()) {
      return ALL;
    }
    String normalized = value.trim().toUpperCase(Locale.ROOT);
    return switch (normalized) {
      case "ALL" -> ALL;
      case "FAILED" -> FAILED;
      case "SUCCEEDED" -> SUCCEEDED;
      default -> throw new IllegalArgumentException("Unsupported status filter: " + value);
    };
  }
}

