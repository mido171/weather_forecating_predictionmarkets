package com.predictionmarkets.weather.kalshiapi.util;

import org.springframework.web.util.UriBuilder;

public final class KalshiQueryUtils {

  private KalshiQueryUtils() {
  }

  public static void addParam(UriBuilder uriBuilder, String name, Object value) {
    if (uriBuilder == null || name == null || name.isBlank() || value == null) {
      return;
    }
    uriBuilder.queryParam(name, value);
  }
}
