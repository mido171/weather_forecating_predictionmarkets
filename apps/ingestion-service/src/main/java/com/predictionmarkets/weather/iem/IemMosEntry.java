package com.predictionmarkets.weather.iem;

import java.time.Instant;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

public record IemMosEntry(
    Instant runtimeUtc,
    Instant forecastTimeUtc,
    Map<String, IemMosValue> values) {

  public static final String KEY_TMAX = "n_x";

  public IemMosEntry {
    if (values == null) {
      values = Map.of();
    } else if (values.isEmpty()) {
      values = Map.of();
    } else {
      values = Collections.unmodifiableMap(new java.util.HashMap<>(values));
    }
  }

  public IemMosValue value(String key) {
    if (key == null || key.isBlank()) {
      return null;
    }
    return values.get(key.trim().toLowerCase(Locale.ROOT));
  }

  public java.math.BigDecimal numericValue(String key) {
    IemMosValue value = value(key);
    return value == null ? null : value.numericValue();
  }

  public String rawValue(String key) {
    IemMosValue value = value(key);
    return value == null ? null : value.rawValue();
  }

  public Map<String, IemMosValue> values() {
    return values;
  }

  @Override
  public String toString() {
    return "IemMosEntry{runtimeUtc=" + runtimeUtc
        + ", forecastTimeUtc=" + forecastTimeUtc
        + ", values=" + values.size() + "}";
  }
}
