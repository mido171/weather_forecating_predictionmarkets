package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.models.GribstreamMetric;
import java.io.BufferedReader;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class GribstreamDailyFeatureCsv {
  public static final List<String> HEADER = List.of(
      "id",
      "station_id",
      "zone_id",
      "target_date_local",
      "asof_utc",
      "model_code",
      "metric",
      "value_f",
      "value_k",
      "source_forecasted_at_utc",
      "window_start_utc",
      "window_end_utc",
      "min_horizon_hours",
      "max_horizon_hours",
      "request_json",
      "request_sha256",
      "response_sha256",
      "retrieved_at_utc",
      "notes");

  private static final List<String> REQUIRED_COLUMNS = List.of(
      "station_id",
      "zone_id",
      "target_date_local",
      "asof_utc",
      "model_code",
      "metric",
      "window_start_utc",
      "window_end_utc",
      "min_horizon_hours",
      "max_horizon_hours",
      "request_json",
      "request_sha256",
      "response_sha256",
      "retrieved_at_utc");

  private GribstreamDailyFeatureCsv() {
  }

  public static String headerLine() {
    return formatRow(HEADER);
  }

  public static String toCsvLine(GribstreamDailyFeatureEntity entity) {
    List<String> values = new ArrayList<>(HEADER.size());
    values.add(formatLong(entity.getId()));
    values.add(entity.getStationId());
    values.add(entity.getZoneId());
    values.add(formatDate(entity.getTargetDateLocal()));
    values.add(formatInstant(entity.getAsofUtc()));
    values.add(entity.getModelCode());
    values.add(formatMetric(entity.getMetric()));
    values.add(formatDouble(entity.getValueF()));
    values.add(formatDouble(entity.getValueK()));
    values.add(formatInstant(entity.getSourceForecastedAtUtc()));
    values.add(formatInstant(entity.getWindowStartUtc()));
    values.add(formatInstant(entity.getWindowEndUtc()));
    values.add(Integer.toString(entity.getMinHorizonHours()));
    values.add(Integer.toString(entity.getMaxHorizonHours()));
    values.add(entity.getRequestJson());
    values.add(entity.getRequestSha256());
    values.add(entity.getResponseSha256());
    values.add(formatInstant(entity.getRetrievedAtUtc()));
    values.add(entity.getNotes());
    return formatRow(values);
  }

  public static String readRecord(BufferedReader reader) throws IOException {
    String line = reader.readLine();
    if (line == null) {
      return null;
    }
    StringBuilder record = new StringBuilder(line);
    while (hasUnclosedQuotes(record)) {
      String next = reader.readLine();
      if (next == null) {
        break;
      }
      record.append('\n').append(next);
    }
    return record.toString();
  }

  public static List<String> parseRecord(String record) {
    List<String> values = new ArrayList<>();
    StringBuilder current = new StringBuilder();
    boolean inQuotes = false;
    for (int i = 0; i < record.length(); i++) {
      char currentChar = record.charAt(i);
      if (inQuotes) {
        if (currentChar == '"') {
          if (i + 1 < record.length() && record.charAt(i + 1) == '"') {
            current.append('"');
            i++;
          } else {
            inQuotes = false;
          }
        } else {
          current.append(currentChar);
        }
        continue;
      }
      if (currentChar == '"') {
        inQuotes = true;
      } else if (currentChar == ',') {
        values.add(current.toString());
        current.setLength(0);
      } else {
        current.append(currentChar);
      }
    }
    values.add(current.toString());
    return values;
  }

  public static Map<String, Integer> headerIndex(List<String> headerValues) {
    Map<String, Integer> index = new HashMap<>();
    for (int i = 0; i < headerValues.size(); i++) {
      String normalized = normalizeHeader(headerValues.get(i));
      if (!normalized.isEmpty() && !index.containsKey(normalized)) {
        index.put(normalized, i);
      }
    }
    return index;
  }

  public static void validateHeader(Map<String, Integer> headerIndex) {
    List<String> missing = new ArrayList<>();
    for (String required : REQUIRED_COLUMNS) {
      if (!headerIndex.containsKey(required)) {
        missing.add(required);
      }
    }
    if (!missing.isEmpty()) {
      throw new IllegalArgumentException("Missing required CSV columns: " + missing);
    }
  }

  public static GribstreamDailyFeatureEntity toEntity(Map<String, Integer> headerIndex,
                                                      List<String> values) {
    GribstreamDailyFeatureEntity entity = new GribstreamDailyFeatureEntity();
    entity.setStationId(requireValue(headerIndex, values, "station_id"));
    entity.setZoneId(requireValue(headerIndex, values, "zone_id"));
    entity.setTargetDateLocal(parseDate(requireValue(headerIndex, values, "target_date_local"),
        "target_date_local"));
    entity.setAsofUtc(parseInstant(requireValue(headerIndex, values, "asof_utc"), "asof_utc"));
    entity.setModelCode(requireValue(headerIndex, values, "model_code"));
    entity.setMetric(parseMetric(requireValue(headerIndex, values, "metric"), "metric"));
    entity.setValueF(parseDouble(optionalValue(headerIndex, values, "value_f"), "value_f"));
    entity.setValueK(parseDouble(optionalValue(headerIndex, values, "value_k"), "value_k"));
    entity.setSourceForecastedAtUtc(parseInstant(
        optionalValue(headerIndex, values, "source_forecasted_at_utc"),
        "source_forecasted_at_utc"));
    entity.setWindowStartUtc(parseInstant(requireValue(headerIndex, values, "window_start_utc"),
        "window_start_utc"));
    entity.setWindowEndUtc(parseInstant(requireValue(headerIndex, values, "window_end_utc"),
        "window_end_utc"));
    entity.setMinHorizonHours(parseInt(requireValue(headerIndex, values, "min_horizon_hours"),
        "min_horizon_hours"));
    entity.setMaxHorizonHours(parseInt(requireValue(headerIndex, values, "max_horizon_hours"),
        "max_horizon_hours"));
    entity.setRequestJson(requireRawValue(headerIndex, values, "request_json"));
    entity.setRequestSha256(requireValue(headerIndex, values, "request_sha256"));
    entity.setResponseSha256(requireValue(headerIndex, values, "response_sha256"));
    entity.setRetrievedAtUtc(parseInstant(requireValue(headerIndex, values, "retrieved_at_utc"),
        "retrieved_at_utc"));
    entity.setNotes(optionalValue(headerIndex, values, "notes"));
    return entity;
  }

  private static String normalizeHeader(String value) {
    if (value == null) {
      return "";
    }
    return value.trim().toLowerCase(Locale.ROOT);
  }

  private static String requireValue(Map<String, Integer> headerIndex,
                                     List<String> values,
                                     String column) {
    String raw = rawValue(headerIndex, values, column);
    if (raw == null || raw.isBlank()) {
      throw new IllegalArgumentException("Missing value for column " + column);
    }
    return raw.trim();
  }

  private static String requireRawValue(Map<String, Integer> headerIndex,
                                        List<String> values,
                                        String column) {
    String raw = rawValue(headerIndex, values, column);
    if (raw == null || raw.isEmpty()) {
      throw new IllegalArgumentException("Missing value for column " + column);
    }
    return raw;
  }

  private static String optionalValue(Map<String, Integer> headerIndex,
                                      List<String> values,
                                      String column) {
    String raw = rawValue(headerIndex, values, column);
    if (raw == null || raw.isEmpty()) {
      return null;
    }
    return raw.trim();
  }

  private static String rawValue(Map<String, Integer> headerIndex,
                                 List<String> values,
                                 String column) {
    Integer index = headerIndex.get(column);
    if (index == null) {
      return null;
    }
    if (index >= values.size()) {
      return "";
    }
    return values.get(index);
  }

  private static Instant parseInstant(String value, String column) {
    if (value == null || value.isEmpty()) {
      return null;
    }
    try {
      return Instant.parse(value.trim());
    } catch (Exception ex) {
      throw new IllegalArgumentException("Invalid timestamp for column " + column + ": " + value, ex);
    }
  }

  private static LocalDate parseDate(String value, String column) {
    if (value == null || value.isEmpty()) {
      return null;
    }
    try {
      return LocalDate.parse(value.trim());
    } catch (Exception ex) {
      throw new IllegalArgumentException("Invalid date for column " + column + ": " + value, ex);
    }
  }

  private static int parseInt(String value, String column) {
    if (value == null || value.isEmpty()) {
      throw new IllegalArgumentException("Missing integer for column " + column);
    }
    try {
      return Integer.parseInt(value.trim());
    } catch (Exception ex) {
      throw new IllegalArgumentException("Invalid integer for column " + column + ": " + value, ex);
    }
  }

  private static Double parseDouble(String value, String column) {
    if (value == null || value.isEmpty()) {
      return null;
    }
    try {
      return Double.parseDouble(value.trim());
    } catch (Exception ex) {
      throw new IllegalArgumentException("Invalid number for column " + column + ": " + value, ex);
    }
  }

  private static GribstreamMetric parseMetric(String value, String column) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("Missing metric for column " + column);
    }
    try {
      return GribstreamMetric.valueOf(value.trim().toUpperCase(Locale.ROOT));
    } catch (Exception ex) {
      throw new IllegalArgumentException("Invalid metric for column " + column + ": " + value, ex);
    }
  }

  private static String formatRow(List<String> values) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < values.size(); i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append(escape(values.get(i)));
    }
    return builder.toString();
  }

  private static String escape(String value) {
    if (value == null) {
      return "";
    }
    boolean needsQuotes = value.indexOf(',') >= 0
        || value.indexOf('"') >= 0
        || value.indexOf('\n') >= 0
        || value.indexOf('\r') >= 0;
    if (!needsQuotes) {
      return value;
    }
    String escaped = value.replace("\"", "\"\"");
    return "\"" + escaped + "\"";
  }

  private static boolean hasUnclosedQuotes(CharSequence record) {
    boolean inQuotes = false;
    for (int i = 0; i < record.length(); i++) {
      char currentChar = record.charAt(i);
      if (currentChar == '"') {
        if (i + 1 < record.length() && record.charAt(i + 1) == '"') {
          i++;
          continue;
        }
        inQuotes = !inQuotes;
      }
    }
    return inQuotes;
  }

  private static String formatLong(Long value) {
    return value == null ? "" : value.toString();
  }

  private static String formatDate(LocalDate value) {
    return value == null ? "" : value.toString();
  }

  private static String formatInstant(Instant value) {
    return value == null ? "" : value.toString();
  }

  private static String formatDouble(Double value) {
    return value == null ? "" : value.toString();
  }

  private static String formatMetric(GribstreamMetric metric) {
    return metric == null ? "" : metric.name();
  }
}
