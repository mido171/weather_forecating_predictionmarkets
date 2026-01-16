package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class GribstreamGenericResponseParser {
  private GribstreamGenericResponseParser() {
  }

  public static List<GribstreamValueRow> parseRows(ObjectMapper mapper,
                                                   byte[] responseBytes,
                                                   String modelCode,
                                                   String requestSha256) {
    if (responseBytes == null || responseBytes.length == 0) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, ""));
    }
    String body = new String(responseBytes, StandardCharsets.UTF_8);
    if (body.isBlank()) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, ""));
    }
    String trimmed = body.trim();
    if (trimmed.startsWith("[")) {
      return parseArray(mapper, trimmed, modelCode, requestSha256);
    }
    if (trimmed.startsWith("{")) {
      return parseNdjson(mapper, body, modelCode, requestSha256);
    }
    if (trimmed.startsWith("forecasted_at")) {
      return parseCsv(body, modelCode, requestSha256);
    }
    throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, snippet(body)));
  }

  private static List<GribstreamValueRow> parseArray(ObjectMapper mapper,
                                                     String json,
                                                     String modelCode,
                                                     String requestSha256) {
    try {
      JsonNode root = mapper.readTree(json);
      if (!root.isArray()) {
        throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, snippet(json)));
      }
      List<GribstreamValueRow> rows = new ArrayList<>(root.size());
      for (JsonNode entry : root) {
        rows.add(parseRow(entry, modelCode, requestSha256, snippet(json)));
      }
      return rows;
    } catch (IOException ex) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, snippet(json)), ex);
    }
  }

  private static List<GribstreamValueRow> parseNdjson(ObjectMapper mapper,
                                                      String body,
                                                      String modelCode,
                                                      String requestSha256) {
    List<GribstreamValueRow> rows = new ArrayList<>();
    try (BufferedReader reader = new BufferedReader(new StringReader(body))) {
      String line;
      while ((line = reader.readLine()) != null) {
        String trimmed = line.trim();
        if (trimmed.isEmpty()) {
          continue;
        }
        JsonNode entry = mapper.readTree(trimmed);
        rows.add(parseRow(entry, modelCode, requestSha256, snippet(body)));
      }
    } catch (IOException ex) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, snippet(body)), ex);
    }
    return rows;
  }

  private static List<GribstreamValueRow> parseCsv(String body,
                                                   String modelCode,
                                                   String requestSha256) {
    List<GribstreamValueRow> rows = new ArrayList<>();
    String[] lines = body.split("\\r?\\n");
    if (lines.length < 2) {
      return rows;
    }
    String header = lines[0];
    String[] columns = header.split(",", -1);
    Map<String, Integer> index = new HashMap<>();
    for (int i = 0; i < columns.length; i++) {
      index.put(columns[i].trim().toLowerCase(Locale.ROOT), i);
    }
    int idxForecastedAt = index.getOrDefault("forecasted_at", -1);
    int idxForecastedTime = index.getOrDefault("forecasted_time", -1);
    int idxMember = index.getOrDefault("member", -1);
    for (int i = 1; i < lines.length; i++) {
      String line = lines[i];
      if (line == null || line.isBlank()) {
        continue;
      }
      String[] values = line.split(",", -1);
      Instant forecastedAt = parseInstant(values, idxForecastedAt);
      Instant forecastedTime = parseInstant(values, idxForecastedTime);
      Integer member = parseInt(values, idxMember);
      Map<String, String> payload = new HashMap<>();
      for (int c = 0; c < columns.length; c++) {
        String key = columns[c];
        if (isMetaKey(key)) {
          continue;
        }
        String raw = c < values.length ? values[c] : null;
        if (raw != null && !raw.isBlank()) {
          payload.put(key, raw.trim());
        }
      }
      rows.add(new GribstreamValueRow(forecastedAt, forecastedTime, member, payload));
    }
    return rows;
  }

  private static GribstreamValueRow parseRow(JsonNode entry,
                                             String modelCode,
                                             String requestSha256,
                                             String bodySnippet) {
    if (entry == null || !entry.isObject()) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet));
    }
    Instant forecastedAt = parseInstant(entry.get("forecasted_at"));
    Instant forecastedTime = parseInstant(entry.get("forecasted_time"));
    Integer member = parseInt(entry.get("member"));
    Map<String, String> values = new HashMap<>();
    Iterator<Map.Entry<String, JsonNode>> fields = entry.fields();
    while (fields.hasNext()) {
      Map.Entry<String, JsonNode> field = fields.next();
      String key = field.getKey();
      if (isMetaKey(key)) {
        continue;
      }
      JsonNode value = field.getValue();
      if (value == null || value.isNull()) {
        continue;
      }
      values.put(key, value.asText());
    }
    return new GribstreamValueRow(forecastedAt, forecastedTime, member, values);
  }

  private static boolean isMetaKey(String key) {
    if (key == null) {
      return true;
    }
    String normalized = key.trim().toLowerCase(Locale.ROOT);
    return normalized.equals("forecasted_at")
        || normalized.equals("forecasted_time")
        || normalized.equals("lat")
        || normalized.equals("lon")
        || normalized.equals("name")
        || normalized.equals("member");
  }

  private static Instant parseInstant(JsonNode node) {
    if (node == null || node.isNull()) {
      return null;
    }
    String text = node.asText().trim();
    if (text.isEmpty()) {
      return null;
    }
    return Instant.parse(text);
  }

  private static Instant parseInstant(String[] values, int index) {
    if (index < 0 || index >= values.length) {
      return null;
    }
    String text = values[index];
    if (text == null || text.isBlank()) {
      return null;
    }
    return Instant.parse(text.trim());
  }

  private static Integer parseInt(JsonNode node) {
    if (node == null || node.isNull()) {
      return null;
    }
    if (node.isNumber()) {
      return node.asInt();
    }
    String text = node.asText().trim();
    if (text.isEmpty()) {
      return null;
    }
    try {
      return Integer.parseInt(text);
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private static Integer parseInt(String[] values, int index) {
    if (index < 0 || index >= values.length) {
      return null;
    }
    String text = values[index];
    if (text == null || text.isBlank()) {
      return null;
    }
    try {
      return Integer.parseInt(text.trim());
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private static String errorPrefix(String modelCode, String requestSha256, String bodySnippet) {
    String safeModel = modelCode == null ? "unknown" : modelCode;
    String safeRequest = requestSha256 == null ? "unknown" : requestSha256;
    String snippet = bodySnippet == null ? "" : bodySnippet;
    return "Gribstream response parse error model=" + safeModel
        + " requestSha256=" + safeRequest
        + " bodySnippet=" + snippet;
  }

  private static String snippet(String body) {
    if (body == null || body.isEmpty()) {
      return "";
    }
    String trimmed = body.trim();
    if (trimmed.length() <= 500) {
      return trimmed;
    }
    return trimmed.substring(0, 500);
  }
}
