package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class GribstreamResponseParser {
  private static final int SNIPPET_LIMIT = 500;

  private GribstreamResponseParser() {
  }

  public static List<GribstreamRow> parseRows(ObjectMapper mapper,
                                              byte[] responseBytes,
                                              String modelCode,
                                              String requestSha256) {
    Objects.requireNonNull(mapper, "mapper is required");
    String safeModel = modelCode == null ? "unknown" : modelCode;
    String safeRequestHash = requestSha256 == null ? "unknown" : requestSha256;
    if (responseBytes == null || responseBytes.length == 0) {
      throw new GribstreamResponseException(errorPrefix(safeModel, safeRequestHash, "")
          + " empty response body");
    }
    String bodySnippet = snippet(responseBytes);
    int firstNonWhitespace = firstNonWhitespaceIndex(responseBytes);
    if (firstNonWhitespace < 0) {
      throw new GribstreamResponseException(errorPrefix(safeModel, safeRequestHash, bodySnippet)
          + " empty response body");
    }
    byte firstByte = responseBytes[firstNonWhitespace];
    if (firstByte == '[') {
      JsonNode root = readTree(mapper, responseBytes, safeModel, safeRequestHash, bodySnippet);
      if (root == null || !root.isArray()) {
        throw new GribstreamResponseException(errorPrefix(safeModel, safeRequestHash, bodySnippet)
            + " expected JSON array body");
      }
      return parseArray(root, safeModel, safeRequestHash, bodySnippet);
    }
    return parseNdjson(mapper, responseBytes, safeModel, safeRequestHash, bodySnippet);
  }

  private static List<GribstreamRow> parseArray(JsonNode root,
                                                String modelCode,
                                                String requestSha256,
                                                String bodySnippet) {
    List<GribstreamRow> rows = new ArrayList<>(root.size());
    for (JsonNode entry : root) {
      GribstreamRow row = parseRow(entry, modelCode, requestSha256, bodySnippet);
      if (row != null) {
        rows.add(row);
      }
    }
    return rows;
  }

  private static List<GribstreamRow> parseNdjson(ObjectMapper mapper,
                                                 byte[] responseBytes,
                                                 String modelCode,
                                                 String requestSha256,
                                                 String bodySnippet) {
    String body = new String(responseBytes, StandardCharsets.UTF_8);
    String[] lines = body.split("\\r?\\n");
    List<GribstreamRow> rows = new ArrayList<>(lines.length);
    for (String line : lines) {
      if (line == null) {
        continue;
      }
      String trimmed = line.trim();
      if (trimmed.isEmpty()) {
        continue;
      }
      JsonNode entry = readTree(mapper, trimmed.getBytes(StandardCharsets.UTF_8),
          modelCode, requestSha256, bodySnippet);
      if (entry.isArray()) {
        rows.addAll(parseArray(entry, modelCode, requestSha256, bodySnippet));
      } else {
        GribstreamRow row = parseRow(entry, modelCode, requestSha256, bodySnippet);
        if (row != null) {
          rows.add(row);
        }
      }
    }
    if (rows.isEmpty()) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
          + " empty response body");
    }
    return rows;
  }

  private static GribstreamRow parseRow(JsonNode entry,
                                        String modelCode,
                                        String requestSha256,
                                        String bodySnippet) {
    if (entry == null || !entry.isObject()) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
          + " expected object rows");
    }
    Instant forecastedAt = requireInstant(entry, "forecasted_at", modelCode, requestSha256, bodySnippet);
    Instant forecastedTime = requireInstant(entry, "forecasted_time", modelCode, requestSha256, bodySnippet);
    Double tmpk = optionalDouble(entry, "tmpk", modelCode, requestSha256, bodySnippet);
    if (tmpk == null) {
      return null;
    }
    Integer member = optionalMember(entry);
    return new GribstreamRow(forecastedAt, forecastedTime, tmpk, member);
  }

  private static int firstNonWhitespaceIndex(byte[] responseBytes) {
    for (int i = 0; i < responseBytes.length; i++) {
      if (!isWhitespace(responseBytes[i])) {
        return i;
      }
    }
    return -1;
  }

  private static boolean isWhitespace(byte value) {
    return value == ' ' || value == '\n' || value == '\r' || value == '\t';
  }

  private static JsonNode readTree(ObjectMapper mapper,
                                   byte[] responseBytes,
                                   String modelCode,
                                   String requestSha256,
                                   String bodySnippet) {
    try {
      return mapper.readTree(responseBytes);
    } catch (Exception ex) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
          + " failed to parse JSON body", ex);
    }
  }

  private static Instant requireInstant(JsonNode node,
                                        String field,
                                        String modelCode,
                                        String requestSha256,
                                        String bodySnippet) {
    JsonNode value = node.get(field);
    if (value == null || !value.isTextual() || value.asText().isBlank()) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
          + " missing text field: " + field);
    }
    String text = value.asText();
    try {
      return Instant.parse(text);
    } catch (DateTimeParseException ex) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
          + " invalid instant field: " + field + " value=" + text, ex);
    }
  }

  private static double requireDouble(JsonNode node,
                                      String field,
                                      String modelCode,
                                      String requestSha256,
                                      String bodySnippet) {
    JsonNode value = node.get(field);
    if (value == null || value.isNull()) {
      throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
          + " missing numeric field: " + field);
    }
    if (value.isNumber()) {
      return value.asDouble();
    }
    if (value.isTextual()) {
      String text = value.asText().trim();
      if (text.isEmpty()) {
        throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
            + " missing numeric field: " + field);
      }
      try {
        return Double.parseDouble(text);
      } catch (NumberFormatException ex) {
        throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
            + " invalid numeric field: " + field + " value=" + text, ex);
      }
    }
    throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
        + " invalid numeric field: " + field);
  }

  private static Double optionalDouble(JsonNode node,
                                       String field,
                                       String modelCode,
                                       String requestSha256,
                                       String bodySnippet) {
    JsonNode value = node.get(field);
    if (value == null || value.isNull()) {
      return null;
    }
    if (value.isNumber()) {
      return value.asDouble();
    }
    if (value.isTextual()) {
      String text = value.asText().trim();
      if (text.isEmpty()) {
        return null;
      }
      try {
        return Double.parseDouble(text);
      } catch (NumberFormatException ex) {
        throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
            + " invalid numeric field: " + field + " value=" + text, ex);
      }
    }
    throw new GribstreamResponseException(errorPrefix(modelCode, requestSha256, bodySnippet)
        + " invalid numeric field: " + field);
  }

  private static Integer optionalMember(JsonNode node) {
    JsonNode value = node.get("member");
    if (value == null || value.isNull()) {
      value = node.get("member_id");
    }
    if (value == null || value.isNull()) {
      value = node.get("ens_member");
    }
    if (value == null || value.isNull()) {
      return null;
    }
    if (value.isInt()) {
      return value.asInt();
    }
    if (value.isNumber()) {
      return value.numberValue().intValue();
    }
    if (value.isTextual()) {
      String text = value.asText().trim();
      if (text.isEmpty()) {
        return null;
      }
      try {
        return Integer.parseInt(text);
      } catch (NumberFormatException ex) {
        return null;
      }
    }
    return null;
  }

  private static String errorPrefix(String modelCode, String requestSha256, String bodySnippet) {
    return "Gribstream response parse error model=" + modelCode
        + " requestSha256=" + requestSha256 + " bodySnippet=" + bodySnippet;
  }

  private static String snippet(byte[] responseBytes) {
    String text = new String(responseBytes, StandardCharsets.UTF_8);
    if (text.length() <= SNIPPET_LIMIT) {
      return text;
    }
    return text.substring(0, SNIPPET_LIMIT);
  }
}
