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
    JsonNode root = readTree(mapper, responseBytes, safeModel, safeRequestHash, bodySnippet);
    if (root == null || !root.isArray()) {
      throw new GribstreamResponseException(errorPrefix(safeModel, safeRequestHash, bodySnippet)
          + " expected JSON array body");
    }
    List<GribstreamRow> rows = new ArrayList<>(root.size());
    for (JsonNode entry : root) {
      if (entry == null || !entry.isObject()) {
        throw new GribstreamResponseException(errorPrefix(safeModel, safeRequestHash, bodySnippet)
            + " expected object rows");
      }
      Instant forecastedAt = requireInstant(entry, "forecasted_at", safeModel, safeRequestHash, bodySnippet);
      Instant forecastedTime = requireInstant(entry, "forecasted_time", safeModel, safeRequestHash, bodySnippet);
      double tmpk = requireDouble(entry, "tmpk", safeModel, safeRequestHash, bodySnippet);
      Integer member = optionalMember(entry);
      rows.add(new GribstreamRow(forecastedAt, forecastedTime, tmpk, member));
    }
    return rows;
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
