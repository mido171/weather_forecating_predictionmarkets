package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.MosModel;
import java.nio.charset.StandardCharsets;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

public record IemMosPayload(
    String stationId,
    MosModel model,
    List<IemMosEntry> entries,
    String rawJson,
    String rawPayloadHash) {

  public static IemMosPayload parse(ObjectMapper mapper,
                                    byte[] rawBytes,
                                    String expectedStationId,
                                    MosModel expectedModel) {
    Objects.requireNonNull(mapper, "mapper");
    if (rawBytes == null || rawBytes.length == 0) {
      throw new IllegalArgumentException("Raw JSON payload is required");
    }
    String rawJson = new String(rawBytes, StandardCharsets.UTF_8);
    if (rawJson.isBlank()) {
      throw new IllegalArgumentException("Raw JSON payload is required");
    }
    String normalizedStation = normalizeStation(expectedStationId);
    if (expectedModel == null) {
      throw new IllegalArgumentException("expectedModel is required");
    }
    JsonNode root = readTree(mapper, rawJson);
    JsonNode array = requireArray(root);
    List<IemMosEntry> entries = new ArrayList<>(array.size());
    for (JsonNode entry : array) {
      if (entry == null || !entry.isObject()) {
        throw new IllegalArgumentException("MOS entry must be an object");
      }
      String station = requireText(entry, "station");
      String normalizedEntryStation = normalizeStation(station);
      if (!normalizedEntryStation.equals(normalizedStation)) {
        throw new IllegalArgumentException("Station mismatch: expected " + normalizedStation
            + " but got " + normalizedEntryStation);
      }
      MosModel entryModel = parseModel(requireText(entry, "model"));
      if (entryModel != expectedModel) {
        throw new IllegalArgumentException("Model mismatch: expected " + expectedModel
            + " but got " + entryModel);
      }
      Instant runtimeUtc = Instant.ofEpochMilli(requireEpochMillis(entry, "runtime"));
      Instant forecastTimeUtc = optionalEpochMillis(entry, "ftime");
      BigDecimal nX = parseDecimal(entry, "n_x");
      entries.add(new IemMosEntry(runtimeUtc, forecastTimeUtc, nX));
    }
    String payloadHash = Hashing.sha256Hex(rawBytes);
    return new IemMosPayload(
        normalizedStation,
        expectedModel,
        Collections.unmodifiableList(entries),
        rawJson,
        payloadHash);
  }

  private static JsonNode readTree(ObjectMapper mapper, String rawJson) {
    try {
      return mapper.readTree(rawJson);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to parse IEM MOS JSON", ex);
    }
  }

  private static JsonNode requireArray(JsonNode node) {
    if (node == null || !node.isArray()) {
      throw new IllegalArgumentException("Expected JSON array payload");
    }
    return node;
  }

  private static String requireText(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || !value.isTextual() || value.asText().isBlank()) {
      throw new IllegalArgumentException("Missing text field: " + field);
    }
    return value.asText();
  }

  private static long requireEpochMillis(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || value.isNull()) {
      throw new IllegalArgumentException("Missing numeric field: " + field);
    }
    return parseEpochMillis(value, field);
  }

  private static Instant optionalEpochMillis(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || value.isNull()) {
      return null;
    }
    long epochMillis = parseEpochMillis(value, field);
    return Instant.ofEpochMilli(epochMillis);
  }

  private static long parseEpochMillis(JsonNode value, String field) {
    if (value.isNumber()) {
      return value.asLong();
    }
    if (value.isTextual()) {
      String text = value.asText().trim();
      if (text.isEmpty()) {
        throw new IllegalArgumentException("Missing numeric field: " + field);
      }
      try {
        return Long.parseLong(text);
      } catch (NumberFormatException ex) {
        throw new IllegalArgumentException("Invalid epoch millis for field: " + field, ex);
      }
    }
    throw new IllegalArgumentException("Expected numeric field: " + field);
  }

  private static BigDecimal parseDecimal(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || value.isNull()) {
      return null;
    }
    if (value.isNumber()) {
      return value.decimalValue();
    }
    if (value.isTextual()) {
      String text = value.asText().trim();
      if (text.isEmpty() || text.equalsIgnoreCase("M") || text.equalsIgnoreCase("T")) {
        return null;
      }
      int slash = text.indexOf('/');
      if (slash > 0) {
        text = text.substring(0, slash).trim();
        if (text.isEmpty() || text.equalsIgnoreCase("M") || text.equalsIgnoreCase("T")) {
          return null;
        }
      }
      try {
        return new BigDecimal(text);
      } catch (NumberFormatException ex) {
        throw new IllegalArgumentException("Invalid decimal for field: " + field + " value=" + text, ex);
      }
    }
    throw new IllegalArgumentException("Expected numeric field: " + field);
  }

  private static MosModel parseModel(String value) {
    String normalized = normalizeText(value);
    try {
      return MosModel.valueOf(normalized);
    } catch (IllegalArgumentException ex) {
      throw new IllegalArgumentException("Unknown MOS model: " + value, ex);
    }
  }

  private static String normalizeStation(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("StationId is required");
    }
    return value.trim().toUpperCase(Locale.ROOT);
  }

  private static String normalizeText(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("Required value missing");
    }
    return value.trim().toUpperCase(Locale.ROOT);
  }
}
