package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record IemCliPayload(
    String stationId,
    List<IemCliDaily> days,
    String rawJson,
    String rawPayloadHash,
    Instant generatedAtUtc) {

  private static final Pattern ISSUE_TIMESTAMP = Pattern.compile("(\\d{12})");
  private static final DateTimeFormatter ISSUE_FORMATTER = DateTimeFormatter.ofPattern("yyyyMMddHHmm");

  public static IemCliPayload parse(ObjectMapper mapper, String rawJson, String expectedStationId) {
    Objects.requireNonNull(mapper, "mapper");
    if (rawJson == null || rawJson.isBlank()) {
      throw new IllegalArgumentException("Raw JSON payload is required");
    }
    String normalizedStation = normalizeStation(expectedStationId);
    JsonNode root = readTree(mapper, rawJson);
    JsonNode results = requireArray(root, "results");
    List<IemCliDaily> days = new ArrayList<>(results.size());
    for (JsonNode entry : results) {
      if (entry == null || !entry.isObject()) {
        throw new IllegalArgumentException("CLI results entry must be an object");
      }
      String station = requireText(entry, "station");
      String normalizedEntryStation = normalizeStation(station);
      if (!normalizedEntryStation.equals(normalizedStation)) {
        throw new IllegalArgumentException("Station mismatch: expected " + normalizedStation
            + " but got " + normalizedEntryStation);
      }
      LocalDate date = parseDate(requireText(entry, "valid"));
      BigDecimal tmax = parseDecimal(entry, "high");
      BigDecimal tmin = parseDecimal(entry, "low");
      Instant reportIssuedAtUtc = parseIssuedAt(entry);
      days.add(new IemCliDaily(normalizedEntryStation, date, tmax, tmin, reportIssuedAtUtc));
    }
    Instant generatedAtUtc = optionalInstant(root, "generated_at");
    String payloadHash = Hashing.sha256Hex(rawJson);
    return new IemCliPayload(normalizedStation, Collections.unmodifiableList(days), rawJson, payloadHash, generatedAtUtc);
  }

  private static JsonNode readTree(ObjectMapper mapper, String rawJson) {
    try {
      return mapper.readTree(rawJson);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to parse IEM CLI JSON", ex);
    }
  }

  private static JsonNode requireArray(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || !value.isArray()) {
      throw new IllegalArgumentException("Missing array field: " + field);
    }
    return value;
  }

  private static String requireText(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || !value.isTextual() || value.asText().isBlank()) {
      throw new IllegalArgumentException("Missing text field: " + field);
    }
    return value.asText();
  }

  private static String optionalText(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || value.isNull()) {
      return null;
    }
    if (!value.isTextual()) {
      throw new IllegalArgumentException("Expected text field: " + field);
    }
    String text = value.asText().trim();
    return text.isEmpty() ? null : text;
  }

  private static Instant optionalInstant(JsonNode node, String field) {
    String value = optionalText(node, field);
    if (value == null) {
      return null;
    }
    try {
      return Instant.parse(value);
    } catch (DateTimeParseException ex) {
      throw new IllegalArgumentException("Invalid instant for field: " + field, ex);
    }
  }

  private static LocalDate parseDate(String value) {
    try {
      return LocalDate.parse(value);
    } catch (DateTimeParseException ex) {
      throw new IllegalArgumentException("Invalid date value: " + value, ex);
    }
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
      try {
        return new BigDecimal(text);
      } catch (NumberFormatException ex) {
        throw new IllegalArgumentException("Invalid decimal for field: " + field + " value=" + text, ex);
      }
    }
    throw new IllegalArgumentException("Expected numeric field: " + field);
  }

  private static Instant parseIssuedAt(JsonNode entry) {
    String product = optionalText(entry, "product");
    Instant issuedAt = parseIssuedAtToken(product);
    if (issuedAt != null) {
      return issuedAt;
    }
    String link = optionalText(entry, "link");
    return parseIssuedAtToken(link);
  }

  private static Instant parseIssuedAtToken(String token) {
    if (token == null) {
      return null;
    }
    Matcher matcher = ISSUE_TIMESTAMP.matcher(token);
    if (!matcher.find()) {
      return null;
    }
    String timestamp = matcher.group(1);
    try {
      LocalDateTime dateTime = LocalDateTime.parse(timestamp, ISSUE_FORMATTER);
      return dateTime.toInstant(ZoneOffset.UTC);
    } catch (DateTimeParseException ex) {
      return null;
    }
  }

  private static String normalizeStation(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("StationId is required");
    }
    return value.trim().toUpperCase(Locale.ROOT);
  }
}
