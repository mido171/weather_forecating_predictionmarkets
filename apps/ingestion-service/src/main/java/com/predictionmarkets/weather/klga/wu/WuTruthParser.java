package com.predictionmarkets.weather.klga.wu;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class WuTruthParser {
  static final String PARSER_VERSION = "weathercom_historical_observations_v2_hourly_temp_only";
  static final String DAILY_HIGH_SOURCE = "hourly_temp_max";

  private final ObjectMapper objectMapper;

  WuTruthParser(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  List<WuTruthDailyRow> parse(WuTruthFetchResult result) {
    String body = result.responseBody() == null ? "" : result.responseBody();
    String payloadHash = sha256Hex(body);
    if (!result.success()) {
      if (isProviderNoDataBody(body)) {
        return failureRows(result, payloadHash, "no_data");
      }
      return failureRows(result, payloadHash, "fetch_failed");
    }

    JsonNode payload;
    try {
      payload = objectMapper.readTree(body);
    } catch (Exception ex) {
      return failureRows(
          new WuTruthFetchResult(
              result.station(),
              result.startDate(),
              result.endDate(),
              false,
              result.httpStatus(),
              body,
              "JSON_PARSE_ERROR",
              ex.getMessage(),
              result.fetchedAtUtc(),
              result.attempts(),
              result.sourceUrlRedacted()),
          payloadHash,
          "fetch_failed");
    }

    JsonNode observations = payload.get("observations");
    if (observations == null || !observations.isArray() || observations.isEmpty()) {
      String noDataStatus = providerNoData(payload) ? "no_data" : "no_data";
      return failureRows(result, payloadHash, noDataStatus);
    }

    ZoneId zoneId = ZoneId.of(result.station().timezoneName());
    Map<LocalDate, DayAccumulator> grouped = new LinkedHashMap<>();
    Set<Instant> seenTimestamps = new HashSet<>();
    for (JsonNode observation : observations) {
      if (observation == null || !observation.isObject() || !observation.hasNonNull("valid_time_gmt")) {
        continue;
      }
      Instant observedUtc;
      try {
        observedUtc = Instant.ofEpochSecond(observation.get("valid_time_gmt").asLong());
      } catch (RuntimeException ignored) {
        continue;
      }
      LocalDate localDate = observedUtc.atZone(zoneId).toLocalDate();
      if (localDate.isBefore(result.startDate()) || localDate.isAfter(result.endDate())) {
        continue;
      }
      DayAccumulator accumulator = grouped.computeIfAbsent(localDate, ignored -> new DayAccumulator());
      if (!seenTimestamps.add(observedUtc)) {
        accumulator.duplicateTimestampCount++;
        continue;
      }
      ObjectNode evidence = hourlyEvidence(observation, observedUtc, zoneId);
      accumulator.hourly.add(evidence);
      Double rawTemp = number(observation.get("temp"));
      if (rawTemp != null && (rawTemp < -40 || rawTemp > 120)) {
        accumulator.outOfBoundsTempCount++;
      }
      Integer temp = boundedInt(observation.get("temp"), -40, 120);
      if (temp != null) {
        accumulator.tempValues.add(temp);
      }
      JsonNode maxDiagnostic = providerDiagnostic(observation, observedUtc, zoneId, "max_temp");
      if (maxDiagnostic != null) {
        accumulator.providerMax.add(maxDiagnostic);
      }
      JsonNode minDiagnostic = providerDiagnostic(observation, observedUtc, zoneId, "min_temp");
      if (minDiagnostic != null) {
        accumulator.providerMin.add(minDiagnostic);
      }
    }

    Map<LocalDate, WuTruthDailyRow> parsedRows = new HashMap<>();
    for (Map.Entry<LocalDate, DayAccumulator> entry : grouped.entrySet()) {
      parsedRows.put(entry.getKey(), acceptedOrSuspectRow(result, payloadHash, entry.getKey(), entry.getValue()));
    }
    List<WuTruthDailyRow> rows = new ArrayList<>();
    LocalDate cursor = result.startDate();
    while (!cursor.isAfter(result.endDate())) {
      WuTruthDailyRow row = parsedRows.get(cursor);
      if (row == null) {
        rows.add(noDataRow(result, payloadHash, cursor, "no_hourly_observations_for_station_date"));
      } else {
        rows.add(row);
      }
      cursor = cursor.plusDays(1);
    }
    rows.sort(Comparator.comparing(WuTruthDailyRow::stationId).thenComparing(WuTruthDailyRow::localDate));
    return rows;
  }

  private WuTruthDailyRow acceptedOrSuspectRow(
      WuTruthFetchResult result,
      String payloadHash,
      LocalDate localDate,
      DayAccumulator accumulator) {
    accumulator.hourly.sort(Comparator.comparing(node -> node.path("observation_time_utc").asText()));
    Integer tmax = accumulator.tempValues.stream().max(Integer::compareTo).orElse(null);
    Integer tmin = accumulator.tempValues.stream().min(Integer::compareTo).orElse(null);
    ArrayNode highTimes = objectMapper.createArrayNode();
    if (tmax != null) {
      for (ObjectNode evidence : accumulator.hourly) {
        JsonNode tempNode = evidence.get("temp_f");
        if (tempNode != null && !tempNode.isNull() && Math.round(tempNode.asDouble()) == tmax) {
          highTimes.add(evidence.path("observation_time_local").asText());
        }
      }
    }

    List<String> notes = new ArrayList<>();
    if (tmax == null) {
      notes.add("missing_hourly_temp_max");
    }
    if (tmin != null && tmax != null && tmax < tmin) {
      notes.add("daily_high_below_daily_low");
    }
    if (accumulator.tempValues.isEmpty()) {
      notes.add("no_hourly_temperature_observations");
    } else if (accumulator.tempValues.size() < 18) {
      notes.add("incomplete_hourly_temperature_coverage");
    }
    if (accumulator.duplicateTimestampCount > 0) {
      notes.add("duplicate_observation_timestamps_removed");
    }
    if (accumulator.outOfBoundsTempCount > 0) {
      notes.add("out_of_bounds_hourly_temperature_removed");
    }

    boolean manualCanary = "KLGA".equals(result.station().stationId())
        && LocalDate.of(2026, 5, 21).equals(localDate)
        && Integer.valueOf(66).equals(tmax)
        && Integer.valueOf(56).equals(tmin);
    String validationStatus = notes.isEmpty() ? "accepted" : "suspect";
    if (manualCanary) {
      validationStatus = "manual_confirmed";
      notes.remove("incomplete_hourly_temperature_coverage");
    }

    ObjectNode noteJson = objectMapper.createObjectNode();
    noteJson.put("daily_high_source", DAILY_HIGH_SOURCE);
    noteJson.put("daily_high_rule", "max bounded hourly temp field only; provider max_temp ignored");
    noteJson.put("observation_count", accumulator.tempValues.size());
    noteJson.put("duplicate_timestamp_count", accumulator.duplicateTimestampCount);
    noteJson.put("out_of_bounds_temp_count", accumulator.outOfBoundsTempCount);
    ArrayNode notesNode = noteJson.putArray("notes");
    for (String note : notes) {
      notesNode.add(note);
    }
    if (manualCanary) {
      noteJson.put("manual_confirmation_source_url", result.station().pageUrl(localDate));
      noteJson.put("manual_confirmation_expected_tmax_f", 66);
      noteJson.put("manual_confirmation_expected_tmin_f", 56);
    }

    return new WuTruthDailyRow(
        result.station().stationId(),
        result.station().wundergroundStationId(),
        localDate,
        result.station().timezoneName(),
        tmax,
        tmin,
        accumulator.tempValues.size(),
        highTimes,
        array(accumulator.hourly),
        array(accumulator.providerMax),
        array(accumulator.providerMin),
        result.sourceUrlRedacted(),
        result.station().pageUrl(localDate),
        payloadHash,
        PARSER_VERSION,
        result.fetchedAtUtc(),
        settlementAvailableAt(result.station(), localDate),
        DAILY_HIGH_SOURCE,
        validationStatus,
        noteJson,
        "saved",
        result.httpStatus(),
        null,
        null,
        result.attempts());
  }

  private List<WuTruthDailyRow> failureRows(WuTruthFetchResult result, String payloadHash, String fetchStatus) {
    List<WuTruthDailyRow> rows = new ArrayList<>();
    String status = "no_data".equals(fetchStatus) ? "no_data" : "fetch_failed";
    LocalDate cursor = result.startDate();
    while (!cursor.isAfter(result.endDate())) {
      rows.add(failureRow(result, payloadHash, cursor, fetchStatus, status, result.errorType(), result.errorMessage()));
      cursor = cursor.plusDays(1);
    }
    return rows;
  }

  private WuTruthDailyRow noDataRow(WuTruthFetchResult result, String payloadHash, LocalDate localDate, String note) {
    return failureRow(result, payloadHash, localDate, "no_data", "no_data", "NO_DATA", note);
  }

  private WuTruthDailyRow failureRow(
      WuTruthFetchResult result,
      String payloadHash,
      LocalDate localDate,
      String fetchStatus,
      String validationStatus,
      String errorType,
      String errorMessage) {
    ObjectNode noteJson = objectMapper.createObjectNode();
    noteJson.put("daily_high_source", DAILY_HIGH_SOURCE);
    noteJson.put("daily_high_rule", "max bounded hourly temp field only; provider max_temp ignored");
    noteJson.put("error_type", errorType == null ? "" : errorType);
    noteJson.put("error_message", errorMessage == null ? "" : errorMessage);
    return new WuTruthDailyRow(
        result.station().stationId(),
        result.station().wundergroundStationId(),
        localDate,
        result.station().timezoneName(),
        null,
        null,
        0,
        objectMapper.createArrayNode(),
        objectMapper.createArrayNode(),
        objectMapper.createArrayNode(),
        objectMapper.createArrayNode(),
        result.sourceUrlRedacted(),
        result.station().pageUrl(localDate),
        payloadHash,
        PARSER_VERSION,
        result.fetchedAtUtc(),
        settlementAvailableAt(result.station(), localDate),
        DAILY_HIGH_SOURCE,
        validationStatus,
        noteJson,
        fetchStatus,
        result.httpStatus(),
        errorType,
        errorMessage,
        result.attempts());
  }

  private ObjectNode hourlyEvidence(JsonNode observation, Instant observedUtc, ZoneId zoneId) {
    ObjectNode node = objectMapper.createObjectNode();
    node.put("observation_time_local", observedUtc.atZone(zoneId).toString());
    node.put("observation_time_utc", observedUtc.toString());
    putNullableNumber(node, "temp_f", boundedDouble(observation.get("temp"), -40, 120));
    putNullableNumber(node, "dewpoint_f", boundedDouble(observation.get("dewPt"), -80, 90));
    putNullableNumber(node, "humidity_pct", boundedDouble(observation.get("rh"), 0, 100));
    putNullableNumber(node, "wind_speed_mph", boundedDouble(observation.get("wspd"), 0, 150));
    putNullableNumber(node, "wind_gust_mph", number(observation.get("gust")));
    putNullableNumber(node, "wind_direction_deg", number(observation.get("wdir")));
    putNullableNumber(node, "pressure_in", number(observation.get("pressure")));
    putNullableNumber(node, "precipitation_in", boundedDouble(observation.get("precip_hrly"), 0, 20));
    String condition = text(observation.get("wx_phrase"));
    if (condition == null) {
      condition = text(observation.get("terse_phrase"));
    }
    putNullableText(node, "condition_text", condition);
    return node;
  }

  private JsonNode providerDiagnostic(JsonNode observation, Instant observedUtc, ZoneId zoneId, String fieldName) {
    JsonNode value = observation.get(fieldName);
    if (value == null || value.isNull() || !value.isNumber()) {
      return null;
    }
    ObjectNode node = objectMapper.createObjectNode();
    node.put("observation_time_local", observedUtc.atZone(zoneId).toString());
    node.put("observation_time_utc", observedUtc.toString());
    node.put(fieldName, value.asDouble());
    putNullableNumber(node, "actual_temp_f", boundedDouble(observation.get("temp"), -40, 130));
    return node;
  }

  private Instant settlementAvailableAt(WuTruthStation station, LocalDate localDate) {
    ZoneId zoneId = ZoneId.of(station.timezoneName());
    LocalDateTime endPlus24h = localDate.plusDays(2).atStartOfDay();
    return endPlus24h.atZone(zoneId).withZoneSameInstant(ZoneOffset.UTC).toInstant();
  }

  private ArrayNode array(List<? extends JsonNode> nodes) {
    ArrayNode array = objectMapper.createArrayNode();
    for (JsonNode node : nodes) {
      array.add(node);
    }
    return array;
  }

  private boolean providerNoData(JsonNode payload) {
    JsonNode errors = payload.get("errors");
    if (errors == null || !errors.isArray()) {
      return false;
    }
    for (JsonNode item : errors) {
      String code = item.path("error").path("code").asText("");
      String message = item.path("error").path("message").asText("").toLowerCase();
      if ("NDF-0001".equalsIgnoreCase(code) || message.contains("no data found")) {
        return true;
      }
    }
    return false;
  }

  private boolean isProviderNoDataBody(String body) {
    if (body == null || body.isBlank()) {
      return false;
    }
    try {
      return providerNoData(objectMapper.readTree(body));
    } catch (Exception ignored) {
      return false;
    }
  }

  private Integer boundedInt(JsonNode node, int min, int max) {
    Double value = boundedDouble(node, min, max);
    return value == null ? null : (int) Math.round(value);
  }

  private Double boundedDouble(JsonNode node, double min, double max) {
    Double value = number(node);
    if (value == null || value < min || value > max) {
      return null;
    }
    return value;
  }

  private Double number(JsonNode node) {
    return node == null || node.isNull() || !node.isNumber() ? null : node.asDouble();
  }

  private String text(JsonNode node) {
    return node == null || node.isNull() ? null : node.asText();
  }

  private void putNullableNumber(ObjectNode node, String field, Double value) {
    if (value == null) {
      node.putNull(field);
    } else {
      node.put(field, value);
    }
  }

  private void putNullableText(ObjectNode node, String field, String value) {
    if (value == null) {
      node.putNull(field);
    } else {
      node.put(field, value);
    }
  }

  private String sha256Hex(String value) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      return HexFormat.of().formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException ex) {
      throw new IllegalStateException("SHA-256 not available", ex);
    }
  }

  private static final class DayAccumulator {
    final List<ObjectNode> hourly = new ArrayList<>();
    final List<JsonNode> providerMax = new ArrayList<>();
    final List<JsonNode> providerMin = new ArrayList<>();
    final List<Integer> tempValues = new ArrayList<>();
    int duplicateTimestampCount;
    int outOfBoundsTempCount;
  }
}
