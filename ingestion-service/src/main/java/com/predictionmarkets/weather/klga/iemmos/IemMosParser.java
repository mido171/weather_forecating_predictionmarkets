package com.predictionmarkets.weather.klga.iemmos;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.predictionmarkets.weather.common.Hashing;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Component;

@Component
public class IemMosParser {
  public static final String PARSER_VERSION = "iem_mos_structured_json_v1";
  private static final Map<String, String> MODEL_ALIASES = Map.of(
      "ETA", "NAM",
      "AVN", "GFS");

  private final ObjectMapper objectMapper;

  public IemMosParser(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  public List<IemMosForecastRow> parse(byte[] payload,
                                       IemMosChunk chunk,
                                       IemMosStoredRequest storedRequest) {
    JsonNode root = readTree(payload);
    if (!root.isArray()) {
      throw new IllegalArgumentException("Expected IEM MOS JSON array");
    }
    String rawPayloadHash = Hashing.sha256Hex(payload);
    List<IemMosForecastRow> rows = new ArrayList<>(root.size());
    for (JsonNode node : root) {
      if (!node.isObject()) {
        throw new IllegalArgumentException("IEM MOS row must be a JSON object");
      }
      String rowStation = normalizeText(requiredText(node, "station"));
      if (!rowStation.equals(chunk.station().stationId())
          && !rowStation.equals(chunk.station().mosStationId())) {
        throw new IllegalArgumentException("Station mismatch for " + chunk.chunkId()
            + ": expected " + chunk.station().stationId() + " or "
            + chunk.station().mosStationId() + " but got " + rowStation);
      }
      String model = normalizeModel(requiredText(node, "model"));
      if (!model.equals(chunk.product().endpointModel().name())) {
        throw new IllegalArgumentException("Model mismatch for " + chunk.chunkId()
            + ": expected " + chunk.product().endpointModel().name() + " but got " + model);
      }
      Instant runtime = parseInstant(node.get("runtime"), "runtime");
      Instant forecastTime = parseInstant(node.get("ftime"), "ftime");
      ObjectNode values = extractValues(node);
      String rawValuesJson = writeJson(values);
      Instant availableAt = runtime.plus(Duration.ofHours(2));
      Double forecastHour = Duration.between(runtime, forecastTime).toMinutes() / 60.0;
      String rawRowHash = rowHash(chunk, runtime, forecastTime, rawValuesJson);
      rows.add(new IemMosForecastRow(
          chunk.station().stationId(),
          chunk.station().mosStationId(),
          chunk.product().productCode(),
          chunk.product().endpointModel().name(),
          chunk.cutoffId(),
          runtime,
          forecastTime,
          forecastHour,
          "point",
          firstNumeric(values, "n_x", "x_n", "max"),
          firstNumeric(values, "tmp"),
          firstNumeric(values, "dpt"),
          firstNumeric(values, "wdr"),
          firstNumeric(values, "wsp"),
          firstNumeric(values, "gst"),
          firstNumeric(values, "sky", "cld"),
          firstNumeric(values, "pop", "p06", "p12", "p24"),
          firstNumeric(values, "qpf", "q06", "q12", "q24"),
          firstNumeric(values, "t03", "t06", "t12", "t24"),
          rawValuesJson,
          rawPayloadHash,
          availableAt,
          availableAt,
          "conservative_lag_rule",
          storedRequest.sourceRequestId(),
          storedRequest.sourceRecordId(),
          chunk.requestSha256(),
          rawRowHash,
          PARSER_VERSION,
          "ok",
          null));
    }
    return List.copyOf(rows);
  }

  private JsonNode readTree(byte[] payload) {
    try {
      return objectMapper.readTree(new String(payload, StandardCharsets.UTF_8));
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to parse IEM MOS JSON", ex);
    }
  }

  private ObjectNode extractValues(JsonNode row) {
    ObjectNode values = objectMapper.createObjectNode();
    Iterator<Map.Entry<String, JsonNode>> fields = row.fields();
    while (fields.hasNext()) {
      Map.Entry<String, JsonNode> field = fields.next();
      String normalized = normalizeKey(field.getKey());
      if (normalized.equals("station")
          || normalized.equals("model")
          || normalized.equals("runtime")
          || normalized.equals("ftime")) {
        continue;
      }
      values.set(normalized, field.getValue());
    }
    return values;
  }

  private Double firstNumeric(ObjectNode values, String... keys) {
    for (String key : keys) {
      JsonNode value = values.get(normalizeKey(key));
      Double parsed = numeric(value);
      if (parsed != null) {
        return parsed;
      }
    }
    return null;
  }

  private Double numeric(JsonNode value) {
    if (value == null || value.isNull()) {
      return null;
    }
    if (value.isNumber()) {
      return value.asDouble();
    }
    if (!value.isTextual()) {
      return null;
    }
    String raw = value.asText().trim();
    if (raw.isEmpty()
        || raw.equalsIgnoreCase("M")
        || raw.equalsIgnoreCase("T")
        || raw.equalsIgnoreCase("X")
        || raw.equals("-")) {
      return null;
    }
    int slash = raw.indexOf('/');
    if (slash > 0) {
      raw = raw.substring(0, slash).trim();
    }
    try {
      return new BigDecimal(raw).doubleValue();
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private Instant parseInstant(JsonNode value, String fieldName) {
    if (value == null || value.isNull()) {
      throw new IllegalArgumentException("Missing required IEM MOS timestamp field: " + fieldName);
    }
    if (value.isNumber()) {
      return Instant.ofEpochMilli(value.asLong());
    }
    if (!value.isTextual()) {
      throw new IllegalArgumentException("Invalid IEM MOS timestamp field: " + fieldName);
    }
    String text = value.asText().trim();
    if (text.isEmpty()) {
      throw new IllegalArgumentException("Blank IEM MOS timestamp field: " + fieldName);
    }
    try {
      return Instant.ofEpochMilli(Long.parseLong(text));
    } catch (NumberFormatException ex) {
      Long isoMillis = parseIsoMillis(text);
      if (isoMillis == null) {
        throw new IllegalArgumentException("Invalid IEM MOS timestamp field: " + fieldName, ex);
      }
      return Instant.ofEpochMilli(isoMillis);
    }
  }

  private Long parseIsoMillis(String text) {
    try {
      return Instant.parse(text).toEpochMilli();
    } catch (DateTimeParseException ex) {
      // fall through for UTC strings without zone suffix.
    }
    String normalized = text.endsWith("Z") ? text.substring(0, text.length() - 1) : text;
    try {
      DateTimeFormatter formatter = normalized.contains(".")
          ? DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS")
          : DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss");
      return LocalDateTime.parse(normalized, formatter).toInstant(ZoneOffset.UTC).toEpochMilli();
    } catch (DateTimeParseException ex) {
      return null;
    }
  }

  private String rowHash(IemMosChunk chunk, Instant runtime, Instant forecastTime, String rawValuesJson) {
    String identity = chunk.station().stationId() + "|"
        + chunk.product().productCode() + "|"
        + chunk.cutoffId() + "|"
        + runtime + "|"
        + forecastTime + "|"
        + rawValuesJson;
    return Hashing.sha256Hex(identity);
  }

  private String writeJson(JsonNode node) {
    try {
      return objectMapper.writeValueAsString(node);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize IEM MOS row values", ex);
    }
  }

  private String requiredText(JsonNode node, String fieldName) {
    JsonNode value = node.get(fieldName);
    if (value == null || value.isNull() || !value.isTextual() || value.asText().isBlank()) {
      throw new IllegalArgumentException("Missing required IEM MOS field: " + fieldName);
    }
    return value.asText();
  }

  private String normalizeModel(String value) {
    String normalized = normalizeText(value);
    return MODEL_ALIASES.getOrDefault(normalized, normalized);
  }

  private static String normalizeText(String value) {
    return value == null ? "" : value.trim().toUpperCase(Locale.ROOT);
  }

  private static String normalizeKey(String value) {
    return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
  }
}
