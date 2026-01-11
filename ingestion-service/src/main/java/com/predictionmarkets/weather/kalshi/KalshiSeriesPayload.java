package com.predictionmarkets.weather.kalshi;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import java.util.Objects;

public record KalshiSeriesPayload(
    String seriesTicker,
    String title,
    String category,
    String settlementSourceName,
    String settlementSourceUrl,
    String contractTermsUrl,
    String contractUrl,
    String rawJson,
    String rawPayloadHash) {

  public static KalshiSeriesPayload parse(ObjectMapper mapper, String rawJson) {
    Objects.requireNonNull(mapper, "mapper");
    if (rawJson == null || rawJson.isBlank()) {
      throw new IllegalArgumentException("Raw JSON payload is required");
    }
    JsonNode root = readTree(mapper, rawJson);
    JsonNode series = requireObject(root, "series");
    String ticker = requireText(series, "ticker");
    String title = requireText(series, "title");
    String category = requireText(series, "category");
    JsonNode settlementSources = requireArray(series, "settlement_sources");
    if (settlementSources.isEmpty()) {
      throw new IllegalArgumentException("Settlement sources are empty");
    }
    JsonNode primarySettlement = settlementSources.get(0);
    String settlementName = requireText(primarySettlement, "name");
    String settlementUrl = requireText(primarySettlement, "url");
    String contractTermsUrl = optionalText(series, "contract_terms_url");
    String contractUrl = optionalText(series, "contract_url");
    String payloadHash = Hashing.sha256Hex(rawJson);
    return new KalshiSeriesPayload(
        ticker,
        title,
        category,
        settlementName,
        settlementUrl,
        contractTermsUrl,
        contractUrl,
        rawJson,
        payloadHash);
  }

  private static JsonNode readTree(ObjectMapper mapper, String rawJson) {
    try {
      return mapper.readTree(rawJson);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to parse Kalshi series JSON", ex);
    }
  }

  private static JsonNode requireObject(JsonNode node, String field) {
    JsonNode value = node.get(field);
    if (value == null || !value.isObject()) {
      throw new IllegalArgumentException("Missing object field: " + field);
    }
    return value;
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
    String text = value.asText();
    return text.isBlank() ? null : text;
  }
}
