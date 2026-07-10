package com.predictionmarkets.weather.kalshiapi.http;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.model.ApiError;
import org.springframework.stereotype.Component;

@Component
public class KalshiErrorParser {

  private final ObjectMapper objectMapper;

  public KalshiErrorParser(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  public ApiError parse(String responseBody) {
    if (responseBody == null || responseBody.isBlank()) {
      return null;
    }

    try {
      JsonNode root = objectMapper.readTree(responseBody);
      JsonNode errorNode = root.has("error") ? root.get("error") : root;
      if (errorNode == null || errorNode.isMissingNode()) {
        return null;
      }

      Integer code = errorNode.has("code") && errorNode.get("code").canConvertToInt()
          ? errorNode.get("code").asInt()
          : null;

      String message = errorNode.hasNonNull("message") ? errorNode.get("message").asText() : null;
      String msg = errorNode.hasNonNull("msg") ? errorNode.get("msg").asText() : null;

      JsonNode details = errorNode.has("details") ? errorNode.get("details") : null;
      String service = errorNode.hasNonNull("service") ? errorNode.get("service").asText() : null;

      return new ApiError(code, msg, message, details, service);
    } catch (Exception ignored) {
      return null;
    }
  }
}
