package com.predictionmarkets.weather.kalshiapi.model;

import com.fasterxml.jackson.databind.JsonNode;

public record ApiError(Integer code, String msg, String message, JsonNode details, String service) {
  public String resolvedMessage() {
    if (message != null && !message.isBlank()) {
      return message;
    }
    if (msg != null && !msg.isBlank()) {
      return msg;
    }
    return null;
  }
}
