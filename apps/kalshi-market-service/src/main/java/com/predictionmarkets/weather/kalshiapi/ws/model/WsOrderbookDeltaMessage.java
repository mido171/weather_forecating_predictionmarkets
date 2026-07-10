package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.JsonNode;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsOrderbookDeltaMessage(String type, Integer sid, Long seq, JsonNode msg) {
}
