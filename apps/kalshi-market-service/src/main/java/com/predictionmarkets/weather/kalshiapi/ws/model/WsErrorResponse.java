package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsErrorResponse(Integer id, String type, WsErrorDetails msg) {
}
