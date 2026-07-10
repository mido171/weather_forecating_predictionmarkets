package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsErrorDetails(Integer code, String msg) {
}
