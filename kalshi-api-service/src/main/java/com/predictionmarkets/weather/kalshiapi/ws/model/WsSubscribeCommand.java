package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsSubscribeCommand(int id, String cmd, WsSubscribeParams params) {
  public WsSubscribeCommand(int id, WsSubscribeParams params) {
    this(id, "subscribe", params);
  }
}
