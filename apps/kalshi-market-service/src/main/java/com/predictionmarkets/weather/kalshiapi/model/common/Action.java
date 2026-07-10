package com.predictionmarkets.weather.kalshiapi.model.common;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum Action {
  @JsonProperty("buy")
  BUY,
  @JsonProperty("sell")
  SELL
}
