package com.predictionmarkets.weather.kalshiapi.model.common;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum OrderType {
  @JsonProperty("limit")
  LIMIT,
  @JsonProperty("market")
  MARKET
}
