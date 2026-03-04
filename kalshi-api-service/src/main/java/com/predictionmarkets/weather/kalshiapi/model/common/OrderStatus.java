package com.predictionmarkets.weather.kalshiapi.model.common;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum OrderStatus {
  @JsonProperty("resting")
  RESTING,
  @JsonProperty("canceled")
  CANCELED,
  @JsonProperty("executed")
  EXECUTED
}
