package com.predictionmarkets.weather.kalshiapi.model.common;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum TimeInForce {
  @JsonProperty("fill_or_kill")
  FILL_OR_KILL,
  @JsonProperty("good_till_canceled")
  GOOD_TILL_CANCELED,
  @JsonProperty("immediate_or_cancel")
  IMMEDIATE_OR_CANCEL
}
