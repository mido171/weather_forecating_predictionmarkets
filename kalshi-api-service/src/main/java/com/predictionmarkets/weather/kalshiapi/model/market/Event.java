package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Event(
    @JsonProperty("event_ticker") String eventTicker,
    String title,
    @JsonProperty("sub_title") String subTitle
) {
}
