package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record MarketPosition(
    @JsonProperty("market_ticker") String marketTicker,
    Integer position,
    @JsonProperty("position_fp") String positionFp
) {
}
