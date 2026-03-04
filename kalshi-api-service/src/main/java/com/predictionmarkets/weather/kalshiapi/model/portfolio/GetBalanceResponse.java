package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record GetBalanceResponse(
    Long balance,
    @JsonProperty("portfolio_value") Long portfolioValue,
    @JsonProperty("updated_ts") Long updatedTs
) {
}
