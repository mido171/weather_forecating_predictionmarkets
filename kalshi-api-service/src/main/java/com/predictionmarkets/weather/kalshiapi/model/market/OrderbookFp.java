package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record OrderbookFp(
    @JsonProperty("yes_dollars") List<DollarPriceLevelFp> yesDollars,
    @JsonProperty("no_dollars") List<DollarPriceLevelFp> noDollars
) {
}
