package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Orderbook(
    List<IntPriceLevel> yes,
    List<IntPriceLevel> no,
    @JsonProperty("yes_dollars") List<DollarPriceLevel> yesDollars,
    @JsonProperty("no_dollars") List<DollarPriceLevel> noDollars,
    @JsonProperty("yes_dollars_fp") List<DollarPriceLevelFp> yesDollarsFp,
    @JsonProperty("no_dollars_fp") List<DollarPriceLevelFp> noDollarsFp
) {
}
