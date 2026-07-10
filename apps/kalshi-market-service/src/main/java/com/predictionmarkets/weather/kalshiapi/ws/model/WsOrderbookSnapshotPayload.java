package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.predictionmarkets.weather.kalshiapi.model.market.DollarPriceLevel;
import com.predictionmarkets.weather.kalshiapi.model.market.DollarPriceLevelFp;
import com.predictionmarkets.weather.kalshiapi.model.market.IntPriceLevel;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsOrderbookSnapshotPayload(
    @JsonProperty("market_ticker") String marketTicker,
    List<IntPriceLevel> yes,
    List<IntPriceLevel> no,
    @JsonProperty("yes_dollars") List<DollarPriceLevel> yesDollars,
    @JsonProperty("no_dollars") List<DollarPriceLevel> noDollars,
    @JsonProperty("yes_dollars_fp") List<DollarPriceLevelFp> yesDollarsFp,
    @JsonProperty("no_dollars_fp") List<DollarPriceLevelFp> noDollarsFp
) {
}
