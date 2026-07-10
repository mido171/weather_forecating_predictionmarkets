package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record GetOrderbookResponse(
    Orderbook orderbook,
    @JsonProperty("orderbook_fp") OrderbookFp orderbookFp
) {
}
