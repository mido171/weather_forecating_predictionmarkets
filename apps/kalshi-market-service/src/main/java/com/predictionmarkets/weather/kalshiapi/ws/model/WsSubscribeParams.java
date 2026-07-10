package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsSubscribeParams(
    List<String> channels,
    @JsonProperty("market_ticker") String marketTicker,
    @JsonProperty("market_tickers") List<String> marketTickers
) {
}
