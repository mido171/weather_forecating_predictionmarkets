package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsSubscription(
    String channel,
    Integer sid,
    @JsonProperty("market_ticker") String marketTicker,
    @JsonProperty("market_tickers") List<String> marketTickers
) {
}
