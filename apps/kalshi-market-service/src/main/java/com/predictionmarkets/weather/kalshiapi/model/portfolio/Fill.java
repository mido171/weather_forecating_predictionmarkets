package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Fill(
    @JsonProperty("fill_id") String fillId,
    @JsonProperty("trade_id") String tradeId,
    @JsonProperty("order_id") String orderId,
    String ticker,
    @JsonProperty("market_ticker") String marketTicker,
    Side side,
    Action action,
    Integer count,
    @JsonProperty("count_fp") String countFp,
    @JsonProperty("client_order_id") String clientOrderId,
    Long ts
) {
}
