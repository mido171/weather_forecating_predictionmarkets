package com.predictionmarkets.weather.kalshiapi.trading;

import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;

public record TradeOrderRequest(
    String marketTicker,
    Side side,
    int contractsRequested,
    OrderType orderType,
    Integer limitPriceCents,
    Integer buyMaxCostCents,
    String intentId
) {

  public TradeOrderRequest {
    if (marketTicker == null || marketTicker.isBlank()) {
      throw new IllegalArgumentException("marketTicker is required");
    }
    if (side == null) {
      throw new IllegalArgumentException("side is required");
    }
    if (contractsRequested < 1) {
      throw new IllegalArgumentException("contractsRequested must be >= 1");
    }
    if (orderType == null) {
      throw new IllegalArgumentException("orderType is required");
    }
    if (intentId == null || intentId.isBlank()) {
      throw new IllegalArgumentException("intentId is required");
    }
    if (orderType == OrderType.LIMIT && (limitPriceCents == null || limitPriceCents < 1 || limitPriceCents > 99)) {
      throw new IllegalArgumentException("limitPriceCents must be set between 1 and 99 for limit orders");
    }
    if (orderType == OrderType.MARKET && (buyMaxCostCents == null || buyMaxCostCents < 1)) {
      throw new IllegalArgumentException("buyMaxCostCents must be >= 1 for market orders");
    }
  }
}
