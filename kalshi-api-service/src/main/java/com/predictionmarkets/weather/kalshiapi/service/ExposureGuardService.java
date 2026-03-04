package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.trading.ExposureSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradePreflightDecision;

public interface ExposureGuardService {
  TradePreflightDecision preflightBuy(TradeOrderRequest request);

  ExposureSnapshot recomputeExposure(String marketTicker, Side side, int allowedContracts);

  void verifyPostSubmitOrHalt(String marketTicker, Side side, int allowedContracts);

  boolean isTradingHalted(String marketTicker, Side side);

  void haltTrading(String marketTicker, Side side, String reason);

  int configuredCapForMarket(String marketTicker);

  void resetTradingHalt(String marketTicker, Side side);
}
