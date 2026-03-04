package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CancelOrderResponse;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderResult;

public interface KalshiTradingService {
  TradeOrderResult placeMarketBuy(TradeOrderRequest request);

  TradeOrderResult placeLimitBuy(TradeOrderRequest request);

  CancelOrderResponse cancelOrder(String orderId);

  void resetTradingHalt(String marketTicker, Side side);
}
