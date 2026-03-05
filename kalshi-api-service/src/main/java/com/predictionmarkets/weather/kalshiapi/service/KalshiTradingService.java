package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CancelOrderResponse;
import com.predictionmarkets.weather.kalshiapi.trading.KalshiAccountBalance;
import com.predictionmarkets.weather.kalshiapi.trading.KalshiAccountSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.KalshiPositionExposure;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderResult;
import java.util.List;

public interface KalshiTradingService {
  TradeOrderResult placeMarketBuy(TradeOrderRequest request);

  TradeOrderResult placeLimitBuy(TradeOrderRequest request);

  CancelOrderResponse cancelOrder(String orderId);

  void resetTradingHalt(String marketTicker, Side side);

  KalshiAccountBalance getAccountBalance();

  List<KalshiPositionExposure> getOpenPositionsWithExposure();

  KalshiAccountSnapshot getAccountSnapshot();
}
