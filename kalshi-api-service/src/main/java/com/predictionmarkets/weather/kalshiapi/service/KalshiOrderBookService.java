package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.trading.OrderBookSnapshotView;
import java.util.Set;

public interface KalshiOrderBookService {
  void connect();

  void disconnect();

  void replaceTrackedMarkets(Set<String> marketTickers);

  OrderBookSnapshotView snapshot(String marketTicker);
}
