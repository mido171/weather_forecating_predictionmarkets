package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.model.market.GetOrderbookResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.GetTradesQuery;
import com.predictionmarkets.weather.kalshiapi.model.market.GetTradesResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.MarketResponse;

public interface KalshiMarketDataService {
  MarketResponse getMarket(String ticker);

  GetOrderbookResponse getOrderbook(String ticker, int depth);

  GetTradesResponse getTrades(GetTradesQuery query);
}
