package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.api.KalshiMarketDataApi;
import com.predictionmarkets.weather.kalshiapi.model.market.GetOrderbookResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.GetTradesQuery;
import com.predictionmarkets.weather.kalshiapi.model.market.GetTradesResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.MarketResponse;
import org.springframework.stereotype.Service;

@Service
public class DefaultKalshiMarketDataService implements KalshiMarketDataService {

  private final KalshiMarketDataApi marketDataApi;

  public DefaultKalshiMarketDataService(KalshiMarketDataApi marketDataApi) {
    this.marketDataApi = marketDataApi;
  }

  @Override
  public MarketResponse getMarket(String ticker) {
    return marketDataApi.getMarket(ticker);
  }

  @Override
  public GetOrderbookResponse getOrderbook(String ticker, int depth) {
    return marketDataApi.getOrderbook(ticker, depth);
  }

  @Override
  public GetTradesResponse getTrades(GetTradesQuery query) {
    return marketDataApi.getTrades(query);
  }
}
