package com.predictionmarkets.weather.kalshiapi.api;

import com.predictionmarkets.weather.kalshiapi.http.KalshiHttpClient;
import com.predictionmarkets.weather.kalshiapi.model.market.GetOrderbookResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.GetTradesQuery;
import com.predictionmarkets.weather.kalshiapi.model.market.GetTradesResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.MarketResponse;
import org.springframework.stereotype.Component;

@Component
public class KalshiMarketDataApi {

  private final KalshiHttpClient httpClient;

  public KalshiMarketDataApi(KalshiHttpClient httpClient) {
    this.httpClient = httpClient;
  }

  public MarketResponse getMarket(String ticker) {
    return httpClient.getPublic("/markets/" + ticker, null, MarketResponse.class);
  }

  public GetOrderbookResponse getOrderbook(String ticker) {
    return getOrderbook(ticker, 0);
  }

  public GetOrderbookResponse getOrderbook(String ticker, int depth) {
    int normalizedDepth = Math.max(0, depth);
    if (normalizedDepth > 100) {
      throw new IllegalArgumentException("depth must be between 0 and 100");
    }
    return httpClient.getPublic(
        "/markets/" + ticker + "/orderbook",
        uriBuilder -> uriBuilder.queryParam("depth", normalizedDepth),
        GetOrderbookResponse.class);
  }

  public GetTradesResponse getTrades(GetTradesQuery query) {
    return httpClient.getPublic("/markets/trades", query == null ? null : query::applyTo, GetTradesResponse.class);
  }
}
