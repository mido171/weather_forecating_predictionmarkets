package com.predictionmarkets.weather.kalshiapi.api;

import com.predictionmarkets.weather.kalshiapi.http.KalshiHttpClient;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetBalanceResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetFillsQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetFillsResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetPositionsQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetPositionsResponse;
import org.springframework.stereotype.Component;

@Component
public class KalshiPortfolioApi {

  private final KalshiHttpClient httpClient;

  public KalshiPortfolioApi(KalshiHttpClient httpClient) {
    this.httpClient = httpClient;
  }

  public GetBalanceResponse getBalance() {
    return httpClient.getAuth("/portfolio/balance", null, GetBalanceResponse.class);
  }

  public GetPositionsResponse getPositions(GetPositionsQuery query) {
    return httpClient.getAuth("/portfolio/positions", query == null ? null : query::applyTo, GetPositionsResponse.class);
  }

  public GetFillsResponse getFills(GetFillsQuery query) {
    return httpClient.getAuth("/portfolio/fills", query == null ? null : query::applyTo, GetFillsResponse.class);
  }
}
