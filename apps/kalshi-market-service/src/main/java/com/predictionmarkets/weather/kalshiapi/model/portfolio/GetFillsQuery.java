package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.predictionmarkets.weather.kalshiapi.util.KalshiQueryUtils;
import org.springframework.web.util.UriBuilder;

public record GetFillsQuery(
    String ticker,
    String orderId,
    Long minTs,
    Long maxTs,
    Integer limit,
    String cursor,
    Integer subaccount
) {

  public void applyTo(UriBuilder uriBuilder) {
    validate();
    KalshiQueryUtils.addParam(uriBuilder, "ticker", ticker);
    KalshiQueryUtils.addParam(uriBuilder, "order_id", orderId);
    KalshiQueryUtils.addParam(uriBuilder, "min_ts", minTs);
    KalshiQueryUtils.addParam(uriBuilder, "max_ts", maxTs);
    KalshiQueryUtils.addParam(uriBuilder, "limit", limit);
    KalshiQueryUtils.addParam(uriBuilder, "cursor", cursor);
    KalshiQueryUtils.addParam(uriBuilder, "subaccount", subaccount);
  }

  private void validate() {
    if (limit != null && (limit < 1 || limit > 200)) {
      throw new IllegalArgumentException("limit must be between 1 and 200");
    }
    if (subaccount != null && subaccount < 0) {
      throw new IllegalArgumentException("subaccount must be non-negative");
    }
  }
}
