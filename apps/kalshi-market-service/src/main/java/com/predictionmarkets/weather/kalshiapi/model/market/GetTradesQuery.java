package com.predictionmarkets.weather.kalshiapi.model.market;

import com.predictionmarkets.weather.kalshiapi.util.KalshiQueryUtils;
import org.springframework.web.util.UriBuilder;

public record GetTradesQuery(Integer limit, String cursor, String ticker, Long minTs, Long maxTs) {

  public void applyTo(UriBuilder uriBuilder) {
    validate();
    KalshiQueryUtils.addParam(uriBuilder, "limit", limit);
    KalshiQueryUtils.addParam(uriBuilder, "cursor", cursor);
    KalshiQueryUtils.addParam(uriBuilder, "ticker", ticker);
    KalshiQueryUtils.addParam(uriBuilder, "min_ts", minTs);
    KalshiQueryUtils.addParam(uriBuilder, "max_ts", maxTs);
  }

  private void validate() {
    if (limit != null && (limit < 1 || limit > 1_000)) {
      throw new IllegalArgumentException("limit must be between 1 and 1000");
    }
  }
}
