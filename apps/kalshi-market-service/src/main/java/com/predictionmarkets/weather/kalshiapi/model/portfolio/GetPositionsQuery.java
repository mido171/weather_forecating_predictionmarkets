package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.predictionmarkets.weather.kalshiapi.util.KalshiQueryUtils;
import org.springframework.web.util.UriBuilder;

public record GetPositionsQuery(String cursor, Integer limit, String countFilter) {

  public void applyTo(UriBuilder uriBuilder) {
    validate();
    KalshiQueryUtils.addParam(uriBuilder, "cursor", cursor);
    KalshiQueryUtils.addParam(uriBuilder, "limit", limit);
    KalshiQueryUtils.addParam(uriBuilder, "count_filter", countFilter);
  }

  private void validate() {
    if (limit != null && (limit < 1 || limit > 1_000)) {
      throw new IllegalArgumentException("limit must be between 1 and 1000");
    }
  }
}
