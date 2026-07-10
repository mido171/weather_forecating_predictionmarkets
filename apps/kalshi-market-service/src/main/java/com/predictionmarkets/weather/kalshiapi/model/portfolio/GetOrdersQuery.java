package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.predictionmarkets.weather.kalshiapi.model.common.OrderStatus;
import com.predictionmarkets.weather.kalshiapi.util.KalshiQueryUtils;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.web.util.UriBuilder;

public record GetOrdersQuery(Integer limit, String cursor, List<OrderStatus> status) {

  public void applyTo(UriBuilder uriBuilder) {
    validate();
    KalshiQueryUtils.addParam(uriBuilder, "limit", limit);
    KalshiQueryUtils.addParam(uriBuilder, "cursor", cursor);
    if (status != null && !status.isEmpty()) {
      String statusParam = status.stream()
          .map(value -> value.name().toLowerCase())
          .collect(Collectors.joining(","));
      KalshiQueryUtils.addParam(uriBuilder, "status", statusParam);
    }
  }

  private void validate() {
    if (limit != null && (limit < 1 || limit > 1_000)) {
      throw new IllegalArgumentException("limit must be between 1 and 1000");
    }
  }
}
