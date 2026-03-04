package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Collections;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record BatchCancelOrdersResponse(
    @JsonProperty("orders") List<BatchCancelOrderResult> orders,
    @JsonProperty("results") List<BatchCancelOrderResult> results
) {
  public List<BatchCancelOrderResult> effectiveResults() {
    if (orders != null) {
      return orders;
    }
    if (results != null) {
      return results;
    }
    return Collections.emptyList();
  }
}
