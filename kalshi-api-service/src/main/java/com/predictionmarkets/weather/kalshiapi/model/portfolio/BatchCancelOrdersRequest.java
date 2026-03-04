package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record BatchCancelOrdersRequest(List<String> ids) {
  public BatchCancelOrdersRequest {
    if (ids == null || ids.isEmpty()) {
      throw new IllegalArgumentException("ids must not be empty");
    }
    if (ids.size() > 20) {
      throw new IllegalArgumentException("Batch cancel supports at most 20 order ids");
    }
  }
}
