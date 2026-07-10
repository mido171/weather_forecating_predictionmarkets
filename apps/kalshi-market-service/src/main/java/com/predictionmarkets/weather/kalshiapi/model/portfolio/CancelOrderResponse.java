package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record CancelOrderResponse(
    Order order,
    @JsonProperty("reduced_by") Integer reducedBy,
    @JsonProperty("reduced_by_fp") String reducedByFp
) {
}
