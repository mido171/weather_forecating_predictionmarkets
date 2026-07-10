package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record CreateOrderResponse(Order order) {
}
