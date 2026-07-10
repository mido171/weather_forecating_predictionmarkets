package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsSubscriptionAck(Integer id, String type, List<WsSubscription> subscriptions) {
}
