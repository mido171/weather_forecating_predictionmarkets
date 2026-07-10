package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.common.TimeInForce;

@JsonIgnoreProperties(ignoreUnknown = true)
public record CreateOrderRequest(
    String ticker,
    Side side,
    Action action,
    @JsonProperty("client_order_id") String clientOrderId,
    Integer count,
    @JsonProperty("count_fp") String countFp,
    OrderType type,
    @JsonProperty("yes_price") Integer yesPrice,
    @JsonProperty("no_price") Integer noPrice,
    @JsonProperty("expiration_ts") Long expirationTs,
    @JsonProperty("time_in_force") TimeInForce timeInForce,
    @JsonProperty("buy_max_cost") Integer buyMaxCost,
    @JsonProperty("post_only") Boolean postOnly,
    @JsonProperty("reduce_only") Boolean reduceOnly,
    @JsonProperty("cancel_order_on_pause") Boolean cancelOrderOnPause,
    Integer subaccount
) {

  public CreateOrderRequest {
    if (ticker == null || ticker.isBlank()) {
      throw new IllegalArgumentException("ticker is required");
    }
    if (side == null) {
      throw new IllegalArgumentException("side is required");
    }
    if (action == null) {
      throw new IllegalArgumentException("action is required");
    }
    if (count == null || count < 1) {
      throw new IllegalArgumentException("count must be at least 1");
    }
    if (type == null) {
      type = OrderType.LIMIT;
    }
    if (subaccount == null) {
      subaccount = 0;
    }

    int priceFields = 0;
    if (yesPrice != null) {
      validatePriceCents(yesPrice, "yes_price");
      priceFields++;
    }
    if (noPrice != null) {
      validatePriceCents(noPrice, "no_price");
      priceFields++;
    }

    if (type == OrderType.LIMIT) {
      if (priceFields != 1) {
        throw new IllegalArgumentException("A limit order requires exactly one side price field");
      }
    } else {
      if (priceFields != 0) {
        throw new IllegalArgumentException("A market order must not include side price fields");
      }
      if (buyMaxCost == null || buyMaxCost < 1) {
        throw new IllegalArgumentException("A market order requires buy_max_cost >= 1");
      }
    }

    if (buyMaxCost != null && buyMaxCost < 1) {
      throw new IllegalArgumentException("buy_max_cost must be >= 1 when set");
    }
  }

  private static void validatePriceCents(Integer cents, String fieldName) {
    if (cents < 1 || cents > 99) {
      throw new IllegalArgumentException(fieldName + " must be between 1 and 99 cents");
    }
  }
}
