package com.predictionmarkets.weather.kalshiapi.model.portfolio;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.common.TimeInForce;
import com.predictionmarkets.weather.kalshiapi.util.FlexibleOffsetDateTimeDeserializer;
import java.time.OffsetDateTime;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Order(
    @JsonProperty("order_id") String orderId,
    String ticker,
    @JsonProperty("market_ticker") String marketTicker,
    Side side,
    Action action,
    OrderType type,
    String status,
    @JsonProperty("client_order_id") String clientOrderId,
    Integer count,
    @JsonProperty("count_fp") String countFp,
    @JsonProperty("remaining_count") Integer remainingCount,
    @JsonProperty("remaining_count_fp") String remainingCountFp,
    @JsonProperty("fill_count") Integer fillCount,
    @JsonProperty("fill_count_fp") String fillCountFp,
    @JsonProperty("yes_price") Integer yesPrice,
    @JsonProperty("no_price") Integer noPrice,
    @JsonProperty("time_in_force") TimeInForce timeInForce,
    @JsonProperty("buy_max_cost") Integer buyMaxCost,
    @JsonProperty("post_only") Boolean postOnly,
    @JsonProperty("reduce_only") Boolean reduceOnly,
    @JsonProperty("queue_position") Long queuePosition,
    Integer subaccount,
    @JsonProperty("created_time")
    @JsonDeserialize(using = FlexibleOffsetDateTimeDeserializer.class)
    OffsetDateTime createdTime,
    @JsonProperty("updated_time")
    @JsonDeserialize(using = FlexibleOffsetDateTimeDeserializer.class)
    OffsetDateTime updatedTime
) {
}
