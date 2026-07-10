package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.predictionmarkets.weather.kalshiapi.util.FlexibleOffsetDateTimeDeserializer;
import java.time.OffsetDateTime;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Trade(
    @JsonProperty("trade_id") String tradeId,
    String ticker,
    Integer price,
    Integer count,
    @JsonProperty("count_fp") String countFp,
    @JsonProperty("yes_price") Integer yesPrice,
    @JsonProperty("no_price") Integer noPrice,
    @JsonProperty("yes_price_dollars") String yesPriceDollars,
    @JsonProperty("no_price_dollars") String noPriceDollars,
    @JsonProperty("taker_side") String takerSide,
    @JsonProperty("created_time") @JsonDeserialize(using = FlexibleOffsetDateTimeDeserializer.class) OffsetDateTime createdTime,
    Long ts
) {
}
