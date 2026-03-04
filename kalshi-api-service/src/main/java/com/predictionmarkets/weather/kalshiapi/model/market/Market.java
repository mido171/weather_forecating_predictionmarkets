package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.predictionmarkets.weather.kalshiapi.util.FlexibleOffsetDateTimeDeserializer;
import java.time.OffsetDateTime;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Market(
    String ticker,
    @JsonProperty("event_ticker") String eventTicker,
    String title,
    String subtitle,
    String status,
    @JsonProperty("close_time") @JsonDeserialize(using = FlexibleOffsetDateTimeDeserializer.class) OffsetDateTime closeTime,
    @JsonProperty("yes_ask") Integer yesAsk,
    @JsonProperty("no_ask") Integer noAsk,
    @JsonProperty("yes_ask_dollars") String yesAskDollars,
    @JsonProperty("no_ask_dollars") String noAskDollars,
    @JsonProperty("yes_bid") Integer yesBid,
    @JsonProperty("no_bid") Integer noBid,
    @JsonProperty("yes_bid_dollars") String yesBidDollars,
    @JsonProperty("no_bid_dollars") String noBidDollars,
    @JsonProperty("tick_size") Integer tickSize,
    @JsonProperty("tick_size_dollars") String tickSizeDollars,
    @JsonProperty("open_interest") Long openInterest,
    @JsonProperty("open_interest_fp") String openInterestFp,
    Long volume,
    @JsonProperty("volume_fp") String volumeFp
) {
}
