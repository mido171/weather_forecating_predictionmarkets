package com.predictionmarkets.weather.kalshiapi.live;

public record LiveOpportunityView(
    String stationId,
    String marketTicker,
    String bucketLabel,
    String side,
    Double modelWinProbability,
    Double marketPriceProbability,
    Integer entryPriceCents,
    Double ev
) {
}

