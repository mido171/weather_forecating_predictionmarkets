package com.predictionmarkets.weather.kalshiapi.trading;

public record KalshiAccountBalance(
    Long balanceCents,
    Double balanceDollars,
    Long portfolioValueCents,
    Double portfolioValueDollars,
    Long updatedTs
) {
}
