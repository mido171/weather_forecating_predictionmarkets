package com.predictionmarkets.weather.kalshiapi.trading;

public record KalshiPositionExposure(
    String marketTicker,
    Integer netContracts,
    String netSide,
    Long exposureCents,
    Double exposureDollars
) {
}
