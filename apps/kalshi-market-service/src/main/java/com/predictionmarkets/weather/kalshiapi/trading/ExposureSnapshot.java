package com.predictionmarkets.weather.kalshiapi.trading;

public record ExposureSnapshot(
    int filledContracts,
    int restingBuyContracts,
    int allowedContracts,
    int remainingAllowance
) {
}
