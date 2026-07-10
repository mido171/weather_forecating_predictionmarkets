package com.predictionmarkets.weather.kalshiapi.trading;

public record TradePreflightDecision(
    boolean blocked,
    boolean halted,
    String reason,
    int configuredCap,
    int effectiveCap,
    int requestedContracts,
    int allowedContractsToSend,
    String deterministicClientOrderId,
    ExposureSnapshot exposureSnapshot,
    String marketSideKey
) {
}
