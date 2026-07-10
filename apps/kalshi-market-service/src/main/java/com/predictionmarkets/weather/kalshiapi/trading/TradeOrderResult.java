package com.predictionmarkets.weather.kalshiapi.trading;

public record TradeOrderResult(
    boolean submitted,
    boolean blockedByGuardrail,
    boolean halted,
    String reason,
    String orderId,
    String clientOrderId,
    int requestedContracts,
    int submittedContracts,
    ExposureSnapshot exposureSnapshot
) {
}
