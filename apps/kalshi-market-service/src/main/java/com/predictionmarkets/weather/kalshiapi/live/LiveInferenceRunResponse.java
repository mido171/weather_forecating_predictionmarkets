package com.predictionmarkets.weather.kalshiapi.live;

public record LiveInferenceRunResponse(
    String targetDateLocal,
    String status,
    Integer exitCode,
    String reportPath,
    String message
) {
}
