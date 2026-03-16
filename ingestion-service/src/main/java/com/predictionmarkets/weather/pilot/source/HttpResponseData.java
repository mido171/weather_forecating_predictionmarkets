package com.predictionmarkets.weather.pilot.source;

import java.util.Map;

public record HttpResponseData(
    int statusCode,
    byte[] body,
    double durationMs,
    String finalUrl,
    Map<String, String> headers,
    int retryCount) {
}
