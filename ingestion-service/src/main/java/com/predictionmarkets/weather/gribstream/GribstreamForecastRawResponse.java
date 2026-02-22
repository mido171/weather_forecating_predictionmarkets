package com.predictionmarkets.weather.gribstream;

import java.time.Instant;

public record GribstreamForecastRawResponse(
    String requestJson,
    String requestSha256,
    String responseSha256,
    Instant retrievedAtUtc,
    int statusCode,
    byte[] responseBytes) {
}
