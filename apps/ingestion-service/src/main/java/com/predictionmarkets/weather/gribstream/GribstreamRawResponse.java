package com.predictionmarkets.weather.gribstream;

import java.time.Instant;

public record GribstreamRawResponse(
    String requestJson,
    String requestSha256,
    String responseSha256,
    Instant retrievedAtUtc,
    byte[] responseBytes) {
}
