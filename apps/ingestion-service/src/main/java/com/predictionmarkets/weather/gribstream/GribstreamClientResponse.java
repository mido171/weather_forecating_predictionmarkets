package com.predictionmarkets.weather.gribstream;

import java.time.Instant;
import java.util.List;

public record GribstreamClientResponse(
    String requestJson,
    String requestSha256,
    String responseSha256,
    Instant retrievedAtUtc,
    List<GribstreamRow> rows) {
}
