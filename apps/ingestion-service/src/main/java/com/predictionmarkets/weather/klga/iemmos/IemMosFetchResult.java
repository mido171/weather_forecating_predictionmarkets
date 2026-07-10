package com.predictionmarkets.weather.klga.iemmos;

import java.time.Instant;

public record IemMosFetchResult(
    int httpStatus,
    byte[] body,
    String contentType,
    Instant retrievedAtUtc,
    String url) {
}
