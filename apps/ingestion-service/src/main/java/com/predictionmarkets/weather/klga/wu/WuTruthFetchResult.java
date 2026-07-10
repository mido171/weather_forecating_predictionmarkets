package com.predictionmarkets.weather.klga.wu;

import java.time.Instant;
import java.time.LocalDate;

record WuTruthFetchResult(
    WuTruthStation station,
    LocalDate startDate,
    LocalDate endDate,
    boolean success,
    int httpStatus,
    String responseBody,
    String errorType,
    String errorMessage,
    Instant fetchedAtUtc,
    int attempts,
    String sourceUrlRedacted
) {
}
