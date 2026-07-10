package com.predictionmarkets.weather.iem;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;

public record IemCliDaily(
    String stationId,
    LocalDate targetDateLocal,
    BigDecimal tmaxF,
    BigDecimal tminF,
    Instant reportIssuedAtUtc,
    String truthSourceUrl) {
}
