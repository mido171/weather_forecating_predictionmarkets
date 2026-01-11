package com.predictionmarkets.weather.iem;

import java.math.BigDecimal;
import java.time.Instant;

public record IemMosEntry(
    Instant runtimeUtc,
    Instant forecastTimeUtc,
    BigDecimal nX) {
}
