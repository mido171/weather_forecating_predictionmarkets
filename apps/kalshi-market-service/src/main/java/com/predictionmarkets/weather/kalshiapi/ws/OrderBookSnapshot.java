package com.predictionmarkets.weather.kalshiapi.ws;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.NavigableMap;

public record OrderBookSnapshot(
    String marketTicker,
    NavigableMap<Integer, BigDecimal> yesSide,
    NavigableMap<Integer, BigDecimal> noSide,
    Instant asOfUtc
) {
}
