package com.predictionmarkets.weather.weathercom.web.dto;

import java.math.BigDecimal;
import java.time.Instant;

public record WeatherComObservationResponse(
    Long id,
    Long apiCallId,
    String requestLocationId,
    String obsId,
    String obsName,
    long validTimeGmt,
    Instant validTimeUtc,
    Integer temp,
    Integer dewPt,
    Integer rh,
    BigDecimal pressure,
    Integer wspd,
    String wxPhrase,
    Instant createdAtUtc,
    Instant updatedAtUtc
) {
}

