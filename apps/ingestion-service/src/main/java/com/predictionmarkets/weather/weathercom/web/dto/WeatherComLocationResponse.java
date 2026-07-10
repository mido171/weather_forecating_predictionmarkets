package com.predictionmarkets.weather.weathercom.web.dto;

import java.time.Instant;

public record WeatherComLocationResponse(
    Long id,
    String locationId,
    String displayName,
    boolean active,
    Instant createdAtUtc,
    Instant updatedAtUtc
) {
}

