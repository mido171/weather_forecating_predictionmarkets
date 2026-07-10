package com.predictionmarkets.weather.weathercom.web.dto;

import java.time.Instant;

public record WeatherComIngestionRunResponse(
    Long id,
    String status,
    Instant startedAtUtc,
    Instant finishedAtUtc,
    String requestedBy,
    int totalTasks,
    int succeededTasks,
    int failedTasks,
    Instant createdAtUtc,
    Instant updatedAtUtc
) {
}

