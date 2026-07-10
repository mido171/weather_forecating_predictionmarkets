package com.predictionmarkets.weather.weathercom.web.dto;

import java.time.Instant;
import java.time.LocalDate;

public record WeatherComApiCallResponse(
    Long id,
    Long ingestionRunId,
    String requestLocationId,
    String units,
    LocalDate startDate,
    LocalDate endDate,
    Integer httpStatus,
    String errorType,
    String errorMessage,
    Integer durationMs,
    Instant fetchedAtUtc,
    String responseLocationId,
    String responseUnits,
    String responseLanguage,
    String transactionId,
    String apiVersion,
    Long expireTimeGmt
) {
}

