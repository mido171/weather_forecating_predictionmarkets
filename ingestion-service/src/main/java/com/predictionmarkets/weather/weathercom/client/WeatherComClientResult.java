package com.predictionmarkets.weather.weathercom.client;

import com.predictionmarkets.weather.weathercom.client.dto.WeatherComHistoricalResponse;
import java.time.Instant;

public record WeatherComClientResult(
    boolean success,
    int httpStatus,
    WeatherComHistoricalResponse payload,
    String responseBody,
    String errorType,
    String errorMessage,
    Instant fetchedAtUtc,
    int durationMs,
    int attempts
) {
}

