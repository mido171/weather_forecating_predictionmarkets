package com.predictionmarkets.weather.weathercom.web.dto;

import java.time.Instant;

public record WeatherComErrorResponse(
    Instant timestampUtc,
    int status,
    String error,
    String message,
    String path
) {
}

