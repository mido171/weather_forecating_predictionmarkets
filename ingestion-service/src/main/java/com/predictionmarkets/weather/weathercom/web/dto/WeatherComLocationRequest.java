package com.predictionmarkets.weather.weathercom.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;

public record WeatherComLocationRequest(
    @NotBlank
    @Pattern(regexp = "^[^:]+(:[^:]+){2,}$",
        message = "locationId must contain at least three colon-delimited segments")
    String locationId,
    String displayName,
    Boolean active
) {
}

