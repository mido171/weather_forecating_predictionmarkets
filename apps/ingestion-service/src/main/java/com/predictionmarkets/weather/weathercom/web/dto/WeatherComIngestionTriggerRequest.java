package com.predictionmarkets.weather.weathercom.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import java.util.List;

public record WeatherComIngestionTriggerRequest(
    List<@Pattern(
        regexp = "^[^:]+(:[^:]+){2,}$",
        message = "Each locationId must contain at least three colon-delimited segments") String> locationIds,
    @NotBlank
    @Pattern(regexp = "^\\d{8}$", message = "startDate must be yyyyMMdd")
    String startDate,
    @NotBlank
    @Pattern(regexp = "^\\d{8}$", message = "endDate must be yyyyMMdd")
    String endDate,
    @NotBlank
    @Pattern(regexp = "^[eEmMhHsS]$", message = "units must be one of e,m,h,s")
    String units,
    String requestedBy
) {
}

