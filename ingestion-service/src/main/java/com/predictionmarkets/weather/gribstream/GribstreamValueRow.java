package com.predictionmarkets.weather.gribstream;

import java.time.Instant;
import java.util.Map;

public record GribstreamValueRow(
    Instant forecastedAt,
    Instant forecastedTime,
    Integer member,
    Map<String, String> values) {
}
