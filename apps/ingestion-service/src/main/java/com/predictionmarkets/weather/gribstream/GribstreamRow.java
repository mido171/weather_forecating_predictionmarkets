package com.predictionmarkets.weather.gribstream;

import java.time.Instant;

public record GribstreamRow(
    Instant forecastedAt,
    Instant forecastedTime,
    double tmpk,
    Integer member) {
}
