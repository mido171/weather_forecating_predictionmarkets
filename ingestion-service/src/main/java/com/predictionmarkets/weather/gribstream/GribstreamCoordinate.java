package com.predictionmarkets.weather.gribstream;

public record GribstreamCoordinate(
    double lat,
    double lon,
    String name) {
}
