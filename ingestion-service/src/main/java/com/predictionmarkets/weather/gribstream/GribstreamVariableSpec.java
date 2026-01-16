package com.predictionmarkets.weather.gribstream;

public record GribstreamVariableSpec(
    String name,
    String level,
    String info,
    int minHorizonHours,
    int maxHorizonHours) {
}
