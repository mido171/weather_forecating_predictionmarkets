package com.predictionmarkets.weather.gribstream;

public record GribstreamVariable(
    String name,
    String level,
    String info,
    String alias) {
}
