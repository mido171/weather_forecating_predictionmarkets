package com.predictionmarkets.weather.gribstream;

import java.util.Map;

public record GribstreamDailyOpinionResult(
    Map<String, Double> tmaxByModelF,
    Double gefsSpreadF) {
}
