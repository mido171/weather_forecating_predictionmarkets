package com.predictionmarkets.weather.kalshiapi.live;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

public record LiveStationOrderbookView(
    String stationId,
    String displayName,
    String seriesTicker,
    String zoneId,
    LocalDate targetDateLocal,
    String eventTicker,
    Instant resolvedAtUtc,
    String inferenceRuntimeUtc,
    Double predictionPointTmaxF,
    Map<String, Double> predictionQuantiles,
    List<LiveBucketOrderbookView> buckets
) {
}
