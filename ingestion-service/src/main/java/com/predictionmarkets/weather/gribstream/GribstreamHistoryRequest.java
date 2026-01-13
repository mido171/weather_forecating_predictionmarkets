package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record GribstreamHistoryRequest(
    String fromTime,
    String untilTime,
    String asOf,
    int minHorizon,
    int maxHorizon,
    List<GribstreamCoordinate> coordinates,
    List<GribstreamVariable> variables,
    List<Integer> members) {
}
