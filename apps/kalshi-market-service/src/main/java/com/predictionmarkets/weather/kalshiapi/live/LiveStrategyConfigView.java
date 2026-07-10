package com.predictionmarkets.weather.kalshiapi.live;

import java.util.List;

public record LiveStrategyConfigView(
    String referenceLabel,
    String periodLabel,
    List<String> stationIds,
    double minWinProbability,
    double minEv,
    double minSidePriceProbability,
    String sizingMode,
    double kellyFraction,
    double stakeCapUsd,
    String entryRule,
    String predictionSource
) {
}
