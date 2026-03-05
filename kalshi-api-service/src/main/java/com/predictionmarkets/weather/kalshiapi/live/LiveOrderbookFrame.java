package com.predictionmarkets.weather.kalshiapi.live;

import java.time.Instant;
import java.util.List;

public record LiveOrderbookFrame(
    Instant asOfUtc,
    List<LiveStationOrderbookView> stations,
    List<LiveOpportunityView> opportunities
) {
}
