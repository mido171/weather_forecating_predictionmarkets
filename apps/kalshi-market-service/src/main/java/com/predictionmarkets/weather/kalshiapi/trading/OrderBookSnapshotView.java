package com.predictionmarkets.weather.kalshiapi.trading;

import java.time.Instant;
import java.util.List;

public record OrderBookSnapshotView(
    String marketTicker,
    List<OrderBookLevelView> yesLevels,
    List<OrderBookLevelView> noLevels,
    Integer bestYesBidCents,
    Integer bestNoBidCents,
    Integer impliedYesAskCents,
    Integer impliedNoAskCents,
    Instant asOfUtc
) {
}
