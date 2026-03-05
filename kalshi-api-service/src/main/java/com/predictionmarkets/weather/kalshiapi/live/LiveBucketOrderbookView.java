package com.predictionmarkets.weather.kalshiapi.live;

import com.predictionmarkets.weather.kalshiapi.trading.OrderBookLevelView;
import java.time.Instant;
import java.util.List;

public record LiveBucketOrderbookView(
    String marketTicker,
    String bucketLabel,
    String marketStatus,
    Integer yesBidCents,
    Integer yesAskCents,
    Integer yesSpreadCents,
    Double yesModelWinProbability,
    Double yesEv,
    Integer noBidCents,
    Integer noAskCents,
    Integer noSpreadCents,
    Double noModelWinProbability,
    Double noEv,
    Integer midYesCents,
    Instant bookAsOfUtc,
    List<OrderBookLevelView> yesTopLevels,
    List<OrderBookLevelView> noTopLevels
) {
}
