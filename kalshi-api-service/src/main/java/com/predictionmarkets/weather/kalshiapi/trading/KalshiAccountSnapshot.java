package com.predictionmarkets.weather.kalshiapi.trading;

import java.time.Instant;
import java.util.List;

public record KalshiAccountSnapshot(
    Instant asOfUtc,
    KalshiAccountBalance balance,
    Long totalExposureCents,
    Double totalExposureDollars,
    List<KalshiPositionExposure> positions
) {
}
