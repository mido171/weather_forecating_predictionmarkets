package com.predictionmarkets.weather.kalshiapi.trading;

import java.math.BigDecimal;

public record OrderBookLevelView(int priceCents, BigDecimal quantity) {
}
