package com.predictionmarkets.weather.iem;

import java.math.BigDecimal;

public record IemMosValue(String rawValue, BigDecimal numericValue) {
}
