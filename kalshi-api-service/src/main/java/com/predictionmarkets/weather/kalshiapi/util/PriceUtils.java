package com.predictionmarkets.weather.kalshiapi.util;

import java.math.BigDecimal;
import java.math.RoundingMode;
import org.springframework.util.StringUtils;

public final class PriceUtils {

  private PriceUtils() {
  }

  public static Integer dollarsToCents(String dollars) {
    if (!StringUtils.hasText(dollars)) {
      return null;
    }
    try {
      BigDecimal value = new BigDecimal(dollars.trim());
      return dollarsToCents(value);
    } catch (Exception ex) {
      return null;
    }
  }

  public static Integer dollarsToCents(BigDecimal dollars) {
    if (dollars == null) {
      return null;
    }
    BigDecimal cents = dollars.movePointRight(2);
    return cents.setScale(0, RoundingMode.HALF_UP).intValue();
  }

  public static BigDecimal parseDecimal(String value) {
    if (!StringUtils.hasText(value)) {
      return BigDecimal.ZERO;
    }
    try {
      return new BigDecimal(value.trim());
    } catch (NumberFormatException ex) {
      return BigDecimal.ZERO;
    }
  }

  public static int clampPrice(int priceCents) {
    if (priceCents < 1) {
      return 1;
    }
    if (priceCents > 99) {
      return 99;
    }
    return priceCents;
  }
}
