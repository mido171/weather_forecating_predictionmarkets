package com.predictionmarkets.weather.kalshiapi.ws;

import com.fasterxml.jackson.databind.JsonNode;
import com.predictionmarkets.weather.kalshiapi.model.market.DollarPriceLevel;
import com.predictionmarkets.weather.kalshiapi.model.market.DollarPriceLevelFp;
import com.predictionmarkets.weather.kalshiapi.model.market.IntPriceLevel;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsOrderbookSnapshotPayload;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Comparator;
import java.util.List;
import java.util.NavigableMap;
import java.util.TreeMap;

public class OrderBook {

  private final String marketTicker;
  private final NavigableMap<Integer, BigDecimal> yesSide = new TreeMap<>(Comparator.reverseOrder());
  private final NavigableMap<Integer, BigDecimal> noSide = new TreeMap<>(Comparator.reverseOrder());

  public OrderBook(String marketTicker) {
    this.marketTicker = marketTicker;
  }

  public String marketTicker() {
    return marketTicker;
  }

  public synchronized void applySnapshot(WsOrderbookSnapshotPayload payload) {
    yesSide.clear();
    noSide.clear();

    applyIntLevels(payload.yes(), yesSide);
    applyIntLevels(payload.no(), noSide);
    applyDollarLevels(payload.yesDollars(), yesSide);
    applyDollarLevels(payload.noDollars(), noSide);
    applyDollarFpLevels(payload.yesDollarsFp(), yesSide);
    applyDollarFpLevels(payload.noDollarsFp(), noSide);
  }

  public synchronized void applyDelta(JsonNode deltaMsg) {
    if (deltaMsg == null || deltaMsg.isNull()) {
      return;
    }
    applyDeltaForKey(deltaMsg, "yes", yesSide, false);
    applyDeltaForKey(deltaMsg, "no", noSide, false);
    applyDeltaForKey(deltaMsg, "yes_dollars", yesSide, true);
    applyDeltaForKey(deltaMsg, "no_dollars", noSide, true);
    applyDeltaForKey(deltaMsg, "yes_dollars_fp", yesSide, true);
    applyDeltaForKey(deltaMsg, "no_dollars_fp", noSide, true);
  }

  public synchronized NavigableMap<Integer, BigDecimal> yesSideSnapshot() {
    return new TreeMap<>(yesSide);
  }

  public synchronized NavigableMap<Integer, BigDecimal> noSideSnapshot() {
    return new TreeMap<>(noSide);
  }

  private void applyIntLevels(List<IntPriceLevel> levels, NavigableMap<Integer, BigDecimal> side) {
    if (levels == null) {
      return;
    }
    for (IntPriceLevel level : levels) {
      if (level == null || level.price() == null || level.quantity() == null) {
        continue;
      }
      upsertLevel(side, level.price(), BigDecimal.valueOf(level.quantity()));
    }
  }

  private void applyDollarLevels(List<DollarPriceLevel> levels, NavigableMap<Integer, BigDecimal> side) {
    if (levels == null) {
      return;
    }
    for (DollarPriceLevel level : levels) {
      if (level == null || level.price() == null || level.quantity() == null) {
        continue;
      }
      Integer priceCents = dollarsToCentsSafe(level.price());
      if (priceCents == null) {
        continue;
      }
      upsertLevel(side, priceCents, BigDecimal.valueOf(level.quantity()));
    }
  }

  private void applyDollarFpLevels(List<DollarPriceLevelFp> levels, NavigableMap<Integer, BigDecimal> side) {
    if (levels == null) {
      return;
    }
    for (DollarPriceLevelFp level : levels) {
      if (level == null || level.price() == null || level.quantity() == null) {
        continue;
      }
      Integer priceCents = dollarsToCentsSafe(level.price());
      if (priceCents == null) {
        continue;
      }
      BigDecimal quantity = parseDecimal(level.quantity());
      upsertLevel(side, priceCents, quantity);
    }
  }

  private void applyDeltaForKey(JsonNode deltaMsg,
                                String key,
                                NavigableMap<Integer, BigDecimal> side,
                                boolean dollarsVariant) {
    JsonNode levelsNode = deltaMsg.get(key);
    if (levelsNode == null || !levelsNode.isArray()) {
      return;
    }
    for (JsonNode levelNode : levelsNode) {
      if (!levelNode.isArray() || levelNode.size() < 2) {
        continue;
      }
      JsonNode priceNode = levelNode.get(0);
      JsonNode quantityNode = levelNode.get(1);
      Integer priceCents = parsePrice(priceNode, dollarsVariant);
      BigDecimal quantity = parseQuantity(quantityNode);
      if (priceCents == null || quantity == null) {
        continue;
      }
      upsertLevel(side, priceCents, quantity);
    }
  }

  private void upsertLevel(NavigableMap<Integer, BigDecimal> side, int priceCents, BigDecimal quantity) {
    if (quantity.signum() <= 0) {
      side.remove(priceCents);
    } else {
      side.put(priceCents, quantity);
    }
  }

  private Integer parsePrice(JsonNode priceNode, boolean dollarsVariant) {
    if (priceNode == null || priceNode.isNull()) {
      return null;
    }
    if (!dollarsVariant && priceNode.isNumber()) {
      return priceNode.asInt();
    }
    if (!dollarsVariant && priceNode.isTextual()) {
      try {
        return new BigDecimal(priceNode.asText()).intValue();
      } catch (NumberFormatException ex) {
        return null;
      }
    }

    if (priceNode.isNumber()) {
      return dollarsToCents(priceNode.decimalValue());
    }
    if (priceNode.isTextual()) {
      return dollarsToCentsSafe(priceNode.asText());
    }
    return null;
  }

  private BigDecimal parseQuantity(JsonNode quantityNode) {
    if (quantityNode == null || quantityNode.isNull()) {
      return null;
    }
    if (quantityNode.isNumber()) {
      return quantityNode.decimalValue();
    }
    if (quantityNode.isTextual()) {
      return parseDecimal(quantityNode.asText());
    }
    return null;
  }

  private BigDecimal parseDecimal(String value) {
    try {
      return new BigDecimal(value);
    } catch (NumberFormatException ex) {
      return BigDecimal.ZERO;
    }
  }

  private Integer dollarsToCentsSafe(String dollars) {
    try {
      return dollarsToCents(new BigDecimal(dollars));
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private int dollarsToCents(BigDecimal dollars) {
    BigDecimal cents = dollars.movePointRight(2);
    try {
      return cents.intValueExact();
    } catch (ArithmeticException ex) {
      return cents.setScale(0, RoundingMode.HALF_UP).intValue();
    }
  }
}
