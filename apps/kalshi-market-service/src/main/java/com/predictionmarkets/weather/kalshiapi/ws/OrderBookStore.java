package com.predictionmarkets.weather.kalshiapi.ws;

import com.fasterxml.jackson.databind.JsonNode;
import com.predictionmarkets.weather.kalshiapi.model.market.GetOrderbookResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.Orderbook;
import com.predictionmarkets.weather.kalshiapi.model.market.OrderbookFp;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsOrderbookSnapshotPayload;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

@Component
public class OrderBookStore {

  private final Map<String, OrderBook> booksByMarket = new ConcurrentHashMap<>();

  public void applySnapshot(WsOrderbookSnapshotPayload payload) {
    if (payload == null || !StringUtils.hasText(payload.marketTicker())) {
      return;
    }
    OrderBook book = booksByMarket.computeIfAbsent(payload.marketTicker(), OrderBook::new);
    book.applySnapshot(payload);
  }

  public void applyDelta(String marketTicker, JsonNode deltaMsg) {
    if (!StringUtils.hasText(marketTicker) || deltaMsg == null) {
      return;
    }
    OrderBook book = booksByMarket.computeIfAbsent(marketTicker, OrderBook::new);
    book.applyDelta(deltaMsg);
  }

  public void applyDelta(JsonNode deltaMsg) {
    if (deltaMsg == null) {
      return;
    }
    JsonNode marketTickerNode = deltaMsg.get("market_ticker");
    if (marketTickerNode == null || marketTickerNode.isNull()) {
      return;
    }
    applyDelta(marketTickerNode.asText(), deltaMsg);
  }

  public void applyRestSnapshot(String marketTicker, GetOrderbookResponse response) {
    if (!StringUtils.hasText(marketTicker) || response == null) {
      return;
    }
    Orderbook orderbook = response.orderbook();
    OrderbookFp orderbookFp = response.orderbookFp();
    WsOrderbookSnapshotPayload payload = new WsOrderbookSnapshotPayload(
        marketTicker,
        orderbook == null ? null : orderbook.yes(),
        orderbook == null ? null : orderbook.no(),
        orderbook == null ? null : orderbook.yesDollars(),
        orderbook == null ? null : orderbook.noDollars(),
        orderbookFp == null ? null : orderbookFp.yesDollars(),
        orderbookFp == null ? null : orderbookFp.noDollars());
    applySnapshot(payload);
  }

  public Optional<OrderBookSnapshot> snapshot(String marketTicker) {
    OrderBook book = booksByMarket.get(marketTicker);
    if (book == null) {
      return Optional.empty();
    }
    return Optional.of(new OrderBookSnapshot(
        marketTicker,
        book.yesSideSnapshot(),
        book.noSideSnapshot(),
        Instant.now()));
  }

  public void clear(String marketTicker) {
    if (StringUtils.hasText(marketTicker)) {
      booksByMarket.remove(marketTicker);
    }
  }

  public void clearAll() {
    booksByMarket.clear();
  }
}
