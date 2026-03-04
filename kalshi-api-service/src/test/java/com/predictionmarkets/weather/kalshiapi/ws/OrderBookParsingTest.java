package com.predictionmarkets.weather.kalshiapi.ws;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.model.market.IntPriceLevel;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsOrderbookSnapshotPayload;
import java.util.List;
import org.junit.jupiter.api.Test;

class OrderBookParsingTest {

  private final ObjectMapper objectMapper = new ObjectMapper();

  @Test
  void appliesSnapshotAndDeltaAcrossFormats() throws Exception {
    OrderBook orderBook = new OrderBook("MKT1");
    WsOrderbookSnapshotPayload snapshot = new WsOrderbookSnapshotPayload(
        "MKT1",
        List.of(new IntPriceLevel(60, 10), new IntPriceLevel(59, 8)),
        List.of(new IntPriceLevel(40, 12)),
        null,
        null,
        null,
        null);
    orderBook.applySnapshot(snapshot);

    assertThat(orderBook.yesSideSnapshot()).containsEntry(60, java.math.BigDecimal.TEN);
    assertThat(orderBook.noSideSnapshot()).containsEntry(40, java.math.BigDecimal.valueOf(12));

    String deltaJson = """
        {
          "market_ticker":"MKT1",
          "yes":[[60,0],[58,5]],
          "no":[[40,10]],
          "yes_dollars":[["0.57",3]]
        }
        """;
    orderBook.applyDelta(objectMapper.readTree(deltaJson));

    assertThat(orderBook.yesSideSnapshot()).doesNotContainKey(60);
    assertThat(orderBook.yesSideSnapshot()).containsEntry(58, java.math.BigDecimal.valueOf(5));
    assertThat(orderBook.yesSideSnapshot()).containsEntry(57, java.math.BigDecimal.valueOf(3));
    assertThat(orderBook.noSideSnapshot()).containsEntry(40, java.math.BigDecimal.valueOf(10));
  }
}
