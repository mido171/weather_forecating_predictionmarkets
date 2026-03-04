package com.predictionmarkets.weather.kalshiapi.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.api.KalshiMarketDataApi;
import com.predictionmarkets.weather.kalshiapi.auth.KalshiSignerProvider;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.model.market.GetOrderbookResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.IntPriceLevel;
import com.predictionmarkets.weather.kalshiapi.model.market.Orderbook;
import com.predictionmarkets.weather.kalshiapi.model.market.OrderbookFp;
import com.predictionmarkets.weather.kalshiapi.ws.OrderBookStore;
import com.predictionmarkets.weather.kalshiapi.ws.SequenceTracker;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class DefaultKalshiOrderBookServiceTest {

  private DefaultKalshiOrderBookService service;

  @AfterEach
  void tearDown() throws Exception {
    if (service != null) {
      service.destroy();
    }
  }

  @Test
  void sequenceGapTriggersRestResync() throws Exception {
    KalshiExecutionProperties properties = new KalshiExecutionProperties();
    properties.getWebSocket().setEnabled(false);
    properties.setAuthEnabled(false);

    KalshiMarketDataApi marketDataApi = mock(KalshiMarketDataApi.class);
    when(marketDataApi.getOrderbook("MKT1", 0)).thenReturn(new GetOrderbookResponse(
        new Orderbook(List.of(new IntPriceLevel(55, 7)), List.of(new IntPriceLevel(45, 6)), null, null, null, null),
        new OrderbookFp(null, null)));

    service = new DefaultKalshiOrderBookService(
        properties,
        mock(KalshiSignerProvider.class),
        new ObjectMapper(),
        new SequenceTracker(new SimpleMeterRegistry()),
        new OrderBookStore(),
        marketDataApi);

    service.replaceTrackedMarkets(Set.of("MKT1"));

    service.onMessage("""
        {"type":"ok","subscriptions":[{"channel":"orderbook_delta","sid":1}]}
        """);
    service.onMessage("""
        {"type":"orderbook_snapshot","sid":1,"seq":1,"msg":{"market_ticker":"MKT1","yes":[[60,5]],"no":[[40,4]]}}
        """);
    service.onMessage("""
        {"type":"orderbook_delta","sid":1,"seq":3,"msg":{"market_ticker":"MKT1","yes":[[60,4]]}}
        """);

    long deadline = System.nanoTime() + Duration.ofSeconds(3).toNanos();
    while (System.nanoTime() < deadline) {
      if (service.snapshot("MKT1").bestYesBidCents() != null
          && service.snapshot("MKT1").bestYesBidCents() == 55) {
        break;
      }
      Thread.sleep(25L);
    }

    verify(marketDataApi).getOrderbook("MKT1", 0);
    assertThat(service.snapshot("MKT1").bestYesBidCents()).isEqualTo(55);
    assertThat(service.snapshot("MKT1").bestNoBidCents()).isEqualTo(45);
  }
}
