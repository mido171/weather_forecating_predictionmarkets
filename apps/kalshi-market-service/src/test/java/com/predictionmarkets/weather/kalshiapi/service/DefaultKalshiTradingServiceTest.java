package com.predictionmarkets.weather.kalshiapi.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.predictionmarkets.weather.kalshiapi.api.KalshiOrdersApi;
import com.predictionmarkets.weather.kalshiapi.api.KalshiPortfolioApi;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CreateOrderResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.Order;
import com.predictionmarkets.weather.kalshiapi.trading.ExposureSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradePreflightDecision;
import org.junit.jupiter.api.Test;

class DefaultKalshiTradingServiceTest {

  @Test
  void duplicateIntentDoesNotCreateSecondOrder() {
    KalshiOrdersApi ordersApi = mock(KalshiOrdersApi.class);
    KalshiPortfolioApi portfolioApi = mock(KalshiPortfolioApi.class);
    ExposureGuardService guardService = mock(ExposureGuardService.class);
    KalshiExecutionProperties properties = props();

    TradePreflightDecision preflight = new TradePreflightDecision(
        false,
        false,
        "ok",
        1,
        1,
        1,
        1,
        "cid-1",
        new ExposureSnapshot(0, 0, 1, 1),
        "MKT1|YES");

    when(guardService.preflightBuy(any())).thenReturn(preflight);
    doNothing().when(guardService).verifyPostSubmitOrHalt(any(), any(), any(Integer.class));
    when(guardService.recomputeExposure(any(), any(), any(Integer.class)))
        .thenReturn(new ExposureSnapshot(1, 0, 1, 0));
    when(guardService.isTradingHalted(any(), any())).thenReturn(false);
    when(ordersApi.createOrder(any())).thenReturn(new CreateOrderResponse(
        order("O1", "MKT1", Side.YES, "cid-1")));

    DefaultKalshiTradingService service = new DefaultKalshiTradingService(ordersApi, portfolioApi, guardService, properties);
    TradeOrderRequest request = new TradeOrderRequest("MKT1", Side.YES, 1, OrderType.MARKET, null, 100, "intent-1");

    var first = service.placeMarketBuy(request);
    var second = service.placeMarketBuy(request);

    assertThat(first.submitted()).isTrue();
    assertThat(second.submitted()).isFalse();
    assertThat(second.reason()).contains("duplicate_intent");
    verify(ordersApi, times(1)).createOrder(any());
  }

  @Test
  void unknownWriteOutcomeFailsClosedWithoutResubmit() {
    KalshiOrdersApi ordersApi = mock(KalshiOrdersApi.class);
    KalshiPortfolioApi portfolioApi = mock(KalshiPortfolioApi.class);
    ExposureGuardService guardService = mock(ExposureGuardService.class);
    KalshiExecutionProperties properties = props();
    properties.getGuardrails().setFailClosedOnUnknownOrderState(true);

    TradePreflightDecision preflight = new TradePreflightDecision(
        false,
        false,
        "ok",
        1,
        1,
        1,
        1,
        "cid-1",
        new ExposureSnapshot(0, 0, 1, 1),
        "MKT1|YES");

    when(guardService.preflightBuy(any())).thenReturn(preflight);
    when(guardService.recomputeExposure(any(), any(), any(Integer.class)))
        .thenReturn(new ExposureSnapshot(0, 0, 1, 1));
    when(ordersApi.createOrder(any())).thenThrow(new RuntimeException("timed out"));
    when(ordersApi.getOrders(any())).thenReturn(new GetOrdersResponse(java.util.List.of(), null));

    DefaultKalshiTradingService service = new DefaultKalshiTradingService(ordersApi, portfolioApi, guardService, properties);
    TradeOrderRequest request = new TradeOrderRequest("MKT1", Side.YES, 1, OrderType.MARKET, null, 100, "intent-1");

    var first = service.placeMarketBuy(request);
    var second = service.placeMarketBuy(request);

    assertThat(first.submitted()).isFalse();
    assertThat(first.reason()).contains("unknown_submit_outcome_no_resubmit");
    assertThat(second.submitted()).isFalse();
    assertThat(second.reason()).contains("unknown_outcome_pending_reconcile");
    verify(ordersApi, times(1)).createOrder(any());
    verify(guardService, times(1)).haltTrading("MKT1", Side.YES, "unknown_submit_outcome");
    verify(guardService, never()).verifyPostSubmitOrHalt(any(), any(), any(Integer.class));
  }

  private static KalshiExecutionProperties props() {
    KalshiExecutionProperties properties = new KalshiExecutionProperties();
    properties.getGuardrails().setEnabled(true);
    properties.getGuardrails().setDefaultMaxBuyContracts(1);
    return properties;
  }

  private static Order order(String orderId, String marketTicker, Side side, String clientOrderId) {
    return new Order(
        orderId,
        marketTicker,
        marketTicker,
        side,
        Action.BUY,
        OrderType.MARKET,
        "executed",
        clientOrderId,
        1,
        null,
        0,
        null,
        1,
        null,
        side == Side.YES ? 50 : null,
        side == Side.NO ? 50 : null,
        null,
        100,
        null,
        null,
        null,
        0,
        null,
        null);
  }
}
