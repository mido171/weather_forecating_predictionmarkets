package com.predictionmarkets.weather.kalshiapi.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.predictionmarkets.weather.kalshiapi.api.KalshiOrdersApi;
import com.predictionmarkets.weather.kalshiapi.api.KalshiPortfolioApi;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetFillsResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetPositionsResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.MarketPosition;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.Order;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import org.junit.jupiter.api.Test;

class DefaultExposureGuardServiceTest {

  @Test
  void guardrailBlocksWhenFilledPlusRestingAtCap() {
    KalshiPortfolioApi portfolioApi = mock(KalshiPortfolioApi.class);
    KalshiOrdersApi ordersApi = mock(KalshiOrdersApi.class);
    KalshiExecutionProperties properties = props(3, false);

    when(portfolioApi.getPositions(any())).thenReturn(new GetPositionsResponse(
        java.util.List.of(new MarketPosition("MKT1", 2, null)),
        null));
    when(portfolioApi.getFills(any())).thenReturn(new GetFillsResponse(java.util.List.of(), null));
    when(ordersApi.getOrders(any())).thenReturn(new GetOrdersResponse(
        java.util.List.of(restingBuy("O1", "MKT1", Side.YES, 1)),
        null));

    DefaultExposureGuardService service = new DefaultExposureGuardService(portfolioApi, ordersApi, properties);
    TradeOrderRequest request = new TradeOrderRequest("MKT1", Side.YES, 5, OrderType.MARKET, null, 100, "i1");

    var decision = service.preflightBuy(request);
    assertThat(decision.blocked()).isTrue();
    assertThat(decision.allowedContractsToSend()).isZero();
    assertThat(decision.exposureSnapshot().filledContracts()).isEqualTo(2);
    assertThat(decision.exposureSnapshot().restingBuyContracts()).isEqualTo(1);
  }

  @Test
  void guardrailTrimsToRemainingAllowance() {
    KalshiPortfolioApi portfolioApi = mock(KalshiPortfolioApi.class);
    KalshiOrdersApi ordersApi = mock(KalshiOrdersApi.class);
    KalshiExecutionProperties properties = props(5, false);

    when(portfolioApi.getPositions(any())).thenReturn(new GetPositionsResponse(
        java.util.List.of(new MarketPosition("MKT1", 1, null)),
        null));
    when(portfolioApi.getFills(any())).thenReturn(new GetFillsResponse(java.util.List.of(), null));
    when(ordersApi.getOrders(any())).thenReturn(new GetOrdersResponse(
        java.util.List.of(restingBuy("O1", "MKT1", Side.YES, 1)),
        null));

    DefaultExposureGuardService service = new DefaultExposureGuardService(portfolioApi, ordersApi, properties);
    TradeOrderRequest request = new TradeOrderRequest("MKT1", Side.YES, 10, OrderType.MARKET, null, 100, "i1");
    var decision = service.preflightBuy(request);

    assertThat(decision.blocked()).isFalse();
    assertThat(decision.effectiveCap()).isEqualTo(5);
    assertThat(decision.allowedContractsToSend()).isEqualTo(3);
  }

  @Test
  void startupReconcileDetectsRestingExposureAndHalts() {
    KalshiPortfolioApi portfolioApi = mock(KalshiPortfolioApi.class);
    KalshiOrdersApi ordersApi = mock(KalshiOrdersApi.class);
    KalshiExecutionProperties properties = props(1, true);

    when(portfolioApi.getPositions(any())).thenReturn(new GetPositionsResponse(
        java.util.List.of(new MarketPosition("MKT1", 1, null)),
        null));
    when(portfolioApi.getFills(any())).thenReturn(new GetFillsResponse(java.util.List.of(), null));
    when(ordersApi.getOrders(any())).thenReturn(new GetOrdersResponse(
        java.util.List.of(restingBuy("O1", "MKT1", Side.YES, 1)),
        null));

    DefaultExposureGuardService service = new DefaultExposureGuardService(portfolioApi, ordersApi, properties);
    service.startupReconcile();

    assertThat(service.isTradingHalted("MKT1", Side.YES)).isTrue();
    verify(ordersApi, atLeastOnce()).batchCancelOrders(any());
  }

  @Test
  void capBreachTriggersCancelAndHalt() {
    KalshiPortfolioApi portfolioApi = mock(KalshiPortfolioApi.class);
    KalshiOrdersApi ordersApi = mock(KalshiOrdersApi.class);
    KalshiExecutionProperties properties = props(2, false);

    when(portfolioApi.getPositions(any())).thenReturn(new GetPositionsResponse(
        java.util.List.of(new MarketPosition("MKT1", 2, null)),
        null));
    when(portfolioApi.getFills(any())).thenReturn(new GetFillsResponse(java.util.List.of(), null));
    when(ordersApi.getOrders(any())).thenReturn(new GetOrdersResponse(
        java.util.List.of(restingBuy("O1", "MKT1", Side.YES, 1)),
        null));

    DefaultExposureGuardService service = new DefaultExposureGuardService(portfolioApi, ordersApi, properties);
    service.verifyPostSubmitOrHalt("MKT1", Side.YES, 2);

    assertThat(service.isTradingHalted("MKT1", Side.YES)).isTrue();
    verify(ordersApi, atLeastOnce()).batchCancelOrders(any());
  }

  private static KalshiExecutionProperties props(int defaultCap, boolean startupReconcile) {
    KalshiExecutionProperties properties = new KalshiExecutionProperties();
    properties.getGuardrails().setEnabled(true);
    properties.getGuardrails().setDefaultMaxBuyContracts(defaultCap);
    properties.getGuardrails().setStartupReconcile(startupReconcile);
    properties.getGuardrails().setCancelRestingOnCapBreach(true);
    properties.getGuardrails().setFailClosedOnUnknownOrderState(true);
    return properties;
  }

  private static Order restingBuy(String orderId, String marketTicker, Side side, int remaining) {
    return new Order(
        orderId,
        marketTicker,
        marketTicker,
        side,
        Action.BUY,
        OrderType.LIMIT,
        "resting",
        "cid-" + orderId,
        remaining,
        null,
        remaining,
        null,
        0,
        null,
        side == Side.YES ? 50 : null,
        side == Side.NO ? 50 : null,
        null,
        null,
        null,
        null,
        null,
        0,
        null,
        null);
  }
}
