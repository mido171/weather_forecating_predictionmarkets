package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.api.KalshiOrdersApi;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.http.KalshiApiException;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.common.TimeInForce;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CancelOrderResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CreateOrderRequest;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CreateOrderResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.Order;
import com.predictionmarkets.weather.kalshiapi.trading.ExposureSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderResult;
import com.predictionmarkets.weather.kalshiapi.trading.TradePreflightDecision;
import java.time.Duration;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
public class DefaultKalshiTradingService implements KalshiTradingService {

  private static final Logger log = LoggerFactory.getLogger(DefaultKalshiTradingService.class);
  private static final int ORDERS_PAGE_SIZE = 1_000;
  private static final int MAX_PAGES = 50;

  private final KalshiOrdersApi ordersApi;
  private final ExposureGuardService exposureGuardService;
  private final KalshiExecutionProperties properties;

  private final Map<String, ReentrantLock> lockByMarketSide = new ConcurrentHashMap<>();
  private final Map<String, String> inFlightIntentByMarketSide = new ConcurrentHashMap<>();
  private final Map<String, TradeOrderResult> resultByClientOrderId = new ConcurrentHashMap<>();
  private final Map<String, String> unknownOutcomeByClientOrderId = new ConcurrentHashMap<>();

  public DefaultKalshiTradingService(KalshiOrdersApi ordersApi,
                                     ExposureGuardService exposureGuardService,
                                     KalshiExecutionProperties properties) {
    this.ordersApi = ordersApi;
    this.exposureGuardService = exposureGuardService;
    this.properties = properties;
  }

  @Override
  public TradeOrderResult placeMarketBuy(TradeOrderRequest request) {
    validateBuyRequestType(request, OrderType.MARKET);
    return placeBuyInternal(request);
  }

  @Override
  public TradeOrderResult placeLimitBuy(TradeOrderRequest request) {
    validateBuyRequestType(request, OrderType.LIMIT);
    return placeBuyInternal(request);
  }

  @Override
  public CancelOrderResponse cancelOrder(String orderId) {
    return ordersApi.cancelOrder(orderId);
  }

  @Override
  public void resetTradingHalt(String marketTicker, Side side) {
    exposureGuardService.resetTradingHalt(marketTicker, side);
  }

  private TradeOrderResult placeBuyInternal(TradeOrderRequest request) {
    String marketSideKey = marketSideKey(request.marketTicker(), request.side());
    ReentrantLock lock = lockByMarketSide.computeIfAbsent(marketSideKey, key -> new ReentrantLock());
    lock.lock();
    try {
      String existingIntent = inFlightIntentByMarketSide.putIfAbsent(marketSideKey, request.intentId());
      if (existingIntent != null) {
        return blocked(request, false, "active_buy_intent_in_progress", null, 0, null);
      }
      return placeBuyUnderLock(request, marketSideKey);
    } finally {
      inFlightIntentByMarketSide.remove(marketSideKey, request.intentId());
      lock.unlock();
    }
  }

  private TradeOrderResult placeBuyUnderLock(TradeOrderRequest request, String marketSideKey) {
    TradePreflightDecision preflight = exposureGuardService.preflightBuy(request);
    if (preflight.blocked() || preflight.allowedContractsToSend() <= 0) {
      return new TradeOrderResult(
          false,
          true,
          preflight.halted(),
          preflight.reason(),
          null,
          preflight.deterministicClientOrderId(),
          preflight.requestedContracts(),
          0,
          preflight.exposureSnapshot());
    }

    String clientOrderId = preflight.deterministicClientOrderId();
    TradeOrderResult cached = resultByClientOrderId.get(clientOrderId);
    if (cached != null) {
      return new TradeOrderResult(
          false,
          false,
          cached.halted(),
          "duplicate_intent_cached",
          cached.orderId(),
          cached.clientOrderId(),
          request.contractsRequested(),
          0,
          preflight.exposureSnapshot());
    }

    if (unknownOutcomeByClientOrderId.containsKey(clientOrderId)) {
      String unknownReason = unknownOutcomeByClientOrderId.get(clientOrderId);
      return blocked(
          request,
          properties.getGuardrails().isFailClosedOnUnknownOrderState(),
          "unknown_outcome_pending_reconcile: " + unknownReason,
          preflight.exposureSnapshot(),
          0,
          clientOrderId);
    }

    int contractsToSubmit = preflight.allowedContractsToSend();
    CreateOrderRequest createOrderRequest = toCreateOrderRequest(request, contractsToSubmit, clientOrderId);

    try {
      CreateOrderResponse response = ordersApi.createOrder(createOrderRequest);
      String orderId = response != null && response.order() != null ? response.order().orderId() : null;

      exposureGuardService.verifyPostSubmitOrHalt(
          request.marketTicker(),
          request.side(),
          preflight.effectiveCap());

      ExposureSnapshot exposureSnapshot = exposureGuardService.recomputeExposure(
          request.marketTicker(),
          request.side(),
          preflight.effectiveCap());
      boolean halted = exposureGuardService.isTradingHalted(request.marketTicker(), request.side());
      TradeOrderResult result = new TradeOrderResult(
          true,
          false,
          halted,
          "submitted",
          orderId,
          clientOrderId,
          request.contractsRequested(),
          contractsToSubmit,
          exposureSnapshot);
      resultByClientOrderId.put(clientOrderId, result);
      return result;
    } catch (Exception ex) {
      return handleUnknownSubmitOutcome(request, preflight, clientOrderId, ex, marketSideKey);
    }
  }

  private TradeOrderResult handleUnknownSubmitOutcome(TradeOrderRequest request,
                                                      TradePreflightDecision preflight,
                                                      String clientOrderId,
                                                      Exception ex,
                                                      String marketSideKey) {
    log.error("Kalshi order create outcome unknown for key={} clientOrderId={}: {}",
        marketSideKey, clientOrderId, ex.toString());

    Order reconciledOrder = findOrderByClientOrderId(
        request.marketTicker(),
        request.side(),
        clientOrderId);
    if (reconciledOrder != null) {
      String orderId = reconciledOrder.orderId();
      ExposureSnapshot exposureSnapshot = exposureGuardService.recomputeExposure(
          request.marketTicker(),
          request.side(),
          preflight.effectiveCap());
      boolean halted = exposureGuardService.isTradingHalted(request.marketTicker(), request.side());
      TradeOrderResult result = new TradeOrderResult(
          true,
          false,
          halted,
          "submitted_reconciled_after_unknown",
          orderId,
          clientOrderId,
          request.contractsRequested(),
          preflight.allowedContractsToSend(),
          exposureSnapshot);
      resultByClientOrderId.put(clientOrderId, result);
      return result;
    }

    String reason = unknownReason(ex);
    unknownOutcomeByClientOrderId.put(clientOrderId, reason);
    if (properties.getGuardrails().isFailClosedOnUnknownOrderState()) {
      exposureGuardService.haltTrading(request.marketTicker(), request.side(), "unknown_submit_outcome");
    }

    ExposureSnapshot exposureSnapshot = exposureGuardService.recomputeExposure(
        request.marketTicker(),
        request.side(),
        preflight.effectiveCap());
    return blocked(
        request,
        properties.getGuardrails().isFailClosedOnUnknownOrderState(),
        "unknown_submit_outcome_no_resubmit: " + reason,
        exposureSnapshot,
        0,
        clientOrderId);
  }

  private Order findOrderByClientOrderId(String marketTicker, Side side, String clientOrderId) {
    String cursor = null;
    int pages = 0;
    while (pages < MAX_PAGES) {
      GetOrdersResponse response = ordersApi.getOrders(new GetOrdersQuery(ORDERS_PAGE_SIZE, cursor, null));
      for (Order order : response == null || response.orders() == null ? Collections.<Order>emptyList() : response.orders()) {
        if (order == null) {
          continue;
        }
        if (!Objects.equals(order.clientOrderId(), clientOrderId)) {
          continue;
        }
        if (!matchesMarket(order.marketTicker(), marketTicker)) {
          continue;
        }
        if (order.action() != Action.BUY || order.side() != side) {
          continue;
        }
        return order;
      }
      cursor = response == null ? null : response.cursor();
      if (!StringUtils.hasText(cursor)) {
        break;
      }
      pages++;
    }
    return null;
  }

  private CreateOrderRequest toCreateOrderRequest(TradeOrderRequest request, int contractsToSubmit, String clientOrderId) {
    Integer yesPrice = null;
    Integer noPrice = null;
    Integer buyMaxCost = null;
    TimeInForce timeInForce;
    if (request.orderType() == OrderType.MARKET) {
      buyMaxCost = request.buyMaxCostCents();
      timeInForce = TimeInForce.IMMEDIATE_OR_CANCEL;
    } else {
      if (request.side() == Side.YES) {
        yesPrice = request.limitPriceCents();
      } else {
        noPrice = request.limitPriceCents();
      }
      timeInForce = TimeInForce.GOOD_TILL_CANCELED;
    }
    return new CreateOrderRequest(
        request.marketTicker(),
        request.side(),
        Action.BUY,
        clientOrderId,
        contractsToSubmit,
        null,
        request.orderType(),
        yesPrice,
        noPrice,
        null,
        timeInForce,
        buyMaxCost,
        null,
        false,
        null,
        null);
  }

  private void validateBuyRequestType(TradeOrderRequest request, OrderType expected) {
    if (request == null) {
      throw new IllegalArgumentException("request is required");
    }
    if (request.orderType() != expected) {
      throw new IllegalArgumentException("Expected " + expected + " order type, got " + request.orderType());
    }
  }

  private TradeOrderResult blocked(TradeOrderRequest request,
                                   boolean halted,
                                   String reason,
                                   ExposureSnapshot exposureSnapshot,
                                   int submittedContracts,
                                   String clientOrderId) {
    return new TradeOrderResult(
        false,
        true,
        halted,
        reason,
        null,
        clientOrderId,
        request.contractsRequested(),
        submittedContracts,
        exposureSnapshot);
  }

  private String unknownReason(Exception ex) {
    if (ex instanceof KalshiApiException apiException) {
      return "api_status_" + apiException.getStatusCode();
    }
    String msg = ex.getMessage();
    if (!StringUtils.hasText(msg)) {
      return ex.getClass().getSimpleName();
    }
    String normalized = msg.toLowerCase(Locale.ROOT);
    if (normalized.contains("timeout") || normalized.contains("timed out")) {
      return "timeout";
    }
    if (normalized.contains("connection reset") || normalized.contains("connection closed")) {
      return "connection_interrupted";
    }
    return ex.getClass().getSimpleName();
  }

  private static boolean matchesMarket(String candidate, String expectedTicker) {
    if (!StringUtils.hasText(candidate) || !StringUtils.hasText(expectedTicker)) {
      return false;
    }
    return candidate.trim().equalsIgnoreCase(expectedTicker.trim());
  }

  private static String marketSideKey(String marketTicker, Side side) {
    return marketTicker.trim().toUpperCase(Locale.ROOT) + "|" + side.name();
  }
}
