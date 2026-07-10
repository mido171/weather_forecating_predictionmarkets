package com.predictionmarkets.weather.kalshiapi.service;

import com.predictionmarkets.weather.kalshiapi.api.KalshiOrdersApi;
import com.predictionmarkets.weather.kalshiapi.api.KalshiPortfolioApi;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.model.common.Action;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderStatus;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.BatchCancelOrdersRequest;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.Fill;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetFillsQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetFillsResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetPositionsQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetPositionsResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.MarketPosition;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.Order;
import com.predictionmarkets.weather.kalshiapi.trading.ExposureSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradePreflightDecision;
import com.predictionmarkets.weather.kalshiapi.util.PriceUtils;
import jakarta.annotation.PostConstruct;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
public class DefaultExposureGuardService implements ExposureGuardService {

  private static final Logger log = LoggerFactory.getLogger(DefaultExposureGuardService.class);
  private static final int ORDERS_PAGE_SIZE = 1_000;
  private static final int FILLS_PAGE_SIZE = 200;
  private static final int MAX_PAGES = 500;
  private static final int BATCH_CANCEL_LIMIT = 20;

  private final KalshiPortfolioApi portfolioApi;
  private final KalshiOrdersApi ordersApi;
  private final KalshiExecutionProperties properties;

  private final Set<String> haltedKeys = ConcurrentHashMap.newKeySet();
  private final Map<String, String> haltReasonsByKey = new ConcurrentHashMap<>();
  private final AtomicBoolean globallyHalted = new AtomicBoolean(false);
  private final AtomicReference<String> globalHaltReason = new AtomicReference<>();

  public DefaultExposureGuardService(KalshiPortfolioApi portfolioApi,
                                     KalshiOrdersApi ordersApi,
                                     KalshiExecutionProperties properties) {
    this.portfolioApi = portfolioApi;
    this.ordersApi = ordersApi;
    this.properties = properties;
  }

  @PostConstruct
  void startupReconcile() {
    if (!properties.getGuardrails().isEnabled() || !properties.getGuardrails().isStartupReconcile()) {
      return;
    }
    try {
      Set<String> keysToReconcile = new HashSet<>();
      for (Order resting : fetchAllRestingOrders()) {
        if (!isBuyOrder(resting) || resting.side() == null || !StringUtils.hasText(resting.marketTicker())) {
          continue;
        }
        keysToReconcile.add(marketSideKey(resting.marketTicker(), resting.side()));
      }
      for (String key : keysToReconcile) {
        String[] parts = key.split("\\|", 2);
        String marketTicker = parts[0];
        Side side = Side.valueOf(parts[1]);
        int cap = configuredCapForMarket(marketTicker);
        verifyPostSubmitOrHalt(marketTicker, side, cap);
      }
      if (!keysToReconcile.isEmpty()) {
        log.info("Kalshi guardrail startup reconcile complete for {} market-side keys", keysToReconcile.size());
      }
    } catch (Exception ex) {
      if (properties.getGuardrails().isFailClosedOnUnknownOrderState()) {
        globallyHalted.set(true);
        globalHaltReason.set("startup_reconcile_failed");
        log.error("Kalshi guardrail startup reconcile failed; fail-closed behavior enabled", ex);
      } else {
        log.warn("Kalshi guardrail startup reconcile failed: {}", ex.toString());
      }
    }
  }

  @Override
  public TradePreflightDecision preflightBuy(TradeOrderRequest request) {
    String marketTicker = normalizeMarketTicker(request.marketTicker());
    Side side = request.side();
    int configuredCap = configuredCapForMarket(marketTicker);
    int effectiveCap = Math.min(request.contractsRequested(), configuredCap);
    String key = marketSideKey(marketTicker, side);

    if (globallyHalted.get()) {
      ExposureSnapshot snapshot = new ExposureSnapshot(0, 0, effectiveCap, 0);
      return new TradePreflightDecision(
          true,
          true,
          "trading_halted: " + globalHaltReason.get(),
          configuredCap,
          effectiveCap,
          request.contractsRequested(),
          0,
          deterministicClientOrderId(request.intentId(), marketTicker, side, effectiveCap),
          snapshot,
          key);
    }

    if (isTradingHalted(marketTicker, side)) {
      String reason = "trading_halted: " + haltReasonsByKey.getOrDefault(key, "manual_or_guardrail");
      ExposureSnapshot snapshot = recomputeExposure(marketTicker, side, effectiveCap);
      return new TradePreflightDecision(
          true,
          true,
          reason,
          configuredCap,
          effectiveCap,
          request.contractsRequested(),
          0,
          deterministicClientOrderId(request.intentId(), marketTicker, side, effectiveCap),
          snapshot,
          key);
    }

    ExposureSnapshot snapshot = recomputeExposure(marketTicker, side, effectiveCap);
    int remainingAllowance = snapshot.remainingAllowance();
    if (remainingAllowance <= 0) {
      return new TradePreflightDecision(
          true,
          false,
          "allowance_exhausted",
          configuredCap,
          effectiveCap,
          request.contractsRequested(),
          0,
          deterministicClientOrderId(request.intentId(), marketTicker, side, effectiveCap),
          snapshot,
          key);
    }

    int allowedToSend = Math.min(request.contractsRequested(), remainingAllowance);
    return new TradePreflightDecision(
        false,
        false,
        "ok",
        configuredCap,
        effectiveCap,
        request.contractsRequested(),
        allowedToSend,
        deterministicClientOrderId(request.intentId(), marketTicker, side, effectiveCap),
        snapshot,
        key);
  }

  @Override
  public ExposureSnapshot recomputeExposure(String marketTicker, Side side, int allowedContracts) {
    String normalizedTicker = normalizeMarketTicker(marketTicker);
    int normalizedAllowed = Math.max(0, allowedContracts);
    int filledContracts = Math.max(
        computeFilledContractsFromPositions(normalizedTicker, side),
        computeFilledContractsFromFills(normalizedTicker, side));
    int restingBuyContracts = computeRestingBuyContracts(normalizedTicker, side);
    int remainingAllowance = normalizedAllowed - (filledContracts + restingBuyContracts);
    return new ExposureSnapshot(
        filledContracts,
        restingBuyContracts,
        normalizedAllowed,
        remainingAllowance);
  }

  @Override
  public void verifyPostSubmitOrHalt(String marketTicker, Side side, int allowedContracts) {
    ExposureSnapshot snapshot = recomputeExposure(marketTicker, side, allowedContracts);
    int total = snapshot.filledContracts() + snapshot.restingBuyContracts();
    if (total <= snapshot.allowedContracts()) {
      return;
    }

    String key = marketSideKey(marketTicker, side);
    String reason = "cap_breach total=" + total + " allowed=" + snapshot.allowedContracts();
    log.error("Kalshi guardrail breach key={} {}", key, reason);

    if (properties.getGuardrails().isCancelRestingOnCapBreach()) {
      cancelAllRestingBuys(marketTicker, side);
    }
    haltTrading(marketTicker, side, reason);
  }

  @Override
  public boolean isTradingHalted(String marketTicker, Side side) {
    return globallyHalted.get() || haltedKeys.contains(marketSideKey(marketTicker, side));
  }

  @Override
  public void haltTrading(String marketTicker, Side side, String reason) {
    String key = marketSideKey(marketTicker, side);
    haltedKeys.add(key);
    haltReasonsByKey.put(key, StringUtils.hasText(reason) ? reason : "unspecified");
    log.error("Kalshi guardrail halted buys for key={} reason={}", key, haltReasonsByKey.get(key));
  }

  @Override
  public int configuredCapForMarket(String marketTicker) {
    String normalizedTicker = normalizeMarketTicker(marketTicker);
    Integer override = properties.getGuardrails().getMarketOverrides().get(normalizedTicker);
    if (override == null) {
      override = properties.getGuardrails().getMarketOverrides().get(marketTicker);
    }
    int cap = override == null ? properties.getGuardrails().getDefaultMaxBuyContracts() : override;
    return Math.max(1, cap);
  }

  @Override
  public void resetTradingHalt(String marketTicker, Side side) {
    String key = marketSideKey(marketTicker, side);
    haltedKeys.remove(key);
    haltReasonsByKey.remove(key);
    log.info("Kalshi guardrail halt reset for key={}", key);
  }

  public static String deterministicClientOrderId(String intentId,
                                                  String marketTicker,
                                                  Side side,
                                                  int effectiveCap) {
    String seed = intentId + "|" + normalizeMarketTicker(marketTicker) + "|" + side.name().toLowerCase(Locale.ROOT)
        + "|" + effectiveCap;
    String digest = sha256Hex(seed);
    return "intent-" + digest.substring(0, 32);
  }

  private List<Order> fetchAllRestingOrders() {
    List<Order> orders = new ArrayList<>();
    String cursor = null;
    int pages = 0;
    while (pages < MAX_PAGES) {
      GetOrdersResponse response = ordersApi.getOrders(new GetOrdersQuery(
          ORDERS_PAGE_SIZE,
          cursor,
          List.of(OrderStatus.RESTING)));
      if (response == null || response.orders() == null || response.orders().isEmpty()) {
        break;
      }
      orders.addAll(response.orders());
      cursor = response.cursor();
      if (!StringUtils.hasText(cursor)) {
        break;
      }
      pages++;
    }
    return orders;
  }

  private int computeFilledContractsFromPositions(String marketTicker, Side side) {
    int filled = 0;
    String cursor = null;
    int pages = 0;
    while (pages < MAX_PAGES) {
      GetPositionsResponse response = portfolioApi.getPositions(new GetPositionsQuery(cursor, ORDERS_PAGE_SIZE, null));
      List<MarketPosition> positions = response == null ? Collections.emptyList() : response.marketPositions();
      if (positions != null) {
        for (MarketPosition position : positions) {
          if (position == null || !matchesMarket(position.marketTicker(), marketTicker)) {
            continue;
          }
          int net = parseContracts(position.position(), position.positionFp());
          if (side == Side.YES) {
            filled = Math.max(filled, Math.max(0, net));
          } else {
            filled = Math.max(filled, Math.max(0, -net));
          }
        }
      }
      cursor = response == null ? null : response.cursor();
      if (!StringUtils.hasText(cursor)) {
        break;
      }
      pages++;
    }
    return filled;
  }

  private int computeFilledContractsFromFills(String marketTicker, Side side) {
    int net = 0;
    String cursor = null;
    int pages = 0;
    while (pages < MAX_PAGES) {
      GetFillsResponse response = portfolioApi.getFills(new GetFillsQuery(
          null,
          null,
          null,
          null,
          FILLS_PAGE_SIZE,
          cursor,
          null));
      List<Fill> fills = response == null ? Collections.emptyList() : response.fills();
      if (fills != null) {
        for (Fill fill : fills) {
          if (fill == null || fill.side() != side || !matchesMarket(fill.marketTicker(), marketTicker)) {
            continue;
          }
          int count = parseContracts(fill.count(), fill.countFp());
          if (count <= 0) {
            continue;
          }
          if (fill.action() == Action.SELL) {
            net -= count;
          } else {
            net += count;
          }
        }
      }
      cursor = response == null ? null : response.cursor();
      if (!StringUtils.hasText(cursor)) {
        break;
      }
      pages++;
    }
    return Math.max(0, net);
  }

  private int computeRestingBuyContracts(String marketTicker, Side side) {
    int total = 0;
    for (Order order : fetchAllRestingOrders()) {
      if (order == null || !isBuyOrder(order) || order.side() != side || !matchesMarket(order.marketTicker(), marketTicker)) {
        continue;
      }

      int remaining = parseContracts(order.remainingCount(), order.remainingCountFp());
      if (remaining <= 0) {
        int totalCount = parseContracts(order.count(), order.countFp());
        if (isResting(order.status())) {
          remaining = totalCount;
        } else if (totalCount > 0 && properties.getGuardrails().isFailClosedOnUnknownOrderState()) {
          remaining = totalCount;
        }
      }
      total += Math.max(0, remaining);
    }
    return total;
  }

  private void cancelAllRestingBuys(String marketTicker, Side side) {
    List<String> ids = new ArrayList<>();
    for (Order order : fetchAllRestingOrders()) {
      if (order == null || !isBuyOrder(order) || order.side() != side || !matchesMarket(order.marketTicker(), marketTicker)) {
        continue;
      }
      if (StringUtils.hasText(order.orderId())) {
        ids.add(order.orderId());
      }
    }
    if (ids.isEmpty()) {
      return;
    }
    for (int i = 0; i < ids.size(); i += BATCH_CANCEL_LIMIT) {
      int end = Math.min(ids.size(), i + BATCH_CANCEL_LIMIT);
      List<String> batch = ids.subList(i, end);
      ordersApi.batchCancelOrders(new BatchCancelOrdersRequest(batch));
      log.warn("Kalshi guardrail canceled {} resting buy orders for {} {}", batch.size(), marketTicker, side);
    }
  }

  private static int parseContracts(Integer intCount, String fpCount) {
    if (intCount != null && intCount >= 0) {
      return intCount;
    }
    if (!StringUtils.hasText(fpCount)) {
      return 0;
    }
    return PriceUtils.parseDecimal(fpCount).max(java.math.BigDecimal.ZERO).setScale(0, RoundingMode.DOWN).intValue();
  }

  private boolean isBuyOrder(Order order) {
    return order != null && order.action() == Action.BUY;
  }

  private boolean isResting(String status) {
    if (!StringUtils.hasText(status)) {
      return false;
    }
    String normalized = status.trim().toLowerCase(Locale.ROOT);
    return normalized.equals("resting");
  }

  private boolean matchesMarket(String candidate, String expectedTicker) {
    return StringUtils.hasText(candidate)
        && StringUtils.hasText(expectedTicker)
        && normalizeMarketTicker(candidate).equals(normalizeMarketTicker(expectedTicker));
  }

  private static String marketSideKey(String marketTicker, Side side) {
    return normalizeMarketTicker(marketTicker) + "|" + side.name();
  }

  private static String normalizeMarketTicker(String marketTicker) {
    if (!StringUtils.hasText(marketTicker)) {
      return "";
    }
    return marketTicker.trim().toUpperCase(Locale.ROOT);
  }

  private static String sha256Hex(String value) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      byte[] hash = digest.digest(value.getBytes(StandardCharsets.UTF_8));
      StringBuilder sb = new StringBuilder(hash.length * 2);
      for (byte b : hash) {
        sb.append(String.format("%02x", b));
      }
      return sb.toString();
    } catch (NoSuchAlgorithmException ex) {
      throw new IllegalStateException("SHA-256 not available", ex);
    }
  }
}
