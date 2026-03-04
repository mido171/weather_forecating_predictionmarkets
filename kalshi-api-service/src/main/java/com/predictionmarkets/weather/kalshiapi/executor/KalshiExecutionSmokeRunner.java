package com.predictionmarkets.weather.kalshiapi.executor;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.api.KalshiPortfolioApi;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.model.common.OrderType;
import com.predictionmarkets.weather.kalshiapi.model.common.Side;
import com.predictionmarkets.weather.kalshiapi.service.ExposureGuardService;
import com.predictionmarkets.weather.kalshiapi.service.KalshiOrderBookService;
import com.predictionmarkets.weather.kalshiapi.service.KalshiTradingService;
import com.predictionmarkets.weather.kalshiapi.trading.ExposureSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.OrderBookSnapshotView;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderRequest;
import com.predictionmarkets.weather.kalshiapi.trading.TradeOrderResult;
import java.time.Instant;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

@Component
public class KalshiExecutionSmokeRunner implements CommandLineRunner {

  private static final Logger log = LoggerFactory.getLogger(KalshiExecutionSmokeRunner.class);

  private final KalshiExecutionProperties properties;
  private final KalshiPortfolioApi portfolioApi;
  private final KalshiOrderBookService orderBookService;
  private final KalshiTradingService tradingService;
  private final ExposureGuardService exposureGuardService;
  private final ObjectMapper objectMapper;

  public KalshiExecutionSmokeRunner(KalshiExecutionProperties properties,
                                    KalshiPortfolioApi portfolioApi,
                                    KalshiOrderBookService orderBookService,
                                    KalshiTradingService tradingService,
                                    ExposureGuardService exposureGuardService,
                                    ObjectMapper objectMapper) {
    this.properties = properties;
    this.portfolioApi = portfolioApi;
    this.orderBookService = orderBookService;
    this.tradingService = tradingService;
    this.exposureGuardService = exposureGuardService;
    this.objectMapper = objectMapper;
  }

  @Override
  public void run(String... args) throws Exception {
    if (!properties.getSmoke().isEnabled()) {
      return;
    }

    String marketTicker = properties.getSmoke().getMarketTicker();
    if (!StringUtils.hasText(marketTicker)) {
      throw new IllegalStateException("kalshi.smoke.market-ticker is required when smoke runner is enabled");
    }

    Side side = parseSide(properties.getSmoke().getSide());
    int desiredContracts = properties.getSmoke().getDesiredContracts();
    int configuredCap = exposureGuardService.configuredCapForMarket(marketTicker);
    int effectiveCap = Math.min(desiredContracts, configuredCap);

    log.info("Kalshi smoke starting marketTicker={} side={} desiredContracts={} configuredCap={} effectiveCap={}",
        marketTicker, side, desiredContracts, configuredCap, effectiveCap);

    var balance = portfolioApi.getBalance();
    log.info("Kalshi smoke preflight balance={} portfolioValue={} updatedTs={}",
        balance == null ? null : balance.balance(),
        balance == null ? null : balance.portfolioValue(),
        balance == null ? null : balance.updatedTs());

    orderBookService.connect();
    orderBookService.replaceTrackedMarkets(java.util.Set.of(marketTicker));

    int warmupSeconds = Math.max(0, properties.getSmoke().getOrderbookWarmupSeconds());
    if (warmupSeconds > 0) {
      Thread.sleep(warmupSeconds * 1_000L);
    }

    OrderBookSnapshotView orderBookSnapshot = orderBookService.snapshot(marketTicker);

    TradeOrderRequest request = new TradeOrderRequest(
        marketTicker,
        side,
        desiredContracts,
        OrderType.MARKET,
        null,
        properties.getSmoke().getBuyMaxCostCents(),
        "smoke-" + Instant.now());

    TradeOrderResult tradeResult = tradingService.placeMarketBuy(request);
    ExposureSnapshot finalExposure = exposureGuardService.recomputeExposure(marketTicker, side, effectiveCap);
    boolean capBreached = finalExposure.filledContracts() + finalExposure.restingBuyContracts() > effectiveCap;

    log.info("Kalshi smoke result={}", toJson(Map.of(
        "marketTicker", marketTicker,
        "side", side.name().toLowerCase(),
        "desiredContracts", desiredContracts,
        "configuredCap", configuredCap,
        "effectiveCap", effectiveCap,
        "orderbookSnapshot", orderBookSnapshot,
        "tradeResult", tradeResult,
        "finalExposure", finalExposure,
        "capBreached", capBreached)));

    if (capBreached) {
      throw new IllegalStateException("Smoke run detected cap breach after order submit");
    }
  }

  private Side parseSide(String rawSide) {
    if (!StringUtils.hasText(rawSide)) {
      return Side.YES;
    }
    String normalized = rawSide.trim().toLowerCase();
    return switch (normalized) {
      case "yes" -> Side.YES;
      case "no" -> Side.NO;
      default -> throw new IllegalStateException("Unsupported kalshi.smoke.side value: " + rawSide);
    };
  }

  private String toJson(Object value) {
    try {
      return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(value);
    } catch (JsonProcessingException ex) {
      return String.valueOf(value);
    }
  }
}
