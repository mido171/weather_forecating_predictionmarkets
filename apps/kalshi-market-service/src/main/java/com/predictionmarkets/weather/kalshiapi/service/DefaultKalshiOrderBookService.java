package com.predictionmarkets.weather.kalshiapi.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.api.KalshiMarketDataApi;
import com.predictionmarkets.weather.kalshiapi.auth.KalshiSignerProvider;
import com.predictionmarkets.weather.kalshiapi.auth.SignedHeaders;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.trading.OrderBookLevelView;
import com.predictionmarkets.weather.kalshiapi.trading.OrderBookSnapshotView;
import com.predictionmarkets.weather.kalshiapi.ws.OrderBookSnapshot;
import com.predictionmarkets.weather.kalshiapi.ws.OrderBookStore;
import com.predictionmarkets.weather.kalshiapi.ws.SequenceTracker;
import com.predictionmarkets.weather.kalshiapi.ws.SequenceTracker.SequenceResult;
import com.predictionmarkets.weather.kalshiapi.ws.SequenceTracker.SequenceStatus;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsErrorResponse;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsOrderbookDeltaMessage;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsOrderbookSnapshotMessage;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsSubscribeCommand;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsSubscribeParams;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsSubscription;
import com.predictionmarkets.weather.kalshiapi.ws.model.WsSubscriptionAck;
import io.netty.channel.ChannelOption;
import java.math.BigDecimal;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.reactive.socket.CloseStatus;
import org.springframework.web.reactive.socket.WebSocketHandler;
import org.springframework.web.reactive.socket.WebSocketMessage;
import org.springframework.web.reactive.socket.WebSocketSession;
import org.springframework.web.reactive.socket.client.ReactorNettyWebSocketClient;
import org.springframework.web.reactive.socket.client.WebSocketClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.netty.http.client.HttpClient;

@Service
public class DefaultKalshiOrderBookService implements KalshiOrderBookService, DisposableBean {

  private static final Logger log = LoggerFactory.getLogger(DefaultKalshiOrderBookService.class);
  private static final String CHANNEL_ORDERBOOK_DELTA = "orderbook_delta";
  private static final String TYPE_ORDERBOOK_SNAPSHOT = "orderbook_snapshot";
  private static final String TYPE_ORDERBOOK_DELTA = "orderbook_delta";
  private static final String TYPE_OK = "ok";
  private static final String TYPE_SUBSCRIBED = "subscribed";
  private static final String TYPE_ERROR = "error";
  private static final int REST_FALLBACK_REFRESH_SECONDS = 2;

  private final KalshiExecutionProperties properties;
  private final KalshiSignerProvider signerProvider;
  private final ObjectMapper objectMapper;
  private final SequenceTracker sequenceTracker;
  private final OrderBookStore orderBookStore;
  private final KalshiMarketDataApi marketDataApi;
  private final WebSocketClient webSocketClient;
  private final ScheduledExecutorService scheduler;

  private final AtomicReference<WebSocketSession> sessionRef = new AtomicReference<>();
  private final AtomicReference<Sinks.Many<String>> outboundSinkRef = new AtomicReference<>(newOutboundSink());
  private final AtomicBoolean connecting = new AtomicBoolean(false);
  private final AtomicBoolean reconnectScheduled = new AtomicBoolean(false);
  private final AtomicBoolean shuttingDown = new AtomicBoolean(false);
  private final AtomicInteger reconnectAttempts = new AtomicInteger(0);
  private final AtomicLong lastInboundNanos = new AtomicLong(System.nanoTime());
  private final AtomicLong snapshotMsgCount = new AtomicLong(0);
  private final AtomicLong deltaMsgCount = new AtomicLong(0);
  private final AtomicLong deltaAppliedCount = new AtomicLong(0);
  private final AtomicLong deltaDroppedCount = new AtomicLong(0);
  private final AtomicInteger commandId = new AtomicInteger(1);

  private final Set<String> trackedMarkets = ConcurrentHashMap.newKeySet();
  private final Map<Integer, Set<String>> marketTickersBySid = new ConcurrentHashMap<>();
  private final Set<Integer> resyncInProgress = ConcurrentHashMap.newKeySet();

  private volatile boolean connected = false;

  public DefaultKalshiOrderBookService(KalshiExecutionProperties properties,
                                       KalshiSignerProvider signerProvider,
                                       ObjectMapper objectMapper,
                                       SequenceTracker sequenceTracker,
                                       OrderBookStore orderBookStore,
                                       KalshiMarketDataApi marketDataApi) {
    this.properties = properties;
    this.signerProvider = signerProvider;
    this.objectMapper = objectMapper;
    this.sequenceTracker = sequenceTracker;
    this.orderBookStore = orderBookStore;
    this.marketDataApi = marketDataApi;

    HttpClient httpClient = HttpClient.create()
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, properties.getTimeouts().getConnectTimeoutMs());
    this.webSocketClient = new ReactorNettyWebSocketClient(httpClient);

    this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
      Thread t = new Thread(r, "kalshi-orderbook-ws-scheduler");
      t.setDaemon(true);
      return t;
    });
    startWatchdog();
    startRestFallbackRefresh();
  }

  @Override
  public void connect() {
    if (!properties.getWebSocket().isEnabled()) {
      log.warn("Kalshi websocket disabled by configuration");
      return;
    }
    connectAsync();
  }

  @Override
  public void disconnect() {
    shuttingDown.set(true);
    reconnectScheduled.set(false);
    connected = false;
    WebSocketSession session = sessionRef.getAndSet(null);
    if (session != null && session.isOpen()) {
      session.close(CloseStatus.NORMAL).subscribe();
    }
    outboundSinkRef.set(newOutboundSink());
  }

  @Override
  public void replaceTrackedMarkets(Set<String> marketTickers) {
    trackedMarkets.clear();
    if (marketTickers != null) {
      for (String ticker : marketTickers) {
        if (StringUtils.hasText(ticker)) {
          trackedMarkets.add(ticker.trim().toUpperCase());
        }
      }
    }

    if (!isSessionOpen()) {
      return;
    }

    subscribeOrderbookDelta();
    hydrateTrackedMarketsFromRestAsync("tracked_market_update");
  }

  @Override
  public OrderBookSnapshotView snapshot(String marketTicker) {
    return orderBookStore.snapshot(marketTicker)
        .map(this::toView)
        .orElseGet(() -> new OrderBookSnapshotView(
            marketTicker,
            List.of(),
            List.of(),
            null,
            null,
            null,
            null,
            null));
  }

  void onMessage(String payload) {
    try {
      JsonNode root = objectMapper.readTree(payload);
      lastInboundNanos.set(System.nanoTime());
      String type = root.path("type").asText(null);
      if (!StringUtils.hasText(type)) {
        return;
      }
      switch (type) {
        case TYPE_OK, TYPE_SUBSCRIBED -> handleSubscriptionAck(root);
        case TYPE_ORDERBOOK_SNAPSHOT -> handleOrderbookSnapshot(root);
        case TYPE_ORDERBOOK_DELTA -> handleOrderbookDelta(root);
        case TYPE_ERROR -> handleError(root);
        default -> {
          // ignore unrelated channels/messages
        }
      }
    } catch (Exception ex) {
      log.warn("Failed to parse Kalshi websocket payload: {}", ex.toString());
    }
  }

  private void connectAsync() {
    if (shuttingDown.get()) {
      return;
    }
    if (!connecting.compareAndSet(false, true)) {
      return;
    }

    URIHolder uri = new URIHolder(properties.resolvedWsUrl());
    HttpHeaders headers = buildHeaders();

    webSocketClient.execute(uri.uri(), headers, webSocketHandler())
        .doOnSuccess(unused -> handleDisconnect("normal_close", null))
        .doOnError(error -> handleDisconnect("error", error))
        .doFinally(signal -> connecting.set(false))
        .subscribe();
  }

  private WebSocketHandler webSocketHandler() {
    return session -> {
      connected = true;
      reconnectAttempts.set(0);
      sessionRef.set(session);
      sequenceTracker.resetAll();
      marketTickersBySid.clear();
      resyncInProgress.clear();
      outboundSinkRef.set(newOutboundSink());

      subscribeOrderbookDelta();

      Flux<WebSocketMessage> outbound = outboundSinkRef.get().asFlux()
          .map(session::textMessage);

      Mono<Void> send = session.send(outbound);
      Mono<Void> receive = session.receive()
          .map(WebSocketMessage::getPayloadAsText)
          .doOnNext(this::onMessage)
          .then();
      return Mono.when(send, receive);
    };
  }

  private void subscribeOrderbookDelta() {
    if (trackedMarkets.isEmpty()) {
      return;
    }
    for (String marketTicker : List.copyOf(trackedMarkets)) {
      int id = commandId.getAndIncrement();
      WsSubscribeParams params = new WsSubscribeParams(
          List.of(CHANNEL_ORDERBOOK_DELTA),
          null,
          List.of(marketTicker));
      WsSubscribeCommand command = new WsSubscribeCommand(id, params);
      sendCommand(command);
    }
  }

  private void sendCommand(Object command) {
    if (!isSessionOpen()) {
      return;
    }
    try {
      String payload = objectMapper.writeValueAsString(command);
      log.debug("Kalshi WS send command {}", payload);
      outboundSinkRef.get().emitNext(payload, Sinks.EmitFailureHandler.FAIL_FAST);
    } catch (Exception ex) {
      log.warn("Failed to serialize websocket command: {}", ex.toString());
    }
  }

  private void handleSubscriptionAck(JsonNode root) {
    WsSubscriptionAck ack = objectMapper.convertValue(root, WsSubscriptionAck.class);
    if (ack == null || ack.subscriptions() == null) {
      return;
    }
    for (WsSubscription subscription : ack.subscriptions()) {
      if (subscription == null || subscription.sid() == null) {
        continue;
      }
      if (CHANNEL_ORDERBOOK_DELTA.equals(subscription.channel())) {
        Set<String> mappedTickers = resolveSubscriptionTickers(subscription);
        if (!mappedTickers.isEmpty()) {
          marketTickersBySid.put(subscription.sid(), mappedTickers);
          log.info("Kalshi WS subscription sid={} mappedTickers={}", subscription.sid(), mappedTickers);
        } else {
          log.warn("Kalshi WS subscription sid={} had no resolved tickers", subscription.sid());
        }
        hydrateTrackedMarketsFromRestAsync("subscription_ack");
      }
    }
  }

  private void handleOrderbookSnapshot(JsonNode root) {
    WsOrderbookSnapshotMessage snapshot = objectMapper.convertValue(root, WsOrderbookSnapshotMessage.class);
    if (snapshot == null || snapshot.msg() == null) {
      return;
    }
    long count = snapshotMsgCount.incrementAndGet();
    if (snapshot.sid() != null) {
      sequenceTracker.reset(snapshot.sid());
      sequenceTracker.evaluate(snapshot.sid(), snapshot.seq());
    }
    orderBookStore.applySnapshot(snapshot.msg());
    if (count <= 5 || count % 250 == 0) {
      log.debug("Kalshi WS snapshot count={} sid={} seq={} market={}",
          count, snapshot.sid(), snapshot.seq(), snapshot.msg().marketTicker());
    }
  }

  private void handleOrderbookDelta(JsonNode root) {
    WsOrderbookDeltaMessage delta = objectMapper.convertValue(root, WsOrderbookDeltaMessage.class);
    if (delta == null) {
      return;
    }
    long count = deltaMsgCount.incrementAndGet();

    Integer sid = delta.sid();
    Long seq = delta.seq();
    SequenceResult result = sequenceTracker.evaluate(sid, seq);
    if (result.status() == SequenceStatus.GAP || result.status() == SequenceStatus.OUT_OF_ORDER) {
      handleSequenceIssue(sid, seq, result);
      return;
    }

    String marketTicker = extractMarketTicker(delta.msg(), sid);
    if (!StringUtils.hasText(marketTicker)) {
      long dropped = deltaDroppedCount.incrementAndGet();
      log.debug("Dropping orderbook delta with unresolved market sid={} mappedMarkets={}",
          sid, sid == null ? Set.of() : marketTickersBySid.getOrDefault(sid, Set.of()));
      if (dropped % 250 == 0) {
        log.info("Kalshi WS deltas seen={} applied={} dropped={}",
            count, deltaAppliedCount.get(), dropped);
      }
      return;
    }
    orderBookStore.applyDelta(marketTicker, delta.msg());
    long applied = deltaAppliedCount.incrementAndGet();
    if (applied <= 5 || applied % 250 == 0) {
      log.debug("Applied orderbook delta count={} sid={} seq={} market={}",
          applied, sid, seq, marketTicker);
    }
  }

  private void handleSequenceIssue(Integer sid, Long seq, SequenceResult result) {
    if (sid == null || seq == null) {
      return;
    }
    if (!resyncInProgress.add(sid)) {
      return;
    }
    log.warn("Orderbook sequence issue sid={} status={} prev={} current={}",
        sid, result.status(), result.previousSeq(), seq);

    scheduler.execute(() -> {
      try {
        sequenceTracker.reset(sid);
        sequenceTracker.evaluate(sid, seq);
        for (String marketTicker : marketTickersBySid.getOrDefault(sid, Set.of())) {
          var restOrderbook = marketDataApi.getOrderbook(marketTicker, 0);
          orderBookStore.applyRestSnapshot(marketTicker, restOrderbook);
        }
      } catch (Exception ex) {
        log.warn("Failed to resync orderbook for sid {}: {}", sid, ex.toString());
      } finally {
        resyncInProgress.remove(sid);
      }
    });
  }

  private void hydrateTrackedMarketsFromRestAsync(String reason) {
    Set<String> marketSnapshot = Set.copyOf(trackedMarkets);
    if (marketSnapshot.isEmpty()) {
      return;
    }
    scheduler.execute(() -> {
      for (String marketTicker : marketSnapshot) {
        try {
          var restOrderbook = marketDataApi.getOrderbook(marketTicker, 0);
          orderBookStore.applyRestSnapshot(marketTicker, restOrderbook);
        } catch (Exception ex) {
          log.debug("Skipping REST depth hydrate market={} reason={} error={}",
              marketTicker, reason, ex.toString());
        }
      }
    });
  }

  private void handleError(JsonNode root) {
    WsErrorResponse errorResponse = objectMapper.convertValue(root, WsErrorResponse.class);
    if (errorResponse == null || errorResponse.msg() == null) {
      return;
    }
    log.warn("Kalshi websocket error code={} msg={}", errorResponse.msg().code(), errorResponse.msg().msg());
  }

  private String extractMarketTicker(JsonNode msgNode, Integer sid) {
    if (msgNode != null && msgNode.hasNonNull("market_ticker")) {
      return msgNode.get("market_ticker").asText();
    }
    if (sid == null) {
      return null;
    }
    Set<String> marketTickers = marketTickersBySid.get(sid);
    if (marketTickers == null || marketTickers.size() != 1) {
      return null;
    }
    return marketTickers.iterator().next();
  }

  private Set<String> resolveSubscriptionTickers(WsSubscription subscription) {
    Set<String> resolved = new LinkedHashSet<>();

    if (StringUtils.hasText(subscription.marketTicker())) {
      resolved.add(subscription.marketTicker().trim().toUpperCase());
    }
    if (subscription.marketTickers() != null) {
      for (String ticker : subscription.marketTickers()) {
        if (StringUtils.hasText(ticker)) {
          resolved.add(ticker.trim().toUpperCase());
        }
      }
    }
    if (resolved.isEmpty() && trackedMarkets.size() == 1) {
      resolved.add(trackedMarkets.iterator().next());
    }

    return Set.copyOf(resolved);
  }

  private HttpHeaders buildHeaders() {
    HttpHeaders headers = new HttpHeaders();
    if (properties.isAuthEnabled()) {
      SignedHeaders signedHeaders = signerProvider.getSigner().signWebSocketHandshake();
      signedHeaders.apply(headers);
    }
    return headers;
  }

  private void startWatchdog() {
    long timeoutSeconds = properties.getWebSocket().getWatchdogTimeoutSeconds();
    long periodSeconds = Math.max(5, timeoutSeconds / 2);
    scheduler.scheduleAtFixedRate(this::runWatchdog, periodSeconds, periodSeconds, TimeUnit.SECONDS);
  }

  private void startRestFallbackRefresh() {
    scheduler.scheduleAtFixedRate(
        this::runRestFallbackRefresh,
        REST_FALLBACK_REFRESH_SECONDS,
        REST_FALLBACK_REFRESH_SECONDS,
        TimeUnit.SECONDS);
  }

  private void runWatchdog() {
    if (shuttingDown.get() || !isSessionOpen()) {
      return;
    }
    long elapsedSeconds = Duration.ofNanos(System.nanoTime() - lastInboundNanos.get()).toSeconds();
    if (elapsedSeconds <= properties.getWebSocket().getWatchdogTimeoutSeconds()) {
      return;
    }
    log.warn("Kalshi websocket watchdog triggered after {} seconds without inbound data", elapsedSeconds);
    handleDisconnect("watchdog", null);
  }

  private void runRestFallbackRefresh() {
    if (shuttingDown.get()) {
      return;
    }

    Set<String> marketSnapshot = Set.copyOf(trackedMarkets);
    if (marketSnapshot.isEmpty()) {
      return;
    }

    for (String marketTicker : marketSnapshot) {
      try {
        var restOrderbook = marketDataApi.getOrderbook(marketTicker, 0);
        orderBookStore.applyRestSnapshot(marketTicker, restOrderbook);
      } catch (Exception ex) {
        log.debug("Skipping fallback REST refresh market={} error={}", marketTicker, ex.toString());
      }
    }
  }

  private boolean isSessionOpen() {
    WebSocketSession session = sessionRef.get();
    return connected && session != null && session.isOpen();
  }

  private void handleDisconnect(String reason, Throwable error) {
    WebSocketSession session = sessionRef.getAndSet(null);
    connected = false;
    connecting.set(false);
    marketTickersBySid.clear();
    resyncInProgress.clear();
    sequenceTracker.resetAll();
    outboundSinkRef.set(newOutboundSink());

    if (session != null && session.isOpen()) {
      session.close().subscribe();
    }

    if (shuttingDown.get()) {
      return;
    }
    scheduleReconnect(reason, error);
  }

  private void scheduleReconnect(String reason, Throwable error) {
    if (!reconnectScheduled.compareAndSet(false, true)) {
      return;
    }

    int maxAttempts = properties.getWebSocket().getMaxReconnectAttempts();
    int attempts = reconnectAttempts.incrementAndGet();
    if (maxAttempts > 0 && attempts > maxAttempts) {
      reconnectScheduled.set(false);
      log.error("Kalshi websocket reconnect attempts exhausted maxAttempts={}", maxAttempts);
      return;
    }

    long delayMs = computeReconnectDelayMs(attempts);
    scheduler.schedule(() -> {
      reconnectScheduled.set(false);
      connectAsync();
    }, delayMs, TimeUnit.MILLISECONDS);

    log.warn("Scheduling Kalshi websocket reconnect reason={} attempt={} delayMs={} error={}",
        reason, attempts, delayMs, error == null ? "-" : error.toString());
  }

  private long computeReconnectDelayMs(int attempts) {
    long base = properties.getWebSocket().getReconnectBaseBackoffMs();
    long multiplier = 1L << Math.min(Math.max(attempts - 1, 0), 6);
    return Math.min(base * multiplier, 30_000L);
  }

  private OrderBookSnapshotView toView(OrderBookSnapshot snapshot) {
    List<OrderBookLevelView> yesLevels = toLevels(snapshot.yesSide());
    List<OrderBookLevelView> noLevels = toLevels(snapshot.noSide());

    Integer bestYesBid = yesLevels.isEmpty() ? null : yesLevels.get(0).priceCents();
    Integer bestNoBid = noLevels.isEmpty() ? null : noLevels.get(0).priceCents();
    Integer impliedYesAsk = bestNoBid == null ? null : Math.max(1, 100 - bestNoBid);
    Integer impliedNoAsk = bestYesBid == null ? null : Math.max(1, 100 - bestYesBid);

    return new OrderBookSnapshotView(
        snapshot.marketTicker(),
        yesLevels,
        noLevels,
        bestYesBid,
        bestNoBid,
        impliedYesAsk,
        impliedNoAsk,
        snapshot.asOfUtc());
  }

  private List<OrderBookLevelView> toLevels(Map<Integer, BigDecimal> sideLevels) {
    List<OrderBookLevelView> result = new ArrayList<>();
    sideLevels.entrySet().stream()
        .sorted(Map.Entry.comparingByKey(Comparator.reverseOrder()))
        .forEach(entry -> result.add(new OrderBookLevelView(entry.getKey(), entry.getValue())));
    return result;
  }

  private Sinks.Many<String> newOutboundSink() {
    return Sinks.many().multicast().onBackpressureBuffer();
  }

  @Override
  public void destroy() {
    disconnect();
    scheduler.shutdownNow();
  }

  private record URIHolder(java.net.URI uri) {
    URIHolder(String wsUrl) {
      this(java.net.URI.create(wsUrl));
    }
  }
}
