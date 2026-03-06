package com.predictionmarkets.weather.kalshiapi.live;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.api.KalshiMarketDataApi;
import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import com.predictionmarkets.weather.kalshiapi.http.KalshiApiException;
import com.predictionmarkets.weather.kalshiapi.model.market.Event;
import com.predictionmarkets.weather.kalshiapi.model.market.EventResponse;
import com.predictionmarkets.weather.kalshiapi.model.market.Market;
import com.predictionmarkets.weather.kalshiapi.service.KalshiOrderBookService;
import com.predictionmarkets.weather.kalshiapi.trading.OrderBookLevelView;
import com.predictionmarkets.weather.kalshiapi.trading.OrderBookSnapshotView;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Clock;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.beans.factory.DisposableBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

@Service
public class LiveOrderbookStreamService implements DisposableBean {

  private static final Logger log = LoggerFactory.getLogger(LiveOrderbookStreamService.class);
  private static final DateTimeFormatter MONTH_FORMAT = DateTimeFormatter.ofPattern("MMM", Locale.ENGLISH);
  private static final Pattern TITLE_DATE_PATTERN =
      Pattern.compile("\\bon\\s+([A-Za-z]+)\\s+(\\d{1,2}),\\s*(\\d{4})\\b", Pattern.CASE_INSENSITIVE);
  private static final DateTimeFormatter TITLE_DATE_FORMATTER =
      new DateTimeFormatterBuilder()
          .parseCaseInsensitive()
          .appendPattern("MMM d uuuu")
          .parseDefaulting(ChronoField.HOUR_OF_DAY, 0)
          .toFormatter(Locale.ENGLISH);
  private static final Pattern QUESTION_BUCKET_PATTERN =
      Pattern.compile("(?i)\\bbe\\s+(.+?)\\s+on\\s+[A-Za-z]+\\s+\\d{1,2},\\s*\\d{4}\\??");
  private static final Pattern LESS_THAN_BUCKET_PATTERN =
      Pattern.compile("^<\\s*(\\d+)\\s*$");
  private static final Pattern GREATER_THAN_BUCKET_PATTERN =
      Pattern.compile("^>\\s*(\\d+)\\s*$");
  private static final Pattern RANGE_BUCKET_PATTERN =
      Pattern.compile("^(\\d+)\\s*(?:-|to)\\s*(\\d+)\\s*$", Pattern.CASE_INSENSITIVE);
  private static final Pattern NUMERIC_PATTERN = Pattern.compile("(\\d+)");
  private static final int PMF_SUPPORT_LO = -20;
  private static final int PMF_SUPPORT_HI = 130;

  private final KalshiOrderBookService orderBookService;
  private final KalshiMarketDataApi marketDataApi;
  private final LiveTradingProperties properties;
  private final Clock clock;
  private final ObjectMapper objectMapper;
  private final ScheduledExecutorService scheduler;
  private final LocalTime nextTargetDateCutoffLocalTime;

  private final AtomicBoolean started = new AtomicBoolean(false);
  private final AtomicReference<Map<String, StationRuntimeState>> stationStateById =
      new AtomicReference<>(Map.of());
  private final AtomicReference<Map<LocalDate, Map<String, StationRuntimeState>>> stationStateByDate =
      new AtomicReference<>(Map.of());
  private final AtomicReference<Set<String>> trackedMarkets =
      new AtomicReference<>(Set.of());
  private final AtomicReference<InferenceIndex> latestInference =
      new AtomicReference<>(InferenceIndex.empty());
  private final AtomicReference<LiveOrderbookFrame> latestFrame = new AtomicReference<>();
  private final Sinks.Many<LiveOrderbookFrame> sink = Sinks.many().replay().latest();

  public LiveOrderbookStreamService(KalshiOrderBookService orderBookService,
                                    KalshiMarketDataApi marketDataApi,
                                    LiveTradingProperties properties,
                                    Clock clock,
                                    ObjectMapper objectMapper) {
    this.orderBookService = orderBookService;
    this.marketDataApi = marketDataApi;
    this.properties = properties;
    this.clock = clock;
    this.objectMapper = objectMapper;
    this.nextTargetDateCutoffLocalTime = parseCutoffTime(properties.getNextTargetDateCutoffLocalTime());
    this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
      Thread t = new Thread(r, "kalshi-live-orderbook-stream");
      t.setDaemon(true);
      return t;
    });
  }

  @EventListener(ApplicationReadyEvent.class)
  public void onApplicationReady() {
    startIfEnabled();
  }

  public Flux<LiveOrderbookFrame> stream() {
    return sink.asFlux();
  }

  public LiveOrderbookFrame currentSnapshot() {
    LiveOrderbookFrame frame = latestFrame.get();
    if (frame != null) {
      return frame;
    }
    return new LiveOrderbookFrame(Instant.now(clock), List.of(), List.of(), strategyConfigView());
  }

  public LiveOrderbookFrame currentSnapshot(LocalDate targetDateLocal) {
    if (targetDateLocal == null) {
      return currentSnapshot();
    }
    Map<String, StationRuntimeState> stationStates = stationStateByDate.get().get(targetDateLocal);
    if (stationStates == null || stationStates.isEmpty()) {
      stationStates = resolveStationsForDate(targetDateLocal);
    }
    if (stationStates == null || stationStates.isEmpty()) {
      return new LiveOrderbookFrame(Instant.now(clock), List.of(), List.of(), strategyConfigView());
    }
    return buildFrame(stationStates, targetDateLocal);
  }

  private void startIfEnabled() {
    if (!properties.isEnabled()) {
      log.info("Live orderbook streaming is disabled (kalshi.live-trading.enabled=false)");
      return;
    }
    if (!started.compareAndSet(false, true)) {
      return;
    }
    orderBookService.connect();
    refreshMarketsSafely();
    refreshInferenceSafely();
    publishFrameSafely();

    scheduler.scheduleWithFixedDelay(
        this::refreshMarketsSafely,
        properties.getMarketResolveIntervalSeconds(),
        properties.getMarketResolveIntervalSeconds(),
        TimeUnit.SECONDS);
    scheduler.scheduleWithFixedDelay(
        this::refreshInferenceSafely,
        properties.getInferenceRefreshIntervalSeconds(),
        properties.getInferenceRefreshIntervalSeconds(),
        TimeUnit.SECONDS);
    scheduler.scheduleAtFixedRate(
        this::publishFrameSafely,
        properties.getPublishIntervalMillis(),
        properties.getPublishIntervalMillis(),
        TimeUnit.MILLISECONDS);
  }

  private void refreshMarketsSafely() {
    try {
      refreshMarkets();
    } catch (Exception ex) {
      log.warn("Failed to refresh live market mappings: {}", ex.toString());
    }
  }

  private void refreshInferenceSafely() {
    try {
      refreshInference();
    } catch (Exception ex) {
      log.warn("Failed to refresh live inference snapshot: {}", ex.toString());
    }
  }

  private void publishFrameSafely() {
    try {
      LiveOrderbookFrame frame = buildFrame();
      latestFrame.set(frame);
      Sinks.EmitResult emitResult = sink.tryEmitNext(frame);
      if (emitResult.isFailure()) {
        log.debug("Live frame emit skipped due to sink state={}", emitResult);
      }
    } catch (Exception ex) {
      log.warn("Failed to publish live orderbook frame: {}", ex.toString());
    }
  }

  private void refreshMarkets() {
    Map<String, StationRuntimeState> previousPrimaryState = stationStateById.get();
    Map<LocalDate, Map<String, StationRuntimeState>> previousByDate = stationStateByDate.get();
    Map<String, StationRuntimeState> nextPrimaryState = new LinkedHashMap<>();
    Map<LocalDate, Map<String, StationRuntimeState>> nextByDateMutable = new LinkedHashMap<>();
    Set<String> nextTrackedMarkets = new HashSet<>();

    for (LiveTradingProperties.Station station : properties.getStations()) {
      String stationId = normalizeStationId(station.getStationId());
      if (!StringUtils.hasText(stationId)) {
        continue;
      }
      ZoneId zoneId = parseStationZone(stationId, station.getZoneId());
      if (zoneId == null) {
        continue;
      }
      LocalDate todayLocal = LocalDate.now(clock.withZone(zoneId));
      LocalDate tomorrowLocal = todayLocal.plusDays(1);
      LocalDate primaryDate = primaryTargetDate(zoneId, todayLocal);
      LocalDate secondaryDate = primaryDate.equals(todayLocal) ? tomorrowLocal : todayLocal;

      for (LocalDate targetDateLocal : List.of(primaryDate, secondaryDate)) {
        StationRuntimeState resolved = resolveStation(station, zoneId, targetDateLocal);
        if (resolved == null) {
          resolved = previousByDate.getOrDefault(targetDateLocal, Map.of()).get(stationId);
        }
        if (resolved == null && targetDateLocal.equals(primaryDate)) {
          resolved = previousPrimaryState.get(stationId);
        }
        if (resolved == null) {
          continue;
        }

        nextByDateMutable
            .computeIfAbsent(targetDateLocal, ignored -> new LinkedHashMap<>())
            .put(stationId, resolved);
        if (targetDateLocal.equals(primaryDate)) {
          nextPrimaryState.put(stationId, resolved);
        }
        for (MarketDescriptor market : resolved.markets()) {
          if (StringUtils.hasText(market.marketTicker())) {
            nextTrackedMarkets.add(market.marketTicker());
          }
        }
      }
    }

    Map<LocalDate, Map<String, StationRuntimeState>> immutableByDate = new LinkedHashMap<>();
    for (Map.Entry<LocalDate, Map<String, StationRuntimeState>> entry : nextByDateMutable.entrySet()) {
      immutableByDate.put(entry.getKey(), Map.copyOf(entry.getValue()));
    }

    stationStateByDate.set(Map.copyOf(immutableByDate));
    stationStateById.set(Map.copyOf(nextPrimaryState));
    Set<String> normalizedTracked = Set.copyOf(nextTrackedMarkets);
    if (!Objects.equals(trackedMarkets.get(), normalizedTracked)) {
      trackedMarkets.set(normalizedTracked);
      orderBookService.replaceTrackedMarkets(normalizedTracked);
      log.info("Live tracked markets updated count={}", normalizedTracked.size());
    }
  }

  private void refreshInference() throws IOException {
    List<Path> reportPaths = findInferenceReportCandidates();
    if (reportPaths.isEmpty()) {
      latestInference.set(InferenceIndex.empty());
      return;
    }

    Map<String, Long> signature = new LinkedHashMap<>();
    for (Path reportPath : reportPaths) {
      signature.put(reportPath.toAbsolutePath().normalize().toString(), lastModifiedMillisSafe(reportPath));
    }

    InferenceIndex current = latestInference.get();
    if (current.matches(signature)) {
      return;
    }

    Map<String, Map<String, StationInferenceVersioned>> byDateMutable = new LinkedHashMap<>();
    int loadedReports = 0;
    for (Path reportPath : reportPaths) {
      FileTime modified = Files.getLastModifiedTime(reportPath);
      JsonNode report = objectMapper.readTree(reportPath.toFile());
      String targetDateLocal = textOrNull(report.path("target_date_local"));
      if (!StringUtils.hasText(targetDateLocal)) {
        continue;
      }
      JsonNode inferenceByStation = report.path("inference_by_station");
      if (!inferenceByStation.isObject()) {
        continue;
      }
      loadedReports += 1;
      Map<String, StationInferenceVersioned> byStation =
          byDateMutable.computeIfAbsent(targetDateLocal, ignored -> new LinkedHashMap<>());
      inferenceByStation.fields().forEachRemaining(entry -> {
        JsonNode stationNode = entry.getValue();
        String stationId = normalizeStationId(textOrNull(stationNode.path("station_id")));
        if (!StringUtils.hasText(stationId)) {
          stationId = normalizeStationId(entry.getKey());
        }
        if (!StringUtils.hasText(stationId)) {
          return;
        }
        Map<String, Double> quantilesByLabel = parseQuantilesByLabel(stationNode.path("quantiles"));
        Map<Double, Double> quantilesByTau = parseQuantilesByTau(quantilesByLabel);
        StationInference nextInference = new StationInference(
            stationId,
            textOrNull(stationNode.path("target_date_local")),
            textOrNull(stationNode.path("runtime_utc")),
            numberOrNull(stationNode.path("prediction_point_tmax_f")),
            Map.copyOf(quantilesByLabel),
            Map.copyOf(quantilesByTau));
        StationInferenceVersioned previous = byStation.get(stationId);
        if (previous == null || modified.compareTo(previous.reportModifiedTime()) >= 0) {
          byStation.put(stationId, new StationInferenceVersioned(nextInference, modified));
        }
      });
    }

    Map<String, Map<String, StationInference>> byDate = new LinkedHashMap<>();
    int loadedStations = 0;
    for (Map.Entry<String, Map<String, StationInferenceVersioned>> entry : byDateMutable.entrySet()) {
      Map<String, StationInference> immutableByStation = new LinkedHashMap<>();
      for (Map.Entry<String, StationInferenceVersioned> stationEntry : entry.getValue().entrySet()) {
        immutableByStation.put(stationEntry.getKey(), stationEntry.getValue().stationInference());
      }
      loadedStations += immutableByStation.size();
      byDate.put(entry.getKey(), Map.copyOf(immutableByStation));
    }

    latestInference.set(new InferenceIndex(Map.copyOf(signature), Map.copyOf(byDate)));
    log.info("Live inference index loaded reports={} targetDates={} stationSnapshots={}",
        loadedReports, byDate.size(), loadedStations);
  }

  private List<Path> findInferenceReportCandidates() throws IOException {
    if (!StringUtils.hasText(properties.getInferenceRootDir())) {
      return List.of();
    }
    Path root = Path.of(properties.getInferenceRootDir());
    if (!Files.isDirectory(root)) {
      return List.of();
    }
    String reportFileName = StringUtils.hasText(properties.getInferenceReportFileName())
        ? properties.getInferenceReportFileName().trim()
        : "inference_report.json";

    List<Path> candidates = new ArrayList<>();
    Path directReport = root.resolve(reportFileName);
    if (Files.isRegularFile(directReport)) {
      candidates.add(directReport);
    }

    try (Stream<Path> dirs = Files.list(root)) {
      dirs.filter(Files::isDirectory)
          .map(dir -> dir.resolve(reportFileName))
          .filter(Files::isRegularFile)
          .forEach(candidates::add);
    }

    candidates.sort(Comparator.comparingLong(this::lastModifiedMillisSafe).reversed());
    return candidates;
  }

  private StationRuntimeState resolveStation(LiveTradingProperties.Station station,
                                             ZoneId zoneId,
                                             LocalDate targetDateLocal) {
    String stationId = normalizeStationId(station.getStationId());
    String seriesTicker = normalizeSeries(station.getSeriesTicker());
    if (!StringUtils.hasText(stationId) || !StringUtils.hasText(seriesTicker)) {
      return null;
    }

    List<EventCandidate> candidates = fetchEventCandidates(seriesTicker, targetDateLocal);
    EventCandidate selected = chooseEventCandidate(candidates, targetDateLocal);
    if (selected == null) {
      log.warn("No live event resolved for station={} date={} series={}",
          stationId, targetDateLocal, seriesTicker);
      return null;
    }
    List<MarketDescriptor> markets = selected.markets().stream()
        .filter(Objects::nonNull)
        .filter(m -> StringUtils.hasText(m.ticker()))
        .map(this::toMarketDescriptor)
        .sorted(Comparator.comparing(MarketDescriptor::sortKey).thenComparing(MarketDescriptor::marketTicker))
        .toList();

    return new StationRuntimeState(
        stationId,
        station.getDisplayName(),
        seriesTicker,
        zoneId.getId(),
        targetDateLocal,
        selected.eventTicker(),
        Instant.now(clock),
        markets);
  }

  private Map<String, StationRuntimeState> resolveStationsForDate(LocalDate targetDateLocal) {
    Map<String, StationRuntimeState> resolved = new LinkedHashMap<>();
    for (LiveTradingProperties.Station station : properties.getStations()) {
      String stationId = normalizeStationId(station.getStationId());
      if (!StringUtils.hasText(stationId)) {
        continue;
      }
      ZoneId zoneId = parseStationZone(stationId, station.getZoneId());
      if (zoneId == null) {
        continue;
      }
      StationRuntimeState stationState = resolveStation(station, zoneId, targetDateLocal);
      if (stationState != null) {
        resolved.put(stationId, stationState);
      }
    }
    return resolved;
  }

  private ZoneId parseStationZone(String stationId, String zoneIdRaw) {
    try {
      return ZoneId.of(zoneIdRaw);
    } catch (Exception ex) {
      log.warn("Invalid zoneId for station {}: {}", stationId, zoneIdRaw);
      return null;
    }
  }

  private LocalDate primaryTargetDate(ZoneId zoneId, LocalDate todayLocal) {
    LocalTime nowLocal = LocalTime.now(clock.withZone(zoneId));
    return nowLocal.isBefore(nextTargetDateCutoffLocalTime) ? todayLocal : todayLocal.plusDays(1);
  }

  private List<EventCandidate> fetchEventCandidates(String seriesTicker, LocalDate targetDateLocal) {
    List<EventCandidate> out = new ArrayList<>();
    for (String eventTicker : eventTickerCandidates(seriesTicker, targetDateLocal)) {
      try {
        EventResponse response = marketDataApi.getEvent(eventTicker);
        Event event = response == null ? null : response.event();
        List<Market> markets = response == null ? null : response.markets();
        if (event == null || markets == null || markets.isEmpty()) {
          continue;
        }
        LocalDate titleDate = parseEventTitleDate(event.title()).orElse(null);
        String resolvedEventTicker = StringUtils.hasText(event.eventTicker()) ? event.eventTicker() : eventTicker;
        out.add(new EventCandidate(resolvedEventTicker, titleDate, markets));
      } catch (KalshiApiException ex) {
        if (ex.getStatusCode() != 404) {
          log.warn("Event lookup failed ticker={} status={} details={}",
              eventTicker, ex.getStatusCode(), ex.toString());
        }
      } catch (Exception ex) {
        log.warn("Event lookup failed ticker={} error={}", eventTicker, ex.toString());
      }
    }
    return out;
  }

  private EventCandidate chooseEventCandidate(List<EventCandidate> candidates, LocalDate targetDateLocal) {
    if (candidates.isEmpty()) {
      return null;
    }
    List<EventCandidate> exact = candidates.stream()
        .filter(c -> targetDateLocal.equals(c.titleDate()))
        .toList();
    if (!exact.isEmpty()) {
      return exact.get(0);
    }
    return candidates.get(0);
  }

  private List<String> eventTickerCandidates(String seriesTicker, LocalDate targetDateLocal) {
    String day = String.format(Locale.ROOT, "%02d", targetDateLocal.getDayOfMonth());
    String month = targetDateLocal.format(MONTH_FORMAT).toUpperCase(Locale.ROOT);
    String year2 = String.format(Locale.ROOT, "%02d", targetDateLocal.getYear() % 100);
    String yyMonDd = seriesTicker + "-" + year2 + month + day;
    String ddMonYy = seriesTicker + "-" + day + month + year2;

    LinkedHashSet<String> ordered = new LinkedHashSet<>();
    ordered.add(yyMonDd);
    ordered.add(ddMonYy);
    return new ArrayList<>(ordered);
  }

  private Optional<LocalDate> parseEventTitleDate(String title) {
    if (!StringUtils.hasText(title)) {
      return Optional.empty();
    }
    Matcher matcher = TITLE_DATE_PATTERN.matcher(title);
    if (!matcher.find()) {
      return Optional.empty();
    }
    String text = matcher.group(1) + " " + matcher.group(2) + " " + matcher.group(3);
    try {
      return Optional.of(LocalDate.from(TITLE_DATE_FORMATTER.parse(text)));
    } catch (DateTimeParseException ex) {
      return Optional.empty();
    }
  }

  private MarketDescriptor toMarketDescriptor(Market market) {
    String rawBucketLabel = StringUtils.hasText(market.subtitle())
        ? market.subtitle().trim()
        : (StringUtils.hasText(market.title()) ? market.title().trim() : market.ticker());
    String bucketLabel = canonicalizeBucketLabel(rawBucketLabel);
    return new MarketDescriptor(
        market.ticker().trim().toUpperCase(Locale.ROOT),
        bucketLabel,
        market.status(),
        bucketSortKey(bucketLabel));
  }

  private String canonicalizeBucketLabel(String rawLabel) {
    if (!StringUtils.hasText(rawLabel)) {
      return rawLabel;
    }
    String trimmed = rawLabel.trim();
    Matcher questionMatcher = QUESTION_BUCKET_PATTERN.matcher(trimmed);
    if (!questionMatcher.find()) {
      return trimmed;
    }
    String bucketToken = questionMatcher.group(1)
        .replace("°", "")
        .replace("*", "")
        .trim();

    Matcher lessMatcher = LESS_THAN_BUCKET_PATTERN.matcher(bucketToken);
    if (lessMatcher.matches()) {
      int strike = Integer.parseInt(lessMatcher.group(1));
      return Math.max(0, strike - 1) + "° or below";
    }

    Matcher greaterMatcher = GREATER_THAN_BUCKET_PATTERN.matcher(bucketToken);
    if (greaterMatcher.matches()) {
      int strike = Integer.parseInt(greaterMatcher.group(1));
      return (strike + 1) + "° or above";
    }

    Matcher rangeMatcher = RANGE_BUCKET_PATTERN.matcher(bucketToken);
    if (rangeMatcher.matches()) {
      int lo = Integer.parseInt(rangeMatcher.group(1));
      int hi = Integer.parseInt(rangeMatcher.group(2));
      return Math.min(lo, hi) + "° to " + Math.max(lo, hi) + "°";
    }

    return trimmed;
  }

  private String bucketSortKey(String label) {
    if (!StringUtils.hasText(label)) {
      return "z|9999|9999|";
    }
    String normalized = canonicalizeBucketLabel(label).toLowerCase(Locale.ROOT);
    List<Integer> numbers = new ArrayList<>();
    Matcher matcher = NUMERIC_PATTERN.matcher(normalized);
    while (matcher.find()) {
      numbers.add(Integer.parseInt(matcher.group(1)));
    }

    if (normalized.contains("or below") || normalized.contains("or less")) {
      int hi = numbers.isEmpty() ? 9_999 : numbers.get(0);
      return String.format(Locale.ROOT, "a|%04d|%04d|%s", hi, 0, normalized);
    }
    if (normalized.contains("or above") || normalized.contains("or higher")) {
      int lo = numbers.isEmpty() ? 9_999 : numbers.get(0);
      return String.format(Locale.ROOT, "c|%04d|%04d|%s", lo, 9_999, normalized);
    }
    if (numbers.size() >= 2) {
      int lo = Math.min(numbers.get(0), numbers.get(1));
      int hi = Math.max(numbers.get(0), numbers.get(1));
      return String.format(Locale.ROOT, "b|%04d|%04d|%s", lo, hi, normalized);
    }
    int fallback = numbers.isEmpty() ? 9_999 : numbers.get(0);
    return String.format(Locale.ROOT, "d|%04d|%04d|%s", fallback, fallback, normalized);
  }

  private LiveOrderbookFrame buildFrame() {
    return buildFrame(stationStateById.get(), null);
  }

  private LiveOrderbookFrame buildFrame(Map<String, StationRuntimeState> stationStateByStationId,
                                        LocalDate targetDateOverride) {
    Instant asOfUtc = Instant.now(clock);
    InferenceIndex inferenceIndex = latestInference.get();
    List<LiveStationOrderbookView> stations = new ArrayList<>();
    List<LiveOpportunityView> opportunities = new ArrayList<>();
    Map<String, StationRuntimeState> resolvedStates =
        stationStateByStationId == null ? Map.of() : stationStateByStationId;
    for (LiveTradingProperties.Station cfg : properties.getStations()) {
      String stationId = normalizeStationId(cfg.getStationId());
      StationRuntimeState stationState = resolvedStates.get(stationId);
      LocalDate inferenceTargetDate = stationState == null ? targetDateOverride : stationState.targetDateLocal();
      StationInference stationInference = selectStationInference(
          inferenceIndex,
          stationId,
          inferenceTargetDate);
      Map<Integer, Double> stationPmf = stationInference == null || stationInference.quantilesByTau().isEmpty()
          ? Map.of()
          : pmfIntFromQuantiles(stationInference.quantilesByTau(), PMF_SUPPORT_LO, PMF_SUPPORT_HI);
      if (stationState == null) {
        stations.add(new LiveStationOrderbookView(
            stationId,
            cfg.getDisplayName(),
            normalizeSeries(cfg.getSeriesTicker()),
            cfg.getZoneId(),
            targetDateOverride,
            null,
            null,
            stationInference == null ? null : stationInference.runtimeUtc(),
            stationInference == null ? null : stationInference.predictionPointTmaxF(),
            stationInference == null ? Map.of() : stationInference.quantilesByLabel(),
            List.of()));
        continue;
      }
      List<LiveBucketOrderbookView> buckets = new ArrayList<>();
      for (MarketDescriptor market : stationState.markets()) {
        OrderBookSnapshotView snapshot = orderBookService.snapshot(market.marketTicker());
        List<OrderBookLevelView> yesTop = topLevels(snapshot.yesLevels());
        List<OrderBookLevelView> noTop = topLevels(snapshot.noLevels());
        Integer yesBid = snapshot.bestYesBidCents();
        Integer yesAsk = snapshot.impliedYesAskCents();
        Integer noBid = snapshot.bestNoBidCents();
        Integer noAsk = snapshot.impliedNoAskCents();
        Double yesModelWinProb = null;
        Double yesEv = null;
        Double noModelWinProb = null;
        Double noEv = null;

        BucketSpec parsedBucket = parseBucketLabel(market.bucketLabel());
        if (parsedBucket != null && !stationPmf.isEmpty()) {
          double computedYes = bucketProbability(stationPmf, parsedBucket);
          double computedNo = 1.0 - computedYes;
          yesModelWinProb = computedYes;
          noModelWinProb = computedNo;

          Double yesMarketProb = normalizeProbFromCents(yesAsk);
          Double noMarketProb = normalizeProbFromCents(noAsk);
          yesEv = evFromModelAndMarket(yesModelWinProb, yesMarketProb);
          noEv = evFromModelAndMarket(noModelWinProb, noMarketProb);

          maybeAddOpportunity(
              opportunities,
              stationState.stationId(),
              market,
              "YES",
              yesModelWinProb,
              yesMarketProb,
              yesAsk,
              yesEv);
          maybeAddOpportunity(
              opportunities,
              stationState.stationId(),
              market,
              "NO",
              noModelWinProb,
              noMarketProb,
              noAsk,
              noEv);
        }

        buckets.add(new LiveBucketOrderbookView(
            market.marketTicker(),
            market.bucketLabel(),
            market.marketStatus(),
            yesBid,
            yesAsk,
            spread(yesBid, yesAsk),
            yesModelWinProb,
            yesEv,
            noBid,
            noAsk,
            spread(noBid, noAsk),
            noModelWinProb,
            noEv,
            midpoint(yesBid, yesAsk),
            snapshot.asOfUtc(),
            yesTop,
            noTop));
      }

      stations.add(new LiveStationOrderbookView(
          stationState.stationId(),
          stationState.displayName(),
          stationState.seriesTicker(),
          stationState.zoneId(),
          stationState.targetDateLocal(),
          stationState.eventTicker(),
          stationState.resolvedAtUtc(),
          stationInference == null ? null : stationInference.runtimeUtc(),
          stationInference == null ? null : stationInference.predictionPointTmaxF(),
          stationInference == null ? Map.of() : stationInference.quantilesByLabel(),
          buckets));
    }
    opportunities.sort(this::compareOpportunities);
    int maxCount = Math.max(1, properties.getOpportunitiesMaxCount());
    List<LiveOpportunityView> limited = opportunities.size() > maxCount
        ? new ArrayList<>(opportunities.subList(0, maxCount))
        : opportunities;
    return new LiveOrderbookFrame(asOfUtc, stations, limited, strategyConfigView());
  }

  private StationInference selectStationInference(InferenceIndex index,
                                                  String stationId,
                                                  LocalDate targetDateLocal) {
    if (index == null || !StringUtils.hasText(stationId)) {
      return null;
    }
    if (targetDateLocal == null) {
      return null;
    }
    Map<String, StationInference> byStation = index.byTargetDate().get(targetDateLocal.toString());
    if (byStation == null || byStation.isEmpty()) {
      return null;
    }
    StationInference inference = byStation.get(normalizeStationId(stationId));
    if (inference == null) {
      return null;
    }
    if (!StringUtils.hasText(inference.targetDateLocal())) {
      return inference;
    }
    return targetDateLocal.toString().equals(inference.targetDateLocal()) ? inference : null;
  }

  private Map<String, Double> parseQuantilesByLabel(JsonNode quantilesNode) {
    if (quantilesNode == null || !quantilesNode.isObject()) {
      return Map.of();
    }
    Map<String, Double> out = new LinkedHashMap<>();
    quantilesNode.fields().forEachRemaining(entry -> {
      Double value = numberOrNull(entry.getValue());
      if (value != null) {
        out.put(entry.getKey(), value);
      }
    });
    return out;
  }

  private Map<Double, Double> parseQuantilesByTau(Map<String, Double> quantilesByLabel) {
    if (quantilesByLabel == null || quantilesByLabel.isEmpty()) {
      return Map.of();
    }
    Map<Double, Double> byTau = new LinkedHashMap<>();
    for (Map.Entry<String, Double> entry : quantilesByLabel.entrySet()) {
      if (!StringUtils.hasText(entry.getKey()) || entry.getValue() == null) {
        continue;
      }
      String key = entry.getKey().trim().toLowerCase(Locale.ROOT);
      if (!key.startsWith("q_")) {
        continue;
      }
      try {
        double tau = Double.parseDouble(key.substring(2));
        if (tau < 0.0 || tau > 1.0 || !Double.isFinite(tau)) {
          continue;
        }
        byTau.put(tau, entry.getValue());
      } catch (NumberFormatException ignored) {
        // ignore malformed quantile keys
      }
    }
    return byTau;
  }

  private Map<Integer, Double> pmfIntFromQuantiles(Map<Double, Double> quantilesByTau,
                                                   int supportLo,
                                                   int supportHi) {
    if (quantilesByTau == null || quantilesByTau.size() < 2) {
      return Map.of();
    }
    Map<Integer, Double> out = new LinkedHashMap<>();
    double total = 0.0;
    for (int temp = supportLo; temp <= supportHi; temp += 1) {
      double p = cdfFromQuantiles(quantilesByTau, temp + 0.5) - cdfFromQuantiles(quantilesByTau, temp - 0.5);
      double clamped = Math.max(0.0, p);
      out.put(temp, clamped);
      total += clamped;
    }
    if (!Double.isFinite(total) || total <= 0.0) {
      int width = supportHi - supportLo + 1;
      double uniform = 1.0 / width;
      Map<Integer, Double> uniformOut = new LinkedHashMap<>();
      for (int temp = supportLo; temp <= supportHi; temp += 1) {
        uniformOut.put(temp, uniform);
      }
      return uniformOut;
    }
    Map<Integer, Double> normalized = new LinkedHashMap<>();
    for (Map.Entry<Integer, Double> entry : out.entrySet()) {
      normalized.put(entry.getKey(), entry.getValue() / total);
    }
    return normalized;
  }

  private double cdfFromQuantiles(Map<Double, Double> quantilesByTau, double x) {
    List<Map.Entry<Double, Double>> sorted = quantilesByTau.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .toList();
    if (sorted.isEmpty()) {
      return 0.0;
    }
    List<Double> taus = new ArrayList<>(sorted.size());
    List<Double> qvals = new ArrayList<>(sorted.size());
    double previous = Double.NEGATIVE_INFINITY;
    for (Map.Entry<Double, Double> entry : sorted) {
      double tau = entry.getKey();
      double q = entry.getValue();
      if (!Double.isFinite(tau) || !Double.isFinite(q)) {
        continue;
      }
      double monotonicQ = Math.max(previous, q);
      taus.add(tau);
      qvals.add(monotonicQ);
      previous = monotonicQ;
    }
    if (taus.isEmpty()) {
      return 0.0;
    }
    if (x < qvals.get(0)) {
      return 0.0;
    }
    int last = qvals.size() - 1;
    if (x > qvals.get(last)) {
      return 1.0;
    }
    if (taus.size() == 1) {
      return Math.max(0.0, Math.min(1.0, taus.get(0)));
    }
    for (int idx = 0; idx < qvals.size() - 1; idx += 1) {
      double qLo = qvals.get(idx);
      double qHi = qvals.get(idx + 1);
      double tLo = taus.get(idx);
      double tHi = taus.get(idx + 1);
      if (x > qHi) {
        continue;
      }
      if (x <= qLo) {
        return Math.max(0.0, Math.min(1.0, tLo));
      }
      if (qHi <= qLo) {
        return Math.max(0.0, Math.min(1.0, tHi));
      }
      double w = (x - qLo) / (qHi - qLo);
      double interp = tLo + (w * (tHi - tLo));
      return Math.max(0.0, Math.min(1.0, interp));
    }
    return Math.max(0.0, Math.min(1.0, taus.get(last)));
  }

  private BucketSpec parseBucketLabel(String label) {
    if (!StringUtils.hasText(label)) {
      return null;
    }
    String normalized = canonicalizeBucketLabel(label).trim().toLowerCase(Locale.ROOT).replace(" to ", "-");
    List<Integer> numbers = new ArrayList<>();
    Matcher matcher = NUMERIC_PATTERN.matcher(normalized);
    while (matcher.find()) {
      numbers.add(Integer.parseInt(matcher.group(1)));
    }
    if ((normalized.contains("or below") || normalized.contains("or less")) && !numbers.isEmpty()) {
      return new BucketSpec(null, numbers.get(0), "or_below");
    }
    if ((normalized.contains("or above") || normalized.contains("or higher")) && !numbers.isEmpty()) {
      return new BucketSpec(numbers.get(0), null, "or_above");
    }
    if (numbers.size() >= 2) {
      int lo = Math.min(numbers.get(0), numbers.get(1));
      int hi = Math.max(numbers.get(0), numbers.get(1));
      return new BucketSpec(lo, hi, "range");
    }
    return null;
  }

  private double bucketProbability(Map<Integer, Double> pmf, BucketSpec bucket) {
    if (pmf == null || pmf.isEmpty() || bucket == null) {
      return 0.0;
    }
    double total = 0.0;
    for (Map.Entry<Integer, Double> entry : pmf.entrySet()) {
      int temp = entry.getKey();
      double prob = entry.getValue();
      if (!Double.isFinite(prob) || prob <= 0.0) {
        continue;
      }
      if ("or_below".equals(bucket.mode()) && bucket.hi() != null && temp <= bucket.hi()) {
        total += prob;
      } else if ("or_above".equals(bucket.mode()) && bucket.lo() != null && temp >= bucket.lo()) {
        total += prob;
      } else if ("range".equals(bucket.mode())
          && bucket.lo() != null
          && bucket.hi() != null
          && temp >= bucket.lo()
          && temp <= bucket.hi()) {
        total += prob;
      }
    }
    return Math.max(0.0, Math.min(1.0, total));
  }

  private void maybeAddOpportunity(List<LiveOpportunityView> opportunities,
                                   String stationId,
                                   MarketDescriptor market,
                                   String side,
                                   Double modelWinProbability,
                                   Double marketPriceProbability,
                                   Integer entryPriceCents,
                                   Double ev) {
    if (modelWinProbability == null || marketPriceProbability == null || ev == null) {
      return;
    }
    if (!Double.isFinite(modelWinProbability) || !Double.isFinite(marketPriceProbability) || !Double.isFinite(ev)) {
      return;
    }
    if (modelWinProbability < properties.getOpportunitiesMinWinProbability()) {
      return;
    }
    if (ev < properties.getOpportunitiesMinEv()) {
      return;
    }
    if (marketPriceProbability < properties.getOpportunitiesMinSidePriceProbability()) {
      return;
    }
    opportunities.add(new LiveOpportunityView(
        stationId,
        market.marketTicker(),
        market.bucketLabel(),
        side,
        modelWinProbability,
        marketPriceProbability,
        entryPriceCents,
        ev));
  }

  private LiveStrategyConfigView strategyConfigView() {
    List<String> stationIds = properties.getStations().stream()
        .map(LiveTradingProperties.Station::getStationId)
        .filter(StringUtils::hasText)
        .map(this::normalizeStationId)
        .toList();
    return new LiveStrategyConfigView(
        properties.getStrategyReferenceLabel(),
        properties.getStrategyPeriodLabel(),
        stationIds,
        properties.getOpportunitiesMinWinProbability(),
        properties.getOpportunitiesMinEv(),
        properties.getOpportunitiesMinSidePriceProbability(),
        properties.getStrategySizingMode(),
        properties.getStrategyKellyFraction(),
        properties.getStrategyStakeCapUsd(),
        properties.getStrategyEntryRule(),
        properties.getStrategyPredictionSource());
  }

  private int compareOpportunities(LiveOpportunityView left, LiveOpportunityView right) {
    int byEv = Double.compare(
        right.ev() == null ? Double.NEGATIVE_INFINITY : right.ev(),
        left.ev() == null ? Double.NEGATIVE_INFINITY : left.ev());
    if (byEv != 0) {
      return byEv;
    }
    int byWin = Double.compare(
        right.modelWinProbability() == null ? Double.NEGATIVE_INFINITY : right.modelWinProbability(),
        left.modelWinProbability() == null ? Double.NEGATIVE_INFINITY : left.modelWinProbability());
    if (byWin != 0) {
      return byWin;
    }
    int leftEntry = left.entryPriceCents() == null ? Integer.MAX_VALUE : left.entryPriceCents();
    int rightEntry = right.entryPriceCents() == null ? Integer.MAX_VALUE : right.entryPriceCents();
    int byEntry = Integer.compare(leftEntry, rightEntry);
    if (byEntry != 0) {
      return byEntry;
    }
    int byStation = String.valueOf(left.stationId()).compareTo(String.valueOf(right.stationId()));
    if (byStation != 0) {
      return byStation;
    }
    int byBucket = String.valueOf(left.bucketLabel()).compareTo(String.valueOf(right.bucketLabel()));
    if (byBucket != 0) {
      return byBucket;
    }
    return String.valueOf(left.side()).compareTo(String.valueOf(right.side()));
  }

  private Double normalizeProbFromCents(Integer cents) {
    if (cents == null) {
      return null;
    }
    double prob = cents / 100.0;
    if (!Double.isFinite(prob) || prob < 0.0 || prob > 1.0) {
      return null;
    }
    return prob;
  }

  private Double evFromModelAndMarket(Double modelProb, Double marketProb) {
    if (modelProb == null || marketProb == null) {
      return null;
    }
    if (!Double.isFinite(modelProb) || !Double.isFinite(marketProb)) {
      return null;
    }
    return modelProb - marketProb;
  }

  private String textOrNull(JsonNode node) {
    if (node == null || node.isMissingNode() || node.isNull()) {
      return null;
    }
    String text = node.asText();
    return StringUtils.hasText(text) ? text.trim() : null;
  }

  private Double numberOrNull(JsonNode node) {
    if (node == null || node.isMissingNode() || node.isNull()) {
      return null;
    }
    double value = node.asDouble(Double.NaN);
    return Double.isFinite(value) ? value : null;
  }

  private long lastModifiedMillisSafe(Path path) {
    try {
      return Files.getLastModifiedTime(path).toMillis();
    } catch (IOException ex) {
      return Long.MIN_VALUE;
    }
  }

  private Integer spread(Integer bid, Integer ask) {
    if (bid == null || ask == null) {
      return null;
    }
    return ask - bid;
  }

  private Integer midpoint(Integer bid, Integer ask) {
    if (bid == null || ask == null) {
      return null;
    }
    return Math.round((bid + ask) / 2.0f);
  }

  private List<OrderBookLevelView> topLevels(List<OrderBookLevelView> levels) {
    if (levels == null || levels.isEmpty()) {
      return List.of();
    }
    int limit = Math.max(1, properties.getTopLevelsPerSide());
    return levels.stream().limit(limit).toList();
  }

  private String normalizeStationId(String stationId) {
    if (!StringUtils.hasText(stationId)) {
      return null;
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private String normalizeSeries(String series) {
    if (!StringUtils.hasText(series)) {
      return null;
    }
    return series.trim().toUpperCase(Locale.ROOT);
  }

  private LocalTime parseCutoffTime(String raw) {
    if (!StringUtils.hasText(raw)) {
      return LocalTime.of(17, 45);
    }
    try {
      return LocalTime.parse(raw.trim());
    } catch (DateTimeParseException ex) {
      log.warn("Invalid kalshi.live-trading.next-target-date-cutoff-local-time '{}', using 17:45", raw);
      return LocalTime.of(17, 45);
    }
  }

  public void shutdown() {
    scheduler.shutdownNow();
    if (!started.compareAndSet(true, false)) {
      return;
    }
    try {
      orderBookService.disconnect();
    } catch (Exception ex) {
      log.warn("Failed to disconnect orderbook service cleanly: {}", ex.toString());
    }
  }

  @Override
  public void destroy() {
    shutdown();
  }

  private record EventCandidate(
      String eventTicker,
      LocalDate titleDate,
      List<Market> markets
  ) {
  }

  private record MarketDescriptor(
      String marketTicker,
      String bucketLabel,
      String marketStatus,
      String sortKey
  ) {
  }

  private record StationRuntimeState(
      String stationId,
      String displayName,
      String seriesTicker,
      String zoneId,
      LocalDate targetDateLocal,
      String eventTicker,
      Instant resolvedAtUtc,
      List<MarketDescriptor> markets
  ) {
  }

  private record BucketSpec(
      Integer lo,
      Integer hi,
      String mode
  ) {
  }

  private record StationInference(
      String stationId,
      String targetDateLocal,
      String runtimeUtc,
      Double predictionPointTmaxF,
      Map<String, Double> quantilesByLabel,
      Map<Double, Double> quantilesByTau
  ) {
  }

  private record StationInferenceVersioned(
      StationInference stationInference,
      FileTime reportModifiedTime
  ) {
  }

  private record InferenceIndex(
      Map<String, Long> sourceSignature,
      Map<String, Map<String, StationInference>> byTargetDate
  ) {
    private static InferenceIndex empty() {
      return new InferenceIndex(Map.of(), Map.of());
    }

    private boolean matches(Map<String, Long> signature) {
      return Objects.equals(sourceSignature, signature);
    }
  }
}
