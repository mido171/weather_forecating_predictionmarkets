package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.cli.CliDailyIngestService;
import com.predictionmarkets.weather.config.CliSettlementIngestionProperties;
import com.predictionmarkets.weather.config.PipelineProperties;
import com.predictionmarkets.weather.kalshi.KalshiSeriesResolver;
import com.predictionmarkets.weather.models.AsofTimeZone;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.mos.MosRunIngestService;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

public final class FullIngestionPipelineExecutor {
  private static final Logger logger = LoggerFactory.getLogger(FullIngestionPipelineExecutor.class);

  private FullIngestionPipelineExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      KalshiSeriesResolver kalshiSeriesResolver = context.getBean(KalshiSeriesResolver.class);
      StationRegistryRepository stationRegistryRepository =
          context.getBean(StationRegistryRepository.class);
      CliDailyIngestService cliDailyIngestService =
          context.getBean(CliDailyIngestService.class);
      CliSettlementIngestionProperties cliSettlementProperties =
          context.getBean(CliSettlementIngestionProperties.class);
      MosRunIngestService mosRunIngestService =
          context.getBean(MosRunIngestService.class);
      IngestCheckpointRepository checkpointRepository =
          context.getBean(IngestCheckpointRepository.class);
      PipelineProperties config = context.getBean(PipelineProperties.class);
      validateConfig(config);
      CliSettlementConfig cliSettlementConfig = resolveCliSettlementConfig(cliSettlementProperties);
      List<MosModel> mosModels = resolveMosModels(config.getModels());

      List<String> stationIdsToRun = parseStationIds(config.getStationIdsToRun());
      List<String> seriesTickers =
          resolveSeriesTickers(config.getSeriesTickers(), stationIdsToRun, stationRegistryRepository);
      snapshot("Full ingestion starting seriesTickers=" + seriesTickers
          + " stationIdsToRun=" + (stationIdsToRun.isEmpty() ? "[ALL]" : stationIdsToRun)
          + " mosModels=" + mosModels
          + " asofLocalTime=" + config.getAsofLocalTime()
          + " asofTimeZone=" + resolveAsofTimeZone(config.getAsofTimeZone())
          + " threadCount=" + config.getThreadCount());

      if (config.isResetCheckpoints()) {
        snapshot("Resetting ingest checkpoints");
        checkpointRepository.deleteAll();
      }

      ExecutorService executor = Executors.newFixedThreadPool(
          config.getThreadCount(), namedThreadFactory());
      try {
        runJobs(executor, "Step 1/3: Kalshi series sync",
            buildKalshiJobs(seriesTickers, kalshiSeriesResolver));

        List<StationContext> stations = resolveStationContexts(
            seriesTickers, stationRegistryRepository, kalshiSeriesResolver, config);

        runJobs(executor, "Step 2/3: CLI ingest",
            buildCliJobs(stations, cliDailyIngestService, cliSettlementConfig));

        runJobs(executor, "Step 3/3: MOS ingest",
            buildMosIngestJobs(stations, mosModels, config.getAsofLocalTime(),
                resolveAsofTimeZone(config.getAsofTimeZone()), mosRunIngestService));

        snapshot("Full ingestion pipeline complete.");
      } finally {
        shutdownExecutor(executor);
      }
    }
  }

  private static DateRange resolveDateRange(PipelineProperties config, StationRegistry station) {
    LocalDate start = config.getDateStartLocal();
    LocalDate end = config.getDateEndLocal();
    if (start != null && end != null) {
      validateDateRange(start, end);
      return new DateRange(start, end);
    }
    ZoneId zoneId = ZoneId.of(station.getZoneId());
    LocalDate defaultEnd = LocalDate.now(zoneId).minusDays(1);
    LocalDate defaultStart = defaultEnd.minusDays(config.getDefaultRangeDays() - 1L);
    LocalDate resolvedStart = start != null ? start : defaultStart;
    LocalDate resolvedEnd = end != null ? end : defaultEnd;
    validateDateRange(resolvedStart, resolvedEnd);
    return new DateRange(resolvedStart, resolvedEnd);
  }

  private static void validateDateRange(LocalDate start, LocalDate end) {
    if (start == null || end == null) {
      throw new IllegalArgumentException("date range must include start and end");
    }
    if (end.isBefore(start)) {
      throw new IllegalArgumentException("dateEndLocal must be >= dateStartLocal");
    }
  }

  private static String normalizeSeriesTicker(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      throw new IllegalArgumentException("seriesTicker is required");
    }
    return seriesTicker.trim().toUpperCase(Locale.ROOT);
  }

  private static List<String> resolveSeriesTickers(List<String> configuredTickers,
                                                   List<String> stationIdsToRun,
                                                   StationRegistryRepository stationRegistryRepository) {
    if (configuredTickers == null || configuredTickers.isEmpty()) {
      throw new IllegalArgumentException("pipeline.series-tickers is required");
    }
    if (stationIdsToRun == null || stationIdsToRun.isEmpty()) {
      return configuredTickers;
    }
    List<String> normalizedConfigured = new ArrayList<>(configuredTickers.size());
    for (String ticker : configuredTickers) {
      normalizedConfigured.add(normalizeSeriesTicker(ticker));
    }
    Set<String> configuredSet = new LinkedHashSet<>(normalizedConfigured);
    Map<String, String> stationToTicker = new HashMap<>();
    for (String stationId : stationIdsToRun) {
      StationRegistry station = stationRegistryRepository.findById(stationId)
          .orElseThrow(() -> new IllegalArgumentException(
              "pipeline.station-ids-to-run includes stationId=" + stationId
                  + " but station_registry is missing"));
      stationToTicker.put(stationId, normalizeSeriesTicker(station.getSeriesTicker()));
    }
    List<String> resolved = new ArrayList<>(stationIdsToRun.size());
    for (String stationId : stationIdsToRun) {
      String seriesTicker = stationToTicker.get(stationId);
      if (!configuredSet.contains(seriesTicker)) {
        throw new IllegalArgumentException(
            "pipeline.station-ids-to-run includes stationId=" + stationId
                + " but seriesTicker=" + seriesTicker
                + " is not listed in pipeline.series-tickers");
      }
      resolved.add(seriesTicker);
    }
    return resolved;
  }

  private static List<String> parseStationIds(String stationIdsToRun) {
    if (stationIdsToRun == null || stationIdsToRun.isBlank()) {
      return List.of();
    }
    Set<String> stationIds = new LinkedHashSet<>();
    for (String token : Arrays.asList(stationIdsToRun.split(","))) {
      String trimmed = token.trim();
      if (!trimmed.isEmpty()) {
        stationIds.add(trimmed.toUpperCase(Locale.ROOT));
      }
    }
    return new ArrayList<>(stationIds);
  }

  private static CliSettlementConfig resolveCliSettlementConfig(
      CliSettlementIngestionProperties properties) {
    if (properties == null) {
      throw new IllegalArgumentException("cli-settlement config is required");
    }
    List<String> stationIds = normalizeCliStationIds(properties.getStationIds());
    LocalDate start = requireCliStartDate(properties.getStartDateLocal());
    LocalDate end = requireCliEndDate(properties.getEndDateLocal());
    validateCliDateRange(start, end);
    return new CliSettlementConfig(stationIds, start, end);
  }

  private static List<String> normalizeCliStationIds(List<String> stationIds) {
    if (stationIds == null || stationIds.isEmpty()) {
      throw new IllegalArgumentException("cli-settlement.station-ids is required");
    }
    Set<String> normalized = new LinkedHashSet<>();
    for (String stationId : stationIds) {
      if (stationId == null || stationId.isBlank()) {
        continue;
      }
      normalized.add(stationId.trim().toUpperCase(Locale.ROOT));
    }
    if (normalized.isEmpty()) {
      throw new IllegalArgumentException("cli-settlement.station-ids is required");
    }
    return new ArrayList<>(normalized);
  }

  private static LocalDate requireCliStartDate(LocalDate start) {
    if (start == null) {
      throw new IllegalArgumentException("cli-settlement.start-date-local is required");
    }
    return start;
  }

  private static LocalDate requireCliEndDate(LocalDate end) {
    if (end == null) {
      throw new IllegalArgumentException("cli-settlement.end-date-local is required");
    }
    return end;
  }

  private static void validateCliDateRange(LocalDate start, LocalDate end) {
    if (end.isBefore(start)) {
      throw new IllegalArgumentException("cli-settlement date range must be start <= end");
    }
  }

  private static void validateConfig(PipelineProperties config) {
    if (config == null) {
      throw new IllegalArgumentException("pipeline config is required");
    }
    if (config.getSeriesTickers() == null || config.getSeriesTickers().isEmpty()) {
      throw new IllegalArgumentException("pipeline.series-tickers is required");
    }
    if (config.getModels() == null || config.getModels().isEmpty()) {
      throw new IllegalArgumentException("pipeline.models is required");
    }
    if (config.getMosWindowDays() < 1) {
      throw new IllegalArgumentException("pipeline.mos-window-days must be >= 1");
    }
    if (config.getThreadCount() < 1) {
      throw new IllegalArgumentException("pipeline.thread-count must be >= 1");
    }
    if (config.getDefaultRangeDays() < 1) {
      throw new IllegalArgumentException("pipeline.default-range-days must be >= 1");
    }
    LocalDate rangeStart = config.getDateStartLocal();
    LocalDate rangeEnd = config.getDateEndLocal();
    if (rangeStart == null || rangeEnd == null) {
      throw new IllegalArgumentException(
          "pipeline.date-start-local and pipeline.date-end-local are required");
    }
    LocalDate expectedStart = LocalDate.of(2007, 1, 1);
    LocalDate expectedEnd = LocalDate.of(2025, 12, 31);
    if (!expectedStart.equals(rangeStart) || !expectedEnd.equals(rangeEnd)) {
      if (!config.isAllowNonstandardRange()) {
        throw new IllegalArgumentException(
            "pipeline date range must be " + expectedStart + ".." + expectedEnd);
      }
      snapshot("Using non-standard pipeline date range " + rangeStart + ".." + rangeEnd);
    }
    LocalTime asofLocalTime = config.getAsofLocalTime();
    if (asofLocalTime == null) {
      throw new IllegalArgumentException("pipeline.asof-local-time is required");
    }
    if (!LocalTime.of(12, 0).equals(asofLocalTime)) {
      throw new IllegalArgumentException(
          "pipeline.asof-local-time must be 12:00 for T-1 12Z as-of");
    }
    AsofTimeZone asofTimeZone = resolveAsofTimeZone(config.getAsofTimeZone());
    if (asofTimeZone != AsofTimeZone.UTC) {
      throw new IllegalArgumentException(
          "pipeline.asof-time-zone must be UTC for T-1 12Z as-of");
    }
  }

  private static AsofTimeZone resolveAsofTimeZone(AsofTimeZone configured) {
    return configured == null ? AsofTimeZone.LOCAL : configured;
  }

  private record DateRange(LocalDate start, LocalDate end) {
  }

  private record StationContext(StationRegistry station,
                                String seriesTicker,
                                DateRange dateRange) {
  }

  private record CliSettlementConfig(List<String> stationIds,
                                     LocalDate startDate,
                                     LocalDate endDate) {
  }

  private record Job(String name, Runnable action) {
  }

  private static void snapshot(String message) {
    String payload = "[FULL-INGESTION] thread=" + Thread.currentThread().getName()
        + " " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private static List<Job> buildKalshiJobs(List<String> seriesTickers,
                                           KalshiSeriesResolver resolver) {
    List<Job> jobs = new ArrayList<>(seriesTickers.size());
    for (String seriesTicker : seriesTickers) {
      String normalized = normalizeSeriesTicker(seriesTicker);
      jobs.add(new Job("kalshi_series_sync ticker=" + normalized,
          () -> resolver.resolveAndUpsert(normalized)));
    }
    return jobs;
  }

  private static List<StationContext> resolveStationContexts(
      List<String> seriesTickers,
      StationRegistryRepository stationRegistryRepository,
      KalshiSeriesResolver kalshiSeriesResolver,
      PipelineProperties config) {
    List<StationContext> contexts = new ArrayList<>(seriesTickers.size());
    for (String seriesTicker : seriesTickers) {
      String normalized = normalizeSeriesTicker(seriesTicker);
      StationRegistry station = stationRegistryRepository.findBySeriesTicker(normalized)
          .orElseGet(() -> kalshiSeriesResolver.resolveAndUpsert(normalized));
      DateRange range = resolveDateRange(config, station);
      contexts.add(new StationContext(station, normalized, range));
    }
    return contexts;
  }

  private static List<Job> buildCliJobs(List<StationContext> stations,
                                        CliDailyIngestService cliDailyIngestService,
                                        CliSettlementConfig cliSettlementConfig) {
    List<Job> jobs = new ArrayList<>();
    boolean matchedStation = false;
    Set<String> cliStationIds = new LinkedHashSet<>(cliSettlementConfig.stationIds());
    for (StationContext station : stations) {
      String stationId = station.station().getStationId();
      if (stationId == null || !cliStationIds.contains(stationId.toUpperCase(Locale.ROOT))) {
        continue;
      }
      matchedStation = true;
      LocalDate rangeStart = cliSettlementConfig.startDate();
      LocalDate rangeEnd = cliSettlementConfig.endDate();
      List<DateRange> slices = buildYearSlices(rangeStart, rangeEnd);
      for (DateRange slice : slices) {
        String name = "cli_ingest_range station=" + station.station().getStationId()
            + " slice=" + slice.start() + ".." + slice.end();
        jobs.add(new Job(name, () ->
            cliDailyIngestService.ingestRange(
                station.station().getStationId(), slice.start(), slice.end())));
      }
    }
    if (!matchedStation) {
      throw new IllegalArgumentException(
          "CLI settlement ingest requires stationId in " + cliStationIds
              + " but they were not configured");
    }
    return jobs;
  }

  private static List<Job> buildMosIngestJobs(List<StationContext> stations,
                                              List<MosModel> mosModels,
                                              LocalTime asofLocalTime,
                                              AsofTimeZone asofTimeZone,
                                              MosRunIngestService mosRunIngestService) {
    List<Job> jobs = new ArrayList<>();
    for (StationContext station : stations) {
      DateRange range = station.dateRange();
      List<LocalDate> targets = buildTargetDates(range.start(), range.end());
      for (MosModel model : mosModels) {
        for (LocalDate targetDate : targets) {
          String name = "mos_ingest_window station=" + station.station().getStationId()
              + " model=" + model.name()
              + " targetDate=" + targetDate;
          jobs.add(new Job(name, () ->
              mosRunIngestService.ingestTargetDateAsOf(
                  station.station().getStationId(),
                  model,
                  targetDate,
                  asofLocalTime,
                  asofTimeZone)));
        }
      }
    }
    return jobs;
  }

  private static List<DateRange> buildYearSlices(LocalDate start, LocalDate end) {
    List<DateRange> slices = new ArrayList<>();
    for (int year = start.getYear(); year <= end.getYear(); year++) {
      LocalDate yearStart = LocalDate.of(year, 1, 1);
      LocalDate yearEnd = LocalDate.of(year, 12, 31);
      LocalDate sliceStart = yearStart.isAfter(start) ? yearStart : start;
      LocalDate sliceEnd = yearEnd.isBefore(end) ? yearEnd : end;
      slices.add(new DateRange(sliceStart, sliceEnd));
    }
    return slices;
  }

  private static List<LocalDate> buildTargetDates(LocalDate start, LocalDate end) {
    List<LocalDate> dates = new ArrayList<>();
    LocalDate cursor = start;
    while (!cursor.isAfter(end)) {
      dates.add(cursor);
      cursor = cursor.plusDays(1);
    }
    return dates;
  }

  private static void runJobs(ExecutorService executor, String stepLabel, List<Job> jobs) {
    if (jobs.isEmpty()) {
      snapshot(stepLabel + " skipped (no jobs)");
      return;
    }
    snapshot(stepLabel + " starting jobs=" + jobs.size());
    AtomicInteger failures = new AtomicInteger();
    List<Future<?>> futures = new ArrayList<>(jobs.size());
    for (Job job : jobs) {
      futures.add(executor.submit(() -> {
        snapshot("job=start " + job.name());
        try {
          job.action().run();
          snapshot("job=complete " + job.name());
        } catch (RuntimeException ex) {
          failures.incrementAndGet();
          snapshot("job=failed " + job.name());
          logger.error("[FULL-INGESTION] job failed {}", job.name(), ex);
        }
        return null;
      }));
    }
    waitAllOrFail(futures, stepLabel);
    if (failures.get() > 0) {
      snapshot(stepLabel + " complete jobs=" + jobs.size() + " failures=" + failures.get());
      return;
    }
    snapshot(stepLabel + " complete jobs=" + jobs.size());
  }

  private static ThreadFactory namedThreadFactory() {
    AtomicInteger counter = new AtomicInteger(1);
    return runnable -> {
      Thread thread = new Thread(runnable);
      thread.setName("full-ingest-" + counter.getAndIncrement());
      thread.setDaemon(false);
      return thread;
    };
  }

  private static void cancelRemaining(List<Future<?>> futures) {
    for (Future<?> future : futures) {
      if (!future.isDone()) {
        future.cancel(true);
      }
    }
  }

  private static void waitAllOrFail(List<Future<?>> futures, String label) {
    RuntimeException failure = null;
    for (Future<?> future : futures) {
      try {
        future.get();
      } catch (InterruptedException ex) {
        Thread.currentThread().interrupt();
        if (failure == null) {
          failure = new IllegalStateException(label + " interrupted", ex);
        }
      } catch (ExecutionException ex) {
        Throwable cause = ex.getCause() == null ? ex : ex.getCause();
        if (cause instanceof RuntimeException runtimeException) {
          if (failure == null) {
            failure = runtimeException;
          }
        } else {
          if (failure == null) {
            failure = new IllegalStateException(label + " failed", cause);
          }
        }
      }
    }
    if (failure != null) {
      logger.error("[FULL-INGESTION] step failed (continuing). step={}", label, failure);
    }
  }

  private static void shutdownExecutor(ExecutorService executor) {
    executor.shutdown();
    try {
      if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
        executor.shutdownNow();
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      executor.shutdownNow();
    }
  }

  private static List<MosModel> resolveMosModels(List<MosModel> configuredModels) {
    if (configuredModels == null || configuredModels.isEmpty()) {
      throw new IllegalArgumentException("pipeline.models is required");
    }
    Set<MosModel> filtered = new LinkedHashSet<>();
    Set<MosModel> ignored = EnumSet.noneOf(MosModel.class);
    for (MosModel model : configuredModels) {
      if (model == null) {
        throw new IllegalArgumentException("pipeline.models includes null");
      }
      if (model == MosModel.GFS || model == MosModel.NAM) {
        filtered.add(model);
      } else {
        ignored.add(model);
      }
    }
    if (!ignored.isEmpty()) {
      snapshot("Ignoring non-GFS/NAM MOS models for full ingestion: " + ignored);
    }
    if (!filtered.contains(MosModel.GFS) || !filtered.contains(MosModel.NAM)) {
      throw new IllegalArgumentException("pipeline.models must include GFS and NAM");
    }
    return List.copyOf(filtered);
  }
}
