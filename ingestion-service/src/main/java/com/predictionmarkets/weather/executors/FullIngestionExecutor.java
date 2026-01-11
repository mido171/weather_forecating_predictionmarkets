package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.backfill.BackfillJobType;
import com.predictionmarkets.weather.backfill.BackfillOrchestrator;
import com.predictionmarkets.weather.backfill.BackfillRequest;
import com.predictionmarkets.weather.models.AsofPolicy;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.AsofPolicyRepository;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
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

public final class FullIngestionExecutor {
  private static final Logger logger = LoggerFactory.getLogger(FullIngestionExecutor.class);

  private FullIngestionExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      BackfillOrchestrator orchestrator = context.getBean(BackfillOrchestrator.class);
      StationRegistryRepository stationRegistryRepository =
          context.getBean(StationRegistryRepository.class);
      AsofPolicyRepository asofPolicyRepository = context.getBean(AsofPolicyRepository.class);
      IngestCheckpointRepository checkpointRepository =
          context.getBean(IngestCheckpointRepository.class);
      PipelineProperties config = context.getBean(PipelineProperties.class);
      validateConfig(config);

      List<String> seriesTickers = config.getSeriesTickers();
      if (seriesTickers.isEmpty()) {
        throw new IllegalArgumentException("seriesTickers is required");
      }
      snapshot("Full ingestion starting seriesTickers=" + seriesTickers
          + " models=" + config.getModels()
          + " mosWindowDays=" + config.getMosWindowDays()
          + " threadCount=" + config.getThreadCount());

      if (config.isResetCheckpoints()) {
        snapshot("Resetting ingest checkpoints");
        checkpointRepository.deleteAll();
      }

      snapshot("Step 1/4: Kalshi series sync starting");
      orchestrator.run(new BackfillRequest(
          BackfillJobType.KALSHI_SERIES_SYNC,
          seriesTickers,
          null,
          null,
          null,
          null,
          0));
      snapshot("Step 1/4: Kalshi series sync complete");

      Long asofPolicyId = resolveAsofPolicyId(config, asofPolicyRepository);
      snapshot("Using asofPolicyId=" + asofPolicyId
          + " asofPolicyName=" + config.getAsofPolicyName()
          + " asofLocalTime=" + config.getAsofLocalTime());
      runStationsInParallel(seriesTickers, stationRegistryRepository, orchestrator,
          config, asofPolicyId);
      snapshot("Full ingestion pipeline complete.");
    }
  }

  private static Long resolveAsofPolicyId(PipelineProperties config,
                                          AsofPolicyRepository repository) {
    if (config.getAsofPolicyId() != null) {
      return config.getAsofPolicyId();
    }
    String policyName = config.getAsofPolicyName();
    if (policyName == null || policyName.isBlank()) {
      throw new IllegalArgumentException("pipeline.asof-policy-name is required");
    }
    LocalTime asofLocalTime = config.getAsofLocalTime();
    if (asofLocalTime == null) {
      throw new IllegalArgumentException("pipeline.asof-local-time is required");
    }
    Optional<AsofPolicy> existing = repository.findByName(policyName);
    if (existing.isPresent()) {
      return existing.get().getId();
    }
    AsofPolicy policy = new AsofPolicy();
    policy.setName(policyName);
    policy.setAsofLocalTime(asofLocalTime);
    policy.setEnabled(true);
    AsofPolicy saved = repository.save(policy);
    snapshot("Created asof_policy name=" + policyName + " id=" + saved.getId());
    return saved.getId();
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
  }

  private record DateRange(LocalDate start, LocalDate end) {
  }

  private static void snapshot(String message) {
    String payload = "[FULL-INGESTION] thread=" + Thread.currentThread().getName()
        + " " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private static void runStationsInParallel(List<String> seriesTickers,
                                            StationRegistryRepository stationRegistryRepository,
                                            BackfillOrchestrator orchestrator,
                                            PipelineProperties config,
                                            Long asofPolicyId) {
    int threadCount = config.getThreadCount();
    if (threadCount == 1) {
      for (String seriesTicker : seriesTickers) {
        runStationPipeline(seriesTicker, stationRegistryRepository, orchestrator, config, asofPolicyId);
      }
      return;
    }
    ExecutorService executor = Executors.newFixedThreadPool(threadCount, namedThreadFactory());
    List<Future<?>> futures = new ArrayList<>();
    try {
      for (String seriesTicker : seriesTickers) {
        futures.add(executor.submit(() -> {
          runStationPipeline(seriesTicker, stationRegistryRepository, orchestrator, config, asofPolicyId);
          return null;
        }));
      }
      for (Future<?> future : futures) {
        try {
          future.get();
        } catch (InterruptedException ex) {
          Thread.currentThread().interrupt();
          throw new IllegalStateException("Full ingestion interrupted", ex);
        } catch (ExecutionException ex) {
          cancelRemaining(futures);
          Throwable cause = ex.getCause() == null ? ex : ex.getCause();
          if (cause instanceof RuntimeException runtimeException) {
            throw runtimeException;
          }
          throw new IllegalStateException("Full ingestion failed", cause);
        }
      }
    } finally {
      shutdownExecutor(executor);
    }
  }

  private static void runStationPipeline(String seriesTicker,
                                         StationRegistryRepository stationRegistryRepository,
                                         BackfillOrchestrator orchestrator,
                                         PipelineProperties config,
                                         Long asofPolicyId) {
    String normalizedTicker = normalizeSeriesTicker(seriesTicker);
    StationRegistry station = stationRegistryRepository.findBySeriesTicker(normalizedTicker)
        .orElseThrow(() -> new IllegalArgumentException(
            "Station not found for series ticker: " + normalizedTicker));

    DateRange dateRange = resolveDateRange(config, station);
    LocalDate start = dateRange.start();
    LocalDate end = dateRange.end();
    snapshot("Station " + station.getStationId()
        + " seriesTicker=" + normalizedTicker
        + " dateRange=" + start + ".." + end
        + " zoneId=" + station.getZoneId());

    snapshot("Step 2/4: CLI ingest starting station=" + station.getStationId());
    orchestrator.run(new BackfillRequest(
        BackfillJobType.CLI_INGEST_YEAR,
        List.of(normalizedTicker),
        start,
        end,
        null,
        null,
        0));
    snapshot("Step 2/4: CLI ingest complete station=" + station.getStationId());

    snapshot("Step 3/4: MOS ingest starting station=" + station.getStationId()
        + " models=" + config.getModels()
        + " windowDays=" + config.getMosWindowDays());
    orchestrator.run(new BackfillRequest(
        BackfillJobType.MOS_INGEST_WINDOW,
        List.of(normalizedTicker),
        start,
        end,
        null,
        config.getModels(),
        config.getMosWindowDays()));
    snapshot("Step 3/4: MOS ingest complete station=" + station.getStationId());

    snapshot("Step 4/4: MOS as-of materialize starting station=" + station.getStationId()
        + " asofPolicyId=" + asofPolicyId);
    orchestrator.run(new BackfillRequest(
        BackfillJobType.MOS_ASOF_MATERIALIZE_RANGE,
        List.of(normalizedTicker),
        start,
        end,
        asofPolicyId,
        config.getModels(),
        0));
    snapshot("Step 4/4: MOS as-of materialize complete station=" + station.getStationId());
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
}
