package com.predictionmarkets.weather.backfill;

import com.predictionmarkets.weather.cli.CliDailyIngestService;
import com.predictionmarkets.weather.kalshi.KalshiSeriesResolver;
import com.predictionmarkets.weather.models.IngestCheckpoint;
import com.predictionmarkets.weather.models.IngestCheckpointStatus;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.mos.MosAsofMaterializeService;
import com.predictionmarkets.weather.mos.MosAsofFeatureReportService;
import com.predictionmarkets.weather.mos.MosRunIngestService;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class BackfillOrchestrator {
  private static final Logger logger = LoggerFactory.getLogger(BackfillOrchestrator.class);
  private final KalshiSeriesResolver kalshiSeriesResolver;
  private final StationRegistryRepository stationRegistryRepository;
  private final CliDailyIngestService cliDailyIngestService;
  private final MosRunIngestService mosRunIngestService;
  private final MosAsofMaterializeService mosAsofMaterializeService;
  private final MosAsofFeatureReportService mosAsofFeatureReportService;
  private final BackfillConcurrencyProperties concurrencyProperties;
  private final IngestCheckpointService checkpointService;

  public BackfillOrchestrator(KalshiSeriesResolver kalshiSeriesResolver,
                              StationRegistryRepository stationRegistryRepository,
                              CliDailyIngestService cliDailyIngestService,
                              MosRunIngestService mosRunIngestService,
                              MosAsofMaterializeService mosAsofMaterializeService,
                              MosAsofFeatureReportService mosAsofFeatureReportService,
                              BackfillConcurrencyProperties concurrencyProperties,
                              IngestCheckpointService checkpointService) {
    this.kalshiSeriesResolver = kalshiSeriesResolver;
    this.stationRegistryRepository = stationRegistryRepository;
    this.cliDailyIngestService = cliDailyIngestService;
    this.mosRunIngestService = mosRunIngestService;
    this.mosAsofMaterializeService = mosAsofMaterializeService;
    this.mosAsofFeatureReportService = mosAsofFeatureReportService;
    this.concurrencyProperties = concurrencyProperties;
    this.checkpointService = checkpointService;
  }

  public void run(BackfillRequest request) {
    Objects.requireNonNull(request, "request is required");
    BackfillJobType jobType = Objects.requireNonNull(request.jobType(), "jobType is required");
    switch (jobType) {
      case KALSHI_SERIES_SYNC -> runKalshiSeriesSync(request.seriesTickers());
      case CLI_INGEST_YEAR -> runCliIngestYear(request);
      case MOS_INGEST_WINDOW -> runMosIngestWindow(request);
      case MOS_ASOF_MATERIALIZE_RANGE -> runMosAsofMaterializeRange(request);
      default -> throw new IllegalArgumentException("Unsupported job: " + jobType);
    }
  }

  private void runKalshiSeriesSync(List<String> seriesTickers) {
    if (seriesTickers == null || seriesTickers.isEmpty()) {
      throw new IllegalArgumentException("seriesTicker is required for kalshi_series_sync");
    }
    int total = seriesTickers.size();
    int index = 0;
    for (String seriesTicker : seriesTickers) {
      int current = ++index;
      String normalized = normalizeSeriesTicker(seriesTicker);
      String stationId = null;
      try {
        StationRegistry station = stationRegistryRepository.findBySeriesTicker(normalized)
            .orElse(null);
        if (station != null) {
          IngestCheckpoint existing = checkpointService
              .findCheckpoint(BackfillJobType.KALSHI_SERIES_SYNC.jobName(),
                  station.getStationId(), null)
              .orElse(null);
          if (existing != null && existing.getStatus() == IngestCheckpointStatus.COMPLETE) {
            snapshot("job=kalshi_series_sync progress=" + current + "/" + total
                + " remaining=" + (total - current)
                + " ticker=" + normalized
                + " station=" + station.getStationId()
                + " status=skipped");
            continue;
          }
        }
        station = kalshiSeriesResolver.resolveAndUpsert(normalized);
        stationId = station.getStationId();
        checkpointService.markRunning(BackfillJobType.KALSHI_SERIES_SYNC.jobName(), stationId, null,
            null, null);
        checkpointService.markComplete(BackfillJobType.KALSHI_SERIES_SYNC.jobName(), stationId, null,
            null, null);
        snapshot("job=kalshi_series_sync progress=" + current + "/" + total
            + " remaining=" + (total - current)
            + " ticker=" + normalized
            + " station=" + stationId
            + " status=complete");
      } catch (RuntimeException ex) {
        StationRegistry existingStation = stationRegistryRepository.findBySeriesTicker(normalized)
            .orElse(null);
        if (existingStation != null) {
          stationId = existingStation.getStationId();
          checkpointService.markFailed(BackfillJobType.KALSHI_SERIES_SYNC.jobName(),
              existingStation.getStationId(), null, null, null, ex);
        }
        if (stationId != null) {
          snapshot("job=kalshi_series_sync progress=" + current + "/" + total
              + " remaining=" + (total - current)
              + " ticker=" + normalized
              + " station=" + stationId
              + " status=failed");
        }
        throw ex;
      }
    }
  }

  private void runCliIngestYear(BackfillRequest request) {
    LocalDate start = requireStartDate(request.dateStartLocal(), BackfillJobType.CLI_INGEST_YEAR);
    LocalDate end = requireEndDate(request.dateEndLocal(), BackfillJobType.CLI_INGEST_YEAR);
    validateDateRange(start, end);
    StationRegistry station = resolveStation(singleSeriesTicker(request.seriesTickers(),
        BackfillJobType.CLI_INGEST_YEAR));
    String jobName = BackfillJobType.CLI_INGEST_YEAR.jobName();
    String stationId = station.getStationId();

    IngestCheckpoint existing = checkpointService.findCheckpoint(jobName, stationId, null)
        .orElse(null);
    LocalDate cursorDate = existing != null ? existing.getCursorDate() : null;
    if (cursorDate != null && !cursorDate.isBefore(end)) {
      snapshot("job=cli_ingest_year station=" + stationId
          + " progress=complete"
          + " cursorDate=" + cursorDate);
      checkpointService.markComplete(jobName, stationId, null, cursorDate, null);
      return;
    }
    LocalDate effectiveStart = start;
    if (cursorDate != null && !cursorDate.isBefore(start)) {
      effectiveStart = cursorDate.plusDays(1);
    }
    if (effectiveStart.isAfter(end)) {
      snapshot("job=cli_ingest_year station=" + stationId
          + " progress=complete"
          + " cursorDate=" + cursorDate);
      checkpointService.markComplete(jobName, stationId, null, cursorDate, null);
      return;
    }
    long totalDays = ChronoUnit.DAYS.between(effectiveStart, end) + 1;
    long processedDays = 0;
    AtomicReference<LocalDate> cursorRef = new AtomicReference<>(cursorDate);
    CheckpointHeartbeat heartbeat = CheckpointHeartbeat.start(() ->
        checkpointService.markRunning(jobName, stationId, null, cursorRef.get(), null));
    try {
      checkpointService.markRunning(jobName, stationId, null, cursorRef.get(), null);
      for (int year = effectiveStart.getYear(); year <= end.getYear(); year++) {
        LocalDate yearStart = LocalDate.of(year, 1, 1);
        LocalDate yearEnd = LocalDate.of(year, 12, 31);
        LocalDate rangeStart = yearStart.isAfter(effectiveStart) ? yearStart : effectiveStart;
        LocalDate rangeEnd = yearEnd.isBefore(end) ? yearEnd : end;
        cliDailyIngestService.ingestRange(stationId, rangeStart, rangeEnd);
        long daysInSlice = ChronoUnit.DAYS.between(rangeStart, rangeEnd) + 1;
        processedDays += daysInSlice;
        cursorDate = rangeEnd;
        cursorRef.set(cursorDate);
        checkpointService.markRunning(jobName, stationId, null, cursorDate, null);
        snapshot("job=cli_ingest_year station=" + stationId
            + " slice=" + rangeStart + ".." + rangeEnd
            + " progress=" + processedDays + "/" + totalDays
            + " remaining=" + (totalDays - processedDays));
      }
      cursorDate = end;
      cursorRef.set(cursorDate);
      heartbeat.close();
      checkpointService.markComplete(jobName, stationId, null, cursorDate, null);
    } catch (RuntimeException ex) {
      heartbeat.close();
      checkpointService.markFailed(jobName, stationId, null, cursorDate, null, ex);
      throw ex;
    }
  }

  private void runMosIngestWindow(BackfillRequest request) {
    LocalDate start = requireStartDate(request.dateStartLocal(), BackfillJobType.MOS_INGEST_WINDOW);
    LocalDate end = requireEndDate(request.dateEndLocal(), BackfillJobType.MOS_INGEST_WINDOW);
    validateDateRange(start, end);
    StationRegistry station = resolveStation(singleSeriesTicker(request.seriesTickers(),
        BackfillJobType.MOS_INGEST_WINDOW));
    List<MosModel> models = requireModels(request.models(), BackfillJobType.MOS_INGEST_WINDOW);
    int windowDays = request.mosWindowDays();
    if (windowDays < 1) {
      throw new IllegalArgumentException("mosWindowDays must be >= 1");
    }
    Instant rangeStartUtc = start.atStartOfDay(ZoneOffset.UTC).toInstant();
    Instant rangeEndUtc = end.plusDays(1).atStartOfDay(ZoneOffset.UTC).toInstant();
    Duration windowSize = Duration.ofDays(windowDays);
    String jobName = BackfillJobType.MOS_INGEST_WINDOW.jobName();
    String stationId = station.getStationId();

    for (MosModel model : models) {
      IngestCheckpoint existing = checkpointService.findCheckpoint(jobName, stationId, model)
          .orElse(null);
      Instant cursorRuntime = existing != null ? existing.getCursorRuntimeUtc() : null;
      Instant effectiveStart = rangeStartUtc;
      if (cursorRuntime != null && cursorRuntime.isAfter(rangeStartUtc)) {
        effectiveStart = cursorRuntime;
      }
      if (!effectiveStart.isBefore(rangeEndUtc)) {
        snapshot("job=mos_ingest_window station=" + stationId
            + " model=" + model.name()
            + " progress=complete"
            + " cursorRuntime=" + cursorRuntime);
        checkpointService.markComplete(jobName, stationId, model, null, cursorRuntime);
        continue;
      }
      List<WindowRange> windows = buildWindows(effectiveStart, rangeEndUtc, windowSize);
      long totalWindows = windows.size();
      long processedWindows = 0;
      AtomicReference<Instant> cursorRef = new AtomicReference<>(cursorRuntime);
      CheckpointHeartbeat heartbeat = CheckpointHeartbeat.start(() ->
          checkpointService.markRunning(jobName, stationId, model, null, cursorRef.get()));
      try {
        checkpointService.markRunning(jobName, stationId, model, null, cursorRef.get());
        int threadCount = Math.max(1, concurrencyProperties.getMosWindowThreads());
        if (threadCount == 1 || windows.size() <= 1) {
          for (WindowRange window : windows) {
            mosRunIngestService.ingestWindow(stationId, model, window.start(), window.end());
            processedWindows += 1;
            snapshot("job=mos_ingest_window station=" + stationId
                + " model=" + model.name()
                + " window=" + window.start() + ".." + window.end()
                + " progress=" + processedWindows + "/" + totalWindows
                + " remaining=" + (totalWindows - processedWindows));
            cursorRuntime = window.end();
            cursorRef.set(cursorRuntime);
            checkpointService.markRunning(jobName, stationId, model, null, cursorRuntime);
          }
        } else {
          ExecutorService executor = Executors.newFixedThreadPool(
              threadCount, namedThreadFactory("mos-window"));
          try {
            int index = 0;
            while (index < windows.size()) {
              int endIndex = Math.min(index + threadCount, windows.size());
              List<WindowRange> batch = windows.subList(index, endIndex);
              List<Future<?>> futures = new ArrayList<>(batch.size());
              for (WindowRange window : batch) {
                futures.add(executor.submit(() -> {
                  mosRunIngestService.ingestWindow(stationId, model, window.start(), window.end());
                  return null;
                }));
              }
              waitAllOrFail(futures);
              processedWindows += batch.size();
              WindowRange last = batch.get(batch.size() - 1);
              snapshot("job=mos_ingest_window station=" + stationId
                  + " model=" + model.name()
                  + " window=" + batch.get(0).start() + ".." + last.end()
                  + " progress=" + processedWindows + "/" + totalWindows
                  + " remaining=" + (totalWindows - processedWindows)
                  + " threads=" + threadCount);
              cursorRuntime = last.end();
              cursorRef.set(cursorRuntime);
              checkpointService.markRunning(jobName, stationId, model, null, cursorRuntime);
              index = endIndex;
            }
          } finally {
            shutdownExecutor(executor);
          }
        }
        cursorRuntime = rangeEndUtc;
        cursorRef.set(cursorRuntime);
        heartbeat.close();
        checkpointService.markComplete(jobName, stationId, model, null, cursorRuntime);
      } catch (RuntimeException ex) {
        heartbeat.close();
        checkpointService.markFailed(jobName, stationId, model, null, cursorRuntime, ex);
        throw ex;
      }
    }
  }

  private void runMosAsofMaterializeRange(BackfillRequest request) {
    LocalDate start = requireStartDate(request.dateStartLocal(),
        BackfillJobType.MOS_ASOF_MATERIALIZE_RANGE);
    LocalDate end = requireEndDate(request.dateEndLocal(),
        BackfillJobType.MOS_ASOF_MATERIALIZE_RANGE);
    validateDateRange(start, end);
    StationRegistry station = resolveStation(singleSeriesTicker(request.seriesTickers(),
        BackfillJobType.MOS_ASOF_MATERIALIZE_RANGE));
    List<MosModel> models = requireModels(request.models(),
        BackfillJobType.MOS_ASOF_MATERIALIZE_RANGE);
    if (request.asofPolicyId() == null) {
      throw new IllegalArgumentException("asofPolicyId is required for mos_asof_materialize_range");
    }
    String jobName = BackfillJobType.MOS_ASOF_MATERIALIZE_RANGE.jobName();
    String stationId = station.getStationId();

    IngestCheckpoint existing = checkpointService.findCheckpoint(jobName, stationId, null)
        .orElse(null);
    LocalDate cursorDate = existing != null ? existing.getCursorDate() : null;
    if (cursorDate != null && !cursorDate.isBefore(end)) {
      snapshot("job=mos_asof_materialize_range station=" + stationId
          + " progress=complete"
          + " cursorDate=" + cursorDate);
      checkpointService.markComplete(jobName, stationId, null, cursorDate, null);
      return;
    }
    LocalDate effectiveStart = start;
    if (cursorDate != null && !cursorDate.isBefore(start)) {
      effectiveStart = cursorDate.plusDays(1);
    }
    if (effectiveStart.isAfter(end)) {
      snapshot("job=mos_asof_materialize_range station=" + stationId
          + " progress=complete"
          + " cursorDate=" + cursorDate);
      checkpointService.markComplete(jobName, stationId, null, cursorDate, null);
      return;
    }
    long totalDays = ChronoUnit.DAYS.between(effectiveStart, end) + 1;
    long processedDays = 0;
    AtomicReference<LocalDate> cursorRef = new AtomicReference<>(cursorDate);
    CheckpointHeartbeat heartbeat = CheckpointHeartbeat.start(() ->
        checkpointService.markRunning(jobName, stationId, null, cursorRef.get(), null));
    try {
      checkpointService.markRunning(jobName, stationId, null, cursorRef.get(), null);
      LocalDate current = effectiveStart;
      int threadCount = Math.max(1, concurrencyProperties.getMosAsofThreads());
      if (threadCount == 1) {
        while (!current.isAfter(end)) {
          mosAsofMaterializeService.materializeForTargetDate(
              stationId, current, request.asofPolicyId(), models);
          processedDays += 1;
          cursorDate = current;
          cursorRef.set(cursorDate);
          checkpointService.markRunning(jobName, stationId, null, cursorDate, null);
          snapshot("job=mos_asof_materialize_range station=" + stationId
              + " targetDate=" + current
              + " progress=" + processedDays + "/" + totalDays
              + " remaining=" + (totalDays - processedDays)
              + " asofPolicyId=" + request.asofPolicyId());
          current = current.plusDays(1);
        }
      } else {
        ExecutorService executor = Executors.newFixedThreadPool(
            threadCount, namedThreadFactory("mos-asof"));
        try {
          List<LocalDate> days = new ArrayList<>();
          LocalDate cursor = current;
          while (!cursor.isAfter(end)) {
            days.add(cursor);
            cursor = cursor.plusDays(1);
          }
          int index = 0;
          while (index < days.size()) {
            int endIndex = Math.min(index + threadCount, days.size());
            List<LocalDate> batch = days.subList(index, endIndex);
            List<Future<?>> futures = new ArrayList<>(batch.size());
            for (LocalDate day : batch) {
              futures.add(executor.submit(() -> {
                mosAsofMaterializeService.materializeForTargetDate(
                    stationId, day, request.asofPolicyId(), models);
                return null;
              }));
            }
            waitAllOrFail(futures);
            processedDays += batch.size();
            LocalDate last = batch.get(batch.size() - 1);
            cursorDate = last;
            cursorRef.set(cursorDate);
            checkpointService.markRunning(jobName, stationId, null, cursorDate, null);
            snapshot("job=mos_asof_materialize_range station=" + stationId
                + " targetDate=" + batch.get(0) + ".." + last
                + " progress=" + processedDays + "/" + totalDays
                + " remaining=" + (totalDays - processedDays)
                + " asofPolicyId=" + request.asofPolicyId()
                + " threads=" + threadCount);
            index = endIndex;
          }
          current = end.plusDays(1);
        } finally {
          shutdownExecutor(executor);
        }
      }
      cursorDate = end;
      cursorRef.set(cursorDate);
      heartbeat.close();
      mosAsofFeatureReportService.logCompletenessReport(
          stationId, start, end, request.asofPolicyId(), models);
      checkpointService.markComplete(jobName, stationId, null, cursorDate, null);
    } catch (RuntimeException ex) {
      heartbeat.close();
      checkpointService.markFailed(jobName, stationId, null, cursorDate, null, ex);
      throw ex;
    }
  }

  private StationRegistry resolveStation(String seriesTicker) {
    String normalized = normalizeSeriesTicker(seriesTicker);
    return stationRegistryRepository.findBySeriesTicker(normalized)
        .orElseGet(() -> kalshiSeriesResolver.resolveAndUpsert(normalized));
  }

  private String singleSeriesTicker(List<String> seriesTickers, BackfillJobType jobType) {
    if (seriesTickers == null || seriesTickers.isEmpty()) {
      throw new IllegalArgumentException("seriesTicker is required for " + jobType.jobName());
    }
    if (seriesTickers.size() > 1) {
      throw new IllegalArgumentException(
          "Only one seriesTicker is supported for " + jobType.jobName());
    }
    return seriesTickers.get(0);
  }

  private List<MosModel> requireModels(List<MosModel> models, BackfillJobType jobType) {
    if (models == null || models.isEmpty()) {
      throw new IllegalArgumentException("models are required for " + jobType.jobName());
    }
    return models;
  }

  private LocalDate requireStartDate(LocalDate start, BackfillJobType jobType) {
    if (start == null) {
      throw new IllegalArgumentException("dateStartLocal is required for " + jobType.jobName());
    }
    return start;
  }

  private LocalDate requireEndDate(LocalDate end, BackfillJobType jobType) {
    if (end == null) {
      throw new IllegalArgumentException("dateEndLocal is required for " + jobType.jobName());
    }
    return end;
  }

  private void validateDateRange(LocalDate start, LocalDate end) {
    if (end.isBefore(start)) {
      throw new IllegalArgumentException("dateEndLocal must be >= dateStartLocal");
    }
  }

  private String normalizeSeriesTicker(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      throw new IllegalArgumentException("seriesTicker is required");
    }
    return seriesTicker.trim().toUpperCase(Locale.ROOT);
  }

  private long windowCount(Instant start, Instant end, Duration windowSize) {
    if (!start.isBefore(end)) {
      return 0;
    }
    long windowMillis = windowSize.toMillis();
    if (windowMillis <= 0) {
      return 0;
    }
    long totalMillis = Duration.between(start, end).toMillis();
    return (totalMillis + windowMillis - 1) / windowMillis;
  }

  private List<WindowRange> buildWindows(Instant start, Instant end, Duration windowSize) {
    if (!start.isBefore(end)) {
      return List.of();
    }
    List<WindowRange> windows = new ArrayList<>();
    Instant cursor = start;
    while (cursor.isBefore(end)) {
      Instant windowEnd = cursor.plus(windowSize);
      if (windowEnd.isAfter(end)) {
        windowEnd = end;
      }
      windows.add(new WindowRange(cursor, windowEnd));
      cursor = windowEnd;
    }
    return windows;
  }

  private void waitAllOrFail(List<Future<?>> futures) {
    for (Future<?> future : futures) {
      try {
        future.get();
      } catch (InterruptedException ex) {
        Thread.currentThread().interrupt();
        cancelRemaining(futures);
        throw new IllegalStateException("Backfill parallel task interrupted", ex);
      } catch (ExecutionException ex) {
        cancelRemaining(futures);
        Throwable cause = ex.getCause() == null ? ex : ex.getCause();
        if (cause instanceof RuntimeException runtimeException) {
          throw runtimeException;
        }
        throw new IllegalStateException("Backfill parallel task failed", cause);
      }
    }
  }

  private void cancelRemaining(List<? extends Future<?>> futures) {
    for (Future<?> future : futures) {
      if (!future.isDone()) {
        future.cancel(true);
      }
    }
  }

  private void shutdownExecutor(ExecutorService executor) {
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

  private ThreadFactory namedThreadFactory(String prefix) {
    AtomicInteger counter = new AtomicInteger(1);
    return runnable -> {
      Thread thread = new Thread(runnable);
      thread.setName(prefix + "-" + counter.getAndIncrement());
      thread.setDaemon(false);
      return thread;
    };
  }

  private record WindowRange(Instant start, Instant end) {
  }

  private void snapshot(String message) {
    String payload = "[BACKFILL] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
