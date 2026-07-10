package com.predictionmarkets.weather.klga.iemmos;

import com.predictionmarkets.weather.common.Hashing;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class IemMosBackfillService {
  private static final Logger logger = LoggerFactory.getLogger(IemMosBackfillService.class);

  private final IemMosBackfillRepository repository;
  private final IemMosPlanner planner;
  private final IemMosHttpClient httpClient;
  private final IemMosParser parser;

  public IemMosBackfillService(IemMosBackfillRepository repository,
                               IemMosPlanner planner,
                               IemMosHttpClient httpClient,
                               IemMosParser parser) {
    this.repository = repository;
    this.planner = planner;
    this.httpClient = httpClient;
    this.parser = parser;
  }

  public void run(IemMosBackfillProperties properties) {
    validate(properties);
    List<IemMosStation> stations = filterStations(properties, repository.loadMosStations());
    if (stations.isEmpty()) {
      throw new IllegalStateException("No MOS stations found in registry.station_registry");
    }
    if (!repository.cutoffExists(properties.getCutoffId())) {
      throw new IllegalStateException("Cutoff " + properties.getCutoffId()
          + " is missing from registry.cutoffs. Run the KLGA Python db migrate/seed first.");
    }
    List<IemMosChunk> plannedChunks = planner.plan(properties, stations);
    repository.initializeJob(properties, plannedChunks);
    snapshot("planned job=" + properties.getJobId()
        + " chunks=" + plannedChunks.size()
        + " stations=" + stations.size()
        + " products=" + (properties.getProducts().isEmpty() ? "all" : properties.getProducts())
        + " through=" + properties.getThrough());
    if ("dry-run".equalsIgnoreCase(properties.getMode())) {
      snapshot("dry-run complete; no HTTP requests were sent.");
      return;
    }
    repository.markJobRunning(properties.getJobId());
    List<IemMosChunk> runnable = repository.chunksToRun(properties.getJobId());
    runChunks(properties, runnable);
    repository.refreshJobSummary(properties.getJobId(), true);
    snapshot("IEM MOS backfill finished: " + formatProgress(repository.progress(properties.getJobId())));
  }

  private void runChunks(IemMosBackfillProperties properties, List<IemMosChunk> chunks) {
    if (chunks.isEmpty()) {
      snapshot("No runnable IEM MOS chunks remain.");
      return;
    }
    int workers = Math.max(1, properties.getThreads());
    IemMosRateLimiter rateLimiter = new IemMosRateLimiter(properties.getRequestSpacingMs());
    ExecutorService executor = Executors.newFixedThreadPool(workers);
    ExecutorCompletionService<Void> completion = new ExecutorCompletionService<>(executor);
    try {
      for (IemMosChunk chunk : chunks) {
        completion.submit(() -> {
          processChunk(properties, rateLimiter, chunk);
          return null;
        });
      }
      for (int i = 0; i < chunks.size(); i++) {
        Future<Void> finished = completion.take();
        try {
          finished.get();
        } catch (ExecutionException ex) {
          Throwable cause = ex.getCause();
          if (cause instanceof IemMosRateLimitException) {
            executor.shutdownNow();
            repository.refreshJobSummary(properties.getJobId(), false);
            throw (IemMosRateLimitException) cause;
          }
          logger.error("IEM MOS worker failed", cause);
        } catch (CancellationException ex) {
          logger.warn("IEM MOS worker cancelled");
        }
        repository.refreshJobSummary(properties.getJobId(), false);
        snapshot(formatProgress(repository.progress(properties.getJobId())));
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      executor.shutdownNow();
      throw new IllegalStateException("IEM MOS backfill interrupted", ex);
    } finally {
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

  private void processChunk(IemMosBackfillProperties properties,
                            IemMosRateLimiter rateLimiter,
                            IemMosChunk chunk) {
    repository.markChunkRunning(chunk);
    int maxAttempts = Math.max(1, properties.getMaxAttempts());
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        rateLimiter.awaitTurn();
        IemMosFetchResult result = httpClient.fetch(chunk);
        if (result.httpStatus() == 200) {
          handleSuccess(properties, chunk, result);
          return;
        }
        if (result.httpStatus() == 404) {
          handleEmptyOrMissing(properties, chunk, result, "provider_no_product_for_station");
          return;
        }
        if (result.httpStatus() == 422 && chunk.windowDays() > 31) {
          splitChunk(chunk, result);
          return;
        }
        if (result.httpStatus() == 429) {
          repository.markChunkFailed(
              chunk,
              "rate_limited",
              result.httpStatus(),
              "HTTP_429",
              "IEM returned HTTP 429; stopping to preserve state");
          throw new IemMosRateLimitException("IEM returned HTTP 429 for " + chunk.chunkId());
        }
        if (isRetryableStatus(result.httpStatus()) && attempt < maxAttempts) {
          sleepRetry(properties, attempt);
          continue;
        }
        repository.markChunkFailed(
            chunk,
            "failed",
            result.httpStatus(),
            "HTTP_" + result.httpStatus(),
            "IEM MOS request failed with HTTP " + result.httpStatus());
        return;
      } catch (IOException ex) {
        if (attempt < maxAttempts) {
          sleepRetry(properties, attempt);
          continue;
        }
        repository.markChunkFailed(chunk, "failed", null, "TRANSPORT_ERROR", ex.getMessage());
        return;
      } catch (RuntimeException ex) {
        repository.markChunkFailed(chunk, "failed", null, ex.getClass().getSimpleName(), ex.getMessage());
        throw ex;
      }
    }
  }

  private void handleSuccess(IemMosBackfillProperties properties,
                             IemMosChunk chunk,
                             IemMosFetchResult result) throws IOException {
    String responseSha = Hashing.sha256Hex(result.body());
    String rawStorageUri = writeRawPayload(properties, chunk, result.body());
    IemMosStoredRequest stored = repository.persistSourceArtifacts(
        chunk,
        result,
        rawStorageUri,
        responseSha);
    List<IemMosForecastRow> rows = parser.parse(result.body(), chunk, stored);
    int rowsUpserted = repository.upsertForecastRows(rows);
    int featureRows = 0;
    if (properties.isBuildDailyFeatures()) {
      repository.materializeTargetInstances(
          chunk.cutoffId(),
          chunk.startDate(),
          chunk.endDateInclusive());
      featureRows = repository.rebuildDailyFeatures(chunk);
      if (properties.isBuildFeatureMatrix()) {
        repository.rebuildFeatureMatrix(chunk.cutoffId(), chunk.startDate(), chunk.endDateInclusive());
      }
    }
    if (rows.isEmpty()) {
      repository.recordGap(
          chunk,
          "empty_response",
          "IEM returned HTTP 200 with no structured rows",
          "{\"http_status\":200}");
    }
    repository.markChunkCompleted(chunk, stored, rowsUpserted, featureRows);
  }

  private void handleEmptyOrMissing(IemMosBackfillProperties properties,
                                    IemMosChunk chunk,
                                    IemMosFetchResult result,
                                    String reason) throws IOException {
    String responseSha = Hashing.sha256Hex(result.body());
    String rawStorageUri = writeRawPayload(properties, chunk, result.body());
    IemMosStoredRequest stored = repository.persistSourceArtifacts(
        chunk,
        result,
        rawStorageUri,
        responseSha);
    repository.recordGap(
        chunk,
        "provider_no_data",
        reason,
        "{\"http_status\":" + result.httpStatus() + "}");
    repository.markChunkCompleted(chunk, stored, 0, 0);
  }

  private void splitChunk(IemMosChunk chunk, IemMosFetchResult result) {
    LocalDate start = chunk.startDate();
    LocalDate endExclusive = LocalDate.ofInstant(chunk.windowEndUtc(), ZoneOffset.UTC);
    LocalDate splitDate = start.plusDays(Math.max(1L, chunk.windowDays() / 2L));
    if (!splitDate.isAfter(start) || !splitDate.isBefore(endExclusive)) {
      repository.markChunkFailed(
          chunk,
          "failed",
          result.httpStatus(),
          "HTTP_422",
          "IEM returned 422 and chunk could not be split further");
      return;
    }
    Instant splitPoint = splitDate.atStartOfDay().toInstant(ZoneOffset.UTC);
    IemMosChunk head = planner.splitChunk(chunk, splitPoint, "split_a");
    IemMosChunk tail = planner.splitChunkTail(chunk, splitPoint, "split_b");
    repository.upsertPlannedChunks(List.of(head, tail));
    repository.recordGap(
        chunk,
        "request_window_split",
        "IEM returned 422; parent chunk split into two child chunks",
        "{\"http_status\":422}");
    repository.markChunkFailed(
        chunk,
        "skipped",
        result.httpStatus(),
        "HTTP_422_SPLIT",
        "Split into child chunks " + head.chunkId() + " and " + tail.chunkId());
  }

  private String writeRawPayload(IemMosBackfillProperties properties,
                                 IemMosChunk chunk,
                                 byte[] payload) throws IOException {
    Path dir = properties.getRawOutputRoot()
        .resolve(properties.getJobId())
        .resolve(chunk.product().productCode())
        .resolve(chunk.station().stationId());
    Files.createDirectories(dir);
    String fileName = chunk.windowStartUtc().toString().replace(":", "")
        + "_"
        + chunk.windowEndUtc().toString().replace(":", "")
        + "_"
        + chunk.requestSha256().substring(0, 16)
        + ".json.gz";
    Path output = dir.resolve(fileName).toAbsolutePath().normalize();
    try (OutputStream file = Files.newOutputStream(output);
         GZIPOutputStream gzip = new GZIPOutputStream(file)) {
      gzip.write(payload);
    }
    return output.toString();
  }

  private void validate(IemMosBackfillProperties properties) {
    if (properties.getJobId() == null || properties.getJobId().isBlank()) {
      throw new IllegalArgumentException("iem-mos.job-id is required");
    }
    if (properties.getCutoffId() == null || properties.getCutoffId().isBlank()) {
      throw new IllegalArgumentException("iem-mos.cutoff-id is required");
    }
    if (!"T_1245UTC".equals(properties.getCutoffId())) {
      throw new IllegalArgumentException("This runner is locked to cutoff T_1245UTC for v1");
    }
    if (properties.getThrough() == null) {
      throw new IllegalArgumentException("iem-mos.through is required");
    }
  }

  private List<IemMosStation> filterStations(IemMosBackfillProperties properties,
                                             List<IemMosStation> allStations) {
    if (properties.getStationIds().isEmpty()) {
      return allStations;
    }
    Set<String> requested = properties.getStationIds().stream()
        .map(value -> value.trim().toUpperCase(Locale.ROOT))
        .filter(value -> !value.isBlank())
        .collect(java.util.stream.Collectors.toSet());
    List<IemMosStation> selected = allStations.stream()
        .filter(station -> requested.contains(station.stationId().toUpperCase(Locale.ROOT))
            || requested.contains(station.mosStationId().toUpperCase(Locale.ROOT)))
        .toList();
    if (selected.isEmpty()) {
      throw new IllegalArgumentException("No MOS stations matched " + requested);
    }
    return selected;
  }

  private boolean isRetryableStatus(int status) {
    return status == 500 || status == 502 || status == 503 || status == 504;
  }

  private void sleepRetry(IemMosBackfillProperties properties, int attempt) {
    long delay = Math.min(
        Duration.ofSeconds(30).toMillis(),
        Math.max(1000L, properties.getRetryBackoffMs()) * (1L << Math.min(attempt - 1, 4)));
    try {
      Thread.sleep(delay);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("IEM MOS retry interrupted", ex);
    }
  }

  private void snapshot(String message) {
    String payload = "[IEM-MOS] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private String formatProgress(IemMosProgress progress) {
    return String.format(
        Locale.ROOT,
        "progress complete=%.2f%% done=%d empty=%d failed=%d remaining=%d total=%d rows=%d features=%d bytes=%.2fMB",
        progress.percentComplete(),
        progress.completed(),
        progress.completedEmpty(),
        progress.failed(),
        progress.remaining(),
        progress.chunksTotal(),
        progress.rowsUpserted(),
        progress.featureRowsUpserted(),
        progress.bytesFetched() / 1_000_000.0);
  }
}
