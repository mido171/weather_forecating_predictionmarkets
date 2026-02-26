package com.predictionmarkets.weather.weathercom.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.models.WeatherComApiCall;
import com.predictionmarkets.weather.models.WeatherComIngestionRun;
import com.predictionmarkets.weather.models.WeatherComIngestionStatus;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import com.predictionmarkets.weather.repository.WundergroundDailyMaxTemperatureUpsertRepository;
import com.predictionmarkets.weather.repository.WeatherComApiCallRepository;
import com.predictionmarkets.weather.repository.WeatherComIngestionRunRepository;
import com.predictionmarkets.weather.repository.WeatherComLocationRepository;
import com.predictionmarkets.weather.repository.WeatherComObservationUpsertRepository;
import com.predictionmarkets.weather.weathercom.client.WeatherComClient;
import com.predictionmarkets.weather.weathercom.client.WeatherComClientResult;
import com.predictionmarkets.weather.weathercom.client.dto.WeatherComHistoricalResponse;
import com.predictionmarkets.weather.weathercom.client.dto.WeatherComObservationPayload;
import com.predictionmarkets.weather.weathercom.client.dto.WeatherComResponseMetadata;
import com.predictionmarkets.weather.weathercom.config.WeatherComProperties;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComApiCallStatusFilter;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Comparator;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.time.ZoneId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

@Service
public class WeatherComIngestionService {
  private static final Logger logger = LoggerFactory.getLogger(WeatherComIngestionService.class);
  private static final Set<String> EASTERN_ZONE_FALLBACK_STATIONS = Set.of(
      "KLGA",
      "KJFK",
      "KEWR",
      "KTEB",
      "KHPN",
      "KISP",
      "KBDR",
      "KMMU");

  private final WeatherComClient client;
  private final WeatherComProperties properties;
  private final ObjectMapper objectMapper;
  private final WeatherComLocationRepository locationRepository;
  private final WeatherComIngestionRunRepository runRepository;
  private final WeatherComApiCallRepository apiCallRepository;
  private final WeatherComObservationUpsertRepository observationUpsertRepository;
  private final WundergroundDailyMaxTemperatureUpsertRepository dailyMaxUpsertRepository;
  private final StationRegistryRepository stationRegistryRepository;
  private final ThreadPoolExecutor taskExecutor;
  private final ExecutorService runExecutor;
  private final TransactionTemplate transactionTemplate;
  private final Map<String, String> locationZoneCache = new ConcurrentHashMap<>();

  public WeatherComIngestionService(WeatherComClient client,
                                    WeatherComProperties properties,
                                    ObjectMapper objectMapper,
                                    WeatherComLocationRepository locationRepository,
                                    WeatherComIngestionRunRepository runRepository,
                                    WeatherComApiCallRepository apiCallRepository,
                                    WeatherComObservationUpsertRepository observationUpsertRepository,
                                    WundergroundDailyMaxTemperatureUpsertRepository dailyMaxUpsertRepository,
                                    StationRegistryRepository stationRegistryRepository,
                                    ThreadPoolExecutor weatherComTaskExecutor,
                                    ExecutorService weatherComRunExecutor,
                                    TransactionTemplate transactionTemplate) {
    this.client = client;
    this.properties = properties;
    this.objectMapper = objectMapper;
    this.locationRepository = locationRepository;
    this.runRepository = runRepository;
    this.apiCallRepository = apiCallRepository;
    this.observationUpsertRepository = observationUpsertRepository;
    this.dailyMaxUpsertRepository = dailyMaxUpsertRepository;
    this.stationRegistryRepository = stationRegistryRepository;
    this.taskExecutor = weatherComTaskExecutor;
    this.runExecutor = weatherComRunExecutor;
    this.transactionTemplate = transactionTemplate;
  }

  public WeatherComIngestionRun triggerIngestion(List<String> locationIds,
                                                 LocalDate startDate,
                                                 LocalDate endDate,
                                                 String units,
                                                 String requestedBy) {
    if (!properties.getIngestion().isEnabled()) {
      throw new IllegalStateException("weathercom.ingestion.enabled=false");
    }
    LocalDate normalizedStartDate = Objects.requireNonNull(startDate, "startDate is required");
    LocalDate normalizedEndDate = Objects.requireNonNull(endDate, "endDate is required");
    if (normalizedEndDate.isBefore(normalizedStartDate)) {
      throw new IllegalArgumentException("endDate must be >= startDate");
    }

    String normalizedUnits = normalizeUnits(units);
    List<String> normalizedLocationIds = resolveLocationIds(locationIds);
    int chunkDays = Math.max(1, properties.getIngestion().getChunkDays());
    List<IngestionTask> tasks = buildTasks(
        normalizedLocationIds,
        normalizedStartDate,
        normalizedEndDate,
        normalizedUnits,
        chunkDays);
    if (tasks.isEmpty()) {
      throw new IllegalArgumentException("No ingestion tasks were produced for the request");
    }

    WeatherComIngestionRun run = new WeatherComIngestionRun();
    Instant now = Instant.now();
    run.setStatus(WeatherComIngestionStatus.RUNNING);
    run.setStartedAtUtc(now);
    run.setRequestedBy(normalizeRequestedBy(requestedBy));
    run.setRequestPayloadJson(serializePayload(normalizedLocationIds, normalizedStartDate, normalizedEndDate,
        normalizedUnits, chunkDays));
    run.setTotalTasks(tasks.size());
    run.setSucceededTasks(0);
    run.setFailedTasks(0);
    run.setCreatedAtUtc(now);
    run.setUpdatedAtUtc(now);
    WeatherComIngestionRun savedRun = runRepository.save(run);

    logger.info(
        "weathercom run created runId={} locations={} dateRange={}..{} units={} tasks={} threadPoolSize={} queueCapacity={} chunkDays={}",
        savedRun.getId(),
        normalizedLocationIds.size(),
        normalizedStartDate,
        normalizedEndDate,
        normalizedUnits,
        tasks.size(),
        properties.getIngestion().getThreadPoolSize(),
        properties.getIngestion().getQueueCapacity(),
        chunkDays);

    runExecutor.submit(() -> executeRun(savedRun.getId(), tasks));
    return savedRun;
  }

  public WeatherComIngestionRun getRun(Long runId) {
    return runRepository.findById(runId)
        .orElseThrow(() -> new WeatherComNotFoundException("weathercom_ingestion_run not found: " + runId));
  }

  public Page<WeatherComApiCall> listApiCalls(Long runId,
                                              WeatherComApiCallStatusFilter statusFilter,
                                              int page,
                                              int size) {
    if (!runRepository.existsById(runId)) {
      throw new WeatherComNotFoundException("weathercom_ingestion_run not found: " + runId);
    }
    Pageable pageable = PageRequest.of(Math.max(0, page), Math.max(1, Math.min(size, 500)));
    return switch (statusFilter) {
      case FAILED -> apiCallRepository.findFailedByRunId(runId, pageable);
      case SUCCEEDED -> apiCallRepository.findSucceededByRunId(runId, pageable);
      case ALL -> apiCallRepository.findByIngestionRunIdOrderByIdDesc(runId, pageable);
    };
  }

  private void executeRun(Long runId, List<IngestionTask> tasks) {
    List<Future<Boolean>> futures = new ArrayList<>(tasks.size());
    for (IngestionTask task : tasks) {
      futures.add(taskExecutor.submit(() -> processTask(runId, task)));
    }

    int totalTasks = tasks.size();
    int progressStep = computeProgressLogStep(totalTasks);
    int completed = 0;
    int succeeded = 0;
    int failed = 0;
    for (Future<Boolean> future : futures) {
      try {
        boolean taskSucceeded = Boolean.TRUE.equals(future.get());
        if (taskSucceeded) {
          succeeded++;
        } else {
          failed++;
        }
      } catch (Exception ex) {
        logger.error("weathercom task crashed unexpectedly runId={}", runId, ex);
        incrementFailedTaskCount(runId);
        failed++;
      } finally {
        completed++;
      }
      if (shouldLogProgress(completed, totalTasks, progressStep)) {
        logProgress(runId, completed, totalTasks, succeeded, failed);
      }
    }

    finishRun(runId);
  }

  private boolean processTask(Long runId, IngestionTask task) {
    long taskStartNanos = System.nanoTime();
    logger.info("weathercom api call started runId={} locationId={} range={}..{} units={}",
        runId, task.requestLocationId(), task.startDate(), task.endDate(), task.units());

    WeatherComClientResult result;
    try {
      result = client.fetchHistoricalObservations(
          task.requestLocationId(), task.units(), task.startDate(), task.endDate());
    } catch (RuntimeException ex) {
      result = new WeatherComClientResult(
          false,
          0,
          null,
          null,
          "CLIENT_ERROR",
          abbreviate(ex.getMessage(), 1000),
          Instant.now(),
          0,
          1);
    }

    boolean success = result.success();
    try {
      persistTaskResultWithRetry(runId, task, result);
    } catch (RuntimeException ex) {
      success = false;
      logger.error("weathercom persistence failed runId={} locationId={} range={}..{}",
          runId, task.requestLocationId(), task.startDate(), task.endDate(), ex);
      try {
        WeatherComClientResult persistenceFailure = new WeatherComClientResult(
            false,
            result.httpStatus(),
            null,
            result.responseBody(),
            "PERSISTENCE_ERROR",
            abbreviate(ex.getMessage(), 1000),
            Instant.now(),
            result.durationMs(),
            result.attempts());
        persistTaskResultWithRetry(runId, task, persistenceFailure);
      } catch (RuntimeException inner) {
        logger.error("weathercom fallback persistence failed runId={}", runId, inner);
      }
    }

    if (success) {
      incrementSucceededTaskCount(runId);
    } else {
      incrementFailedTaskCount(runId);
    }

    long elapsedMs = Math.max(0L, (System.nanoTime() - taskStartNanos) / 1_000_000L);
    logger.info(
        "weathercom api call finished runId={} locationId={} range={}..{} success={} httpStatus={} durationMs={}",
        runId,
        task.requestLocationId(),
        task.startDate(),
        task.endDate(),
        success,
        result.httpStatus(),
        elapsedMs);
    return success;
  }

  private void persistTaskResultWithRetry(Long runId, IngestionTask task, WeatherComClientResult result) {
    int maxAttempts = 3;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        persistTaskResult(runId, task, result);
        return;
      } catch (RuntimeException ex) {
        boolean retryable = isRetryablePersistenceFailure(ex);
        if (!retryable || attempt == maxAttempts) {
          throw ex;
        }
        long backoffMs = 150L * attempt;
        logger.warn(
            "weathercom persistence retry runId={} locationId={} range={}..{} attempt={}/{} backoffMs={} reason={}",
            runId,
            task.requestLocationId(),
            task.startDate(),
            task.endDate(),
            attempt,
            maxAttempts,
            backoffMs,
            abbreviate(ex.getMessage(), 250));
        sleepQuietly(backoffMs);
      }
    }
  }

  private void persistTaskResult(Long runId, IngestionTask task, WeatherComClientResult result) {
    transactionTemplate.executeWithoutResult(status -> {
      WeatherComApiCall apiCall = new WeatherComApiCall();
      apiCall.setIngestionRun(runRepository.getReferenceById(runId));
      apiCall.setRequestLocationId(task.requestLocationId());
      apiCall.setUnits(task.units());
      apiCall.setStartDate(task.startDate());
      apiCall.setEndDate(task.endDate());
      apiCall.setHttpStatus(result.httpStatus());
      apiCall.setFetchedAtUtc(result.fetchedAtUtc() == null ? Instant.now() : result.fetchedAtUtc());
      apiCall.setDurationMs(result.durationMs());
      apiCall.setErrorType(trimToNull(result.errorType()));
      apiCall.setErrorMessage(trimToNull(abbreviate(result.errorMessage(), 4000)));
      String responseBody = normalizeResponseBody(result.responseBody());
      apiCall.setResponseBodyJson(responseBody);
      apiCall.setResponseBodyHash(responseBody == null ? null : Hashing.sha256Hex(responseBody));

      WeatherComHistoricalResponse payload = result.payload();
      WeatherComResponseMetadata metadata = payload == null ? null : payload.getMetadata();
      if (metadata != null) {
        apiCall.setResponseLocationId(firstNonBlank(metadata.getLocationId(), metadata.getLocationKey()));
        apiCall.setResponseUnits(trimToNull(metadata.getUnits()));
        apiCall.setResponseLanguage(trimToNull(metadata.getLanguage()));
        apiCall.setTransactionId(trimToNull(metadata.getTransactionId()));
        apiCall.setApiVersion(trimToNull(metadata.getVersion()));
        apiCall.setExpireTimeGmt(metadata.getExpireTimeGmt());
      }

      Instant now = Instant.now();
      apiCall.setCreatedAtUtc(now);
      apiCall.setUpdatedAtUtc(now);
      WeatherComApiCall savedCall = apiCallRepository.save(apiCall);

      if (!result.success() || payload == null || payload.getObservations() == null) {
        return;
      }

      List<WeatherComObservationUpsertRepository.UpsertRow> rows =
          toObservationRows(savedCall.getId(), task.requestLocationId(), payload.getObservations(), now);
      if (!rows.isEmpty()) {
        observationUpsertRepository.upsertAll(rows, properties.getIngestion().getUpsertBatchSize());
      }

      List<WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow> dailyRows =
          toDailyMaxRows(savedCall.getId(), task.requestLocationId(), payload.getObservations(), now);
      if (!dailyRows.isEmpty()) {
        dailyMaxUpsertRepository.upsertAll(dailyRows);
      }
    });
  }

  private List<WeatherComObservationUpsertRepository.UpsertRow> toObservationRows(
      Long apiCallId,
      String requestLocationId,
      List<WeatherComObservationPayload> observations,
      Instant now) {
    List<WeatherComObservationUpsertRepository.UpsertRow> rows = new ArrayList<>();
    for (WeatherComObservationPayload observation : observations) {
      if (observation == null || observation.getValidTimeGmt() == null) {
        continue;
      }
      String obsId = firstNonBlank(observation.getObsId(), observation.getKey());
      if (obsId == null) {
        continue;
      }
      long validTimeGmt = observation.getValidTimeGmt();
      rows.add(new WeatherComObservationUpsertRepository.UpsertRow(
          apiCallId,
          requestLocationId,
          obsId,
          trimToNull(observation.getKey()),
          trimToNull(observation.getObsName()),
          validTimeGmt,
          Instant.ofEpochSecond(validTimeGmt),
          trimToNull(observation.getDayInd()),
          observation.getTemp(),
          observation.getDewPt(),
          observation.getHeatIndex(),
          observation.getRh(),
          observation.getPressure(),
          observation.getPressureTend(),
          trimToNull(observation.getPressureDesc()),
          observation.getVis(),
          observation.getWc(),
          observation.getWdir(),
          trimToNull(observation.getWdirCardinal()),
          observation.getGust(),
          observation.getWspd(),
          trimToNull(observation.getWxPhrase()),
          observation.getWxIcon(),
          observation.getIconExtd(),
          observation.getPrecipTotal(),
          observation.getPrecipHrly(),
          observation.getSnowHrly(),
          observation.getMaxTemp(),
          observation.getMinTemp(),
          trimToNull(observation.getUvDesc()),
          observation.getUvIndex(),
          observation.getFeelsLike(),
          trimToNull(observation.getClds()),
          trimToNull(observation.getQualifier()),
          trimToNull(observation.getQualifierSvrty()),
          trimToNull(observation.getBluntPhrase()),
          trimToNull(observation.getTersePhrase()),
          trimToNull(observation.getObservationClass()),
          observation.getWaterTemp(),
          observation.getPrimaryWavePeriod(),
          observation.getPrimaryWaveHeight(),
          observation.getPrimarySwellPeriod(),
          observation.getPrimarySwellHeight(),
          observation.getPrimarySwellDirection(),
          observation.getSecondarySwellPeriod(),
          observation.getSecondarySwellHeight(),
          observation.getSecondarySwellDirection(),
          now,
          now));
    }
    return rows;
  }

  private List<WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow> toDailyMaxRows(
      Long apiCallId,
      String requestLocationId,
      List<WeatherComObservationPayload> observations,
      Instant now) {
    if (observations == null || observations.isEmpty()) {
      return List.of();
    }

    ZoneId zoneId = resolveZoneId(requestLocationId);
    String zoneIdText = zoneId.getId();
    Map<DailyMaxKey, DailyMaxCandidate> bestByDay = new HashMap<>();
    Map<DailyMaxKey, Integer> countByDay = new HashMap<>();

    for (WeatherComObservationPayload observation : observations) {
      if (observation == null || observation.getValidTimeGmt() == null) {
        continue;
      }
      String obsId = firstNonBlank(observation.getObsId(), observation.getKey());
      if (obsId == null) {
        continue;
      }
      Integer candidateTemp = observation.getMaxTemp() != null ? observation.getMaxTemp() : observation.getTemp();
      if (candidateTemp == null) {
        continue;
      }

      long validTimeGmt = observation.getValidTimeGmt();
      LocalDate targetDateLocal = Instant.ofEpochSecond(validTimeGmt).atZone(zoneId).toLocalDate();
      DailyMaxKey key = new DailyMaxKey(requestLocationId, obsId, targetDateLocal);
      countByDay.merge(key, 1, Integer::sum);

      String sourceType = observation.getMaxTemp() != null
          ? "REPORTED_MAX_TEMP"
          : "OBSERVED_TEMP_MAX";
      DailyMaxCandidate existing = bestByDay.get(key);
      if (existing == null || candidateTemp > existing.maxTempF()) {
        bestByDay.put(key, new DailyMaxCandidate(candidateTemp, validTimeGmt, sourceType));
      }
    }

    List<WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow> rows =
        new ArrayList<>(bestByDay.size());
    for (Map.Entry<DailyMaxKey, DailyMaxCandidate> entry : bestByDay.entrySet()) {
      DailyMaxKey key = entry.getKey();
      DailyMaxCandidate candidate = entry.getValue();
      rows.add(new WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow(
          key.requestLocationId(),
          key.obsId(),
          zoneIdText,
          key.targetDateLocal(),
          candidate.maxTempF(),
          candidate.validTimeGmt(),
          apiCallId,
          candidate.sourceType(),
          countByDay.getOrDefault(key, 1),
          now,
          now));
    }
    rows.sort(Comparator
        .comparing(WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow::requestLocationId)
        .thenComparing(WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow::obsId)
        .thenComparing(WundergroundDailyMaxTemperatureUpsertRepository.UpsertRow::targetDateLocal));
    return rows;
  }

  private ZoneId resolveZoneId(String requestLocationId) {
    String zoneId = locationZoneCache.computeIfAbsent(
        requestLocationId,
        this::findZoneIdForLocation);
    return ZoneId.of(zoneId);
  }

  private String findZoneIdForLocation(String requestLocationId) {
    String stationId = extractStationId(requestLocationId);
    if (stationId == null) {
      return "UTC";
    }
    if (EASTERN_ZONE_FALLBACK_STATIONS.contains(stationId)) {
      return "America/New_York";
    }
    return stationRegistryRepository.findById(stationId)
        .map(StationRegistry::getZoneId)
        .filter(value -> value != null && !value.isBlank())
        .orElse("UTC");
  }

  private String extractStationId(String requestLocationId) {
    String normalized = trimToNull(requestLocationId);
    if (normalized == null) {
      return null;
    }
    int idx = normalized.indexOf(':');
    String stationId = idx < 0 ? normalized : normalized.substring(0, idx);
    stationId = trimToNull(stationId);
    if (stationId == null) {
      return null;
    }
    return stationId.toUpperCase(Locale.ROOT);
  }

  private void incrementSucceededTaskCount(Long runId) {
    transactionTemplate.executeWithoutResult(status ->
        runRepository.incrementSucceededTasks(runId, 1, Instant.now()));
  }

  private void incrementFailedTaskCount(Long runId) {
    transactionTemplate.executeWithoutResult(status ->
        runRepository.incrementFailedTasks(runId, 1, Instant.now()));
  }

  private void finishRun(Long runId) {
    WeatherComIngestionRun run = getRun(runId);
    WeatherComIngestionStatus finalStatus = determineFinalStatus(run);
    Instant now = Instant.now();
    transactionTemplate.executeWithoutResult(status ->
        runRepository.markFinished(runId, finalStatus, now, now));
    logger.info("weathercom run finished runId={} status={} succeeded={} failed={} total={}",
        runId, finalStatus, run.getSucceededTasks(), run.getFailedTasks(), run.getTotalTasks());
  }

  private int computeProgressLogStep(int totalTasks) {
    return Math.max(1, totalTasks / 100);
  }

  private boolean shouldLogProgress(int completed, int totalTasks, int progressStep) {
    return completed == 1 || completed == totalTasks || completed % progressStep == 0;
  }

  private void logProgress(Long runId,
                           int completed,
                           int totalTasks,
                           int succeeded,
                           int failed) {
    double percent = totalTasks == 0 ? 100.0d : ((double) completed * 100.0d) / totalTasks;
    logger.info(
        "weathercom run progress runId={} completed={}/{} percent={} succeeded={} failed={} executorActive={} executorQueued={}",
        runId,
        completed,
        totalTasks,
        String.format(Locale.ROOT, "%.1f", percent),
        succeeded,
        failed,
        taskExecutor.getActiveCount(),
        taskExecutor.getQueue().size());
  }

  private WeatherComIngestionStatus determineFinalStatus(WeatherComIngestionRun run) {
    if (run.getFailedTasks() == 0) {
      return WeatherComIngestionStatus.SUCCEEDED;
    }
    if (run.getSucceededTasks() > 0) {
      return WeatherComIngestionStatus.PARTIAL_SUCCESS;
    }
    return WeatherComIngestionStatus.FAILED;
  }

  private List<IngestionTask> buildTasks(List<String> locationIds,
                                         LocalDate startDate,
                                         LocalDate endDate,
                                         String units,
                                         int chunkDays) {
    List<IngestionTask> tasks = new ArrayList<>();
    for (String locationId : locationIds) {
      LocalDate cursor = startDate;
      while (!cursor.isAfter(endDate)) {
        LocalDate chunkEnd = cursor.plusDays(chunkDays - 1L);
        if (chunkEnd.isAfter(endDate)) {
          chunkEnd = endDate;
        }
        tasks.add(new IngestionTask(locationId, cursor, chunkEnd, units));
        cursor = chunkEnd.plusDays(1);
      }
    }
    return tasks;
  }

  private List<String> resolveLocationIds(List<String> locationIds) {
    if (locationIds == null || locationIds.isEmpty()) {
      List<String> active = locationRepository.findAllByActiveTrueOrderByLocationIdAsc().stream()
          .map(location -> location.getLocationId())
          .toList();
      if (active.isEmpty()) {
        throw new IllegalArgumentException("No active weathercom_location rows found");
      }
      return active;
    }
    LinkedHashSet<String> normalized = new LinkedHashSet<>();
    for (String locationId : locationIds) {
      normalized.add(WeatherComValidation.normalizeLocationId(locationId));
    }
    return new ArrayList<>(normalized);
  }

  private String serializePayload(List<String> locationIds,
                                  LocalDate startDate,
                                  LocalDate endDate,
                                  String units,
                                  int chunkDays) {
    Map<String, Object> payload = Map.of(
        "locationIds", locationIds,
        "startDate", startDate.toString(),
        "endDate", endDate.toString(),
        "units", units,
        "chunkDays", chunkDays);
    try {
      return objectMapper.writeValueAsString(payload);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize weathercom ingestion request payload", ex);
    }
  }

  private String normalizeRequestedBy(String requestedBy) {
    if (requestedBy == null || requestedBy.isBlank()) {
      return "system";
    }
    return requestedBy.trim();
  }

  private String normalizeUnits(String units) {
    if (units == null || units.isBlank()) {
      throw new IllegalArgumentException("units is required");
    }
    String normalized = units.trim().toLowerCase(Locale.ROOT);
    if (!List.of("e", "m", "h", "s").contains(normalized)) {
      throw new IllegalArgumentException("units must be one of: e, m, h, s");
    }
    return normalized;
  }

  private String normalizeResponseBody(String responseBody) {
    if (!properties.getIngestion().isStoreResponseBody() || responseBody == null) {
      return null;
    }
    int limit = Math.max(1, properties.getIngestion().getMaxResponseBodyChars());
    return responseBody.length() <= limit ? responseBody : responseBody.substring(0, limit);
  }

  private String firstNonBlank(String first, String second) {
    String normalizedFirst = trimToNull(first);
    if (normalizedFirst != null) {
      return normalizedFirst;
    }
    return trimToNull(second);
  }

  private String trimToNull(String value) {
    if (value == null) {
      return null;
    }
    String trimmed = value.trim();
    return trimmed.isEmpty() ? null : trimmed;
  }

  private String abbreviate(String value, int maxLen) {
    if (value == null || value.length() <= maxLen) {
      return value;
    }
    return value.substring(0, maxLen);
  }

  private boolean isRetryablePersistenceFailure(RuntimeException ex) {
    String message = ex.getMessage();
    if (message == null) {
      return false;
    }
    String normalized = message.toLowerCase(Locale.ROOT);
    return normalized.contains("deadlock")
        || normalized.contains("lock wait timeout")
        || normalized.contains("cannot acquire");
  }

  private void sleepQuietly(long millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while waiting to retry weathercom persistence", interrupted);
    }
  }

  private record DailyMaxKey(String requestLocationId,
                             String obsId,
                             LocalDate targetDateLocal) {
  }

  private record DailyMaxCandidate(Integer maxTempF,
                                   Long validTimeGmt,
                                   String sourceType) {
  }

  private record IngestionTask(String requestLocationId,
                               LocalDate startDate,
                               LocalDate endDate,
                               String units) {
  }
}
