package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.common.TimeSemantics;
import com.predictionmarkets.weather.iem.IemMosClient;
import com.predictionmarkets.weather.iem.IemMosEntry;
import com.predictionmarkets.weather.iem.IemMosPayload;
import com.predictionmarkets.weather.iem.IemMosValue;
import com.predictionmarkets.weather.models.AsofTimeZone;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.MosDailyValueUpsertRepository;
import com.predictionmarkets.weather.repository.MosRunUpsertRepository;
import com.predictionmarkets.weather.repository.MosRunUpsertRepository.UpsertRow;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.dao.DeadlockLoserDataAccessException;
import org.springframework.dao.DataAccessException;
import org.springframework.stereotype.Service;

@Service
public class MosRunIngestService {
  private static final Logger logger = LoggerFactory.getLogger(MosRunIngestService.class);
  private static final int MAX_DEADLOCK_RETRIES = 5;
  private static final long BASE_BACKOFF_MS = 250L;
  private static final int MEAN_SCALE = 4;

  private final IemMosClient mosClient;
  private final MosRunUpsertRepository upsertRepository;
  private final MosDailyValueUpsertRepository dailyValueRepository;
  private final StationRegistryRepository stationRegistryRepository;

  public MosRunIngestService(
      IemMosClient mosClient,
      MosRunUpsertRepository upsertRepository,
      MosDailyValueUpsertRepository dailyValueRepository,
      StationRegistryRepository stationRegistryRepository) {
    this.mosClient = mosClient;
    this.upsertRepository = upsertRepository;
    this.dailyValueRepository = dailyValueRepository;
    this.stationRegistryRepository = stationRegistryRepository;
  }

  public int ingestTargetDateAsOf(String stationId,
                                  MosModel model,
                                  LocalDate targetDateLocal,
                                  LocalTime asOfLocalTime,
                                  AsofTimeZone asOfTimeZone) {
    String normalizedStation = normalizeStationId(stationId);
    StationRegistry station = requireStation(normalizedStation);
    if (model == null) {
      throw new IllegalArgumentException("model is required");
    }
    if (targetDateLocal == null) {
      throw new IllegalArgumentException("targetDateLocal is required");
    }
    if (asOfLocalTime == null) {
      throw new IllegalArgumentException("asOfLocalTime is required");
    }
    ZoneId stationZone = ZoneId.of(station.getZoneId());
    ZoneId asOfZone = asOfTimeZone == AsofTimeZone.UTC ? ZoneOffset.UTC : stationZone;
    TimeSemantics.AsOfTimes asOfTimes = TimeSemantics.computeAsOfTimes(
        targetDateLocal, asOfLocalTime, stationZone, asOfZone);
    Instant asOfUtc = asOfTimes.asOfUtc();
    Instant windowStartUtc = targetDateLocal.atStartOfDay(stationZone).toInstant();
    Instant windowEndUtc = targetDateLocal.plusDays(1).atStartOfDay(stationZone).toInstant();
    Instant requestStartUtc = asOfUtc.isBefore(windowStartUtc) ? asOfUtc : windowStartUtc;
    IemMosPayload payload = mosClient.fetchWindow(normalizedStation, model, requestStartUtc, windowEndUtc);
    Instant retrievedAtUtc = Instant.now();
    Instant runtimeUtc = selectRuntimeUtc(payload.entries(), asOfUtc, windowStartUtc, windowEndUtc,
        stationZone, targetDateLocal);
    if (runtimeUtc == null) {
      logMissingRuntime(normalizedStation, model.name(), targetDateLocal, asOfUtc, payload.entries(),
          windowStartUtc, windowEndUtc, stationZone);
      return 0;
    }
    List<MosDailyValueUpsertRepository.UpsertRow> dailyRows =
        buildDailyRowsForRuntime(payload, station, model.name(), runtimeUtc, targetDateLocal,
            windowStartUtc, windowEndUtc, asOfUtc, retrievedAtUtc);
    if (dailyRows.isEmpty()) {
      logger.warn("No MOS rows matched runtime={} asofUtc={} station={} model={} targetDate={}",
          runtimeUtc, asOfUtc, normalizedStation, model.name(), targetDateLocal);
      return 0;
    }
    return withDeadlockRetry(() -> dailyValueRepository.upsertAll(dailyRows));
  }

  public int ingestWindow(String stationId, MosModel model, Instant startUtc, Instant endUtc) {
    String normalizedStation = normalizeStationId(stationId);
    StationRegistry station = requireStation(normalizedStation);
    if (model == null) {
      throw new IllegalArgumentException("model is required");
    }
    if (startUtc == null || endUtc == null) {
      throw new IllegalArgumentException("startUtc and endUtc are required");
    }
    if (!endUtc.isAfter(startUtc)) {
      throw new IllegalArgumentException("endUtc must be after startUtc");
    }
    IemMosPayload payload = mosClient.fetchWindow(normalizedStation, model, startUtc, endUtc);
    List<Instant> runtimes = uniqueRuntimes(payload.entries());
    if (runtimes.isEmpty()) {
      return 0;
    }
    return persistWithRetry(payload, normalizedStation, station.getZoneId(), model.name(), runtimes);
  }

  private StationRegistry requireStation(String stationId) {
    return stationRegistryRepository.findById(stationId)
        .orElseThrow(() -> new IllegalArgumentException(
            "Station not found in registry: " + stationId));
  }

  private List<Instant> uniqueRuntimes(List<IemMosEntry> entries) {
    TreeSet<Instant> runtimes = new TreeSet<>();
    for (IemMosEntry entry : entries) {
      Instant runtimeUtc = entry.runtimeUtc();
      if (runtimeUtc != null) {
        runtimes.add(runtimeUtc);
      }
    }
    return new ArrayList<>(runtimes);
  }

  private int persistWithRetry(IemMosPayload payload,
                               String stationId,
                               String stationZoneid,
                               String modelName,
                               List<Instant> runtimes) {
    Instant retrievedAtUtc = Instant.now();
    List<UpsertRow> runRows = buildRunRows(stationId, modelName, runtimes,
        payload.rawPayloadHash(), retrievedAtUtc);
    List<MosDailyValueUpsertRepository.UpsertRow> dailyRows = buildDailyRows(
        payload, stationId, stationZoneid, modelName, retrievedAtUtc);
    int runCount = 0;
    if (!runRows.isEmpty()) {
      runCount = withDeadlockRetry(() -> upsertRepository.upsertAll(runRows));
    }
    if (!dailyRows.isEmpty()) {
      withDeadlockRetry(() -> dailyValueRepository.upsertAll(dailyRows));
    }
    return runCount;
  }

  private List<UpsertRow> buildRunRows(String stationId,
                                       String modelName,
                                       List<Instant> runtimes,
                                       String rawPayloadHash,
                                       Instant retrievedAtUtc) {
    if (runtimes == null || runtimes.isEmpty()) {
      return List.of();
    }
    List<UpsertRow> rows = new ArrayList<>(runtimes.size());
    for (Instant runtimeUtc : runtimes) {
      rows.add(new UpsertRow(
          stationId,
          modelName,
          runtimeUtc,
          rawPayloadHash,
          retrievedAtUtc));
    }
    return rows;
  }

  private List<MosDailyValueUpsertRepository.UpsertRow> buildDailyRows(
      IemMosPayload payload,
      String stationId,
      String stationZoneid,
      String modelName,
      Instant retrievedAtUtc) {
    List<IemMosEntry> entries = payload.entries();
    if (entries == null || entries.isEmpty()) {
      return List.of();
    }
    ZoneId zoneId = ZoneId.of(stationZoneid);
    Map<MosDailyKey, SummaryStats> summaries = new HashMap<>();
    for (IemMosEntry entry : entries) {
      Instant forecastTime = entry.forecastTimeUtc();
      if (forecastTime == null) {
        continue;
      }
      Instant runtimeUtc = entry.runtimeUtc();
      if (runtimeUtc == null) {
        continue;
      }
      LocalDate targetDateLocal = ZonedDateTime.ofInstant(forecastTime, zoneId).toLocalDate();
      for (Map.Entry<String, IemMosValue> kv : entry.values().entrySet()) {
        String variableCode = kv.getKey();
        if (variableCode == null || variableCode.isBlank()) {
          continue;
        }
        IemMosValue value = kv.getValue();
        if (value == null) {
          continue;
        }
        BigDecimal numeric = value.numericValue();
        if (numeric == null) {
          continue;
        }
        MosDailyKey key = new MosDailyKey(runtimeUtc, targetDateLocal, variableCode);
        SummaryStats stats = summaries.computeIfAbsent(key, ignored -> new SummaryStats());
        stats.add(numeric, forecastTime);
      }
    }
    if (summaries.isEmpty()) {
      return List.of();
    }
    List<MosDailyValueUpsertRepository.UpsertRow> rows = new ArrayList<>(summaries.size());
    for (Map.Entry<MosDailyKey, SummaryStats> entry : summaries.entrySet()) {
      SummaryStats stats = entry.getValue();
      if (stats.count == 0) {
        continue;
      }
      BigDecimal mean = stats.sum.divide(BigDecimal.valueOf(stats.count), MEAN_SCALE,
          RoundingMode.HALF_UP);
      BigDecimal median = stats.median(MEAN_SCALE);
      MosDailyKey key = entry.getKey();
      rows.add(new MosDailyValueUpsertRepository.UpsertRow(
          stationId,
          stationZoneid,
          modelName,
          key.runtimeUtc(),
          key.runtimeUtc(),
          key.targetDateLocal(),
          key.variableCode(),
          stats.min,
          stats.max,
          mean,
          median,
          stats.count,
          stats.firstForecastTimeUtc,
          stats.lastForecastTimeUtc,
          payload.rawPayloadHash(),
          retrievedAtUtc));
    }
    if (!rows.isEmpty()) {
      rows.sort(Comparator
          .comparing(MosDailyValueUpsertRepository.UpsertRow::getRuntimeUtc)
          .thenComparing(MosDailyValueUpsertRepository.UpsertRow::getTargetDateLocal)
          .thenComparing(MosDailyValueUpsertRepository.UpsertRow::getVariableCode));
    }
    return rows;
  }

  private List<MosDailyValueUpsertRepository.UpsertRow> buildDailyRowsForRuntime(
      IemMosPayload payload,
      StationRegistry station,
      String modelName,
      Instant runtimeUtc,
      LocalDate targetDateLocal,
      Instant windowStartUtc,
      Instant windowEndUtc,
      Instant asofUtc,
      Instant retrievedAtUtc) {
    List<IemMosEntry> entries = payload.entries();
    if (entries == null || entries.isEmpty()) {
      return List.of();
    }
    ZoneId zoneId = ZoneId.of(station.getZoneId());
    Map<String, SummaryStats> summaries = new HashMap<>();
    for (IemMosEntry entry : entries) {
      Instant forecastTime = entry.forecastTimeUtc();
      if (forecastTime == null
          || forecastTime.isBefore(windowStartUtc)
          || !forecastTime.isBefore(windowEndUtc)) {
        continue;
      }
      if (!runtimeUtc.equals(entry.runtimeUtc())) {
        continue;
      }
      LocalDate entryTarget = ZonedDateTime.ofInstant(forecastTime, zoneId).toLocalDate();
      if (!entryTarget.equals(targetDateLocal)) {
        continue;
      }
      for (Map.Entry<String, IemMosValue> kv : entry.values().entrySet()) {
        String variableCode = kv.getKey();
        if (variableCode == null || variableCode.isBlank()) {
          continue;
        }
        IemMosValue value = kv.getValue();
        if (value == null) {
          continue;
        }
        BigDecimal numeric = value.numericValue();
        if (numeric == null) {
          continue;
        }
        SummaryStats stats = summaries.computeIfAbsent(variableCode, ignored -> new SummaryStats());
        stats.add(numeric, forecastTime);
      }
    }
    if (summaries.isEmpty()) {
      return List.of();
    }
    List<MosDailyValueUpsertRepository.UpsertRow> rows = new ArrayList<>(summaries.size());
    for (Map.Entry<String, SummaryStats> entry : summaries.entrySet()) {
      SummaryStats stats = entry.getValue();
      if (stats.count == 0) {
        continue;
      }
      BigDecimal mean = stats.sum.divide(BigDecimal.valueOf(stats.count), MEAN_SCALE,
          RoundingMode.HALF_UP);
      BigDecimal median = stats.median(MEAN_SCALE);
      rows.add(new MosDailyValueUpsertRepository.UpsertRow(
          station.getStationId(),
          station.getZoneId(),
          modelName,
          asofUtc,
          runtimeUtc,
          targetDateLocal,
          entry.getKey(),
          stats.min,
          stats.max,
          mean,
          median,
          stats.count,
          stats.firstForecastTimeUtc,
          stats.lastForecastTimeUtc,
          payload.rawPayloadHash(),
          retrievedAtUtc));
    }
    if (!rows.isEmpty()) {
      rows.sort(Comparator
          .comparing(MosDailyValueUpsertRepository.UpsertRow::getRuntimeUtc)
          .thenComparing(MosDailyValueUpsertRepository.UpsertRow::getTargetDateLocal)
          .thenComparing(MosDailyValueUpsertRepository.UpsertRow::getVariableCode));
    }
    return rows;
  }

  private Instant selectRuntimeUtc(List<IemMosEntry> entries,
                                   Instant asOfUtc,
                                   Instant windowStartUtc,
                                   Instant windowEndUtc,
                                   ZoneId stationZone,
                                   LocalDate targetDateLocal) {
    if (entries == null || entries.isEmpty() || asOfUtc == null) {
      return null;
    }
    TreeSet<Instant> candidates = new TreeSet<>();
    TreeSet<Instant> covered = new TreeSet<>();
    for (IemMosEntry entry : entries) {
      Instant runtimeUtc = entry.runtimeUtc();
      if (runtimeUtc == null || runtimeUtc.isAfter(asOfUtc)) {
        continue;
      }
      candidates.add(runtimeUtc);
      Instant forecastTime = entry.forecastTimeUtc();
      if (forecastTime == null
          || forecastTime.isBefore(windowStartUtc)
          || !forecastTime.isBefore(windowEndUtc)) {
        continue;
      }
      LocalDate entryTarget = ZonedDateTime.ofInstant(forecastTime, stationZone).toLocalDate();
      if (entryTarget.equals(targetDateLocal)) {
        covered.add(runtimeUtc);
      }
    }
    return covered.isEmpty() ? null : covered.last();
  }

  private void logMissingRuntime(String stationId,
                                 String modelName,
                                 LocalDate targetDateLocal,
                                 Instant asOfUtc,
                                 List<IemMosEntry> entries,
                                 Instant windowStartUtc,
                                 Instant windowEndUtc,
                                 ZoneId stationZone) {
    if (entries == null || entries.isEmpty()) {
      logger.warn("MOS payload empty station={} model={} targetDate={} asofUtc={}",
          stationId, modelName, targetDateLocal, asOfUtc);
      return;
    }
    Instant min = null;
    Instant max = null;
    int runtimeCount = 0;
    int targetCoverage = 0;
    for (IemMosEntry entry : entries) {
      Instant runtimeUtc = entry.runtimeUtc();
      if (runtimeUtc == null) {
        continue;
      }
      runtimeCount++;
      if (min == null || runtimeUtc.isBefore(min)) {
        min = runtimeUtc;
      }
      if (max == null || runtimeUtc.isAfter(max)) {
        max = runtimeUtc;
      }
      Instant forecastTime = entry.forecastTimeUtc();
      if (forecastTime == null
          || forecastTime.isBefore(windowStartUtc)
          || !forecastTime.isBefore(windowEndUtc)) {
        continue;
      }
      LocalDate entryTarget = ZonedDateTime.ofInstant(forecastTime, stationZone).toLocalDate();
      if (entryTarget.equals(targetDateLocal)) {
        targetCoverage += 1;
      }
    }
    logger.warn(
        "No eligible MOS runtime <= asofUtc with target coverage station={} model={} targetDate={} asofUtc={} "
            + "runtimes={} minRuntime={} maxRuntime={} targetCoverageRows={}",
        stationId, modelName, targetDateLocal, asOfUtc, runtimeCount, min, max, targetCoverage);
  }

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private boolean isRetryableDeadlock(DataAccessException ex) {
    if (ex instanceof CannotAcquireLockException || ex instanceof DeadlockLoserDataAccessException) {
      return true;
    }
    Throwable cause = ex;
    while (cause != null) {
      String message = cause.getMessage();
      if (message != null && message.contains("Deadlock found")) {
        return true;
      }
      cause = cause.getCause();
    }
    return false;
  }

  private void backoff(int attempt) {
    long delayMs = BASE_BACKOFF_MS * (1L << attempt);
    delayMs += ThreadLocalRandom.current().nextLong(BASE_BACKOFF_MS);
    try {
      Thread.sleep(delayMs);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private <T> T withDeadlockRetry(Supplier<T> action) {
    int attempt = 0;
    while (true) {
      try {
        return action.get();
      } catch (DataAccessException ex) {
        if (!isRetryableDeadlock(ex) || attempt >= MAX_DEADLOCK_RETRIES) {
          throw ex;
        }
        backoff(attempt);
        attempt += 1;
      }
    }
  }

  private record MosDailyKey(Instant runtimeUtc,
                             LocalDate targetDateLocal,
                             String variableCode) {
    MosDailyKey {
      Objects.requireNonNull(runtimeUtc, "runtimeUtc");
      Objects.requireNonNull(targetDateLocal, "targetDateLocal");
      Objects.requireNonNull(variableCode, "variableCode");
    }
  }

  private static final class SummaryStats {
    private BigDecimal sum = BigDecimal.ZERO;
    private BigDecimal min;
    private BigDecimal max;
    private int count;
    private Instant firstForecastTimeUtc;
    private Instant lastForecastTimeUtc;
    private final List<BigDecimal> values = new ArrayList<>();

    private void add(BigDecimal value, Instant forecastTimeUtc) {
      if (value == null || forecastTimeUtc == null) {
        return;
      }
      if (count == 0) {
        min = value;
        max = value;
        firstForecastTimeUtc = forecastTimeUtc;
        lastForecastTimeUtc = forecastTimeUtc;
      } else {
        if (value.compareTo(min) < 0) {
          min = value;
        }
        if (value.compareTo(max) > 0) {
          max = value;
        }
        if (forecastTimeUtc.isBefore(firstForecastTimeUtc)) {
          firstForecastTimeUtc = forecastTimeUtc;
        }
        if (forecastTimeUtc.isAfter(lastForecastTimeUtc)) {
          lastForecastTimeUtc = forecastTimeUtc;
        }
      }
      sum = sum.add(value);
      count += 1;
      values.add(value);
    }

    private BigDecimal median(int scale) {
      if (values.isEmpty()) {
        return null;
      }
      List<BigDecimal> sorted = new ArrayList<>(values);
      sorted.sort(Comparator.naturalOrder());
      int size = sorted.size();
      int mid = size / 2;
      if (size % 2 == 1) {
        return sorted.get(mid);
      }
      BigDecimal lo = sorted.get(mid - 1);
      BigDecimal hi = sorted.get(mid);
      return lo.add(hi).divide(BigDecimal.valueOf(2), scale, RoundingMode.HALF_UP);
    }
  }
}
