package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.common.StandardTimeClimateWindow;
import com.predictionmarkets.weather.executors.GribstreamTaskExecutor;
import com.predictionmarkets.weather.models.GribstreamMemberStat;
import com.predictionmarkets.weather.models.GribstreamVariableReducer;
import com.predictionmarkets.weather.repository.GribstreamDailyVariableValueUpsertRepository;
import com.predictionmarkets.weather.repository.GribstreamForecastValueUpsertRepository;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class GribstreamVariableIngestService {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamVariableIngestService.class);
  private static final Set<String> ENSEMBLE_MODELS = Set.of("gefsatmos");
  private static final int ALIAS_LIMIT = 64;
  private static final int NAME_LIMIT = 32;
  private static final int LEVEL_LIMIT = 64;
  private static final int INFO_LIMIT = 64;
  private static final int TEXT_LIMIT = 128;
  private static final double ACCUMULATOR_EPS = 1e-6;
  private static final Map<String, Integer> EXPECTED_STEPS_BY_MODEL = Map.of(
      "hrrr", 24,
      "rap", 24,
      "nbm", 24,
      "gefsatmos", 8,
      "gefsatmosmean", 8);
  private static final Set<String> ACCUMULATOR_NAMES = Set.of(
      "APCP", "ACPCP", "ASNOW", "BGRUN", "SSRUN");
  private static final Set<String> LAST_ONLY_NAMES = Set.of(
      "TMAX", "TMIN", "SNOD", "WEASD", "SOILW", "TSOIL", "ICETK", "SNOWC", "MAXRH", "MINRH");
  private static final Set<String> MIN_ONLY_NAMES = Set.of(
      "CIN", "LFTX", "4LFTX", "VIS", "CEIL", "DEPR");
  private static final Set<String> MAX_ONLY_NAMES = Set.of(
      "CAPE", "GUST", "PRATE", "HINDEX", "DSWRF", "USWRF", "SHTFL", "LHTFL");
  private static final Set<String> DIRECTION_NAMES = Set.of("WDIR");
  private static final Set<String> CATEGORICAL_NAMES = Set.of("PTYPE");

  private final GribstreamClient client;
  private final GribstreamProperties gribstreamProperties;
  private final GribstreamVariableIngestProperties ingestProperties;
  private final GribstreamVariableCatalogLoader catalogLoader;
  private final GribstreamVariableWhitelistLoader whitelistLoader;
  private final GribstreamForecastValueUpsertRepository rawRepository;
  private final GribstreamDailyVariableValueUpsertRepository summaryRepository;
  private final GribstreamTaskExecutor taskExecutor;
  private final ObjectMapper objectMapper;

  private final Object batchLock = new Object();
  private volatile Map<String, List<VariableBatch>> cachedBatches;
  private volatile Map<String, List<GribstreamVariableWhitelistEntry>> cachedWhitelist;

  public GribstreamVariableIngestService(GribstreamClient client,
                                         GribstreamProperties gribstreamProperties,
                                         GribstreamVariableIngestProperties ingestProperties,
                                         GribstreamVariableCatalogLoader catalogLoader,
                                         GribstreamVariableWhitelistLoader whitelistLoader,
                                         GribstreamForecastValueUpsertRepository rawRepository,
                                         GribstreamDailyVariableValueUpsertRepository summaryRepository,
                                         GribstreamTaskExecutor taskExecutor,
                                         ObjectMapper objectMapper) {
    this.client = client;
    this.gribstreamProperties = gribstreamProperties;
    this.ingestProperties = ingestProperties;
    this.catalogLoader = catalogLoader;
    this.whitelistLoader = whitelistLoader;
    this.rawRepository = rawRepository;
    this.summaryRepository = summaryRepository;
    this.taskExecutor = taskExecutor;
    this.objectMapper = objectMapper;
  }

  public int ingestForDate(StationSpec station, LocalDate targetDateLocal, Instant asOfUtc) {
    Objects.requireNonNull(station, "station is required");
    Objects.requireNonNull(targetDateLocal, "targetDateLocal is required");
    Objects.requireNonNull(asOfUtc, "asOfUtc is required");
    Map<String, List<VariableBatch>> modelBatches = loadBatches();
    if (modelBatches.isEmpty()) {
      logger.warn("[GRIBSTREAM-VARS] No variable catalog loaded. Skipping.");
      return 0;
    }
    ZoneId zoneId = ZoneId.of(station.zoneId());
    StandardTimeClimateWindow.UtcRange range =
        StandardTimeClimateWindow.computeUtcRange(zoneId, targetDateLocal);
    List<Callable<Integer>> tasks = new ArrayList<>();
    int minHorizon = Math.max(0, gribstreamProperties.getDefaultMinHorizonHours());
    for (Map.Entry<String, List<VariableBatch>> entry : modelBatches.entrySet()) {
      String modelCode = entry.getKey();
      int maxHorizon = resolveMaxHorizon(modelCode);
      for (VariableBatch batch : entry.getValue()) {
        tasks.add(() -> ingestBatch(station, targetDateLocal, asOfUtc, range,
            modelCode, minHorizon, maxHorizon, batch));
      }
    }
    if (tasks.isEmpty()) {
      return 0;
    }
    List<Integer> results = taskExecutor.invokeAllOrFail(tasks);
    return results.stream().mapToInt(Integer::intValue).sum();
  }

  private int ingestBatch(StationSpec station,
                          LocalDate targetDateLocal,
                          Instant asOfUtc,
                          StandardTimeClimateWindow.UtcRange range,
                          String modelCode,
                          int minHorizon,
                          int maxHorizon,
                          VariableBatch batch) {
    List<Integer> members = resolveMembers(modelCode);
    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        range.startUtc().toString(),
        range.endUtc().toString(),
        asOfUtc.toString(),
        minHorizon,
        maxHorizon,
        List.of(new GribstreamCoordinate(station.lat(), station.lon(), station.name())),
        batch.variables(),
        members);
    GribstreamRawResponse response;
    try {
      response = client.fetchHistoryRaw(modelCode, request);
    } catch (GribstreamEmptyResponseException ex) {
      logger.warn("[GRIBSTREAM-VARS] model={} station={} date={} status=empty_response",
          modelCode, station.stationId(), targetDateLocal);
      return 0;
    }
    List<GribstreamValueRow> rows = GribstreamGenericResponseParser.parseRows(
        objectMapper,
        response.responseBytes(),
        modelCode,
        response.requestSha256());
    if (rows.isEmpty()) {
      return 0;
    }
    int sum = 0;
    if (ingestProperties.isStoreRaw()) {
      sum += persistRawValues(station, targetDateLocal, asOfUtc, modelCode, batch, response, rows);
    }
    if (ingestProperties.isStoreSummary()) {
      sum += persistSummaryValues(station, targetDateLocal, asOfUtc, range, modelCode, batch, response, rows);
    }
    if (!ingestProperties.isStoreRaw() && !ingestProperties.isStoreSummary()) {
      logger.warn("[GRIBSTREAM-VARS] model={} station={} date={} status=skipped reason=storage_disabled",
          modelCode, station.stationId(), targetDateLocal);
    }
    return sum;
  }

  private int persistRawValues(StationSpec station,
                               LocalDate targetDateLocal,
                               Instant asOfUtc,
                               String modelCode,
                               VariableBatch batch,
                               GribstreamRawResponse response,
                               List<GribstreamValueRow> rows) {
    List<GribstreamForecastValueUpsertRepository.UpsertRow> upsertRows = new ArrayList<>();
    for (GribstreamValueRow row : rows) {
      if (row.forecastedAt() == null || row.forecastedTime() == null) {
        continue;
      }
      enforceNoLeakage(row.forecastedAt(), asOfUtc, modelCode);
      Integer member = row.member();
      Map<String, String> values = row.values();
      if (values == null || values.isEmpty()) {
        continue;
      }
      for (Map.Entry<String, String> valueEntry : values.entrySet()) {
        String alias = normalizeAlias(valueEntry.getKey());
        if (alias == null) {
          continue;
        }
        GribstreamVariableSpec spec = batch.aliasToSpec().get(alias);
        if (spec == null) {
          continue;
        }
        String raw = valueEntry.getValue();
        if (raw == null || raw.isBlank()) {
          continue;
        }
        Double valueNum = parseDouble(raw);
        String valueText = valueNum == null ? truncate(raw.trim(), TEXT_LIMIT) : null;
        upsertRows.add(new GribstreamForecastValueUpsertRepository.UpsertRow(
            station.stationId(),
            station.zoneId(),
            modelCode,
            asOfUtc,
            row.forecastedAt(),
            row.forecastedTime(),
            member,
            truncate(spec.name(), NAME_LIMIT),
            truncate(spec.level(), LEVEL_LIMIT),
            truncate(spec.info(), INFO_LIMIT),
            truncate(alias, ALIAS_LIMIT),
            valueNum,
            valueText,
            response.requestJson(),
            response.requestSha256(),
            response.responseSha256(),
            response.retrievedAtUtc(),
            null));
      }
    }
    if (upsertRows.isEmpty()) {
      return 0;
    }
    int[] results = rawRepository.upsertAll(upsertRows);
    int sum = 0;
    for (int result : results) {
      sum += result;
    }
    logger.info("[GRIBSTREAM-VARS] model={} station={} date={} batchSize={} rawRowsUpserted={}",
        modelCode, station.stationId(), targetDateLocal, batch.variables().size(), sum);
    return sum;
  }

  private int persistSummaryValues(StationSpec station,
                                   LocalDate targetDateLocal,
                                   Instant asOfUtc,
                                   StandardTimeClimateWindow.UtcRange range,
                                   String modelCode,
                                   VariableBatch batch,
                                   GribstreamRawResponse response,
                                   List<GribstreamValueRow> rows) {
    Map<SeriesKey, List<Sample>> series =
        buildSeries(rows, batch, asOfUtc, range, modelCode);
    if (series.isEmpty()) {
      return 0;
    }
    int expectedSteps = resolveExpectedSteps(modelCode);
    boolean isEnsemble = ENSEMBLE_MODELS.contains(modelCode);
    List<GribstreamDailyVariableValueUpsertRepository.UpsertRow> upsertRows = new ArrayList<>();
    if (isEnsemble && ingestProperties.isCollapseEnsembleMembers()) {
      int expectedMembers = resolveExpectedMembers(modelCode);
      addEnsembleSummaryRows(station, targetDateLocal, asOfUtc, range, modelCode,
          batch, response, series, expectedSteps, expectedMembers, upsertRows);
    } else {
      addSeriesSummaryRows(station, targetDateLocal, asOfUtc, range, modelCode,
          batch, response, series, expectedSteps, isEnsemble, upsertRows);
    }
    if (upsertRows.isEmpty()) {
      return 0;
    }
    int[] results = summaryRepository.upsertAll(upsertRows);
    int sum = 0;
    for (int result : results) {
      sum += result;
    }
    logger.info("[GRIBSTREAM-VARS] model={} station={} date={} batchSize={} summaryRowsUpserted={}",
        modelCode, station.stationId(), targetDateLocal, batch.variables().size(), sum);
    return sum;
  }

  private Map<SeriesKey, List<Sample>> buildSeries(List<GribstreamValueRow> rows,
                                                   VariableBatch batch,
                                                   Instant asOfUtc,
                                                   StandardTimeClimateWindow.UtcRange range,
                                                   String modelCode) {
    Map<SeriesKey, List<Sample>> series = new HashMap<>();
    for (GribstreamValueRow row : rows) {
      if (row.forecastedAt() == null || row.forecastedTime() == null) {
        continue;
      }
      if (row.forecastedTime().isBefore(range.startUtc())
          || !row.forecastedTime().isBefore(range.endUtc())) {
        continue;
      }
      enforceNoLeakage(row.forecastedAt(), asOfUtc, modelCode);
      Integer member = row.member();
      if (ENSEMBLE_MODELS.contains(modelCode) && member == null) {
        throw new IllegalStateException("Missing GEFS member id for model " + modelCode);
      }
      int memberId = member == null ? 0 : member;
      Map<String, String> values = row.values();
      if (values == null || values.isEmpty()) {
        continue;
      }
      for (Map.Entry<String, String> valueEntry : values.entrySet()) {
        String alias = normalizeAlias(valueEntry.getKey());
        if (alias == null) {
          continue;
        }
        GribstreamVariableSpec spec = batch.aliasToSpec().get(alias);
        if (spec == null) {
          continue;
        }
        String raw = valueEntry.getValue();
        if (raw == null || raw.isBlank()) {
          continue;
        }
        String trimmed = raw.trim();
        Double valueNum = parseDouble(trimmed);
        Sample sample = new Sample(row.forecastedTime(), trimmed, valueNum);
        series.computeIfAbsent(new SeriesKey(alias, memberId), ignored -> new ArrayList<>())
            .add(sample);
      }
    }
    return series;
  }

  private void addSeriesSummaryRows(StationSpec station,
                                    LocalDate targetDateLocal,
                                    Instant asOfUtc,
                                    StandardTimeClimateWindow.UtcRange range,
                                    String modelCode,
                                    VariableBatch batch,
                                    GribstreamRawResponse response,
                                    Map<SeriesKey, List<Sample>> series,
                                    int expectedSteps,
                                    boolean perMember,
                                    List<GribstreamDailyVariableValueUpsertRepository.UpsertRow> output) {
    for (Map.Entry<SeriesKey, List<Sample>> entry : series.entrySet()) {
      SeriesKey key = entry.getKey();
      GribstreamVariableSpec spec = batch.aliasToSpec().get(key.alias());
      if (spec == null) {
        continue;
      }
      List<Sample> normalized = normalizeSamples(entry.getValue());
      if (normalized.isEmpty()) {
        continue;
      }
      List<GribstreamVariableReducer> reducers = resolveReducers(spec);
      if (reducers.isEmpty()) {
        continue;
      }
      GribstreamMemberStat memberStat = perMember ? GribstreamMemberStat.MEMBER : GribstreamMemberStat.NONE;
      int memberValue = perMember ? key.member() : 0;
      for (GribstreamVariableReducer reducer : reducers) {
        AggregationResult result = computeReducerResult(reducer, normalized, range);
        if (result == null) {
          continue;
        }
        if (requiresCoverage(spec, reducer) && !hasCoverage(result.sampleCount(), expectedSteps)) {
          continue;
        }
        output.add(new GribstreamDailyVariableValueUpsertRepository.UpsertRow(
            station.stationId(),
            station.zoneId(),
            modelCode,
            targetDateLocal,
            asOfUtc,
            memberStat.name(),
            memberValue,
            truncate(spec.name(), NAME_LIMIT),
            truncate(spec.level(), LEVEL_LIMIT),
            truncate(spec.info(), INFO_LIMIT),
            truncate(key.alias(), ALIAS_LIMIT),
            reducer.name(),
            result.valueNum(),
            result.valueText(),
            result.sampleCount(),
            expectedSteps > 0 ? expectedSteps : null,
            range.startUtc(),
            range.endUtc(),
            response.requestJson(),
            response.requestSha256(),
            response.responseSha256(),
            response.retrievedAtUtc(),
            null));
      }
    }
  }

  private void addEnsembleSummaryRows(StationSpec station,
                                      LocalDate targetDateLocal,
                                      Instant asOfUtc,
                                      StandardTimeClimateWindow.UtcRange range,
                                      String modelCode,
                                      VariableBatch batch,
                                      GribstreamRawResponse response,
                                      Map<SeriesKey, List<Sample>> series,
                                      int expectedSteps,
                                      int expectedMembers,
                                      List<GribstreamDailyVariableValueUpsertRepository.UpsertRow> output) {
    Map<String, Map<Integer, List<Sample>>> byAlias = new HashMap<>();
    for (Map.Entry<SeriesKey, List<Sample>> entry : series.entrySet()) {
      byAlias.computeIfAbsent(entry.getKey().alias(), ignored -> new HashMap<>())
          .put(entry.getKey().member(), entry.getValue());
    }
    for (Map.Entry<String, Map<Integer, List<Sample>>> entry : byAlias.entrySet()) {
      String alias = entry.getKey();
      GribstreamVariableSpec spec = batch.aliasToSpec().get(alias);
      if (spec == null) {
        continue;
      }
      List<GribstreamVariableReducer> reducers = resolveReducers(spec);
      if (reducers.isEmpty()) {
        continue;
      }
      for (GribstreamVariableReducer reducer : reducers) {
        if (reducer == GribstreamVariableReducer.MODE_TEXT) {
          continue;
        }
        List<Double> memberValues = new ArrayList<>();
        for (Map.Entry<Integer, List<Sample>> memberEntry : entry.getValue().entrySet()) {
          List<Sample> normalized = normalizeSamples(memberEntry.getValue());
          if (normalized.isEmpty()) {
            continue;
          }
          AggregationResult result = computeReducerResult(reducer, normalized, range);
          if (result == null) {
            continue;
          }
          if (requiresCoverage(spec, reducer) && !hasCoverage(result.sampleCount(), expectedSteps)) {
            continue;
          }
          if (result.valueNum() != null) {
            memberValues.add(result.valueNum());
          }
        }
        if (memberValues.isEmpty()) {
          continue;
        }
        output.addAll(buildEnsembleRows(
            station,
            targetDateLocal,
            asOfUtc,
            range,
            modelCode,
            spec,
            alias,
            reducer,
            memberValues,
            expectedMembers,
            response));
      }
    }
  }

  private List<GribstreamDailyVariableValueUpsertRepository.UpsertRow> buildEnsembleRows(
      StationSpec station,
      LocalDate targetDateLocal,
      Instant asOfUtc,
      StandardTimeClimateWindow.UtcRange range,
      String modelCode,
      GribstreamVariableSpec spec,
      String alias,
      GribstreamVariableReducer reducer,
      List<Double> memberValues,
      int expectedMembers,
      GribstreamRawResponse response) {
    List<GribstreamDailyVariableValueUpsertRepository.UpsertRow> rows = new ArrayList<>();
    double mean = mean(memberValues);
    rows.add(buildSummaryRow(
        station,
        targetDateLocal,
        asOfUtc,
        range,
        modelCode,
        spec,
        alias,
        reducer,
        GribstreamMemberStat.ENS_MEAN,
        0,
        mean,
        null,
        memberValues.size(),
        expectedMembers,
        response));
    if (memberValues.size() >= 2) {
      double std = stddev(memberValues, mean);
      rows.add(buildSummaryRow(
          station,
          targetDateLocal,
          asOfUtc,
          range,
          modelCode,
          spec,
          alias,
          reducer,
          GribstreamMemberStat.ENS_STD,
          0,
          std,
          null,
          memberValues.size(),
          expectedMembers,
          response));
    }
    if (memberValues.size() >= 3) {
      double p10 = quantile(memberValues, 0.10);
      double p90 = quantile(memberValues, 0.90);
      rows.add(buildSummaryRow(
          station,
          targetDateLocal,
          asOfUtc,
          range,
          modelCode,
          spec,
          alias,
          reducer,
          GribstreamMemberStat.ENS_P10,
          0,
          p10,
          null,
          memberValues.size(),
          expectedMembers,
          response));
      rows.add(buildSummaryRow(
          station,
          targetDateLocal,
          asOfUtc,
          range,
          modelCode,
          spec,
          alias,
          reducer,
          GribstreamMemberStat.ENS_P90,
          0,
          p90,
          null,
          memberValues.size(),
          expectedMembers,
          response));
    }
    return rows;
  }

  private GribstreamDailyVariableValueUpsertRepository.UpsertRow buildSummaryRow(
      StationSpec station,
      LocalDate targetDateLocal,
      Instant asOfUtc,
      StandardTimeClimateWindow.UtcRange range,
      String modelCode,
      GribstreamVariableSpec spec,
      String alias,
      GribstreamVariableReducer reducer,
      GribstreamMemberStat memberStat,
      int member,
      Double valueNum,
      String valueText,
      Integer sampleCount,
      Integer expectedCount,
      GribstreamRawResponse response) {
    return new GribstreamDailyVariableValueUpsertRepository.UpsertRow(
        station.stationId(),
        station.zoneId(),
        modelCode,
        targetDateLocal,
        asOfUtc,
        memberStat.name(),
        member,
        truncate(spec.name(), NAME_LIMIT),
        truncate(spec.level(), LEVEL_LIMIT),
        truncate(spec.info(), INFO_LIMIT),
        truncate(alias, ALIAS_LIMIT),
        reducer.name(),
        valueNum,
        valueText,
        sampleCount,
        expectedCount,
        range.startUtc(),
        range.endUtc(),
        response.requestJson(),
        response.requestSha256(),
        response.responseSha256(),
        response.retrievedAtUtc(),
        null);
  }

  private AggregationResult computeReducerResult(GribstreamVariableReducer reducer,
                                                 List<Sample> samples,
                                                 StandardTimeClimateWindow.UtcRange range) {
    if (reducer == GribstreamVariableReducer.MODE_TEXT) {
      String mode = modeText(samples);
      if (mode == null) {
        return null;
      }
      return new AggregationResult(null, truncate(mode, TEXT_LIMIT), countText(samples));
    }
    List<Sample> numeric = filterNumeric(samples);
    if (numeric.isEmpty()) {
      return null;
    }
    Double value = switch (reducer) {
      case MAX -> max(numeric);
      case MIN -> min(numeric);
      case MEDIAN -> median(numeric);
      case LAST -> last(numeric);
      case TOTAL -> totalAccumulatorOrInterval(numeric);
      case MEAN_TW -> meanTimeWeighted(numeric, range.startUtc(), range.endUtc());
      case CIRCMEAN_DEG -> circularMeanDegrees(numeric);
      default -> null;
    };
    if (value == null || !Double.isFinite(value)) {
      return null;
    }
    return new AggregationResult(value, null, numeric.size());
  }

  private List<GribstreamVariableReducer> resolveReducers(GribstreamVariableSpec spec) {
    String nameKey = normalizeVariableName(spec.name());
    String infoKey = normalizeInfo(spec.info());
    if (infoKey.contains("prob")) {
      return List.of(GribstreamVariableReducer.MAX);
    }
    if (infoKey.contains("% level")) {
      return List.of(GribstreamVariableReducer.LAST);
    }
    if (CATEGORICAL_NAMES.contains(nameKey)) {
      return List.of(GribstreamVariableReducer.MODE_TEXT);
    }
    if (DIRECTION_NAMES.contains(nameKey)) {
      return List.of(GribstreamVariableReducer.CIRCMEAN_DEG);
    }
    if (ACCUMULATOR_NAMES.contains(nameKey)) {
      return List.of(GribstreamVariableReducer.TOTAL);
    }
    if (LAST_ONLY_NAMES.contains(nameKey)) {
      return List.of(GribstreamVariableReducer.LAST);
    }
    if (MIN_ONLY_NAMES.contains(nameKey)) {
      return List.of(GribstreamVariableReducer.MIN);
    }
    if (MAX_ONLY_NAMES.contains(nameKey)) {
      return List.of(GribstreamVariableReducer.MAX);
    }
    return List.of(GribstreamVariableReducer.MEAN_TW, GribstreamVariableReducer.MAX);
  }

  private List<Sample> normalizeSamples(List<Sample> samples) {
    if (samples == null || samples.isEmpty()) {
      return List.of();
    }
    TreeMap<Instant, Sample> byTime = new TreeMap<>();
    for (Sample sample : samples) {
      if (sample.time() == null) {
        continue;
      }
      Sample existing = byTime.get(sample.time());
      byTime.put(sample.time(), mergeSample(existing, sample));
    }
    return new ArrayList<>(byTime.values());
  }

  private Sample mergeSample(Sample existing, Sample candidate) {
    if (existing == null) {
      return candidate;
    }
    Double existingNum = existing.num();
    Double candidateNum = candidate.num();
    if (existingNum != null && candidateNum != null) {
      if (candidateNum > existingNum) {
        return candidate;
      }
      if (candidateNum < existingNum) {
        return existing;
      }
    }
    String existingRaw = existing.raw();
    String candidateRaw = candidate.raw();
    if (existingRaw == null) {
      return candidate;
    }
    if (candidateRaw == null) {
      return existing;
    }
    return candidateRaw.compareTo(existingRaw) >= 0 ? candidate : existing;
  }

  private boolean hasCoverage(int sampleCount, int expectedCount) {
    if (expectedCount <= 0) {
      return true;
    }
    double ratio = sampleCount / (double) expectedCount;
    return ratio >= ingestProperties.getMinCoverageRatio();
  }

  private boolean requiresCoverage(GribstreamVariableSpec spec, GribstreamVariableReducer reducer) {
    if (reducer == GribstreamVariableReducer.LAST || reducer == GribstreamVariableReducer.MODE_TEXT) {
      return false;
    }
    String infoKey = normalizeInfo(spec.info());
    return !infoKey.contains("% level");
  }

  private int resolveExpectedSteps(String modelCode) {
    Integer steps = EXPECTED_STEPS_BY_MODEL.get(modelCode);
    return steps == null ? 0 : steps;
  }

  private int resolveExpectedMembers(String modelCode) {
    if (!ENSEMBLE_MODELS.contains(modelCode)) {
      return 0;
    }
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    return members == null ? 0 : members.size();
  }

  private String normalizeVariableName(String name) {
    if (name == null) {
      return "";
    }
    return name.trim().toUpperCase(Locale.ROOT);
  }

  private String normalizeInfo(String info) {
    if (info == null) {
      return "";
    }
    return info.trim().toLowerCase(Locale.ROOT);
  }

  private List<Sample> filterNumeric(List<Sample> samples) {
    List<Sample> numeric = new ArrayList<>(samples.size());
    for (Sample sample : samples) {
      if (sample.num() != null && Double.isFinite(sample.num())) {
        numeric.add(sample);
      }
    }
    numeric.sort(Comparator.comparing(Sample::time));
    return numeric;
  }

  private int countText(List<Sample> samples) {
    int count = 0;
    for (Sample sample : samples) {
      if (sample.raw() != null && !sample.raw().isBlank()) {
        count++;
      }
    }
    return count;
  }

  private Double max(List<Sample> samples) {
    Double max = null;
    for (Sample sample : samples) {
      Double value = sample.num();
      if (value == null) {
        continue;
      }
      if (max == null || value > max) {
        max = value;
      }
    }
    return max;
  }

  private Double min(List<Sample> samples) {
    Double min = null;
    for (Sample sample : samples) {
      Double value = sample.num();
      if (value == null) {
        continue;
      }
      if (min == null || value < min) {
        min = value;
      }
    }
    return min;
  }

  private Double last(List<Sample> samples) {
    if (samples.isEmpty()) {
      return null;
    }
    return samples.get(samples.size() - 1).num();
  }

  private Double median(List<Sample> samples) {
    if (samples.isEmpty()) {
      return null;
    }
    List<Double> values = new ArrayList<>(samples.size());
    for (Sample sample : samples) {
      if (sample.num() != null) {
        values.add(sample.num());
      }
    }
    if (values.isEmpty()) {
      return null;
    }
    Collections.sort(values);
    int size = values.size();
    int mid = size / 2;
    if (size % 2 == 1) {
      return values.get(mid);
    }
    return (values.get(mid - 1) + values.get(mid)) / 2.0;
  }

  private Double meanTimeWeighted(List<Sample> samples, Instant windowStart, Instant windowEnd) {
    if (samples.isEmpty()) {
      return null;
    }
    double total = 0.0;
    double weight = 0.0;
    for (int i = 0; i < samples.size(); i++) {
      Sample current = samples.get(i);
      Instant start = current.time().isBefore(windowStart) ? windowStart : current.time();
      Instant end = windowEnd;
      if (i + 1 < samples.size()) {
        Instant next = samples.get(i + 1).time();
        end = next.isBefore(windowEnd) ? next : windowEnd;
      }
      if (!end.isAfter(start)) {
        continue;
      }
      double seconds = Duration.between(start, end).toSeconds();
      total += current.num() * seconds;
      weight += seconds;
    }
    if (weight <= 0.0) {
      return null;
    }
    return total / weight;
  }

  private Double circularMeanDegrees(List<Sample> samples) {
    if (samples.isEmpty()) {
      return null;
    }
    double sumSin = 0.0;
    double sumCos = 0.0;
    int count = 0;
    for (Sample sample : samples) {
      Double value = sample.num();
      if (value == null) {
        continue;
      }
      double radians = Math.toRadians(value);
      sumSin += Math.sin(radians);
      sumCos += Math.cos(radians);
      count++;
    }
    if (count == 0) {
      return null;
    }
    double mean = Math.toDegrees(Math.atan2(sumSin / count, sumCos / count));
    if (mean < 0) {
      mean += 360.0;
    }
    return mean;
  }

  private Double totalAccumulatorOrInterval(List<Sample> samples) {
    if (samples.isEmpty()) {
      return null;
    }
    if (samples.size() == 1) {
      return samples.get(0).num();
    }
    int nonDecreasing = 0;
    for (int i = 1; i < samples.size(); i++) {
      Double prev = samples.get(i - 1).num();
      Double next = samples.get(i).num();
      if (prev == null || next == null) {
        continue;
      }
      if (next >= prev - ACCUMULATOR_EPS) {
        nonDecreasing++;
      }
    }
    double ratio = nonDecreasing / (double) (samples.size() - 1);
    if (ratio >= 0.8) {
      Double first = samples.get(0).num();
      Double last = samples.get(samples.size() - 1).num();
      if (first == null || last == null) {
        return null;
      }
      double total = last - first;
      return total < 0 ? 0.0 : total;
    }
    double sum = 0.0;
    for (Sample sample : samples) {
      if (sample.num() != null) {
        sum += sample.num();
      }
    }
    return sum;
  }

  private String modeText(List<Sample> samples) {
    Map<String, Integer> counts = new HashMap<>();
    for (Sample sample : samples) {
      String raw = sample.raw();
      if (raw == null || raw.isBlank()) {
        continue;
      }
      String trimmed = raw.trim();
      counts.put(trimmed, counts.getOrDefault(trimmed, 0) + 1);
    }
    if (counts.isEmpty()) {
      return null;
    }
    String best = null;
    int bestCount = -1;
    for (Map.Entry<String, Integer> entry : counts.entrySet()) {
      int count = entry.getValue();
      String value = entry.getKey();
      if (count > bestCount) {
        bestCount = count;
        best = value;
      } else if (count == bestCount && best != null && value.compareTo(best) < 0) {
        best = value;
      }
    }
    return best;
  }

  private double mean(List<Double> values) {
    double total = 0.0;
    for (Double value : values) {
      total += value;
    }
    return total / values.size();
  }

  private double stddev(List<Double> values, double mean) {
    double variance = 0.0;
    for (Double value : values) {
      double delta = value - mean;
      variance += delta * delta;
    }
    variance /= values.size();
    return Math.sqrt(variance);
  }

  private double quantile(List<Double> values, double q) {
    List<Double> sorted = new ArrayList<>(values);
    Collections.sort(sorted);
    if (sorted.size() == 1) {
      return sorted.get(0);
    }
    double pos = q * (sorted.size() - 1);
    int lower = (int) Math.floor(pos);
    int upper = (int) Math.ceil(pos);
    if (lower == upper) {
      return sorted.get(lower);
    }
    double weight = pos - lower;
    return sorted.get(lower) + (sorted.get(upper) - sorted.get(lower)) * weight;
  }

  private Map<String, List<VariableBatch>> loadBatches() {
    Map<String, List<VariableBatch>> cached = cachedBatches;
    if (cached != null) {
      return cached;
    }
    synchronized (batchLock) {
      if (cachedBatches != null) {
        return cachedBatches;
      }
      List<String> models = resolveModels();
      if (models.isEmpty()) {
        cachedBatches = Collections.emptyMap();
        return cachedBatches;
      }
      Map<String, List<GribstreamVariableWhitelistEntry>> whitelist = resolveWhitelist();
      boolean whitelistEnabled = whitelist != null && !whitelist.isEmpty();
      if (ingestProperties.isWhitelistRequired() && !whitelistEnabled) {
        throw new IllegalStateException("gribstream.variable-ingest.whitelist-resource is required");
      }
      Map<String, List<GribstreamVariableSpec>> catalog =
          catalogLoader.loadCatalog(ingestProperties.getCatalogResource(), models);
      Map<String, List<VariableBatch>> batches = new LinkedHashMap<>();
      for (String model : models) {
        List<GribstreamVariableSpec> specs = catalog.get(model);
        if (specs == null || specs.isEmpty()) {
          logger.warn("[GRIBSTREAM-VARS] No catalog entries found for model={}", model);
          continue;
        }
        List<GribstreamVariableSpec> filtered = filterSpecs(model, specs);
        if (whitelistEnabled) {
          List<GribstreamVariableWhitelistEntry> modelWhitelist = whitelist.get(model);
          if (modelWhitelist == null || modelWhitelist.isEmpty()) {
            logger.warn("[GRIBSTREAM-VARS] Whitelist has no entries for model={}; skipping.", model);
            continue;
          }
          filtered = applyWhitelist(model, filtered, modelWhitelist);
          if (filtered.isEmpty()) {
            logger.warn("[GRIBSTREAM-VARS] Whitelist filtered all variables for model={}", model);
            continue;
          }
        }
        if (filtered.isEmpty()) {
          logger.warn("[GRIBSTREAM-VARS] No usable variables for model={}", model);
          continue;
        }
        List<VariableBatch> modelBatches = buildBatches(filtered);
        batches.put(model, modelBatches);
      }
      cachedBatches = batches;
      return cachedBatches;
    }
  }

  private Map<String, List<GribstreamVariableWhitelistEntry>> resolveWhitelist() {
    Map<String, List<GribstreamVariableWhitelistEntry>> cached = cachedWhitelist;
    if (cached != null) {
      return cached;
    }
    Map<String, List<GribstreamVariableWhitelistEntry>> loaded =
        whitelistLoader.loadWhitelist(ingestProperties.getWhitelistResource());
    cachedWhitelist = loaded;
    return loaded;
  }

  private List<GribstreamVariableSpec> applyWhitelist(String model,
                                                      List<GribstreamVariableSpec> specs,
                                                      List<GribstreamVariableWhitelistEntry> whitelistEntries) {
    Map<String, GribstreamVariableSpec> byKey = new HashMap<>();
    for (GribstreamVariableSpec spec : specs) {
      String key = GribstreamVariableWhitelistLoader.normalizeKey(
          spec.name(), spec.level(), spec.info());
      byKey.put(key, spec);
    }
    List<GribstreamVariableSpec> filtered = new ArrayList<>();
    List<GribstreamVariableWhitelistEntry> missing = new ArrayList<>();
    Set<String> seen = new LinkedHashSet<>();
    for (GribstreamVariableWhitelistEntry entry : whitelistEntries) {
      String key = entry.normalizedKey();
      if (!seen.add(key)) {
        continue;
      }
      GribstreamVariableSpec spec = byKey.get(key);
      if (spec == null) {
        missing.add(entry);
        continue;
      }
      filtered.add(spec);
    }
    if (!missing.isEmpty()) {
      int limit = Math.min(10, missing.size());
      StringBuilder preview = new StringBuilder();
      for (int i = 0; i < limit; i++) {
        if (i > 0) {
          preview.append("; ");
        }
        preview.append(missing.get(i).describe());
      }
      logger.warn("[GRIBSTREAM-VARS] Whitelist entries not found model={} missingCount={} sample={}",
          model, missing.size(), preview);
    }
    return filtered;
  }

  private List<String> resolveModels() {
    List<String> configured = ingestProperties.getModels();
    if (configured != null && !configured.isEmpty()) {
      return normalizeModels(configured);
    }
    if (gribstreamProperties.getModels() != null && !gribstreamProperties.getModels().isEmpty()) {
      return normalizeModels(new ArrayList<>(gribstreamProperties.getModels().keySet()));
    }
    return List.of();
  }

  private List<String> normalizeModels(List<String> models) {
    Set<String> normalized = new LinkedHashSet<>();
    for (String model : models) {
      if (model == null || model.isBlank()) {
        continue;
      }
      normalized.add(model.trim().toLowerCase(Locale.ROOT));
    }
    return new ArrayList<>(normalized);
  }

  private List<GribstreamVariableSpec> filterSpecs(String model, List<GribstreamVariableSpec> specs) {
    int minHorizon = Math.max(0, gribstreamProperties.getDefaultMinHorizonHours());
    int maxHorizon = resolveMaxHorizon(model);
    List<GribstreamVariableSpec> filtered = new ArrayList<>();
    for (GribstreamVariableSpec spec : specs) {
      if (spec == null || spec.name() == null || spec.name().isBlank()) {
        continue;
      }
      if (spec.level() == null || spec.level().isBlank()) {
        continue;
      }
      if (!isWithinHorizon(spec, minHorizon, maxHorizon)) {
        continue;
      }
      filtered.add(spec);
    }
    int maxVariables = ingestProperties.getMaxVariablesPerModel();
    if (maxVariables > 0 && filtered.size() > maxVariables) {
      return new ArrayList<>(filtered.subList(0, maxVariables));
    }
    return filtered;
  }

  private boolean isWithinHorizon(GribstreamVariableSpec spec, int minHorizon, int maxHorizon) {
    int specMin = spec.minHorizonHours();
    int specMax = spec.maxHorizonHours();
    if (specMax > 0 && specMax < minHorizon) {
      return false;
    }
    if (specMin > 0 && specMin > maxHorizon) {
      return false;
    }
    return true;
  }

  private List<VariableBatch> buildBatches(List<GribstreamVariableSpec> specs) {
    int batchSize = Math.max(1, ingestProperties.getBatchSize());
    List<GribstreamVariable> variables = new ArrayList<>(specs.size());
    Map<String, GribstreamVariableSpec> aliasToSpec = new LinkedHashMap<>();
    Map<String, Integer> aliasCounts = new HashMap<>();
    for (GribstreamVariableSpec spec : specs) {
      String alias = buildAlias(spec, aliasCounts);
      aliasToSpec.put(alias, spec);
      variables.add(new GribstreamVariable(spec.name(), spec.level(), spec.info(), alias));
    }
    List<VariableBatch> batches = new ArrayList<>();
    for (int i = 0; i < variables.size(); i += batchSize) {
      int end = Math.min(i + batchSize, variables.size());
      List<GribstreamVariable> batchVars = new ArrayList<>(variables.subList(i, end));
      Map<String, GribstreamVariableSpec> batchSpecs = new LinkedHashMap<>();
      for (GribstreamVariable variable : batchVars) {
        batchSpecs.put(variable.alias(), aliasToSpec.get(variable.alias()));
      }
      batches.add(new VariableBatch(
          Collections.unmodifiableList(batchVars),
          Collections.unmodifiableMap(batchSpecs)));
    }
    return batches;
  }

  private String buildAlias(GribstreamVariableSpec spec, Map<String, Integer> aliasCounts) {
    String name = slug(spec.name());
    String level = slug(spec.level());
    String info = slug(spec.info());
    String base = name + "_" + level;
    if (!info.isEmpty()) {
      base = base + "_" + info;
    }
    String hash = Hashing.sha256Hex(spec.name() + "|" + spec.level() + "|" + spec.info())
        .substring(0, 8);
    String candidate = base;
    if (candidate.length() > ALIAS_LIMIT - 9) {
      candidate = candidate.substring(0, ALIAS_LIMIT - 9);
    }
    candidate = candidate + "_" + hash;
    int count = aliasCounts.getOrDefault(candidate, 0);
    aliasCounts.put(candidate, count + 1);
    if (count == 0) {
      return candidate;
    }
    String suffix = "_" + count;
    int maxBase = ALIAS_LIMIT - suffix.length();
    if (candidate.length() > maxBase) {
      candidate = candidate.substring(0, maxBase);
    }
    return candidate + suffix;
  }

  private String slug(String input) {
    if (input == null) {
      return "";
    }
    String raw = input.trim().toLowerCase(Locale.ROOT);
    if (raw.isEmpty()) {
      return "";
    }
    String normalized = raw.replaceAll("[^a-z0-9]+", "_");
    normalized = normalized.replaceAll("^_+", "").replaceAll("_+$", "");
    return normalized;
  }

  private String normalizeAlias(String alias) {
    if (alias == null || alias.isBlank()) {
      return null;
    }
    return alias.trim();
  }

  private int resolveMaxHorizon(String modelCode) {
    GribstreamProperties.ModelProperties model = gribstreamProperties.getModels().get(modelCode);
    if (model == null) {
      throw new IllegalArgumentException("Missing gribstream.models." + modelCode + ".maxHorizonHours");
    }
    int maxHorizon = model.getMaxHorizonHours();
    if (maxHorizon < 1) {
      throw new IllegalArgumentException("maxHorizonHours must be >= 1 for model " + modelCode);
    }
    return maxHorizon;
  }

  private List<Integer> resolveMembers(String modelCode) {
    if (!ENSEMBLE_MODELS.contains(modelCode)) {
      return null;
    }
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required for model " + modelCode);
    }
    return members;
  }

  private void enforceNoLeakage(Instant forecastedAt, Instant asOfUtc, String modelCode) {
    if (forecastedAt.isAfter(asOfUtc)) {
      throw new IllegalStateException("Gribstream leakage guard model=" + modelCode
          + " forecastedAtUtc=" + forecastedAt + " asOfUtc=" + asOfUtc);
    }
  }

  private Double parseDouble(String raw) {
    String text = raw.trim();
    if (text.isEmpty()) {
      return null;
    }
    if (text.equalsIgnoreCase("nan") || text.equalsIgnoreCase("inf") || text.equalsIgnoreCase("-inf")) {
      return null;
    }
    try {
      return Double.parseDouble(text);
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private String truncate(String value, int limit) {
    if (value == null) {
      return "";
    }
    String trimmed = value.trim();
    if (trimmed.length() <= limit) {
      return trimmed;
    }
    return trimmed.substring(0, limit);
  }

  private record SeriesKey(String alias, int member) {
  }

  private record Sample(Instant time, String raw, Double num) {
  }

  private record AggregationResult(Double valueNum, String valueText, int sampleCount) {
  }

  private record VariableBatch(
      List<GribstreamVariable> variables,
      Map<String, GribstreamVariableSpec> aliasToSpec) {
  }
}
