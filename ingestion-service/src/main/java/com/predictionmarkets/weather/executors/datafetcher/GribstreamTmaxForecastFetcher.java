package com.predictionmarkets.weather.executors.datafetcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.common.StandardTimeClimateWindow;
import com.predictionmarkets.weather.config.GribstreamTmaxForecastProperties;
import com.predictionmarkets.weather.gribstream.GribstreamCoordinate;
import com.predictionmarkets.weather.gribstream.GribstreamForecastClient;
import com.predictionmarkets.weather.gribstream.GribstreamForecastRawResponse;
import com.predictionmarkets.weather.gribstream.GribstreamForecastRequest;
import com.predictionmarkets.weather.gribstream.GribstreamGenericResponseParser;
import com.predictionmarkets.weather.gribstream.GribstreamProperties;
import com.predictionmarkets.weather.gribstream.GribstreamResponseException;
import com.predictionmarkets.weather.gribstream.GribstreamValueRow;
import com.predictionmarkets.weather.gribstream.GribstreamVariable;
import com.predictionmarkets.weather.gribstream.StationSpec;
import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.models.GribstreamMetric;
import com.predictionmarkets.weather.repository.GribstreamDailyFeatureRepository;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Forecast model references:
 * https://gribstream.com/models/nbm
 * https://gribstream.com/models/gfs
 * https://gribstream.com/models/hrrr
 * https://gribstream.com/models/rap
 * https://gribstream.com/models/gefsatmosmean
 * https://gribstream.com/models/gefsatmos
 * Station location reference example: https://acukwik.com/Airport-Info/
 */
@Service
public class GribstreamTmaxForecastFetcher {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamTmaxForecastFetcher.class);
  private static final LocalDate MIN_START_DATE = LocalDate.of(2021, 3, 23);
  private static final LocalTime ASOF_TIME_UTC = LocalTime.of(12, 0);
  private static final String TMP_ALIAS = "tmpK";
  private static final String GEFS_SPREAD_MODEL = "gefsatmos";
  private static final String GEFS_MEAN_MODEL = "gefsatmosmean";
  private static final List<ModelSpec> CORE_MODELS = List.of(
      new ModelSpec("nbm"),
      new ModelSpec("hrrr"),
      new ModelSpec("rap"));
  private static final Pattern MEMBER_KEY_PATTERN =
      Pattern.compile("(?i)tmpk.*?(?:member|mem|ens)?[_-]?(\\d{1,3})$");

  private final GribstreamForecastClient client;
  private final GribstreamProperties gribstreamProperties;
  private final GribstreamTmaxForecastProperties properties;
  private final GribstreamDailyFeatureRepository repository;
  private final ObjectMapper objectMapper;

  public GribstreamTmaxForecastFetcher(GribstreamForecastClient client,
                                       GribstreamProperties gribstreamProperties,
                                       GribstreamTmaxForecastProperties properties,
                                       GribstreamDailyFeatureRepository repository,
                                       ObjectMapper objectMapper) {
    this.client = Objects.requireNonNull(client, "client is required");
    this.gribstreamProperties =
        Objects.requireNonNull(gribstreamProperties, "gribstreamProperties is required");
    this.properties = Objects.requireNonNull(properties, "properties is required");
    this.repository = Objects.requireNonNull(repository, "repository is required");
    this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper is required");
  }

  public void run() {
    StationSpec station = resolveStation(properties.getStationId());
    LocalDate start = requireStartDate(properties);
    LocalDate end = requireEndDate(properties);
    ensureValidDateRange(start, end);
    snapshot("Starting Gribstream Tmax forecast backfill station=" + station.stationId()
        + " dateRange=" + start + ".." + end);

    runSmokeTest(station);
    backfillByDay(station, start, end);
    snapshot("Gribstream Tmax forecast backfill complete.");
  }

  private void runSmokeTest(StationSpec station) {
    Instant smokeAsOfUtc = properties.getSmokeTestForecastedAtUtc();
    if (smokeAsOfUtc == null) {
      throw new IllegalArgumentException("gribstream.tmax-forecast.smoke-test-forecasted-at-utc is required");
    }
    snapshot("Smoke test starting forecastedAtUtc=" + smokeAsOfUtc);
    ZoneId zoneId = ZoneId.of(station.zoneId());
    LocalDate targetDateLocal = toTargetDateLocal(smokeAsOfUtc, zoneId);
    for (ModelSpec spec : CORE_MODELS) {
      runSmokeTestForModel(spec.modelCode(), station, smokeAsOfUtc, targetDateLocal, zoneId);
    }
    runSmokeTestForModel(GEFS_MEAN_MODEL, station, smokeAsOfUtc, targetDateLocal, zoneId);
    runSmokeTestForSpread(station, smokeAsOfUtc, targetDateLocal, zoneId);
    snapshot("Smoke test complete.");
  }

  private void backfillByDay(StationSpec station, LocalDate start, LocalDate end) {
    ZoneId zoneId = ZoneId.of(station.zoneId());
    LocalDate current = start;
    while (!current.isAfter(end)) {
      Instant asOfUtc = computeAsOfUtc(current);
      for (ModelSpec spec : CORE_MODELS) {
        if (!isMissingDailyFeature(station, current, asOfUtc, spec.modelCode(), GribstreamMetric.TMAX_F)) {
          logSkip(station, current, asOfUtc, spec.modelCode(), GribstreamMetric.TMAX_F);
          continue;
        }
        fetchDeterministicModelForDay(spec.modelCode(), station, current, asOfUtc, zoneId);
      }
      boolean needGefsMean =
          isMissingDailyFeature(station, current, asOfUtc, GEFS_MEAN_MODEL, GribstreamMetric.TMAX_F);
      boolean needGefsSpread =
          isMissingDailyFeature(station, current, asOfUtc, GEFS_SPREAD_MODEL, GribstreamMetric.TMP_SPREAD_F);
      if (needGefsMean || needGefsSpread) {
        fetchGefsDerivedForDay(station, current, asOfUtc, zoneId, needGefsMean, needGefsSpread);
      } else {
        logSkip(station, current, asOfUtc, GEFS_MEAN_MODEL, GribstreamMetric.TMAX_F);
        logSkip(station, current, asOfUtc, GEFS_SPREAD_MODEL, GribstreamMetric.TMP_SPREAD_F);
      }
      current = current.plusDays(1L);
    }
  }

  private void fetchDeterministicModelForDay(String modelCode,
                                             StationSpec station,
                                             LocalDate targetDateLocal,
                                             Instant asOfUtc,
                                             ZoneId zoneId) {
    GribstreamForecastRequest request = buildForecastRequest(
        asOfUtc,
        asOfUtc,
        modelCode,
        station,
        null);
    GribstreamForecastRawResponse raw = client.fetchForecastsRaw(modelCode, request);
    ensureSuccessOrThrow(modelCode, raw);
    List<GribstreamValueRow> rows = parseRows(modelCode, raw);
    List<GribstreamValueRow> runRows = selectRowsForAsOf(rows, asOfUtc);
    if (runRows.isEmpty()) {
      logger.warn("Missing forecastedAt rows model={} station={} targetDateLocal={} asofUtc={}",
          modelCode, station.stationId(), targetDateLocal, asOfUtc);
      return;
    }
    DailyValue daily = computeDailyTmax(modelCode, runRows, station, zoneId, targetDateLocal);
    if (daily == null) {
      logger.warn("Missing tmax values model={} station={} targetDateLocal={}",
          modelCode, station.stationId(), targetDateLocal);
      return;
    }
    if (!daily.complete()) {
      logger.warn("Incomplete tmax day model={} station={} targetDateLocal={} reason={}",
          modelCode, station.stationId(), targetDateLocal, daily.reason());
      return;
    }
    String notes = "pointsUsed=" + daily.pointsUsed();
    persistDailyFeature(station, targetDateLocal, asOfUtc, modelCode,
        GribstreamMetric.TMAX_F, daily, request, raw, notes);
  }

  private void fetchGefsDerivedForDay(StationSpec station,
                                      LocalDate targetDateLocal,
                                      Instant asOfUtc,
                                      ZoneId zoneId,
                                      boolean needMean,
                                      boolean needSpread) {
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required");
    }
    boolean meanFromMembers = false;
    if (needMean && !needSpread) {
      MeanFetchResult meanResult = fetchGefsMeanFromApiForDay(station, targetDateLocal, asOfUtc, zoneId);
      DailyValue meanDaily = meanResult.daily();
      if (meanDaily != null && meanDaily.complete()) {
        String notes = "pointsUsed=" + meanDaily.pointsUsed() + " meanSource=gefsatmosmean";
        persistDailyFeature(station, targetDateLocal, asOfUtc, GEFS_MEAN_MODEL,
            GribstreamMetric.TMAX_F, meanDaily, meanResult.request(), meanResult.raw(), notes);
        needMean = false;
      } else {
        meanFromMembers = true;
      }
    } else if (needMean) {
      meanFromMembers = true;
    }

    if (!needSpread && !meanFromMembers) {
      return;
    }

    GefsMemberFetchResult membersResult = fetchGefsMembersForDay(station, asOfUtc, members);
    List<MemberValue> runValues = membersResult.values();
    if (runValues.isEmpty()) {
      logger.warn("Missing GEFS member values station={} targetDateLocal={} asofUtc={}",
          station.stationId(), targetDateLocal, asOfUtc);
      return;
    }

    if (needSpread) {
      DailyValue spread = computeDailySpread(runValues, station, zoneId, targetDateLocal, members);
      if (spread == null) {
        logger.warn("Missing GEFS spread values station={} targetDateLocal={}",
            station.stationId(), targetDateLocal);
      } else if (!spread.complete()) {
        logger.warn("Incomplete GEFS spread day station={} targetDateLocal={} reason={}",
            station.stationId(), targetDateLocal, spread.reason());
      } else {
        String notes = "pointsUsed=" + spread.pointsUsed() + " membersExpected=" + members.size();
        if (membersResult.fallbackUsed()) {
          notes += " memberLayout=per_member";
        }
        persistDailyFeature(station, targetDateLocal, asOfUtc, GEFS_SPREAD_MODEL,
            GribstreamMetric.TMP_SPREAD_F, spread, membersResult.request(),
            membersResult.raw(), notes);
      }
    }

    if (meanFromMembers) {
      DailyValue mean = computeDailyMeanFromMembers(runValues, station, zoneId, targetDateLocal, members);
      if (mean == null) {
        logger.warn("Missing GEFS mean values station={} targetDateLocal={}",
            station.stationId(), targetDateLocal);
      } else if (!mean.complete()) {
        logger.warn("Incomplete GEFS mean day station={} targetDateLocal={} reason={}",
            station.stationId(), targetDateLocal, mean.reason());
      } else {
        String notes = "pointsUsed=" + mean.pointsUsed()
            + " membersExpected=" + members.size()
            + " meanSource=gefsatmos";
        if (membersResult.fallbackUsed()) {
          notes += " memberLayout=per_member";
        }
        persistDailyFeature(station, targetDateLocal, asOfUtc, GEFS_MEAN_MODEL,
            GribstreamMetric.TMAX_F, mean, membersResult.request(),
            membersResult.raw(), notes);
      }
    }
  }

  private MeanFetchResult fetchGefsMeanFromApiForDay(StationSpec station,
                                                    LocalDate targetDateLocal,
                                                    Instant asOfUtc,
                                                    ZoneId zoneId) {
    GribstreamForecastRequest request = buildForecastRequest(
        asOfUtc,
        asOfUtc,
        GEFS_MEAN_MODEL,
        station,
        null);
    GribstreamForecastRawResponse raw = client.fetchForecastsRaw(GEFS_MEAN_MODEL, request);
    ensureSuccessOrThrow(GEFS_MEAN_MODEL, raw);
    List<GribstreamValueRow> rows = parseRows(GEFS_MEAN_MODEL, raw);
    List<GribstreamValueRow> runRows = selectRowsForAsOf(rows, asOfUtc);
    if (runRows.isEmpty()) {
      logger.warn("Missing gefsatmosmean rows station={} targetDateLocal={} asofUtc={}",
          station.stationId(), targetDateLocal, asOfUtc);
      return new MeanFetchResult(null, raw, request);
    }
    DailyValue daily = computeDailyTmax(GEFS_MEAN_MODEL, runRows, station, zoneId, targetDateLocal);
    return new MeanFetchResult(daily, raw, request);
  }

  private GefsMemberFetchResult fetchGefsMembersForDay(StationSpec station,
                                                       Instant asOfUtc,
                                                       List<Integer> members) {
    GribstreamForecastRequest request = buildForecastRequest(
        asOfUtc,
        asOfUtc,
        GEFS_SPREAD_MODEL,
        station,
        members);
    GribstreamForecastRawResponse raw = client.fetchForecastsRaw(GEFS_SPREAD_MODEL, request);
    ensureSuccessOrThrow(GEFS_SPREAD_MODEL, raw);
    List<GribstreamValueRow> rows = parseRows(GEFS_SPREAD_MODEL, raw);
    MemberLayout layout = resolveMemberLayout(rows);
    List<MemberValue> memberValues = new ArrayList<>();
    GribstreamForecastRawResponse rawForPersist = raw;
    boolean fallbackUsed = false;
    if (layout.type() == MemberLayoutType.NONE) {
      logger.warn("GEFS member layout missing; falling back to per-member requests.");
      MemberFetchResult fallback = fetchMembersIndividually(new RunRange(asOfUtc, asOfUtc), station, members);
      memberValues.addAll(fallback.values());
      rawForPersist = fallback.raw();
      fallbackUsed = true;
    } else {
      memberValues.addAll(toMemberValues(rows, layout));
    }
    List<MemberValue> runValues = selectMemberValuesForAsOf(memberValues, asOfUtc);
    return new GefsMemberFetchResult(runValues, rawForPersist, fallbackUsed, request);
  }

  private void fetchGefsSpreadForDay(StationSpec station,
                                     LocalDate targetDateLocal,
                                     Instant asOfUtc,
                                     ZoneId zoneId) {
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required");
    }
    List<MemberValue> memberValues = new ArrayList<>();
    GribstreamForecastRequest request = buildForecastRequest(
        asOfUtc,
        asOfUtc,
        GEFS_SPREAD_MODEL,
        station,
        members);
    GribstreamForecastRawResponse raw = client.fetchForecastsRaw(GEFS_SPREAD_MODEL, request);
    ensureSuccessOrThrow(GEFS_SPREAD_MODEL, raw);
    List<GribstreamValueRow> rows = parseRows(GEFS_SPREAD_MODEL, raw);
    MemberLayout layout = resolveMemberLayout(rows);
    GribstreamForecastRawResponse rawForPersist = raw;
    boolean fallbackUsed = false;
    if (layout.type() == MemberLayoutType.NONE) {
      logger.warn("GEFS member layout missing; falling back to per-member requests.");
      MemberFetchResult fallback =
          fetchMembersIndividually(new RunRange(asOfUtc, asOfUtc), station, members);
      memberValues.addAll(fallback.values());
      rawForPersist = fallback.raw();
      fallbackUsed = true;
    } else {
      memberValues.addAll(toMemberValues(rows, layout));
    }
    List<MemberValue> runValues = selectMemberValuesForAsOf(memberValues, asOfUtc);
    if (runValues.isEmpty()) {
      logger.warn("Missing GEFS member values station={} targetDateLocal={} asofUtc={}",
          station.stationId(), targetDateLocal, asOfUtc);
      return;
    }
    DailyValue daily = computeDailySpread(runValues, station, zoneId, targetDateLocal, members);
    if (daily == null) {
      logger.warn("Missing GEFS spread values station={} targetDateLocal={}",
          station.stationId(), targetDateLocal);
      return;
    }
    if (!daily.complete()) {
      logger.warn("Incomplete GEFS spread day station={} targetDateLocal={} reason={}",
          station.stationId(), targetDateLocal, daily.reason());
      return;
    }
    String notes = "pointsUsed=" + daily.pointsUsed() + " membersExpected=" + members.size();
    if (fallbackUsed) {
      notes += " memberLayout=per_member";
    }
    persistDailyFeature(station, targetDateLocal, asOfUtc, GEFS_SPREAD_MODEL,
        GribstreamMetric.TMP_SPREAD_F, daily, request, rawForPersist, notes);
  }

  private void runSmokeTestForModel(String modelCode,
                                    StationSpec station,
                                    Instant smokeAsOfUtc,
                                    LocalDate targetDateLocal,
                                    ZoneId zoneId) {
    GribstreamForecastRequest request = buildForecastRequest(
        smokeAsOfUtc,
        smokeAsOfUtc,
        modelCode,
        station,
        null);
    GribstreamForecastRawResponse raw = client.fetchForecastsRaw(modelCode, request);
    if (!isSuccessStatus(raw.statusCode())) {
      logSmokeFailure(modelCode, raw);
      if (raw.statusCode() == 401) {
        throw new IllegalStateException("Gribstream unauthorized (401). Check gribstream.apiToken.");
      }
      throw new IllegalStateException("Smoke test failed for model " + modelCode);
    }
    List<GribstreamValueRow> rows = parseRows(modelCode, raw);
    DailyValue daily = computeDailyTmax(modelCode, rows, station, zoneId, targetDateLocal);
    if (daily == null) {
      int rowCount = rows == null ? 0 : rows.size();
      DailyValue missing = new DailyValue(null, null, 0, false, "missing_rows", "rows=" + rowCount);
      logSmokeIncomplete(modelCode, raw, missing);
      snapshot("Smoke test incomplete model=" + modelCode
          + " pointsUsed=0 reason=missing_rows");
      return;
    }
    if (!daily.complete()) {
      logSmokeIncomplete(modelCode, raw, daily);
      snapshot("Smoke test incomplete model=" + modelCode
          + " pointsUsed=" + daily.pointsUsed()
          + " reason=" + daily.reason());
      return;
    }
    snapshot("Smoke test OK model=" + modelCode + " tmax_f=" + format(daily.valueF()));
  }

  private void runSmokeTestForSpread(StationSpec station,
                                     Instant smokeAsOfUtc,
                                     LocalDate targetDateLocal,
                                     ZoneId zoneId) {
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required");
    }
    GribstreamForecastRequest request = buildForecastRequest(
        smokeAsOfUtc,
        smokeAsOfUtc,
        GEFS_SPREAD_MODEL,
        station,
        members);
    GribstreamForecastRawResponse raw = client.fetchForecastsRaw(GEFS_SPREAD_MODEL, request);
    if (!isSuccessStatus(raw.statusCode())) {
      logSmokeFailure(GEFS_SPREAD_MODEL, raw);
      if (raw.statusCode() == 401) {
        throw new IllegalStateException("Gribstream unauthorized (401). Check gribstream.apiToken.");
      }
      throw new IllegalStateException("Smoke test failed for model " + GEFS_SPREAD_MODEL);
    }
    List<GribstreamValueRow> rows = parseRows(GEFS_SPREAD_MODEL, raw);
    List<MemberValue> memberValues = new ArrayList<>();
    MemberLayout layout = resolveMemberLayout(rows);
    if (layout.type() == MemberLayoutType.NONE) {
      logger.warn("GEFS member layout missing in smoke test; falling back to per-member requests.");
      MemberFetchResult fallback =
          fetchMembersIndividually(new RunRange(smokeAsOfUtc, smokeAsOfUtc), station, members);
      memberValues.addAll(fallback.values());
    } else {
      memberValues.addAll(toMemberValues(rows, layout));
    }
    DailyValue spread = computeDailySpread(
        memberValues,
        station,
        zoneId,
        targetDateLocal,
        members);
    if (spread == null) {
      int valueCount = memberValues == null ? 0 : memberValues.size();
      DailyValue missing = new DailyValue(null, null, 0, false, "missing_rows",
          "memberValues=" + valueCount);
      logSmokeIncomplete(GEFS_SPREAD_MODEL, raw, missing);
      snapshot("Smoke test incomplete model=" + GEFS_SPREAD_MODEL
          + " pointsUsed=0 reason=missing_rows");
      return;
    }
    if (!spread.complete()) {
      logSmokeIncomplete(GEFS_SPREAD_MODEL, raw, spread);
      snapshot("Smoke test incomplete model=" + GEFS_SPREAD_MODEL
          + " pointsUsed=" + spread.pointsUsed()
          + " reason=" + spread.reason());
      return;
    }
    snapshot("Smoke test OK model=" + GEFS_SPREAD_MODEL
        + " tmp_spread_f=" + format(spread.valueF()));
  }

  private void fetchDeterministicModel(ModelSpec spec,
                                       StationSpec station,
                                       LocalDate start,
                                       LocalDate end) {
    ZoneId zoneId = ZoneId.of(station.zoneId());
    List<RunRange> runRanges = listRunRanges(start, end);
    for (RunRange runRange : runRanges) {
      GribstreamForecastRequest request = buildForecastRequest(
          runRange.fromUtc(),
          runRange.untilUtc(),
          spec.modelCode(),
          station,
          null);
      GribstreamForecastRawResponse raw = client.fetchForecastsRaw(spec.modelCode(), request);
      ensureSuccessOrThrow(spec.modelCode(), raw);
      List<GribstreamValueRow> rows = parseRows(spec.modelCode(), raw);
      Map<Instant, List<GribstreamValueRow>> byRun = groupByForecastedAt(rows);
      for (Map.Entry<Instant, List<GribstreamValueRow>> entry : byRun.entrySet()) {
        Instant forecastedAt = entry.getKey();
        if (!isTwelveUtc(forecastedAt)) {
          continue;
        }
        if (forecastedAt.isBefore(runRange.fromUtc()) || forecastedAt.isAfter(runRange.untilUtc())) {
          continue;
        }
        LocalDate targetDateLocal = toTargetDateLocal(forecastedAt, zoneId);
        if (targetDateLocal.isBefore(start) || targetDateLocal.isAfter(end)) {
          continue;
        }
        if (!forecastedAt.equals(computeAsOfUtc(targetDateLocal))) {
          logger.warn("Skipping mismatched asof for model={} forecastedAtUtc={} targetDateLocal={}",
              spec.modelCode(), forecastedAt, targetDateLocal);
          continue;
        }
        DailyValue daily = computeDailyTmax(spec.modelCode(), entry.getValue(),
            station, zoneId, targetDateLocal);
        if (daily == null) {
          logger.warn("Missing tmax values model={} station={} targetDateLocal={}",
              spec.modelCode(), station.stationId(), targetDateLocal);
          continue;
        }
        if (!daily.complete()) {
          logger.warn("Incomplete tmax day model={} station={} targetDateLocal={} reason={}",
              spec.modelCode(), station.stationId(), targetDateLocal, daily.reason());
          continue;
        }
        String notes = "pointsUsed=" + daily.pointsUsed();
        persistDailyFeature(station, targetDateLocal, forecastedAt, spec.modelCode(),
            GribstreamMetric.TMAX_F, daily, request, raw, notes);
      }
    }
  }

  private void fetchGefsSpread(StationSpec station,
                               LocalDate start,
                               LocalDate end) {
    ZoneId zoneId = ZoneId.of(station.zoneId());
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required");
    }
    List<RunRange> runRanges = listRunRanges(start, end);
    for (RunRange runRange : runRanges) {
      List<MemberValue> memberValues = new ArrayList<>();
      GribstreamForecastRequest request = buildForecastRequest(
          runRange.fromUtc(),
          runRange.untilUtc(),
          GEFS_SPREAD_MODEL,
          station,
          members);
      GribstreamForecastRawResponse raw = client.fetchForecastsRaw(GEFS_SPREAD_MODEL, request);
      ensureSuccessOrThrow(GEFS_SPREAD_MODEL, raw);
      List<GribstreamValueRow> rows = parseRows(GEFS_SPREAD_MODEL, raw);
      MemberLayout layout = resolveMemberLayout(rows);
      GribstreamForecastRawResponse rawForPersist = raw;
      boolean fallbackUsed = false;
      if (layout.type() == MemberLayoutType.NONE) {
        logger.warn("GEFS member layout missing; falling back to per-member requests.");
        MemberFetchResult fallback = fetchMembersIndividually(runRange, station, members);
        memberValues.addAll(fallback.values());
        rawForPersist = fallback.raw();
        fallbackUsed = true;
      } else {
        memberValues.addAll(toMemberValues(rows, layout));
      }
      Map<Instant, List<MemberValue>> byRun = groupMemberValues(memberValues);
      for (Map.Entry<Instant, List<MemberValue>> entry : byRun.entrySet()) {
        Instant forecastedAt = entry.getKey();
        if (!isTwelveUtc(forecastedAt)) {
          continue;
        }
        if (forecastedAt.isBefore(runRange.fromUtc()) || forecastedAt.isAfter(runRange.untilUtc())) {
          continue;
        }
        LocalDate targetDateLocal = toTargetDateLocal(forecastedAt, zoneId);
        if (targetDateLocal.isBefore(start) || targetDateLocal.isAfter(end)) {
          continue;
        }
        if (!forecastedAt.equals(computeAsOfUtc(targetDateLocal))) {
          logger.warn("Skipping mismatched asof for model={} forecastedAtUtc={} targetDateLocal={}",
              GEFS_SPREAD_MODEL, forecastedAt, targetDateLocal);
          continue;
        }
        DailyValue daily = computeDailySpread(entry.getValue(),
            station, zoneId, targetDateLocal, members);
        if (daily == null) {
          logger.warn("Missing GEFS spread values station={} targetDateLocal={}",
              station.stationId(), targetDateLocal);
          continue;
        }
        if (!daily.complete()) {
          logger.warn("Incomplete GEFS spread day station={} targetDateLocal={} reason={}",
              station.stationId(), targetDateLocal, daily.reason());
          continue;
        }
        String notes = "pointsUsed=" + daily.pointsUsed() + " membersExpected=" + members.size();
        if (fallbackUsed) {
          notes += " memberLayout=per_member";
        }
        persistDailyFeature(station, targetDateLocal, forecastedAt, GEFS_SPREAD_MODEL,
            GribstreamMetric.TMP_SPREAD_F, daily, request, rawForPersist, notes);
      }
    }
  }

  private MemberFetchResult fetchMembersIndividually(RunRange runRange,
                                                     StationSpec station,
                                                     List<Integer> members) {
    List<MemberValue> memberValues = new ArrayList<>();
    List<GribstreamForecastRawResponse> rawResponses = new ArrayList<>();
    for (Integer member : members) {
      if (member == null) {
        continue;
      }
      GribstreamForecastRequest request = buildForecastRequest(
          runRange.fromUtc(),
          runRange.untilUtc(),
          GEFS_SPREAD_MODEL,
          station,
          List.of(member));
      GribstreamForecastRawResponse raw = client.fetchForecastsRaw(GEFS_SPREAD_MODEL, request);
      ensureSuccessOrThrow(GEFS_SPREAD_MODEL, raw);
      rawResponses.add(raw);
      List<GribstreamValueRow> rows = parseRows(GEFS_SPREAD_MODEL, raw);
      for (GribstreamValueRow row : rows) {
        Double tmpK = extractTmpK(row);
        if (tmpK == null) {
          continue;
        }
        memberValues.add(new MemberValue(row.forecastedAt(), row.forecastedTime(), member, tmpK));
      }
    }
    GribstreamForecastRawResponse combined = combineRawResponses(rawResponses);
    return new MemberFetchResult(memberValues, combined);
  }

  private GribstreamForecastRawResponse combineRawResponses(
      List<GribstreamForecastRawResponse> rawResponses) {
    if (rawResponses == null || rawResponses.isEmpty()) {
      throw new IllegalArgumentException("rawResponses are required");
    }
    StringBuilder requestJson = new StringBuilder();
    requestJson.append('[');
    ByteArrayOutputStream combined = new ByteArrayOutputStream();
    Instant retrievedAt = null;
    boolean first = true;
    for (GribstreamForecastRawResponse raw : rawResponses) {
      if (raw == null) {
        continue;
      }
      if (!first) {
        requestJson.append(',');
        combined.write('\n');
      }
      first = false;
      String rawRequestJson = raw.requestJson();
      requestJson.append(rawRequestJson == null ? "{}" : rawRequestJson);
      byte[] responseBytes = raw.responseBytes();
      if (responseBytes != null && responseBytes.length > 0) {
        combined.writeBytes(responseBytes);
      }
      Instant retrieved = raw.retrievedAtUtc();
      if (retrieved != null && (retrievedAt == null || retrieved.isAfter(retrievedAt))) {
        retrievedAt = retrieved;
      }
    }
    requestJson.append(']');
    String requestJsonValue = requestJson.toString();
    String requestSha = Hashing.sha256Hex(requestJsonValue);
    byte[] responseBytes = combined.toByteArray();
    String responseSha = Hashing.sha256Hex(responseBytes);
    Instant retrieved = retrievedAt == null ? Instant.now() : retrievedAt;
    return new GribstreamForecastRawResponse(
        requestJsonValue,
        requestSha,
        responseSha,
        retrieved,
        200,
        responseBytes);
  }

  private List<MemberValue> toMemberValues(List<GribstreamValueRow> rows, MemberLayout layout) {
    List<MemberValue> memberValues = new ArrayList<>();
    for (GribstreamValueRow row : rows) {
      if (layout.type() == MemberLayoutType.ROW_MEMBER) {
        Integer member = row.member();
        if (member == null) {
          continue;
        }
        Double tmpK = extractTmpK(row);
        if (tmpK == null) {
          continue;
        }
        memberValues.add(new MemberValue(row.forecastedAt(), row.forecastedTime(), member, tmpK));
      } else if (layout.type() == MemberLayoutType.COLUMN_MEMBER) {
        for (Map.Entry<String, Integer> entry : layout.memberKeys().entrySet()) {
          String rawValue = row.values() == null ? null : row.values().get(entry.getKey());
          Double tmpK = parseDouble(rawValue);
          if (tmpK == null) {
            continue;
          }
          memberValues.add(new MemberValue(
              row.forecastedAt(),
              row.forecastedTime(),
              entry.getValue(),
              tmpK));
        }
      }
    }
    return memberValues;
  }

  private DailyValue computeDailyTmax(String modelCode,
                                      List<GribstreamValueRow> rows,
                                      StationSpec station,
                                      ZoneId zoneId,
                                      LocalDate targetDateLocal) {
    if (rows == null || rows.isEmpty()) {
      return null;
    }
    List<Double> tempsK = new ArrayList<>();
    for (GribstreamValueRow row : rows) {
      Instant forecastedTime = row.forecastedTime();
      if (forecastedTime == null) {
        continue;
      }
      if (!toLocalDate(forecastedTime, zoneId).equals(targetDateLocal)) {
        continue;
      }
      Double tmpK = extractTmpK(row);
      if (tmpK == null) {
        continue;
      }
      tempsK.add(tmpK);
    }
    int points = tempsK.size();
    if (points < 1) {
      return null;
    }
    int minPoints = resolveMinPointsPerDay(modelCode);
    if (points < minPoints) {
      logger.warn("Incomplete tmax day model={} station={} targetDateLocal={} points={} minPoints={}",
          modelCode, station.stationId(), targetDateLocal, points, minPoints);
      return new DailyValue(null, null, points, false, "insufficient_points",
          "points=" + points + " minPoints=" + minPoints);
    }
    double maxK = tempsK.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
    double maxF = kelvinToF(maxK);
    return new DailyValue(maxF, maxK, points, true, null, null);
  }

  private DailyValue computeDailySpread(List<MemberValue> values,
                                        StationSpec station,
                                        ZoneId zoneId,
                                        LocalDate targetDateLocal,
                                        List<Integer> expectedMembers) {
    if (values == null || values.isEmpty()) {
      return null;
    }
    int expected = expectedMembers.size();
    Map<Instant, Map<Integer, Double>> byTime = new HashMap<>();
    for (MemberValue value : values) {
      if (value.forecastedTime() == null || value.member() == null) {
        continue;
      }
      if (!toLocalDate(value.forecastedTime(), zoneId).equals(targetDateLocal)) {
        continue;
      }
      byTime.computeIfAbsent(value.forecastedTime(), ignored -> new HashMap<>())
          .put(value.member(), value.tmpK());
    }
    List<Double> stddevsK = new ArrayList<>();
    for (Map<Integer, Double> perMember : byTime.values()) {
      if (perMember.size() < expected) {
        continue;
      }
      List<Double> tempsK = new ArrayList<>(perMember.size());
      for (Double tmpK : perMember.values()) {
        tempsK.add(tmpK);
      }
      stddevsK.add(stddev(tempsK));
    }
    int points = stddevsK.size();
    if (points < 1) {
      logger.warn("Missing GEFS members model={} station={} targetDateLocal={} expectedMembers={}",
          GEFS_SPREAD_MODEL, station.stationId(), targetDateLocal, expected);
      return new DailyValue(null, null, points, false, "missing_members",
          "expectedMembers=" + expected);
    }
    int minPoints = resolveMinPointsPerDay(GEFS_SPREAD_MODEL);
    if (points < minPoints) {
      logger.warn("Incomplete spread day model={} station={} targetDateLocal={} points={} minPoints={}",
          GEFS_SPREAD_MODEL, station.stationId(), targetDateLocal,
          points, minPoints);
      return new DailyValue(null, null, points, false, "insufficient_points",
          "points=" + points + " minPoints=" + minPoints);
    }
    double meanStdK = stddevsK.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
    double meanStdF = meanStdK * 9.0 / 5.0;
    return new DailyValue(meanStdF, meanStdK, points, true, null, null);
  }

  private DailyValue computeDailyMeanFromMembers(List<MemberValue> values,
                                                 StationSpec station,
                                                 ZoneId zoneId,
                                                 LocalDate targetDateLocal,
                                                 List<Integer> expectedMembers) {
    if (values == null || values.isEmpty()) {
      return null;
    }
    int expected = expectedMembers.size();
    Map<Instant, Map<Integer, Double>> byTime = new HashMap<>();
    for (MemberValue value : values) {
      if (value.forecastedTime() == null || value.member() == null) {
        continue;
      }
      if (!toLocalDate(value.forecastedTime(), zoneId).equals(targetDateLocal)) {
        continue;
      }
      byTime.computeIfAbsent(value.forecastedTime(), ignored -> new HashMap<>())
          .put(value.member(), value.tmpK());
    }
    List<Double> meansK = new ArrayList<>();
    for (Map<Integer, Double> perMember : byTime.values()) {
      if (perMember.size() < expected) {
        continue;
      }
      double sum = 0.0;
      for (Double tmpK : perMember.values()) {
        sum += tmpK;
      }
      meansK.add(sum / expected);
    }
    int points = meansK.size();
    if (points < 1) {
      logger.warn("Missing GEFS mean members model={} station={} targetDateLocal={} expectedMembers={}",
          GEFS_MEAN_MODEL, station.stationId(), targetDateLocal, expected);
      return new DailyValue(null, null, points, false, "missing_members",
          "expectedMembers=" + expected);
    }
    int minPoints = resolveMinPointsPerDay(GEFS_MEAN_MODEL);
    if (points < minPoints) {
      logger.warn("Incomplete GEFS mean day model={} station={} targetDateLocal={} points={} minPoints={}",
          GEFS_MEAN_MODEL, station.stationId(), targetDateLocal, points, minPoints);
      return new DailyValue(null, null, points, false, "insufficient_points",
          "points=" + points + " minPoints=" + minPoints);
    }
    double maxK = meansK.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
    double maxF = kelvinToF(maxK);
    return new DailyValue(maxF, maxK, points, true, null, null);
  }

  private int resolveMinPointsPerDay(String modelCode) {
    if ("rap".equalsIgnoreCase(modelCode)) {
      return properties.getRapMinPointsPerDay();
    }
    if (GEFS_MEAN_MODEL.equalsIgnoreCase(modelCode) || GEFS_SPREAD_MODEL.equalsIgnoreCase(modelCode)) {
      return properties.getGefsMinPointsPerDay();
    }
    return properties.getMinPointsPerDay();
  }

  private MemberLayout resolveMemberLayout(List<GribstreamValueRow> rows) {
    boolean anyMember = false;
    for (GribstreamValueRow row : rows) {
      if (row.member() != null) {
        anyMember = true;
        break;
      }
    }
    if (anyMember) {
      return new MemberLayout(MemberLayoutType.ROW_MEMBER, Map.of());
    }
    Map<String, Integer> memberKeys = new HashMap<>();
    for (GribstreamValueRow row : rows) {
      if (row.values() == null) {
        continue;
      }
      for (String key : row.values().keySet()) {
        Integer member = parseMemberFromKey(key);
        if (member != null) {
          memberKeys.putIfAbsent(key, member);
        }
      }
      if (!memberKeys.isEmpty()) {
        break;
      }
    }
    if (!memberKeys.isEmpty()) {
      return new MemberLayout(MemberLayoutType.COLUMN_MEMBER, memberKeys);
    }
    return new MemberLayout(MemberLayoutType.NONE, Map.of());
  }

  private Integer parseMemberFromKey(String key) {
    if (key == null) {
      return null;
    }
    Matcher matcher = MEMBER_KEY_PATTERN.matcher(key);
    if (!matcher.find()) {
      return null;
    }
    try {
      return Integer.parseInt(matcher.group(1));
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private Map<Instant, List<GribstreamValueRow>> groupByForecastedAt(List<GribstreamValueRow> rows) {
    Map<Instant, List<GribstreamValueRow>> grouped = new HashMap<>();
    for (GribstreamValueRow row : rows) {
      if (row.forecastedAt() == null) {
        continue;
      }
      grouped.computeIfAbsent(row.forecastedAt(), ignored -> new ArrayList<>()).add(row);
    }
    return grouped;
  }

  private Map<Instant, List<MemberValue>> groupMemberValues(List<MemberValue> values) {
    Map<Instant, List<MemberValue>> grouped = new HashMap<>();
    for (MemberValue value : values) {
      if (value.forecastedAt() == null) {
        continue;
      }
      grouped.computeIfAbsent(value.forecastedAt(), ignored -> new ArrayList<>()).add(value);
    }
    return grouped;
  }

  private List<RunRange> listRunRanges(LocalDate start, LocalDate end) {
    int windowDays = properties.getRunRangeDays();
    if (windowDays <= 0) {
      throw new IllegalArgumentException("gribstream.tmax-forecast.run-range-days must be >= 1");
    }
    List<RunRange> ranges = new ArrayList<>();
    LocalDate current = start;
    while (!current.isAfter(end)) {
      LocalDate chunkEnd = current.plusDays(windowDays - 1L);
      if (chunkEnd.isAfter(end)) {
        chunkEnd = end;
      }
      Instant fromUtc = current.minusDays(1).atTime(ASOF_TIME_UTC).toInstant(ZoneOffset.UTC);
      Instant untilUtc = chunkEnd.minusDays(1).atTime(ASOF_TIME_UTC).toInstant(ZoneOffset.UTC);
      ranges.add(new RunRange(fromUtc, untilUtc));
      current = chunkEnd.plusDays(1L);
    }
    return ranges;
  }

  private GribstreamForecastRequest buildForecastRequest(Instant fromUtc,
                                                         Instant untilUtc,
                                                         String modelCode,
                                                         StationSpec station,
                                                         List<Integer> members) {
    int maxHorizon = resolveMaxHorizon(modelCode);
    int minHorizon = properties.getMinHorizonHours();
    if (minHorizon < 0) {
      throw new IllegalArgumentException("minHorizonHours must be >= 0");
    }
    if (maxHorizon < minHorizon) {
      throw new IllegalArgumentException("maxHorizonHours must be >= minHorizonHours");
    }
    List<GribstreamCoordinate> coordinates = List.of(
        new GribstreamCoordinate(station.lat(), station.lon(), station.stationId()));
    List<GribstreamVariable> variables = List.of(
        new GribstreamVariable("TMP", "2 m above ground", "", TMP_ALIAS));
    return new GribstreamForecastRequest(
        fromUtc.toString(),
        untilUtc.toString(),
        minHorizon,
        maxHorizon,
        coordinates,
        variables,
        members);
  }

  private int resolveMaxHorizon(String modelCode) {
    if ("rap".equalsIgnoreCase(modelCode)) {
      return properties.getRapMaxHorizonHours();
    }
    if ("hrrr".equalsIgnoreCase(modelCode)) {
      return properties.getHrrrMaxHorizonHours();
    }
    return properties.getMaxHorizonHours();
  }

  private StationSpec resolveStation(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("gribstream.tmax-forecast.station-id is required");
    }
    List<GribstreamProperties.StationProperties> stations = gribstreamProperties.getStations();
    if (stations == null || stations.isEmpty()) {
      throw new IllegalArgumentException("gribstream.stations is required");
    }
    for (GribstreamProperties.StationProperties station : stations) {
      if (station.getStationId() != null
          && station.getStationId().equalsIgnoreCase(stationId.trim())) {
        return new StationSpec(
            station.getStationId(),
            station.getZoneId(),
            station.getLatitude(),
            station.getLongitude(),
            station.getName());
      }
    }
    throw new IllegalArgumentException("Station not found in gribstream.stations: " + stationId);
  }

  private LocalDate requireStartDate(GribstreamTmaxForecastProperties properties) {
    LocalDate start = properties.getStartDateLocal();
    if (start == null) {
      throw new IllegalArgumentException("gribstream.tmax-forecast.start-date-local is required");
    }
    return start;
  }

  private LocalDate requireEndDate(GribstreamTmaxForecastProperties properties) {
    LocalDate end = properties.getEndDateLocal();
    if (end == null) {
      throw new IllegalArgumentException("gribstream.tmax-forecast.end-date-local is required");
    }
    return end;
  }

  private void ensureValidDateRange(LocalDate start, LocalDate end) {
    if (start.isBefore(MIN_START_DATE)) {
      throw new IllegalArgumentException(
          "start-date-local must be >= " + MIN_START_DATE + " for the core model set");
    }
    if (end.isBefore(start)) {
      throw new IllegalArgumentException("end-date-local must be >= start-date-local");
    }
  }

  private Instant computeAsOfUtc(LocalDate targetDateLocal) {
    return targetDateLocal.minusDays(1).atTime(ASOF_TIME_UTC).toInstant(ZoneOffset.UTC);
  }

  private LocalDate toTargetDateLocal(Instant forecastedAtUtc, ZoneId zoneId) {
    return forecastedAtUtc.plusSeconds(86400).atZone(zoneId).toLocalDate();
  }

  private LocalDate toLocalDate(Instant instant, ZoneId zoneId) {
    return instant.atZone(zoneId).toLocalDate();
  }

  private boolean isTwelveUtc(Instant instant) {
    if (instant == null) {
      return false;
    }
    return instant.atZone(ZoneOffset.UTC).getHour() == 12
        && instant.atZone(ZoneOffset.UTC).getMinute() == 0
        && instant.atZone(ZoneOffset.UTC).getSecond() == 0;
  }

  private boolean isSuccessStatus(int statusCode) {
    return statusCode >= 200 && statusCode < 300;
  }

  private void ensureSuccessOrThrow(String modelCode, GribstreamForecastRawResponse raw) {
    if (isSuccessStatus(raw.statusCode())) {
      return;
    }
    if (raw.statusCode() == 401) {
      throw new IllegalStateException("Gribstream unauthorized (401). Check gribstream.apiToken.");
    }
    throw new GribstreamResponseException("Forecast fetch failed model=" + modelCode
        + " status=" + raw.statusCode()
        + " requestSha=" + raw.requestSha256());
  }

  private void persistDailyFeature(StationSpec station,
                                   LocalDate targetDateLocal,
                                   Instant asOfUtc,
                                   String modelCode,
                                   GribstreamMetric metric,
                                   DailyValue daily,
                                   GribstreamForecastRequest request,
                                   GribstreamForecastRawResponse raw,
                                   String notes) {
    ZoneId zoneId = ZoneId.of(station.zoneId());
    StandardTimeClimateWindow.UtcRange window =
        StandardTimeClimateWindow.computeUtcRange(zoneId, targetDateLocal);
    GribstreamDailyFeatureEntity entity = new GribstreamDailyFeatureEntity();
    entity.setStationId(station.stationId());
    entity.setZoneId(station.zoneId());
    entity.setTargetDateLocal(targetDateLocal);
    entity.setAsofUtc(asOfUtc);
    entity.setModelCode(modelCode);
    entity.setMetric(metric);
    entity.setValueF(daily.valueF());
    entity.setValueK(daily.valueK());
    entity.setSourceForecastedAtUtc(asOfUtc);
    entity.setWindowStartUtc(window.startUtc());
    entity.setWindowEndUtc(window.endUtc());
    entity.setMinHorizonHours(request.minHorizon());
    entity.setMaxHorizonHours(request.maxHorizon());
    entity.setRequestJson(raw.requestJson());
    entity.setRequestSha256(raw.requestSha256());
    entity.setResponseSha256(raw.responseSha256());
    entity.setRetrievedAtUtc(raw.retrievedAtUtc());
    entity.setNotes(notes);
    upsertFeature(entity);
  }

  private void upsertFeature(GribstreamDailyFeatureEntity candidate) {
    GribstreamDailyFeatureEntity current = repository
        .findByStationIdAndTargetDateLocalAndAsofUtcAndModelCodeAndMetric(
            candidate.getStationId(),
            candidate.getTargetDateLocal(),
            candidate.getAsofUtc(),
            candidate.getModelCode(),
            candidate.getMetric())
        .orElse(null);
    if (current == null) {
      repository.save(candidate);
      return;
    }
    if (isEquivalent(current, candidate)) {
      return;
    }
    current.setZoneId(candidate.getZoneId());
    current.setValueF(candidate.getValueF());
    current.setValueK(candidate.getValueK());
    current.setSourceForecastedAtUtc(candidate.getSourceForecastedAtUtc());
    current.setWindowStartUtc(candidate.getWindowStartUtc());
    current.setWindowEndUtc(candidate.getWindowEndUtc());
    current.setMinHorizonHours(candidate.getMinHorizonHours());
    current.setMaxHorizonHours(candidate.getMaxHorizonHours());
    current.setRequestJson(candidate.getRequestJson());
    current.setRequestSha256(candidate.getRequestSha256());
    current.setResponseSha256(candidate.getResponseSha256());
    current.setRetrievedAtUtc(candidate.getRetrievedAtUtc());
    current.setNotes(candidate.getNotes());
    repository.save(current);
  }

  private boolean isEquivalent(GribstreamDailyFeatureEntity left,
                               GribstreamDailyFeatureEntity right) {
    return Objects.equals(left.getZoneId(), right.getZoneId())
        && Objects.equals(left.getSourceForecastedAtUtc(), right.getSourceForecastedAtUtc())
        && Objects.equals(left.getWindowStartUtc(), right.getWindowStartUtc())
        && Objects.equals(left.getWindowEndUtc(), right.getWindowEndUtc())
        && left.getMinHorizonHours() == right.getMinHorizonHours()
        && left.getMaxHorizonHours() == right.getMaxHorizonHours()
        && Objects.equals(left.getRequestSha256(), right.getRequestSha256())
        && Objects.equals(left.getResponseSha256(), right.getResponseSha256())
        && Objects.equals(left.getRequestJson(), right.getRequestJson())
        && Objects.equals(left.getNotes(), right.getNotes())
        && almostEqual(left.getValueF(), right.getValueF())
        && almostEqual(left.getValueK(), right.getValueK());
  }

  private boolean almostEqual(Double left, Double right) {
    if (left == null || right == null) {
      return left == null && right == null;
    }
    return Math.abs(left - right) <= 1e-6;
  }

  private List<GribstreamValueRow> parseRows(String modelCode, GribstreamForecastRawResponse raw) {
    return GribstreamGenericResponseParser.parseRows(
        objectMapper,
        raw.responseBytes(),
        modelCode,
        raw.requestSha256());
  }

  private Double extractTmpK(GribstreamValueRow row) {
    if (row.values() == null || row.values().isEmpty()) {
      return null;
    }
    String value = row.values().get(TMP_ALIAS);
    if (value == null) {
      for (Map.Entry<String, String> entry : row.values().entrySet()) {
        if (entry.getKey() != null && entry.getKey().equalsIgnoreCase(TMP_ALIAS)) {
          value = entry.getValue();
          break;
        }
      }
    }
    return parseDouble(value);
  }

  private Double parseDouble(String value) {
    if (value == null) {
      return null;
    }
    String trimmed = value.trim();
    if (trimmed.isEmpty()) {
      return null;
    }
    try {
      return Double.parseDouble(trimmed);
    } catch (NumberFormatException ex) {
      throw new IllegalArgumentException("Invalid numeric value: " + value, ex);
    }
  }

  private double kelvinToF(double tempK) {
    return (tempK - 273.15) * 9.0 / 5.0 + 32.0;
  }

  private double stddev(List<Double> values) {
    int n = values.size();
    if (n == 0) {
      return Double.NaN;
    }
    double mean = 0.0;
    for (double value : values) {
      mean += value;
    }
    mean /= n;
    double variance = 0.0;
    for (double value : values) {
      double delta = value - mean;
      variance += delta * delta;
    }
    variance /= n;
    return Math.sqrt(variance);
  }

  private void logSmokeFailure(String modelCode,
                               GribstreamForecastRawResponse raw) {
    String url = buildForecastUrl(modelCode);
    String body = new String(raw.responseBytes(), StandardCharsets.UTF_8);
    String payload = raw.requestJson();
    String message = "[GRIBSTREAM-SMOKE] model=" + modelCode
        + " url=" + url
        + " status=" + raw.statusCode()
        + " payload=" + payload
        + " body=" + body;
    logger.error(message);
    System.out.println(message);
  }

  private void logSmokeIncomplete(String modelCode,
                                  GribstreamForecastRawResponse raw,
                                  DailyValue daily) {
    String url = buildForecastUrl(modelCode);
    String body = new String(raw.responseBytes(), StandardCharsets.UTF_8);
    String payload = raw.requestJson();
    String message = "[GRIBSTREAM-SMOKE] model=" + modelCode
        + " url=" + url
        + " status=" + raw.statusCode()
        + " payload=" + payload
        + " reason=" + daily.reason()
        + " details=" + daily.details()
        + " body=" + body;
    logger.warn(message);
    System.out.println(message);
  }

  private String buildForecastUrl(String modelCode) {
    String baseUrl = gribstreamProperties.getBaseUrl();
    if (baseUrl.endsWith("/")) {
      return baseUrl + "api/v2/" + modelCode + "/forecasts";
    }
    return baseUrl + "/api/v2/" + modelCode + "/forecasts";
  }

  private String format(Double value) {
    return value == null ? "" : Double.toString(value);
  }

  private void snapshot(String message) {
    String payload = "[GRIBSTREAM-TMAX-FORECAST] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private boolean isMissingDailyFeature(StationSpec station,
                                        LocalDate targetDateLocal,
                                        Instant asOfUtc,
                                        String modelCode,
                                        GribstreamMetric metric) {
    return repository
        .findByStationIdAndTargetDateLocalAndAsofUtcAndModelCodeAndMetric(
            station.stationId(),
            targetDateLocal,
            asOfUtc,
            modelCode,
            metric)
        .isEmpty();
  }

  private void logSkip(StationSpec station,
                       LocalDate targetDateLocal,
                       Instant asOfUtc,
                       String modelCode,
                       GribstreamMetric metric) {
    if (!logger.isDebugEnabled()) {
      return;
    }
    logger.debug("Skipping existing feature model={} metric={} station={} targetDateLocal={} asofUtc={}",
        modelCode, metric, station.stationId(), targetDateLocal, asOfUtc);
  }

  private List<GribstreamValueRow> selectRowsForAsOf(List<GribstreamValueRow> rows,
                                                     Instant asOfUtc) {
    if (rows == null || rows.isEmpty()) {
      return List.of();
    }
    Map<Instant, List<GribstreamValueRow>> grouped = groupByForecastedAt(rows);
    List<GribstreamValueRow> matched = grouped.get(asOfUtc);
    return matched == null ? List.of() : matched;
  }

  private List<MemberValue> selectMemberValuesForAsOf(List<MemberValue> values, Instant asOfUtc) {
    if (values == null || values.isEmpty()) {
      return List.of();
    }
    List<MemberValue> filtered = new ArrayList<>();
    for (MemberValue value : values) {
      if (asOfUtc.equals(value.forecastedAt())) {
        filtered.add(value);
      }
    }
    return filtered;
  }

  private static final class ModelSpec {
    private final String modelCode;

    private ModelSpec(String modelCode) {
      this.modelCode = modelCode;
    }

    private String modelCode() {
      return modelCode;
    }
  }

  private record RunRange(Instant fromUtc, Instant untilUtc) {
  }

  private record DailyValue(Double valueF,
                            Double valueK,
                            int pointsUsed,
                            boolean complete,
                            String reason,
                            String details) {
  }

  private record MeanFetchResult(DailyValue daily,
                                 GribstreamForecastRawResponse raw,
                                 GribstreamForecastRequest request) {
  }

  private record GefsMemberFetchResult(List<MemberValue> values,
                                       GribstreamForecastRawResponse raw,
                                       boolean fallbackUsed,
                                       GribstreamForecastRequest request) {
  }

  private record MemberFetchResult(List<MemberValue> values,
                                   GribstreamForecastRawResponse raw) {
  }

  private record MemberValue(Instant forecastedAt,
                             Instant forecastedTime,
                             Integer member,
                             Double tmpK) {
  }

  private record MemberLayout(MemberLayoutType type,
                              Map<String, Integer> memberKeys) {
  }

  private enum MemberLayoutType {
    ROW_MEMBER,
    COLUMN_MEMBER,
    NONE
  }
}

