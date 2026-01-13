package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.common.StandardTimeClimateWindow;
import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.models.GribstreamMetric;
import com.predictionmarkets.weather.repository.GribstreamDailyFeatureRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Callable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class GribstreamDailyTmaxService {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamDailyTmaxService.class);
  private static final String GEFS_MEAN_MODEL = "gefsatmosmean";
  private static final List<String> DETERMINISTIC_MODELS =
      List.of("hrrr", "nbm", "rap", GEFS_MEAN_MODEL);
  private static final String GEFS_SPREAD_MODEL = "gefsatmos";
  private static final int GEFS_MIN_MEMBERS = 10;
  private static final double VALUE_TOLERANCE = 1e-6;

  private final GribstreamClient client;
  private final GribstreamProperties properties;
  private final GribstreamDailyFeatureRepository repository;
  private final GribstreamTaskExecutor taskExecutor;

  public GribstreamDailyTmaxService(GribstreamClient client,
                                    GribstreamProperties properties,
                                    GribstreamDailyFeatureRepository repository,
                                    GribstreamTaskExecutor taskExecutor) {
    this.client = client;
    this.properties = properties;
    this.repository = repository;
    this.taskExecutor = taskExecutor;
  }

  public GribstreamDailyOpinionResult computeAndPersistDailyOpinions(StationSpec station,
                                                                     LocalDate targetDateLocal,
                                                                     Instant asOfUtc) {
    Objects.requireNonNull(station, "station is required");
    Objects.requireNonNull(targetDateLocal, "targetDateLocal is required");
    Objects.requireNonNull(asOfUtc, "asOfUtc is required");
    ZoneId zoneId = ZoneId.of(station.zoneId());
    StandardTimeClimateWindow.UtcRange range =
        StandardTimeClimateWindow.computeUtcRange(zoneId, targetDateLocal);
    List<GribstreamCoordinate> coordinates = List.of(
        new GribstreamCoordinate(station.lat(), station.lon(), station.name()));
    List<GribstreamVariable> variables = List.of(
        new GribstreamVariable("TMP", "2 m above ground", "", "tmpk"));
    int minHorizon = Math.max(0, properties.getDefaultMinHorizonHours());

    List<Callable<ModelResult>> tasks = new ArrayList<>();
    for (String model : DETERMINISTIC_MODELS) {
      tasks.add(() -> computeDeterministic(model, station, targetDateLocal, asOfUtc, range,
          minHorizon, coordinates, variables));
    }
    tasks.add(() -> computeGefsSpread(station, targetDateLocal, asOfUtc, range, minHorizon,
        coordinates, variables));

    List<ModelResult> results = taskExecutor.invokeAllOrFail(tasks);
    Map<String, Double> tmaxByModel = new HashMap<>();
    Double spreadF = null;
    for (ModelResult result : results) {
      if (result == null) {
        continue;
      }
      if (result.metric() == GribstreamMetric.TMAX_F) {
        tmaxByModel.put(result.modelCode(), result.valueF());
      } else if (result.metric() == GribstreamMetric.TMP_SPREAD_F) {
        spreadF = result.valueF();
      }
    }
    return new GribstreamDailyOpinionResult(Map.copyOf(tmaxByModel), spreadF);
  }

  private ModelResult computeDeterministic(String modelCode,
                                           StationSpec station,
                                           LocalDate targetDateLocal,
                                           Instant asOfUtc,
                                           StandardTimeClimateWindow.UtcRange range,
                                           int minHorizon,
                                           List<GribstreamCoordinate> coordinates,
                                           List<GribstreamVariable> variables) {
    String normalizedModel = normalizeModelCode(modelCode);
    int maxHorizon = resolveMaxHorizon(normalizedModel);
    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        range.startUtc().toString(),
        range.endUtc().toString(),
        asOfUtc.toString(),
        minHorizon,
        maxHorizon,
        coordinates,
        variables,
        null);
    GribstreamClientResponse response;
    try {
      response = client.fetchHistory(normalizedModel, request);
    } catch (GribstreamEmptyResponseException ex) {
      if (GEFS_MEAN_MODEL.equals(normalizedModel)) {
        return computeDerivedGefsMean(station, targetDateLocal, asOfUtc, range, minHorizon,
            coordinates, variables);
      }
      logger.warn("[GRIBSTREAM] model={} status=skipped reason=empty_response", normalizedModel);
      return null;
    }
    enforceNoLeakage(response.rows(), asOfUtc, normalizedModel);
    GribstreamDailyMetrics.TmaxResult tmax = GribstreamDailyMetrics.computeTmax(response.rows());
    return upsertTmaxFeature(station, targetDateLocal, asOfUtc, range, minHorizon, maxHorizon,
        normalizedModel, response, tmax, null);
  }

  private ModelResult computeGefsSpread(StationSpec station,
                                        LocalDate targetDateLocal,
                                        Instant asOfUtc,
                                        StandardTimeClimateWindow.UtcRange range,
                                        int minHorizon,
                                        List<GribstreamCoordinate> coordinates,
                                        List<GribstreamVariable> variables) {
    int maxHorizon = resolveMaxHorizon(GEFS_SPREAD_MODEL);
    List<Integer> members = properties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required");
    }
    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        range.startUtc().toString(),
        range.endUtc().toString(),
        asOfUtc.toString(),
        minHorizon,
        maxHorizon,
        coordinates,
        variables,
        members);
    GribstreamClientResponse response;
    try {
      response = client.fetchHistory(GEFS_SPREAD_MODEL, request);
    } catch (GribstreamEmptyResponseException ex) {
      logger.warn("[GRIBSTREAM] model={} status=skipped reason=empty_response", GEFS_SPREAD_MODEL);
      return null;
    }
    enforceNoLeakage(response.rows(), asOfUtc, GEFS_SPREAD_MODEL);
    ensureMembersPresent(response.rows(), GEFS_SPREAD_MODEL);
    GribstreamDailyMetrics.SpreadResult spread =
        GribstreamDailyMetrics.computeSpread(response.rows(), GEFS_MIN_MEMBERS);
    GribstreamDailyFeatureEntity entity = new GribstreamDailyFeatureEntity();
    entity.setStationId(station.stationId());
    entity.setZoneId(station.zoneId());
    entity.setTargetDateLocal(targetDateLocal);
    entity.setAsofUtc(asOfUtc);
    entity.setModelCode(GEFS_SPREAD_MODEL);
    entity.setMetric(GribstreamMetric.TMP_SPREAD_F);
    entity.setValueF(spread.spreadF());
    entity.setValueK(spread.spreadK());
    entity.setSourceForecastedAtUtc(null);
    entity.setWindowStartUtc(range.startUtc());
    entity.setWindowEndUtc(range.endUtc());
    entity.setMinHorizonHours(minHorizon);
    entity.setMaxHorizonHours(maxHorizon);
    entity.setRequestJson(response.requestJson());
    entity.setRequestSha256(response.requestSha256());
    entity.setResponseSha256(response.responseSha256());
    entity.setRetrievedAtUtc(response.retrievedAtUtc());
    entity.setNotes("gefsMembersUsed=" + members.size()
        + " minMembers=" + GEFS_MIN_MEMBERS
        + " timesUsed=" + spread.timesUsed());
    upsertFeature(entity);
    return new ModelResult(GEFS_SPREAD_MODEL, GribstreamMetric.TMP_SPREAD_F, spread.spreadF());
  }

  private ModelResult computeDerivedGefsMean(StationSpec station,
                                             LocalDate targetDateLocal,
                                             Instant asOfUtc,
                                             StandardTimeClimateWindow.UtcRange range,
                                             int minHorizon,
                                             List<GribstreamCoordinate> coordinates,
                                             List<GribstreamVariable> variables) {
    int maxHorizon = resolveMaxHorizon(GEFS_MEAN_MODEL);
    List<Integer> members = properties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required");
    }
    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        range.startUtc().toString(),
        range.endUtc().toString(),
        asOfUtc.toString(),
        minHorizon,
        maxHorizon,
        coordinates,
        variables,
        members);
    GribstreamClientResponse response;
    try {
      response = client.fetchHistory(GEFS_SPREAD_MODEL, request);
    } catch (GribstreamEmptyResponseException ex) {
      logger.warn("[GRIBSTREAM] model={} status=skipped reason=empty_response", GEFS_MEAN_MODEL);
      return null;
    }
    enforceNoLeakage(response.rows(), asOfUtc, GEFS_SPREAD_MODEL);
    ensureMembersPresent(response.rows(), GEFS_SPREAD_MODEL);
    GribstreamDailyMetrics.EnsembleMeanResult meanResult =
        GribstreamDailyMetrics.computeEnsembleMeanTmax(response.rows(), GEFS_MIN_MEMBERS);
    GribstreamDailyMetrics.TmaxResult tmax = new GribstreamDailyMetrics.TmaxResult(
        meanResult.tmaxK(),
        meanResult.tmaxF(),
        meanResult.forecastedAtUtc());
    String notes = "derivedFrom=" + GEFS_SPREAD_MODEL
        + " membersUsed=" + meanResult.membersUsed()
        + " timesUsed=" + meanResult.timesUsed();
    return upsertTmaxFeature(station, targetDateLocal, asOfUtc, range, minHorizon, maxHorizon,
        GEFS_MEAN_MODEL, response, tmax, notes);
  }

  private ModelResult upsertTmaxFeature(StationSpec station,
                                        LocalDate targetDateLocal,
                                        Instant asOfUtc,
                                        StandardTimeClimateWindow.UtcRange range,
                                        int minHorizon,
                                        int maxHorizon,
                                        String modelCode,
                                        GribstreamClientResponse response,
                                        GribstreamDailyMetrics.TmaxResult tmax,
                                        String notes) {
    GribstreamDailyFeatureEntity entity = new GribstreamDailyFeatureEntity();
    entity.setStationId(station.stationId());
    entity.setZoneId(station.zoneId());
    entity.setTargetDateLocal(targetDateLocal);
    entity.setAsofUtc(asOfUtc);
    entity.setModelCode(modelCode);
    entity.setMetric(GribstreamMetric.TMAX_F);
    entity.setValueF(tmax.tmaxF());
    entity.setValueK(tmax.tmaxK());
    entity.setSourceForecastedAtUtc(tmax.forecastedAtUtc());
    entity.setWindowStartUtc(range.startUtc());
    entity.setWindowEndUtc(range.endUtc());
    entity.setMinHorizonHours(minHorizon);
    entity.setMaxHorizonHours(maxHorizon);
    entity.setRequestJson(response.requestJson());
    entity.setRequestSha256(response.requestSha256());
    entity.setResponseSha256(response.responseSha256());
    entity.setRetrievedAtUtc(response.retrievedAtUtc());
    entity.setNotes(notes);
    upsertFeature(entity);
    return new ModelResult(modelCode, GribstreamMetric.TMAX_F, tmax.tmaxF());
  }

  private void enforceNoLeakage(List<GribstreamRow> rows, Instant asOfUtc, String modelCode) {
    for (GribstreamRow row : rows) {
      Instant forecastedAt = row.forecastedAt();
      if (forecastedAt != null && forecastedAt.isAfter(asOfUtc)) {
        throw new IllegalStateException("Gribstream leakage guard model=" + modelCode
            + " forecastedAtUtc=" + forecastedAt + " asOfUtc=" + asOfUtc);
      }
    }
  }

  private void ensureMembersPresent(List<GribstreamRow> rows, String modelCode) {
    for (GribstreamRow row : rows) {
      if (row.member() == null) {
        throw new IllegalStateException("Missing GEFS member id for model " + modelCode);
      }
    }
  }

  private int resolveMaxHorizon(String modelCode) {
    GribstreamProperties.ModelProperties model = properties.getModels().get(modelCode);
    if (model == null) {
      throw new IllegalArgumentException("Missing gribstream.models." + modelCode + ".maxHorizonHours");
    }
    int maxHorizon = model.getMaxHorizonHours();
    if (maxHorizon < 1) {
      throw new IllegalArgumentException("maxHorizonHours must be >= 1 for model " + modelCode);
    }
    return maxHorizon;
  }

  private String normalizeModelCode(String modelCode) {
    if (modelCode == null || modelCode.isBlank()) {
      throw new IllegalArgumentException("modelCode is required");
    }
    return modelCode.trim().toLowerCase(Locale.ROOT);
  }

  private void upsertFeature(GribstreamDailyFeatureEntity candidate) {
    Optional<GribstreamDailyFeatureEntity> existing =
        repository.findByStationIdAndTargetDateLocalAndAsofUtcAndModelCodeAndMetric(
            candidate.getStationId(),
            candidate.getTargetDateLocal(),
            candidate.getAsofUtc(),
            candidate.getModelCode(),
            candidate.getMetric());
    if (existing.isEmpty()) {
      repository.save(candidate);
      return;
    }
    GribstreamDailyFeatureEntity current = existing.get();
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
    return equalsNullable(left.getZoneId(), right.getZoneId())
        && equalsNullable(left.getSourceForecastedAtUtc(), right.getSourceForecastedAtUtc())
        && equalsNullable(left.getWindowStartUtc(), right.getWindowStartUtc())
        && equalsNullable(left.getWindowEndUtc(), right.getWindowEndUtc())
        && left.getMinHorizonHours() == right.getMinHorizonHours()
        && left.getMaxHorizonHours() == right.getMaxHorizonHours()
        && equalsNullable(left.getRequestSha256(), right.getRequestSha256())
        && equalsNullable(left.getResponseSha256(), right.getResponseSha256())
        && equalsNullable(left.getRequestJson(), right.getRequestJson())
        && equalsNullable(left.getNotes(), right.getNotes())
        && almostEqual(left.getValueF(), right.getValueF())
        && almostEqual(left.getValueK(), right.getValueK());
  }

  private boolean almostEqual(Double left, Double right) {
    if (left == null || right == null) {
      return left == null && right == null;
    }
    return Math.abs(left - right) <= VALUE_TOLERANCE;
  }

  private boolean equalsNullable(Object left, Object right) {
    return Objects.equals(left, right);
  }

  private record ModelResult(String modelCode, GribstreamMetric metric, double valueF) {
  }
}
