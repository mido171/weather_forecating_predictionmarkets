package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.common.TimeSemantics;
import com.predictionmarkets.weather.iem.IemMosClient;
import com.predictionmarkets.weather.iem.IemMosEntry;
import com.predictionmarkets.weather.iem.IemMosPayload;
import com.predictionmarkets.weather.models.AsofPolicy;
import com.predictionmarkets.weather.models.AsofTimeZone;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.MosRun;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.AsofPolicyRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureUpsertRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureUpsertRepository.UpsertRow;
import com.predictionmarkets.weather.repository.MosRunRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class DefaultMosAsofMaterializeService implements MosAsofMaterializeService {
  private static final Logger logger = LoggerFactory.getLogger(DefaultMosAsofMaterializeService.class);
  private static final Duration RUN_QUERY_WINDOW = Duration.ofHours(1);
  private static final String MISSING_NO_ELIGIBLE_RUN = "NO_ELIGIBLE_RUN";
  private static final String MISSING_NO_RUN_PAYLOAD = "NO_RUN_PAYLOAD";
  private static final String MISSING_NO_FORECAST_FOR_DATE = "NO_FORECAST_FOR_DATE";
  private static final String MISSING_NO_TMAX_VALUE = "NO_TMAX_VALUE";

  private final IemMosClient mosClient;
  private final MosRunRepository mosRunRepository;
  private final MosAsofFeatureUpsertRepository upsertRepository;
  private final StationRegistryRepository stationRegistryRepository;
  private final AsofPolicyRepository asofPolicyRepository;

  public DefaultMosAsofMaterializeService(
      IemMosClient mosClient,
      MosRunRepository mosRunRepository,
      MosAsofFeatureUpsertRepository upsertRepository,
      StationRegistryRepository stationRegistryRepository,
      AsofPolicyRepository asofPolicyRepository) {
    this.mosClient = mosClient;
    this.mosRunRepository = mosRunRepository;
    this.upsertRepository = upsertRepository;
    this.stationRegistryRepository = stationRegistryRepository;
    this.asofPolicyRepository = asofPolicyRepository;
  }

  @Override
  @Transactional
  public void materializeForTargetDate(String stationId,
                                       LocalDate targetDate,
                                       Long asofPolicyId,
                                       List<MosModel> models) {
    String normalizedStation = normalizeStationId(stationId);
    Objects.requireNonNull(targetDate, "targetDate is required");
    if (asofPolicyId == null) {
      throw new IllegalArgumentException("asofPolicyId is required");
    }
    List<MosModel> requestedModels = normalizeModels(models);
    StationRegistry station = stationRegistryRepository.findById(normalizedStation)
        .orElseThrow(() -> new IllegalArgumentException(
            "Station not found in registry: " + normalizedStation));
    AsofPolicy policy = asofPolicyRepository.findById(asofPolicyId)
        .orElseThrow(() -> new IllegalArgumentException("asofPolicyId not found: " + asofPolicyId));

    ZoneId zoneId = ZoneId.of(station.getZoneId());
    ZoneId asOfZoneId = resolveAsofZone(policy.getAsofTimeZone(), zoneId);
    TimeSemantics.AsOfTimes asOfTimes = TimeSemantics.computeAsOfTimes(
        targetDate, policy.getAsofLocalTime(), zoneId, asOfZoneId);
    Instant asOfUtc = asOfTimes.asOfUtc();
    LocalDateTime asOfLocal = asOfTimes.asOfLocalZdt().toLocalDateTime();
    String stationZoneid = station.getZoneId();

    Instant now = Instant.now();
    List<UpsertRow> rows = new ArrayList<>(requestedModels.size());
    for (MosModel model : requestedModels) {
      Optional<MosRun> chosen = mosRunRepository
          .findTopByIdStationIdAndIdModelAndIdRuntimeUtcLessThanEqualOrderByIdRuntimeUtcDesc(
              normalizedStation, model, asOfUtc);
      if (chosen.isEmpty()) {
        rows.add(new UpsertRow(
            normalizedStation,
            targetDate,
            asofPolicyId,
            model.name(),
            asOfUtc,
            asOfLocal,
            stationZoneid,
            null,
            null,
            MISSING_NO_ELIGIBLE_RUN,
            null,
            now));
        continue;
      }

      MosRun run = chosen.get();
      Instant runtimeUtc = run.getId().getRuntimeUtc();
      try {
        TimeSemantics.assertRuntimeNotAfterAsOf(runtimeUtc, asOfUtc);
      } catch (IllegalArgumentException ex) {
        logger.error("MOS leakage guard triggered station={} model={} runtimeUtc={} asofUtc={}",
            normalizedStation, model.name(), runtimeUtc, asOfUtc, ex);
        throw ex;
      }

      IemMosPayload payload = mosClient.fetchWindow(
          normalizedStation, model, runtimeUtc, runtimeUtc.plus(RUN_QUERY_WINDOW));
      TMaxSelection selection = selectTmax(payload.entries(), runtimeUtc, targetDate, zoneId);

      rows.add(new UpsertRow(
          normalizedStation,
          targetDate,
          asofPolicyId,
          model.name(),
          asOfUtc,
          asOfLocal,
          stationZoneid,
          runtimeUtc,
          selection.tmaxF(),
          selection.missingReason(),
          run.getRawPayloadHash(),
          run.getRetrievedAtUtc()));
    }
    if (!rows.isEmpty()) {
      upsertRepository.upsertAll(rows);
    }
  }

  private TMaxSelection selectTmax(List<IemMosEntry> entries,
                                   Instant runtimeUtc,
                                   LocalDate targetDate,
                                   ZoneId zoneId) {
    if (entries == null || entries.isEmpty()) {
      return new TMaxSelection(null, MISSING_NO_RUN_PAYLOAD);
    }
    List<IemMosEntry> runtimeEntries = new ArrayList<>();
    for (IemMosEntry entry : entries) {
      if (entry.runtimeUtc() != null && entry.runtimeUtc().equals(runtimeUtc)) {
        runtimeEntries.add(entry);
      }
    }
    if (runtimeEntries.isEmpty()) {
      return new TMaxSelection(null, MISSING_NO_RUN_PAYLOAD);
    }
    List<BigDecimal> values = new ArrayList<>();
    for (IemMosEntry entry : runtimeEntries) {
      if (entry.forecastTimeUtc() == null) {
        continue;
      }
      LocalDate forecastDate = ZonedDateTime.ofInstant(entry.forecastTimeUtc(), zoneId)
          .toLocalDate();
      if (!forecastDate.equals(targetDate)) {
        continue;
      }
      values.add(entry.nX());
    }
    if (values.isEmpty()) {
      return new TMaxSelection(null, MISSING_NO_FORECAST_FOR_DATE);
    }
    BigDecimal max = null;
    for (BigDecimal value : values) {
      if (value == null) {
        continue;
      }
      if (max == null || value.compareTo(max) > 0) {
        max = value;
      }
    }
    if (max == null) {
      return new TMaxSelection(null, MISSING_NO_TMAX_VALUE);
    }
    return new TMaxSelection(max, null);
  }

  private List<MosModel> normalizeModels(List<MosModel> models) {
    if (models == null || models.isEmpty()) {
      throw new IllegalArgumentException("models are required");
    }
    Set<MosModel> unique = new LinkedHashSet<>();
    for (MosModel model : models) {
      if (model == null) {
        throw new IllegalArgumentException("model is required");
      }
      unique.add(model);
    }
    return List.copyOf(unique);
  }

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private ZoneId resolveAsofZone(AsofTimeZone asofTimeZone, ZoneId stationZoneId) {
    if (asofTimeZone == null || asofTimeZone == AsofTimeZone.LOCAL) {
      return stationZoneId;
    }
    return ZoneOffset.UTC;
  }

  private record TMaxSelection(BigDecimal tmaxF, String missingReason) {
  }
}
