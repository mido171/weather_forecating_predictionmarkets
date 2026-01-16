package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.iem.IemMosClient;
import com.predictionmarkets.weather.iem.IemMosEntry;
import com.predictionmarkets.weather.iem.IemMosPayload;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.iem.IemMosValue;
import com.predictionmarkets.weather.repository.MosForecastValueUpsertRepository;
import com.predictionmarkets.weather.repository.MosRunUpsertRepository;
import com.predictionmarkets.weather.repository.MosRunUpsertRepository.UpsertRow;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class MosRunIngestService {
  private final IemMosClient mosClient;
  private final MosRunUpsertRepository upsertRepository;
  private final MosForecastValueUpsertRepository forecastValueRepository;
  private final StationRegistryRepository stationRegistryRepository;

  public MosRunIngestService(
      IemMosClient mosClient,
      MosRunUpsertRepository upsertRepository,
      MosForecastValueUpsertRepository forecastValueRepository,
      StationRegistryRepository stationRegistryRepository) {
    this.mosClient = mosClient;
    this.upsertRepository = upsertRepository;
    this.forecastValueRepository = forecastValueRepository;
    this.stationRegistryRepository = stationRegistryRepository;
  }

  @Transactional
  public int ingestWindow(String stationId, MosModel model, Instant startUtc, Instant endUtc) {
    String normalizedStation = normalizeStationId(stationId);
    ensureStationExists(normalizedStation);
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
    Set<Instant> runtimes = uniqueRuntimes(payload.entries());
    if (runtimes.isEmpty()) {
      return 0;
    }
    Instant retrievedAtUtc = Instant.now();
    List<UpsertRow> rows = new ArrayList<>(runtimes.size());
    for (Instant runtimeUtc : runtimes) {
      rows.add(new UpsertRow(
          normalizedStation,
          model.name(),
          runtimeUtc,
          payload.rawPayloadHash(),
          retrievedAtUtc));
    }
    int[] results = upsertRepository.upsertAll(rows);
    upsertForecastValues(payload, normalizedStation, model.name(), retrievedAtUtc);
    return Arrays.stream(results).sum();
  }

  private void ensureStationExists(String stationId) {
    if (!stationRegistryRepository.existsById(stationId)) {
      throw new IllegalArgumentException("Station not found in registry: " + stationId);
    }
  }

  private Set<Instant> uniqueRuntimes(List<IemMosEntry> entries) {
    Set<Instant> runtimes = new LinkedHashSet<>();
    for (IemMosEntry entry : entries) {
      runtimes.add(entry.runtimeUtc());
    }
    return runtimes;
  }

  private void upsertForecastValues(IemMosPayload payload,
                                    String stationId,
                                    String modelName,
                                    Instant retrievedAtUtc) {
    List<IemMosEntry> entries = payload.entries();
    if (entries == null || entries.isEmpty()) {
      return;
    }
    List<MosForecastValueUpsertRepository.UpsertRow> valueRows = new ArrayList<>();
    for (IemMosEntry entry : entries) {
      Instant forecastTime = entry.forecastTimeUtc();
      if (forecastTime == null) {
        continue;
      }
      Instant runtimeUtc = entry.runtimeUtc();
      if (runtimeUtc == null) {
        continue;
      }
      for (Map.Entry<String, IemMosValue> kv : entry.values().entrySet()) {
        String variableCode = kv.getKey();
        if (variableCode == null || variableCode.isBlank()) {
          continue;
        }
        IemMosValue value = kv.getValue();
        if (value == null || value.rawValue() == null || value.rawValue().isBlank()) {
          continue;
        }
        String raw = value.rawValue();
        String text = value.numericValue() == null ? raw : null;
        valueRows.add(new MosForecastValueUpsertRepository.UpsertRow(
            stationId,
            modelName,
            runtimeUtc,
            forecastTime,
            variableCode,
            value.numericValue(),
            text,
            raw,
            payload.rawPayloadHash(),
            retrievedAtUtc));
      }
    }
    if (!valueRows.isEmpty()) {
      forecastValueRepository.upsertAll(valueRows);
    }
  }

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }
}
