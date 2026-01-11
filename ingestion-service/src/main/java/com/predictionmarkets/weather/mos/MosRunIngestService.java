package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.iem.IemMosClient;
import com.predictionmarkets.weather.iem.IemMosEntry;
import com.predictionmarkets.weather.iem.IemMosPayload;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.repository.MosRunUpsertRepository;
import com.predictionmarkets.weather.repository.MosRunUpsertRepository.UpsertRow;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class MosRunIngestService {
  private final IemMosClient mosClient;
  private final MosRunUpsertRepository upsertRepository;
  private final StationRegistryRepository stationRegistryRepository;

  public MosRunIngestService(
      IemMosClient mosClient,
      MosRunUpsertRepository upsertRepository,
      StationRegistryRepository stationRegistryRepository) {
    this.mosClient = mosClient;
    this.upsertRepository = upsertRepository;
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

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }
}
