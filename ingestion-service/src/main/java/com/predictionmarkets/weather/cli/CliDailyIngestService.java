package com.predictionmarkets.weather.cli;

import com.predictionmarkets.weather.iem.IemCliClient;
import com.predictionmarkets.weather.iem.IemCliDaily;
import com.predictionmarkets.weather.iem.IemCliPayload;
import com.predictionmarkets.weather.repository.CliDailyUpsertRepository;
import com.predictionmarkets.weather.repository.CliDailyUpsertRepository.UpsertRow;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import com.predictionmarkets.weather.models.StationRegistry;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class CliDailyIngestService {
  private final IemCliClient cliClient;
  private final CliDailyUpsertRepository upsertRepository;
  private final StationRegistryRepository stationRegistryRepository;

  public CliDailyIngestService(
      IemCliClient cliClient,
      CliDailyUpsertRepository upsertRepository,
      StationRegistryRepository stationRegistryRepository) {
    this.cliClient = cliClient;
    this.upsertRepository = upsertRepository;
    this.stationRegistryRepository = stationRegistryRepository;
  }

  @Transactional
  public int ingestYear(String stationId, int year) {
    String normalizedStation = normalizeStationId(stationId);
    StationRegistry station = requireStation(normalizedStation);
    IemCliPayload payload = cliClient.fetchYear(normalizedStation, year);
    return upsertDays(payload, payload.days(), station);
  }

  @Transactional
  public int ingestRange(String stationId, LocalDate startDate, LocalDate endDate) {
    String normalizedStation = normalizeStationId(stationId);
    StationRegistry station = requireStation(normalizedStation);
    if (startDate == null || endDate == null) {
      throw new IllegalArgumentException("startDate and endDate are required");
    }
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("endDate must be >= startDate");
    }
    int total = 0;
    for (int year = startDate.getYear(); year <= endDate.getYear(); year++) {
      IemCliPayload payload = cliClient.fetchYear(normalizedStation, year);
      List<IemCliDaily> filtered = filterDays(payload.days(), startDate, endDate);
      total += upsertDays(payload, filtered, station);
    }
    return total;
  }

  private StationRegistry requireStation(String stationId) {
    return stationRegistryRepository.findById(stationId)
        .orElseThrow(() -> new IllegalArgumentException(
            "Station not found in registry: " + stationId));
  }

  private List<IemCliDaily> filterDays(List<IemCliDaily> days, LocalDate startDate, LocalDate endDate) {
    List<IemCliDaily> filtered = new ArrayList<>(days.size());
    for (IemCliDaily day : days) {
      LocalDate target = day.targetDateLocal();
      if ((target.isAfter(endDate)) || (target.isBefore(startDate))) {
        continue;
      }
      filtered.add(day);
    }
    return filtered;
  }

  private int upsertDays(IemCliPayload payload, List<IemCliDaily> days, StationRegistry station) {
    if (days.isEmpty()) {
      return 0;
    }
    Instant retrievedAtUtc = Instant.now();
    Instant updatedAtUtc = retrievedAtUtc;
    List<UpsertRow> rows = new ArrayList<>(days.size());
    for (IemCliDaily day : days) {
      String truthSourceUrl = resolveTruthSourceUrl(day, station);
      rows.add(new UpsertRow(
          payload.stationId(),
          day.targetDateLocal(),
          day.tmaxF(),
          day.tminF(),
          day.reportIssuedAtUtc(),
          truthSourceUrl,
          payload.rawPayloadHash(),
          retrievedAtUtc,
          updatedAtUtc));
    }
    int[] results = upsertRepository.upsertAll(rows);
    return Arrays.stream(results).sum();
  }

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private String resolveTruthSourceUrl(IemCliDaily day, StationRegistry station) {
    String fromPayload = day.truthSourceUrl();
    if (fromPayload != null && !fromPayload.isBlank()) {
      return fromPayload.trim();
    }
    String site = station.getWfoSite();
    String issuedby = station.getIssuedby();
    if (site == null || site.isBlank() || issuedby == null || issuedby.isBlank()) {
      return null;
    }
    return "https://forecast.weather.gov/product.php?site=" + site
        + "&product=CLI&issuedby=" + issuedby;
  }
}
