package com.predictionmarkets.weather.cli;

import com.predictionmarkets.weather.iem.IemCliClient;
import com.predictionmarkets.weather.iem.IemCliDaily;
import com.predictionmarkets.weather.iem.IemCliPayload;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.CliDailyUpsertRepository;
import com.predictionmarkets.weather.repository.CliDailyUpsertRepository.UpsertRow;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class CliDailyIngestService {
  private static final Logger logger = LoggerFactory.getLogger(CliDailyIngestService.class);

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

  public List<IemCliDaily> fetchRangeFromSource(String stationId, LocalDate startDate, LocalDate endDate) {
    return fetchRangeFromSource(stationId, startDate, endDate, 1);
  }

  public List<IemCliDaily> fetchRangeFromSource(String stationId,
                                                LocalDate startDate,
                                                LocalDate endDate,
                                                int sourceFetchThreads) {
    String normalizedStation = normalizeStationId(stationId);
    if (startDate == null || endDate == null) {
      throw new IllegalArgumentException("startDate and endDate are required");
    }
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("endDate must be >= startDate");
    }
    if (sourceFetchThreads < 1) {
      throw new IllegalArgumentException("sourceFetchThreads must be >= 1");
    }
    int yearCount = (endDate.getYear() - startDate.getYear()) + 1;
    logger.info(
        "[CLI-SETTLEMENT] source fetch start station={} range={}..{} years={} threads={}",
        normalizedStation,
        startDate,
        endDate,
        yearCount,
        sourceFetchThreads);
    List<IemCliDaily> all = new ArrayList<>();
    if (sourceFetchThreads == 1) {
      for (int year = startDate.getYear(); year <= endDate.getYear(); year++) {
        YearFetchResult result = fetchYearAndFilter(normalizedStation, year, startDate, endDate);
        all.addAll(result.days());
      }
    } else {
      ExecutorService executor = Executors.newFixedThreadPool(sourceFetchThreads);
      try {
        List<CompletableFuture<YearFetchResult>> futures = new ArrayList<>(yearCount);
        for (int year = startDate.getYear(); year <= endDate.getYear(); year++) {
          final int fetchYear = year;
          futures.add(CompletableFuture.supplyAsync(
              () -> fetchYearAndFilter(normalizedStation, fetchYear, startDate, endDate),
              executor));
        }
        int completed = 0;
        for (CompletableFuture<YearFetchResult> future : futures) {
          YearFetchResult result = future.join();
          completed++;
          all.addAll(result.days());
          logger.info(
              "[CLI-SETTLEMENT] source fetch progress station={} completedYears={}/{} lastYear={} lastRows={}",
              normalizedStation,
              completed,
              yearCount,
              result.year(),
              result.days().size());
        }
      } catch (CompletionException ex) {
        throw new IllegalStateException("Failed CLI source fetch for station=" + normalizedStation, ex.getCause());
      } finally {
        shutdownExecutor(executor);
      }
    }
    all.sort(Comparator.comparing(IemCliDaily::targetDateLocal));
    logger.info(
        "[CLI-SETTLEMENT] source fetch complete station={} totalRows={} range={}..{}",
        normalizedStation,
        all.size(),
        startDate,
        endDate);
    return all;
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

  private YearFetchResult fetchYearAndFilter(String stationId,
                                             int year,
                                             LocalDate startDate,
                                             LocalDate endDate) {
    long startNs = System.nanoTime();
    logger.info("[CLI-SETTLEMENT] source fetch year start station={} year={}", stationId, year);
    IemCliPayload payload = cliClient.fetchYear(stationId, year);
    List<IemCliDaily> filtered = filterDays(payload.days(), startDate, endDate);
    long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNs);
    logger.info(
        "[CLI-SETTLEMENT] source fetch year done station={} year={} rows={} elapsedMs={}",
        stationId,
        year,
        filtered.size(),
        elapsedMs);
    return new YearFetchResult(year, filtered);
  }

  private void shutdownExecutor(ExecutorService executor) {
    executor.shutdown();
    try {
      if (!executor.awaitTermination(15, TimeUnit.SECONDS)) {
        executor.shutdownNow();
      }
    } catch (InterruptedException ex) {
      executor.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }

  private record YearFetchResult(int year, List<IemCliDaily> days) {
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
