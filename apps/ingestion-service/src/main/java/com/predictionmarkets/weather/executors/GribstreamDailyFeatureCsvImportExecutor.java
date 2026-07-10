package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.config.GribstreamTransferProperties;
import com.predictionmarkets.weather.gribstream.GribstreamDailyFeatureCsv;
import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.repository.GribstreamDailyFeatureRepository;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

public final class GribstreamDailyFeatureCsvImportExecutor {
  private static final Logger logger =
      LoggerFactory.getLogger(GribstreamDailyFeatureCsvImportExecutor.class);
  private static final double VALUE_TOLERANCE = 1e-6;

  private GribstreamDailyFeatureCsvImportExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamDailyFeatureRepository repository =
          context.getBean(GribstreamDailyFeatureRepository.class);
      GribstreamTransferProperties properties = context.getBean(GribstreamTransferProperties.class);
      GribstreamTransferProperties.Import importConfig = properties.getImport();
      int batchSize = importConfig.getBatchSize();
      if (batchSize < 1) {
        throw new IllegalArgumentException("gribstream.transfer.import.batch-size must be >= 1");
      }
      Path inputPath = Paths.get(importConfig.getInputPath());
      snapshot("CSV import starting inputPath=" + inputPath.toAbsolutePath()
          + " batchSize=" + batchSize
          + " hasHeader=" + importConfig.isHasHeader());
      ImportStats stats = importFromCsv(repository, inputPath, batchSize, importConfig.isHasHeader());
      snapshot("CSV import complete rows=" + stats.total()
          + " inserted=" + stats.inserted()
          + " updated=" + stats.updated()
          + " skipped=" + stats.skipped());
    }
  }

  private static ImportStats importFromCsv(GribstreamDailyFeatureRepository repository,
                                           Path inputPath,
                                           int batchSize,
                                           boolean hasHeader) {
    long processed = 0L;
    long inserted = 0L;
    long updated = 0L;
    long skipped = 0L;
    try (BufferedReader reader = Files.newBufferedReader(inputPath, StandardCharsets.UTF_8)) {
      String record = GribstreamDailyFeatureCsv.readRecord(reader);
      if (record == null) {
        return new ImportStats(0, 0, 0, 0);
      }
      Map<String, Integer> headerIndex;
      if (hasHeader) {
        List<String> headerValues = GribstreamDailyFeatureCsv.parseRecord(record);
        headerIndex = GribstreamDailyFeatureCsv.headerIndex(headerValues);
        GribstreamDailyFeatureCsv.validateHeader(headerIndex);
        record = GribstreamDailyFeatureCsv.readRecord(reader);
      } else {
        headerIndex = GribstreamDailyFeatureCsv.headerIndex(GribstreamDailyFeatureCsv.HEADER);
      }
      while (record != null) {
        List<String> values = GribstreamDailyFeatureCsv.parseRecord(record);
        GribstreamDailyFeatureEntity candidate = GribstreamDailyFeatureCsv.toEntity(headerIndex, values);
        UpsertOutcome outcome = upsert(repository, candidate);
        if (outcome == UpsertOutcome.INSERTED) {
          inserted++;
        } else if (outcome == UpsertOutcome.UPDATED) {
          updated++;
        } else {
          skipped++;
        }
        processed++;
        if (processed % batchSize == 0) {
          snapshot("CSV import progress rows=" + processed
              + " inserted=" + inserted
              + " updated=" + updated
              + " skipped=" + skipped);
        }
        record = GribstreamDailyFeatureCsv.readRecord(reader);
      }
      return new ImportStats(processed, inserted, updated, skipped);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read CSV import from " + inputPath, ex);
    }
  }

  private static UpsertOutcome upsert(GribstreamDailyFeatureRepository repository,
                                      GribstreamDailyFeatureEntity candidate) {
    Optional<GribstreamDailyFeatureEntity> existing =
        repository.findByStationIdAndTargetDateLocalAndAsofUtcAndModelCodeAndMetric(
            candidate.getStationId(),
            candidate.getTargetDateLocal(),
            candidate.getAsofUtc(),
            candidate.getModelCode(),
            candidate.getMetric());
    if (existing.isEmpty()) {
      repository.save(candidate);
      return UpsertOutcome.INSERTED;
    }
    GribstreamDailyFeatureEntity current = existing.get();
    if (isEquivalent(current, candidate)) {
      return UpsertOutcome.SKIPPED;
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
    return UpsertOutcome.UPDATED;
  }

  private static boolean isEquivalent(GribstreamDailyFeatureEntity left,
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

  private static boolean almostEqual(Double left, Double right) {
    if (left == null || right == null) {
      return left == null && right == null;
    }
    return Math.abs(left - right) <= VALUE_TOLERANCE;
  }

  private static void snapshot(String message) {
    String payload = "[GRIBSTREAM-CSV-IMPORT] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private enum UpsertOutcome {
    INSERTED,
    UPDATED,
    SKIPPED
  }

  private record ImportStats(long total, long inserted, long updated, long skipped) {
  }
}
