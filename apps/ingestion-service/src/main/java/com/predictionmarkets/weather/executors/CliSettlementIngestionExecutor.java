package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.cli.CliDailyCsvExportService;
import com.predictionmarkets.weather.cli.CliDailyIngestService;
import com.predictionmarkets.weather.config.CliSettlementIngestionProperties;
import com.predictionmarkets.weather.iem.IemCliDaily;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

public final class CliSettlementIngestionExecutor {
  private static final Logger logger =
      LoggerFactory.getLogger(CliSettlementIngestionExecutor.class);

  private CliSettlementIngestionExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      CliDailyIngestService cliDailyIngestService =
          context.getBean(CliDailyIngestService.class);
      CliDailyCsvExportService csvExportService =
          context.getBean(CliDailyCsvExportService.class);
      CliSettlementIngestionProperties properties =
          context.getBean(CliSettlementIngestionProperties.class);
      List<String> stationIds = normalizeStationIds(properties.getStationIds());
      LocalDate start = requireStartDate(properties.getStartDateLocal());
      LocalDate end = requireEndDate(properties.getEndDateLocal());
      validateDateRange(start, end);
      validateExecutionMode(properties);
      int sourceFetchThreads = requireSourceFetchThreads(properties.getSourceFetchThreads());
      snapshot("CLI settlement ingest starting stations=" + stationIds
          + " range=" + start + ".." + end
          + " ingestEnabled=" + properties.isIngestEnabled()
          + " exportEnabled=" + (properties.getExport() != null && properties.getExport().isEnabled())
          + " sourceFetchThreads=" + sourceFetchThreads);
      for (String stationId : stationIds) {
        if (properties.isIngestEnabled()) {
          int upserted = cliDailyIngestService.ingestRange(stationId, start, end);
          snapshot("CLI settlement ingest complete station=" + stationId
              + " upserted=" + upserted);
        }
        if (properties.getExport() != null && properties.getExport().isEnabled()) {
          List<IemCliDaily> days = cliDailyIngestService.fetchRangeFromSource(
              stationId,
              start,
              end,
              sourceFetchThreads);
          Path outputPath = resolveExportPath(properties, stationId, start, end);
          long exported = csvExportService.exportFetchedDays(
              stationId,
              start,
              end,
              days,
              outputPath,
              properties.getExport().isIncludeHeader());
          snapshot("CLI settlement CSV export complete station=" + stationId
              + " rows=" + exported
              + " outputPath=" + outputPath.toAbsolutePath());
        }
      }
      snapshot("CLI settlement ingest finished.");
    }
  }

  private static int requireSourceFetchThreads(int sourceFetchThreads) {
    if (sourceFetchThreads < 1) {
      throw new IllegalArgumentException("cli-settlement.source-fetch-threads must be >= 1");
    }
    return sourceFetchThreads;
  }

  private static void validateExecutionMode(CliSettlementIngestionProperties properties) {
    boolean ingestEnabled = properties.isIngestEnabled();
    boolean exportEnabled = properties.getExport() != null && properties.getExport().isEnabled();
    if (!ingestEnabled && !exportEnabled) {
      throw new IllegalArgumentException(
          "cli-settlement requires ingest-enabled=true or export.enabled=true");
    }
    if (exportEnabled) {
      String outputDir = properties.getExport().getOutputDir();
      if (outputDir == null || outputDir.isBlank()) {
        throw new IllegalArgumentException("cli-settlement.export.output-dir is required");
      }
    }
  }

  private static Path resolveExportPath(CliSettlementIngestionProperties properties,
                                        String stationId,
                                        LocalDate start,
                                        LocalDate end) {
    String fileName = stationId.toUpperCase(Locale.ROOT) + "_cli_settlement_"
        + start + "_" + end + ".csv";
    return Paths.get(properties.getExport().getOutputDir()).resolve(fileName);
  }

  private static List<String> normalizeStationIds(List<String> stationIds) {
    if (stationIds == null || stationIds.isEmpty()) {
      throw new IllegalArgumentException("cli-settlement.station-ids is required");
    }
    Set<String> normalized = new LinkedHashSet<>();
    for (String stationId : stationIds) {
      if (stationId == null || stationId.isBlank()) {
        continue;
      }
      normalized.add(stationId.trim().toUpperCase(Locale.ROOT));
    }
    if (normalized.isEmpty()) {
      throw new IllegalArgumentException("cli-settlement.station-ids is required");
    }
    return new ArrayList<>(normalized);
  }

  private static LocalDate requireStartDate(LocalDate start) {
    if (start == null) {
      throw new IllegalArgumentException("cli-settlement.start-date-local is required");
    }
    return start;
  }

  private static LocalDate requireEndDate(LocalDate end) {
    if (end == null) {
      throw new IllegalArgumentException("cli-settlement.end-date-local is required");
    }
    return end;
  }

  private static void validateDateRange(LocalDate start, LocalDate end) {
    if (end.isBefore(start)) {
      throw new IllegalArgumentException("cli-settlement date range must be start <= end");
    }
  }

  private static void snapshot(String message) {
    String payload = "[CLI-SETTLEMENT] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
