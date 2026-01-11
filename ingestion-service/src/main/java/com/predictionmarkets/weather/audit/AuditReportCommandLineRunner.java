package com.predictionmarkets.weather.audit;

import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class AuditReportCommandLineRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(AuditReportCommandLineRunner.class);

  private final AuditReportService auditReportService;
  private final AuditReportProperties properties;
  private final StationRegistryRepository stationRegistryRepository;

  public AuditReportCommandLineRunner(AuditReportService auditReportService,
                                      AuditReportProperties properties,
                                      StationRegistryRepository stationRegistryRepository) {
    this.auditReportService = auditReportService;
    this.properties = properties;
    this.stationRegistryRepository = stationRegistryRepository;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    List<String> seriesTickers = splitSeriesTickers(properties.getSeriesTicker());
    if (seriesTickers.isEmpty()) {
      throw new IllegalArgumentException("audit.series-ticker is required");
    }
    LocalDate startDate = Objects.requireNonNull(properties.getDateStartLocal(),
        "audit.date-start-local is required");
    LocalDate endDate = Objects.requireNonNull(properties.getDateEndLocal(),
        "audit.date-end-local is required");
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("audit.date-end-local must be >= audit.date-start-local");
    }
    if (properties.getAsofPolicyId() == null) {
      throw new IllegalArgumentException("audit.asof-policy-id is required");
    }

    List<String> stationIds = new ArrayList<>();
    for (String ticker : seriesTickers) {
      String normalized = normalizeSeriesTicker(ticker);
      StationRegistry station = stationRegistryRepository.findBySeriesTicker(normalized)
          .orElseThrow(() -> new IllegalArgumentException(
              "Station registry not found for series ticker: " + normalized));
      stationIds.add(station.getStationId());
    }

    AuditRequest request = new AuditRequest(
        stationIds,
        startDate,
        endDate,
        properties.getAsofPolicyId(),
        properties.getModels(),
        properties.getMaxForecastDays(),
        properties.getSampleLimit());
    Path outputDir = properties.getOutputDir() == null
        ? Path.of("reports")
        : Path.of(properties.getOutputDir());
    AuditReportArtifacts artifacts = auditReportService.generateAndWriteReport(request, outputDir);
    logger.info("Audit report written markdown={} json={}",
        artifacts.markdownPath(), artifacts.jsonPath());
  }

  private List<String> splitSeriesTickers(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      return List.of();
    }
    String[] parts = seriesTicker.split(",");
    List<String> tickers = new ArrayList<>();
    for (String part : parts) {
      String trimmed = part.trim();
      if (!trimmed.isEmpty()) {
        tickers.add(trimmed);
      }
    }
    return List.copyOf(tickers);
  }

  private String normalizeSeriesTicker(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      throw new IllegalArgumentException("seriesTicker is required");
    }
    return seriesTicker.trim().toUpperCase(Locale.ROOT);
  }
}
