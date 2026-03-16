package com.predictionmarkets.weather.pilot.source.ncei;

import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.catalog.SourceInventoryRecord;
import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.pilot.config.StationAlias;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.manifest.HttpRequestLogRecord;
import com.predictionmarkets.weather.pilot.manifest.ManifestService;
import com.predictionmarkets.weather.pilot.metrics.JobMetricsAccumulator;
import com.predictionmarkets.weather.pilot.source.HttpResponseData;
import com.predictionmarkets.weather.pilot.source.SourceHttpClient;
import com.predictionmarkets.weather.pilot.storage.RawStorageService;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class Lcdv2Fetcher {
  private static final String LCD_BASE_URL =
      "https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/";

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public Lcdv2Fetcher(SourceHttpClient httpClient,
                      ManifestService manifestService,
                      RawStorageService rawStorageService,
                      SqliteCatalogService catalogService,
                      JobRunService jobRunService,
                      PilotIngestionProperties properties) {
    this.httpClient = httpClient;
    this.manifestService = manifestService;
    this.rawStorageService = rawStorageService;
    this.catalogService = catalogService;
    this.jobRunService = jobRunService;
    this.properties = properties;
  }

  public int ingestRange(String jobId,
                         String runId,
                         StationConfig station,
                         String startDate,
                         String endDate,
                         JobMetricsAccumulator metricsAccumulator) {
    LocalDate start = LocalDate.parse(startDate);
    LocalDate end = LocalDate.parse(endDate);
    int totalRows = 0;
    for (int year = start.getYear(); year <= end.getYear(); year++) {
      totalRows += ingestYear(jobId, runId, station, year, start, end, metricsAccumulator);
    }
    return totalRows;
  }

  private int ingestYear(String jobId,
                         String runId,
                         StationConfig station,
                         int year,
                         LocalDate startDate,
                         LocalDate endDate,
                         JobMetricsAccumulator metricsAccumulator) {
    String stationId = aliasValue(station, "ghcnh_or_ghcnd");
    String url = LCD_BASE_URL + year + "/LCD_" + stationId + "_" + year + ".csv";
    HttpResponseData response = httpClient.get("ncei", url, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId,
        jobId,
        "lcdv2_aux",
        "ncei",
        station.getStationKey(),
        "fetch_lcdv2_year",
        url,
        "GET",
        response.statusCode(),
        null,
        null,
        response.durationMs(),
        response.body().length,
        null,
        response.retryCount(),
        response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null,
        null,
        Instant.now().toString()));
    String payload = new String(response.body(), StandardCharsets.UTF_8);
    String checksum = rawStorageService.storeText(
        runId,
        station.getStationKey(),
        "lcdv2_aux",
        "ncei",
        stationId + "::" + year,
        url,
        response.statusCode(),
        payload,
        response.statusCode() == 200 ? "RAW_STORED" : "HTTP_" + response.statusCode(),
        0);
    if (response.statusCode() == 404) {
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          "lcdv2_aux",
          "ncei",
          "annual_file",
          stationId + "::" + year,
          null,
          null,
          "MISSING",
          catalogService.toJson(Map.of("url", url, "year", year)),
          Instant.now().toString(),
          Instant.now().toString()));
      metricsAccumulator.recordMissing("lcdv2_aux");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "SKIPPED");
      return 0;
    }
    if (response.statusCode() < 200 || response.statusCode() >= 300) {
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("LCDv2 request failed with status " + response.statusCode());
    }
    try (CSVParser parser = CSVParser.parse(new StringReader(payload), CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .build())) {
      int hourlyRows = 0;
      int dailyRows = 0;
      Set<String> seenDailyDates = new LinkedHashSet<>();
      ZoneOffset standardOffset = standardOffset(station, year);
      for (CSVRecord record : parser) {
        LocalDateTime localDateTime = parseDateTime(record.get("DATE"));
        if (localDateTime == null) {
          continue;
        }
        LocalDate localDate = localDateTime.toLocalDate();
        if (localDate.isBefore(startDate) || localDate.isAfter(endDate)) {
          continue;
        }
        String validTimeUtc = localDateTime.toInstant(standardOffset).toString();
        String rowJson = catalogService.toJson(record.toMap());
        catalogService.execute("""
            INSERT INTO lcdv2_hourly_aux (
              station_key, valid_time_utc, data_json, source_name, source_family,
              source_identifier, request_url_or_bucket_key, issue_time_utc, ingested_at_utc,
              raw_object_sha256, parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_key, valid_time_utc) DO UPDATE SET
              data_json=excluded.data_json,
              ingested_at_utc=excluded.ingested_at_utc,
              raw_object_sha256=excluded.raw_object_sha256
            """,
            station.getStationKey(),
            validTimeUtc,
            rowJson,
            "lcdv2_aux",
            "ncei",
            stationId + "::" + validTimeUtc,
            url,
            null,
            Instant.now().toString(),
            checksum,
            properties.getParserVersion());
        hourlyRows++;

        if (hasDailyContent(record) && seenDailyDates.add(localDate.toString())) {
          catalogService.execute("""
              INSERT INTO lcdv2_daily_aux (
                station_key, date_local, data_json, source_name, source_family,
                source_identifier, request_url_or_bucket_key, issue_time_utc, ingested_at_utc,
                raw_object_sha256, parser_version
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              ON CONFLICT(station_key, date_local) DO UPDATE SET
                data_json=excluded.data_json,
                ingested_at_utc=excluded.ingested_at_utc,
                raw_object_sha256=excluded.raw_object_sha256
              """,
              station.getStationKey(),
              localDate.toString(),
              catalogService.toJson(dailySubset(record.toMap())),
              "lcdv2_aux",
              "ncei",
              stationId + "::daily::" + localDate,
              url,
              null,
              Instant.now().toString(),
              checksum,
              properties.getParserVersion());
          dailyRows++;
        }
      }
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          "lcdv2_aux",
          "ncei",
          "annual_file",
          stationId + "::" + year,
          null,
          null,
          "INGESTED",
          catalogService.toJson(Map.of(
              "url", url,
              "year", year,
              "hourlyRows", hourlyRows,
              "dailyRows", dailyRows)),
          Instant.now().toString(),
          Instant.now().toString()));
      manifestService.recordNormalizedPartition(
          runId,
          "lcdv2_hourly_aux",
          station.getStationKey(),
          year + "-01-01",
          hourlyRows,
          checksum,
          catalogService.toJson(Map.of("year", year, "url", url)));
      manifestService.recordNormalizedPartition(
          runId,
          "lcdv2_daily_aux",
          station.getStationKey(),
          year + "-01-01",
          dailyRows,
          checksum,
          catalogService.toJson(Map.of("year", year, "url", url)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "lcdv2_aux",
          "normalize_lcdv2_year", "SUCCESS", Map.of(
              "year", year,
              "hourly_rows", hourlyRows,
              "daily_rows", dailyRows));
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, hourlyRows + dailyRows, "SUCCESS");
      return hourlyRows + dailyRows;
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("lcdv2_aux");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse LCDv2 file for year " + year, ex);
    }
  }

  private Map<String, String> dailySubset(Map<String, String> row) {
    Map<String, String> filtered = new LinkedHashMap<>();
    for (Map.Entry<String, String> entry : row.entrySet()) {
      String key = entry.getKey();
      if (key.startsWith("Daily") || "Sunrise".equals(key) || "Sunset".equals(key)
          || "DATE".equals(key) || "REPORT_TYPE".equals(key) || "REM".equals(key)) {
        filtered.put(key, entry.getValue());
      }
    }
    return filtered;
  }

  private boolean hasDailyContent(CSVRecord record) {
    for (String key : record.toMap().keySet()) {
      if (key.startsWith("Daily") || "Sunrise".equals(key) || "Sunset".equals(key)) {
        String value = record.get(key);
        if (value != null && !value.isBlank()) {
          return true;
        }
      }
    }
    return false;
  }

  private LocalDateTime parseDateTime(String raw) {
    if (raw == null || raw.isBlank()) {
      return null;
    }
    return LocalDateTime.parse(raw.trim());
  }

  private ZoneOffset standardOffset(StationConfig station, int year) {
    ZoneId zoneId = ZoneId.of(station.getTimezone());
    return zoneId.getRules().getStandardOffset(LocalDate.of(year, 1, 15).atStartOfDay(zoneId).toInstant());
  }

  private String aliasValue(StationConfig station, String aliasType) {
    return station.getAliases().stream()
        .filter(alias -> aliasType.equalsIgnoreCase(alias.getType()))
        .map(StationAlias::getValue)
        .findFirst()
        .orElseThrow(() -> new IllegalStateException(
            "Missing alias " + aliasType + " for station " + station.getStationKey()))
        .trim()
        .toUpperCase(Locale.ROOT);
  }
}
