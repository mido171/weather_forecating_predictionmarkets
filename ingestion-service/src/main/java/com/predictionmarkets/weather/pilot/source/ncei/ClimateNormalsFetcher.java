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
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class ClimateNormalsFetcher {
  private static final String DAILY_BASE_URL =
      "https://noaa-normals-pds.s3.amazonaws.com/normals-daily/1991-2020/access/";
  private static final String HOURLY_BASE_URL =
      "https://noaa-normals-pds.s3.amazonaws.com/normals-hourly/1991-2020/access/";

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public ClimateNormalsFetcher(SourceHttpClient httpClient,
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

  public int ingestStationNormals(String jobId,
                                  String runId,
                                  StationConfig station,
                                  JobMetricsAccumulator metricsAccumulator) {
    String stationId = aliasValue(station, "ghcnh_or_ghcnd");
    int rows = ingestDaily(jobId, runId, station, stationId, metricsAccumulator);
    rows += ingestHourly(jobId, runId, station, hourlyCandidates(station), metricsAccumulator);
    return rows;
  }

  private int ingestDaily(String jobId,
                          String runId,
                          StationConfig station,
                          String stationId,
                          JobMetricsAccumulator metricsAccumulator) {
    String url = DAILY_BASE_URL + stationId + ".csv";
    HttpResponseData response = httpClient.get("ncei", url, Map.of());
    recordRequest(runId, jobId, station.getStationKey(), "fetch_daily_normals", url, response);
    String payload = new String(response.body(), StandardCharsets.UTF_8);
    String checksum = rawStorageService.storeText(
        runId,
        station.getStationKey(),
        "climate_normals",
        "ncei",
        stationId + "::daily",
        url,
        response.statusCode(),
        payload,
        response.statusCode() == 200 ? "RAW_STORED" : "HTTP_" + response.statusCode(),
        0);
    if (response.statusCode() == 404) {
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          "climate_normals",
          "ncei",
          "daily_station_file",
          stationId,
          null,
          null,
          "MISSING",
          catalogService.toJson(Map.of("url", url)),
          Instant.now().toString(),
          Instant.now().toString()));
      metricsAccumulator.recordMissing("climate_normals_daily");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "SKIPPED");
      return 0;
    }
    if (response.statusCode() < 200 || response.statusCode() >= 300) {
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Daily normals request failed with status " + response.statusCode());
    }
    try (CSVParser parser = CSVParser.parse(new StringReader(payload), CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .build())) {
      int rows = 0;
      for (CSVRecord record : parser) {
        int dayOfYear = toDayOfYear(record.get("DATE"));
        catalogService.execute("""
            INSERT INTO climate_normals_daily (
              station_key, day_of_year, normal_tmax_f, normal_tmin_f, normal_tavg_f,
              static_prior, not_observation, not_forecast, source_name, source_family,
              source_identifier, request_url_or_bucket_key, ingested_at_utc, raw_object_sha256,
              parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_key, day_of_year) DO UPDATE SET
              normal_tmax_f=excluded.normal_tmax_f,
              normal_tmin_f=excluded.normal_tmin_f,
              normal_tavg_f=excluded.normal_tavg_f,
              ingested_at_utc=excluded.ingested_at_utc,
              raw_object_sha256=excluded.raw_object_sha256
            """,
            station.getStationKey(),
            dayOfYear,
            parseDouble(record.get("DLY-TMAX-NORMAL")),
            parseDouble(record.get("DLY-TMIN-NORMAL")),
            parseDouble(record.get("DLY-TAVG-NORMAL")),
            1,
            1,
            1,
            "climate_normals",
            "ncei",
            stationId + "::daily::" + record.get("DATE"),
            url,
            Instant.now().toString(),
            checksum,
            properties.getParserVersion());
        rows++;
      }
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          "climate_normals",
          "ncei",
          "daily_station_file",
          stationId,
          null,
          null,
          "INGESTED",
          catalogService.toJson(Map.of("url", url, "rows", rows)),
          Instant.now().toString(),
          Instant.now().toString()));
      manifestService.recordNormalizedPartition(
          runId,
          "climate_normals_daily",
          station.getStationKey(),
          "1991-2020",
          rows,
          checksum,
          catalogService.toJson(Map.of("stationId", stationId, "url", url)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "climate_normals",
          "normalize_daily_normals", "SUCCESS", Map.of("rows_parsed", rows, "stationId", stationId));
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows, "SUCCESS");
      return rows;
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("climate_normals_daily");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse daily climate normals for " + stationId, ex);
    }
  }

  private int ingestHourly(String jobId,
                           String runId,
                           StationConfig station,
                           List<String> candidateIds,
                           JobMetricsAccumulator metricsAccumulator) {
    for (String candidateId : candidateIds) {
      String url = HOURLY_BASE_URL + candidateId + ".csv";
      HttpResponseData response = httpClient.get("ncei", url, Map.of());
      recordRequest(runId, jobId, station.getStationKey(), "fetch_hourly_normals", url, response);
      String payload = new String(response.body(), StandardCharsets.UTF_8);
      String checksum = rawStorageService.storeText(
          runId,
          station.getStationKey(),
          "climate_normals",
          "ncei",
          candidateId + "::hourly",
          url,
          response.statusCode(),
          payload,
          response.statusCode() == 200 ? "RAW_STORED" : "HTTP_" + response.statusCode(),
          0);
      if (response.statusCode() == 404) {
        metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "SKIPPED");
        continue;
      }
      if (response.statusCode() < 200 || response.statusCode() >= 300) {
        metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
        throw new IllegalStateException("Hourly normals request failed with status " + response.statusCode());
      }
      try (CSVParser parser = CSVParser.parse(new StringReader(payload), CSVFormat.DEFAULT.builder()
          .setHeader()
          .setSkipHeaderRecord(true)
          .build())) {
        int rows = 0;
        for (CSVRecord record : parser) {
          int dayOfYear = toDayOfYear(record.get("DATE"));
          Integer hourLocalStandard = parseInteger(record.get("hour"));
          if (hourLocalStandard == null) {
            continue;
          }
          catalogService.execute("""
              INSERT INTO climate_normals_hourly (
                station_key, day_of_year, hour_local_standard, normal_temp_f,
                static_prior, not_observation, not_forecast, source_name, source_family,
                source_identifier, request_url_or_bucket_key, ingested_at_utc, raw_object_sha256,
                parser_version
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              ON CONFLICT(station_key, day_of_year, hour_local_standard) DO UPDATE SET
                normal_temp_f=excluded.normal_temp_f,
                ingested_at_utc=excluded.ingested_at_utc,
                raw_object_sha256=excluded.raw_object_sha256
              """,
              station.getStationKey(),
              dayOfYear,
              hourLocalStandard,
              parseDouble(record.get("HLY-TEMP-NORMAL")),
              1,
              1,
              1,
              "climate_normals",
              "ncei",
              candidateId + "::hourly::" + record.get("DATE"),
              url,
              Instant.now().toString(),
              checksum,
              properties.getParserVersion());
          rows++;
        }
        manifestService.upsertSourceInventory(new SourceInventoryRecord(
            UUID.randomUUID().toString(),
            station.getStationKey(),
            "climate_normals",
            "ncei",
            "hourly_station_file",
            candidateId,
            null,
            null,
            "INGESTED",
            catalogService.toJson(Map.of("url", url, "rows", rows)),
            Instant.now().toString(),
            Instant.now().toString()));
        manifestService.recordNormalizedPartition(
            runId,
            "climate_normals_hourly",
            station.getStationKey(),
            "1991-2020",
            rows,
            checksum,
            catalogService.toJson(Map.of("stationId", candidateId, "url", url)));
        jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "climate_normals",
            "normalize_hourly_normals", "SUCCESS", Map.of("rows_parsed", rows, "stationId", candidateId));
        metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows, "SUCCESS");
        return rows;
      } catch (Exception ex) {
        metricsAccumulator.recordParserFailure("climate_normals_hourly");
        metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
        throw new IllegalStateException("Failed to parse hourly climate normals for " + candidateId, ex);
      }
    }
    manifestService.upsertSourceInventory(new SourceInventoryRecord(
        UUID.randomUUID().toString(),
        station.getStationKey(),
        "climate_normals",
        "ncei",
        "hourly_station_file",
        station.getStationKey(),
        null,
        null,
        "MISSING",
        catalogService.toJson(Map.of("candidateIds", candidateIds)),
        Instant.now().toString(),
        Instant.now().toString()));
    jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "climate_normals",
        "hourly_normals_missing", "MISSING", Map.of("candidateIds", candidateIds));
    metricsAccumulator.recordMissing("climate_normals_hourly");
    return 0;
  }

  private void recordRequest(String runId,
                             String jobId,
                             String stationKey,
                             String action,
                             String url,
                             HttpResponseData response) {
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId,
        jobId,
        "climate_normals",
        "ncei",
        stationKey,
        action,
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
  }

  private List<String> hourlyCandidates(StationConfig station) {
    LinkedHashSet<String> candidates = new LinkedHashSet<>();
    candidates.add(aliasValue(station, "ghcnh_or_ghcnd"));
    String wban = aliasValue(station, "wban");
    candidates.add("FMW000" + wban);
    candidates.add("USW000" + wban);
    for (StationAlias alias : station.getAliases()) {
      candidates.add(alias.getValue().trim().toUpperCase(Locale.ROOT));
    }
    return new ArrayList<>(candidates);
  }

  private int toDayOfYear(String rawDate) {
    String value = rawDate == null ? "" : rawDate.trim();
    String monthDay = value.length() >= 5 ? value.substring(0, 5) : value;
    String[] parts = monthDay.split("-");
    int month = Integer.parseInt(parts[0]);
    int day = Integer.parseInt(parts[1]);
    int year = month == 2 && day == 29 ? 2000 : 2001;
    return LocalDate.of(year, month, day).getDayOfYear();
  }

  private Double parseDouble(String raw) {
    if (raw == null) {
      return null;
    }
    String value = raw.trim();
    if (value.isEmpty()) {
      return null;
    }
    try {
      double parsed = Double.parseDouble(value);
      return parsed <= -7000.0d ? null : parsed;
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private Integer parseInteger(String raw) {
    if (raw == null || raw.isBlank()) {
      return null;
    }
    try {
      return Integer.valueOf(raw.trim());
    } catch (NumberFormatException ex) {
      return null;
    }
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
