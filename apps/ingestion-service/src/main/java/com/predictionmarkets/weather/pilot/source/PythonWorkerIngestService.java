package com.predictionmarkets.weather.pilot.source;

import com.fasterxml.jackson.databind.JsonNode;
import com.predictionmarkets.weather.pilot.catalog.SourceInventoryRecord;
import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.manifest.ManifestService;
import com.predictionmarkets.weather.pilot.manifest.ObjectManifestRecord;
import com.predictionmarkets.weather.pilot.metrics.JobMetricsAccumulator;
import com.predictionmarkets.weather.pilot.storage.ChecksumService;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class PythonWorkerIngestService {
  private final ManifestService manifestService;
  private final SqliteCatalogService catalogService;
  private final ChecksumService checksumService;
  private final PilotIngestionProperties properties;

  public PythonWorkerIngestService(ManifestService manifestService,
                                   SqliteCatalogService catalogService,
                                   ChecksumService checksumService,
                                   PilotIngestionProperties properties) {
    this.manifestService = manifestService;
    this.catalogService = catalogService;
    this.checksumService = checksumService;
    this.properties = properties;
  }

  public int persistModelPointExtracts(String runId,
                                       String sourceName,
                                       String sourceFamily,
                                       StationConfig station,
                                       JsonNode workerResult,
                                       JobMetricsAccumulator metricsAccumulator) {
    Map<String, String> checksums = persistTouchedObjects(runId, sourceName, sourceFamily, station, workerResult.path("touched_objects"));
    int rows = 0;
    for (JsonNode record : workerResult.path("records")) {
      String requestKey = text(record, "request_url_or_bucket_key");
      Integer forecastHour = intOrNull(record, "forecast_hour");
      String checksum = checksums.getOrDefault(manifestKey(requestKey, forecastHour), firstChecksum(checksums));
      catalogService.execute("""
          INSERT INTO model_point_extract (
            station_key, model_name, cycle_time_utc, valid_time_utc, forecast_hour,
            variable_name, nearest_value, bilinear_value, nbr_mean, nbr_min, nbr_max,
            nbr_std, grid_source_lat, grid_source_lon, grid_distance_km, interpolation_method,
            source_name, source_family, source_identifier, request_url_or_bucket_key,
            issue_time_utc, ingested_at_utc, raw_object_sha256, parser_version
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(station_key, model_name, cycle_time_utc, valid_time_utc, forecast_hour, variable_name) DO UPDATE SET
            nearest_value=excluded.nearest_value,
            bilinear_value=excluded.bilinear_value,
            nbr_mean=excluded.nbr_mean,
            nbr_min=excluded.nbr_min,
            nbr_max=excluded.nbr_max,
            nbr_std=excluded.nbr_std,
            grid_source_lat=excluded.grid_source_lat,
            grid_source_lon=excluded.grid_source_lon,
            grid_distance_km=excluded.grid_distance_km,
            interpolation_method=excluded.interpolation_method,
            ingested_at_utc=excluded.ingested_at_utc,
            raw_object_sha256=excluded.raw_object_sha256
          """,
          station.getStationKey(),
          text(record, "model_name"),
          text(record, "cycle_time_utc"),
          text(record, "valid_time_utc"),
          forecastHour,
          text(record, "variable_name"),
          doubleOrNull(record, "nearest_value"),
          doubleOrNull(record, "bilinear_value"),
          doubleOrNull(record, "nbr_mean"),
          doubleOrNull(record, "nbr_min"),
          doubleOrNull(record, "nbr_max"),
          doubleOrNull(record, "nbr_std"),
          doubleOrNull(record, "grid_source_lat"),
          doubleOrNull(record, "grid_source_lon"),
          doubleOrNull(record, "grid_distance_km"),
          text(record, "interpolation_method"),
          sourceName,
          sourceFamily,
          text(record, "source_identifier"),
          requestKey,
          text(record, "issue_time_utc"),
          Instant.now().toString(),
          checksum,
          properties.getParserVersion());
      rows++;
    }
    manifestService.recordNormalizedPartition(runId, "model_point_extract", station.getStationKey(),
        Instant.now().toString(), rows, firstChecksum(checksums),
        catalogService.toJson(Map.of("sourceName", sourceName, "rowCount", rows)));
    manifestService.upsertSourceInventory(new SourceInventoryRecord(
        UUID.randomUUID().toString(),
        station.getStationKey(),
        sourceName,
        sourceFamily,
        "cycle",
        text(workerResult.path("request"), "cycle_time_utc"),
        text(workerResult.path("request"), "cycle_time_utc"),
        null,
        text(workerResult, "status"),
        catalogService.toJson(Map.of("rows", rows, "warnings", workerResult.path("warnings"))),
        Instant.now().toString(),
        Instant.now().toString()));
    metricsAccumulator.recordRequest(0.0d, totalBytes(workerResult.path("touched_objects")), rows,
        "SUCCESS".equalsIgnoreCase(text(workerResult, "status")) ? "SUCCESS" : "SKIPPED");
    return rows;
  }

  public int persistNdfdHistorical(String runId,
                                   StationConfig station,
                                   JsonNode workerResult,
                                   JobMetricsAccumulator metricsAccumulator) {
    Map<String, String> checksums = persistTouchedObjects(runId, "ndfd_historical", "ncei", station,
        workerResult.path("touched_objects"));
    Map<String, Map<String, Object>> rowsByValidTime = new LinkedHashMap<>();
    for (JsonNode record : workerResult.path("records")) {
      String validTime = text(record, "valid_time_utc");
      Map<String, Object> row = rowsByValidTime.computeIfAbsent(validTime, ignored -> {
        Map<String, Object> created = new LinkedHashMap<>();
        created.put("temp_f", null);
        created.put("maxt_f", null);
        created.put("mint_f", null);
        created.put("sky_pct", null);
        created.put("wind_speed", null);
        created.put("wind_dir", null);
        created.put("qpf", null);
        created.put("pop12", null);
        return created;
      });
      String variableName = text(record, "variable_name");
      Double value = doubleOrNull(record, "bilinear_value");
      if (value == null) {
        value = doubleOrNull(record, "nearest_value");
      }
      switch (variableName) {
        case "temp_f", "temp_2m_f" -> row.put("temp_f", value);
        case "maxt_f", "tmax_2m_f" -> row.put("maxt_f", value);
        case "mint_f" -> row.put("mint_f", value);
        case "sky_pct", "cloud_cover_pct" -> row.put("sky_pct", value);
        case "wind_speed" -> row.put("wind_speed", value);
        case "wind_dir" -> row.put("wind_dir", value);
        case "qpf", "qpf_in" -> row.put("qpf", value);
        case "pop12" -> row.put("pop12", value);
        default -> {
        }
      }
    }
    int rows = 0;
    String issueTimeUtc = text(workerResult.path("request"), "cycle_time_utc");
    for (Map.Entry<String, Map<String, Object>> entry : rowsByValidTime.entrySet()) {
      Map<String, Object> row = entry.getValue();
      Double impliedDailyMax = firstNonNull(asDouble(row.get("maxt_f")), asDouble(row.get("temp_f")));
      catalogService.execute("""
          INSERT INTO ndfd_point_forecast (
            station_key, issue_time_utc, valid_time_utc, live_or_historical,
            temp_f, maxt_f, mint_f, sky_pct, wind_speed, wind_dir, qpf, pop12,
            implied_daily_max_f, source_name, source_family, source_identifier,
            request_url_or_bucket_key, ingested_at_utc, raw_object_sha256, parser_version
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(station_key, issue_time_utc, valid_time_utc, live_or_historical) DO UPDATE SET
            temp_f=excluded.temp_f,
            maxt_f=excluded.maxt_f,
            mint_f=excluded.mint_f,
            sky_pct=excluded.sky_pct,
            wind_speed=excluded.wind_speed,
            wind_dir=excluded.wind_dir,
            qpf=excluded.qpf,
            pop12=excluded.pop12,
            implied_daily_max_f=excluded.implied_daily_max_f,
            ingested_at_utc=excluded.ingested_at_utc,
            raw_object_sha256=excluded.raw_object_sha256
          """,
          station.getStationKey(),
          issueTimeUtc,
          entry.getKey(),
          "historical",
          row.get("temp_f"),
          row.get("maxt_f"),
          row.get("mint_f"),
          row.get("sky_pct"),
          row.get("wind_speed"),
          row.get("wind_dir"),
          row.get("qpf"),
          row.get("pop12"),
          impliedDailyMax,
          "ndfd_historical",
          "ncei",
          station.getStationKey() + "::" + entry.getKey(),
          firstRequestKey(workerResult.path("records")),
          Instant.now().toString(),
          firstChecksum(checksums),
          properties.getParserVersion());
      rows++;
    }
    manifestService.recordNormalizedPartition(runId, "ndfd_point_forecast", station.getStationKey(),
        issueTimeUtc, rows, firstChecksum(checksums),
        catalogService.toJson(Map.of("sourceName", "ndfd_historical", "rowCount", rows)));
    metricsAccumulator.recordRequest(0.0d, totalBytes(workerResult.path("touched_objects")), rows,
        "SUCCESS".equalsIgnoreCase(text(workerResult, "status")) ? "SUCCESS" : "SKIPPED");
    return rows;
  }

  private Map<String, String> persistTouchedObjects(String runId,
                                                    String sourceName,
                                                    String sourceFamily,
                                                    StationConfig station,
                                                    JsonNode touchedObjects) {
    Map<String, String> checksums = new LinkedHashMap<>();
    for (JsonNode objectNode : touchedObjects) {
      String localPathText = text(objectNode, "local_path");
      if (localPathText == null || localPathText.isBlank()) {
        continue;
      }
      Path localPath = Path.of(localPathText);
      if (!Files.exists(localPath)) {
        continue;
      }
      try {
        byte[] payload = Files.readAllBytes(localPath);
        String checksum = checksumService.sha256(payload);
        String requestKey = text(objectNode, "request_url_or_bucket_key");
        Integer forecastHour = intOrNull(objectNode, "forecast_hour");
        manifestService.recordObjectManifest(new ObjectManifestRecord(
            UUID.randomUUID().toString(),
            runId,
            station.getStationKey(),
            sourceName,
            sourceFamily,
            sourceName + "::" + localPath.getFileName(),
            requestKey,
            null,
            null,
            text(objectNode, "cycle_time_utc"),
            forecastHour,
            text(objectNode, "domain_name"),
            200,
            payload.length,
            checksum,
            text(objectNode, "payload_encoding"),
            null,
            payload,
            "RAW_STORED",
            0,
            null,
            "SUCCESS",
            null,
            Instant.now().toString()));
        checksums.put(manifestKey(requestKey, forecastHour), checksum);
      } catch (IOException ex) {
        throw new IllegalStateException("Failed to read worker output file " + localPath, ex);
      }
    }
    return checksums;
  }

  private long totalBytes(JsonNode touchedObjects) {
    long total = 0L;
    for (JsonNode objectNode : touchedObjects) {
      total += objectNode.path("content_length").asLong(0L);
    }
    return total;
  }

  private String manifestKey(String requestKey, Integer forecastHour) {
    return (requestKey == null ? "" : requestKey) + "|" + (forecastHour == null ? "" : forecastHour);
  }

  private String firstChecksum(Map<String, String> checksums) {
    return checksums.values().stream().findFirst().orElse("worker-no-raw-object");
  }

  private String firstRequestKey(JsonNode records) {
    for (JsonNode record : records) {
      String key = text(record, "request_url_or_bucket_key");
      if (key != null && !key.isBlank()) {
        return key;
      }
    }
    return null;
  }

  private String text(JsonNode node, String fieldName) {
    JsonNode child = node.get(fieldName);
    if (child == null || child.isNull()) {
      return null;
    }
    String value = child.asText(null);
    return value == null || value.isBlank() ? null : value;
  }

  private Integer intOrNull(JsonNode node, String fieldName) {
    JsonNode child = node.get(fieldName);
    if (child == null || child.isNull()) {
      return null;
    }
    return child.isNumber() ? child.intValue() : Integer.valueOf(child.asText());
  }

  private Double doubleOrNull(JsonNode node, String fieldName) {
    JsonNode child = node.get(fieldName);
    if (child == null || child.isNull()) {
      return null;
    }
    if (child.isNumber()) {
      return child.doubleValue();
    }
    String value = child.asText(null);
    if (value == null || value.isBlank()) {
      return null;
    }
    return Double.valueOf(value);
  }

  private Double asDouble(Object value) {
    if (value instanceof Number number) {
      return number.doubleValue();
    }
    return null;
  }

  @SafeVarargs
  private final Double firstNonNull(Double... values) {
    for (Double value : values) {
      if (value != null) {
        return value;
      }
    }
    return null;
  }
}
