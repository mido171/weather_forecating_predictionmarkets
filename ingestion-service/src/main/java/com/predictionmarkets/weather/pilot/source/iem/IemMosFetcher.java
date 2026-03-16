package com.predictionmarkets.weather.pilot.source.iem;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class IemMosFetcher {
  private static final DateTimeFormatter RUNTIME_PARAM_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
  private static final DateTimeFormatter API_TIMESTAMP_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS");

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final ObjectMapper objectMapper;
  private final PilotIngestionProperties properties;

  public IemMosFetcher(SourceHttpClient httpClient,
                       ManifestService manifestService,
                       RawStorageService rawStorageService,
                       SqliteCatalogService catalogService,
                       JobRunService jobRunService,
                       ObjectMapper objectMapper,
                       PilotIngestionProperties properties) {
    this.httpClient = httpClient;
    this.manifestService = manifestService;
    this.rawStorageService = rawStorageService;
    this.catalogService = catalogService;
    this.jobRunService = jobRunService;
    this.objectMapper = objectMapper;
    this.properties = properties;
  }

  public int ingestRecentRuntimes(String jobId,
                                  String runId,
                                  StationConfig station,
                                  List<String> models,
                                  int lookbackDays,
                                  JobMetricsAccumulator metricsAccumulator) {
    String icao = aliasValue(station, "icao");
    ZoneId zoneId = ZoneId.of(station.getTimezone());
    Instant startUtc = Instant.now().minusSeconds(Math.max(1, lookbackDays) * 24L * 3600L);
    LocalDateTime cursor = LocalDateTime.ofInstant(startUtc, ZoneOffset.UTC)
        .withMinute(0)
        .withSecond(0)
        .withNano(0);
    int rows = 0;
    while (!cursor.isAfter(LocalDateTime.now(ZoneOffset.UTC))) {
      if (cursor.getHour() % 6 == 0) {
        for (String model : models) {
          rows += ingestRuntime(jobId, runId, station, zoneId, icao, model.toUpperCase(Locale.ROOT),
              cursor, metricsAccumulator);
        }
      }
      cursor = cursor.plusHours(1);
    }
    return rows;
  }

  private int ingestRuntime(String jobId,
                            String runId,
                            StationConfig station,
                            ZoneId zoneId,
                            String icao,
                            String model,
                            LocalDateTime runtimeUtc,
                            JobMetricsAccumulator metricsAccumulator) {
    String sourceName = "iem_mos";
    String runtimeParam = runtimeUtc.format(RUNTIME_PARAM_FORMATTER);
    String url = "https://mesonet.agron.iastate.edu/api/1/mos.json?station=" + icao
        + "&model=" + model + "&runtime=" + runtimeParam.replace(" ", "%20") + "Z";
    HttpResponseData response = httpClient.get("iem", url, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, sourceName, "iem", station.getStationKey(), "fetch_exact_runtime",
        url, "GET", response.statusCode(), runtimeUtc.toInstant(ZoneOffset.UTC).toString(), null,
        response.durationMs(), response.body().length, null, response.retryCount(),
        response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String text = new String(response.body(), StandardCharsets.UTF_8);
    String checksum = rawStorageService.storeText(
        runId, station.getStationKey(), sourceName, "iem", model + "::" + runtimeParam,
        url, response.statusCode(), text, "RAW_STORED", 0);
    try {
      JsonNode root = objectMapper.readTree(text);
      JsonNode data = root.path("data");
      if (!data.isArray() || data.isEmpty()) {
        manifestService.upsertSourceInventory(new SourceInventoryRecord(
            UUID.randomUUID().toString(),
            station.getStationKey(),
            sourceName,
            "iem",
            "runtime",
            model + "::" + runtimeParam,
            runtimeUtc.toInstant(ZoneOffset.UTC).toString(),
            null,
            "EMPTY",
            catalogService.toJson(Map.of("url", url)),
            Instant.now().toString(),
            Instant.now().toString()));
        metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "SKIPPED");
        return 0;
      }
      int rows = 0;
      for (JsonNode row : data) {
        Instant runtimeInstant = parseApiInstant(row.path("runtime_utc").asText(null), row.path("runtime").asText(null));
        Instant validInstant = parseApiInstant(row.path("ftime_utc").asText(null), row.path("ftime").asText(null));
        Double maxTemp = parseDoubleNode(row.get("n_x"));
        Double minTemp = parseDoubleNode(row.get("n_n"));
        Double hourlyTemp = parseDoubleNode(row.get("tmp"));
        Double dewPoint = parseDoubleNode(row.get("dpt"));
        Double windDirection = parseDoubleNode(row.get("wdr"));
        Double windSpeed = parseDoubleNode(row.get("wsp"));
        Double precipProb = firstNonNull(parseDoubleNode(row.get("p06")), parseDoubleNode(row.get("p12")));
        Double thunderProb = firstNonNull(parseDoubleNode(row.get("t06")), parseDoubleNode(row.get("t12")));
        Double qpf = firstNonNull(parseDoubleNode(row.get("q06")), parseDoubleNode(row.get("q12")), parseDoubleNode(row.get("q24")));
        Double ceiling = parseDoubleNode(row.get("cig"));
        Double visibility = parseDoubleNode(row.get("vis"));
        Double thresholdBasis = maxTemp != null ? maxTemp : hourlyTemp;
        LocalDate targetDayLocal = validInstant == null ? null : validInstant.atZone(zoneId).toLocalDate();
        Integer leadHours = runtimeInstant == null || validInstant == null
            ? null : (int) java.time.Duration.between(runtimeInstant, validInstant).toHours();
        catalogService.execute("""
            INSERT INTO mos_station_guidance (
              station_key, model_name, runtime_utc, valid_time_utc, lead_hours, mos_target_day_local,
              max_temp_guidance_f, min_temp_guidance_f, hourly_temp_f, dew_point_f, cloud_category,
              wind_direction_deg, wind_speed_kt, precip_prob, thunder_prob, qpf, ceiling_ft,
              visibility_mi, above_45, above_46, above_47, above_48, above_49, above_50,
              raw_values_json, source_name, source_family, source_identifier, request_url_or_bucket_key,
              issue_time_utc, ingested_at_utc, raw_object_sha256, parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_key, model_name, runtime_utc, valid_time_utc) DO UPDATE SET
              lead_hours=excluded.lead_hours,
              mos_target_day_local=excluded.mos_target_day_local,
              max_temp_guidance_f=excluded.max_temp_guidance_f,
              min_temp_guidance_f=excluded.min_temp_guidance_f,
              hourly_temp_f=excluded.hourly_temp_f,
              dew_point_f=excluded.dew_point_f,
              cloud_category=excluded.cloud_category,
              wind_direction_deg=excluded.wind_direction_deg,
              wind_speed_kt=excluded.wind_speed_kt,
              precip_prob=excluded.precip_prob,
              thunder_prob=excluded.thunder_prob,
              qpf=excluded.qpf,
              ceiling_ft=excluded.ceiling_ft,
              visibility_mi=excluded.visibility_mi,
              above_45=excluded.above_45,
              above_46=excluded.above_46,
              above_47=excluded.above_47,
              above_48=excluded.above_48,
              above_49=excluded.above_49,
              above_50=excluded.above_50,
              raw_values_json=excluded.raw_values_json,
              ingested_at_utc=excluded.ingested_at_utc,
              raw_object_sha256=excluded.raw_object_sha256
            """,
            station.getStationKey(),
            model,
            runtimeInstant == null ? null : runtimeInstant.toString(),
            validInstant == null ? null : validInstant.toString(),
            leadHours,
            targetDayLocal == null ? null : targetDayLocal.toString(),
            maxTemp,
            minTemp,
            hourlyTemp,
            dewPoint,
            row.path("cld").asText(null),
            windDirection,
            windSpeed,
            precipProb,
            thunderProb,
            qpf,
            ceiling,
            visibility,
            flag(thresholdBasis, 45),
            flag(thresholdBasis, 46),
            flag(thresholdBasis, 47),
            flag(thresholdBasis, 48),
            flag(thresholdBasis, 49),
            flag(thresholdBasis, 50),
            row.toString(),
            sourceName,
            "iem",
            model + "::" + runtimeParam,
            url,
            runtimeInstant == null ? null : runtimeInstant.toString(),
            Instant.now().toString(),
            checksum,
            properties.getParserVersion());
        rows++;
      }
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows, "SUCCESS");
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          sourceName,
          "iem",
          "runtime",
          model + "::" + runtimeParam,
          runtimeUtc.toInstant(ZoneOffset.UTC).toString(),
          null,
          "INGESTED",
          catalogService.toJson(Map.of("url", url, "rows", rows)),
          Instant.now().toString(),
          Instant.now().toString()));
      return rows;
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure(sourceName);
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse exact MOS runtime " + model + " " + runtimeParam, ex);
    }
  }

  private Instant parseApiInstant(String rawUtc, String fallback) {
    if (rawUtc != null && !rawUtc.isBlank()) {
      return LocalDateTime.parse(rawUtc, API_TIMESTAMP_FORMATTER).toInstant(ZoneOffset.UTC);
    }
    if (fallback == null || fallback.isBlank()) {
      return null;
    }
    return LocalDateTime.parse(fallback, RUNTIME_PARAM_FORMATTER).toInstant(ZoneOffset.UTC);
  }

  private Double parseDoubleNode(JsonNode node) {
    if (node == null || node.isNull()) {
      return null;
    }
    if (node.isNumber()) {
      return node.doubleValue();
    }
    if (node.isTextual()) {
      String raw = node.asText().trim();
      if (raw.isEmpty() || "M".equalsIgnoreCase(raw)) {
        return null;
      }
      if ("T".equalsIgnoreCase(raw)) {
        return 0.0d;
      }
      if (raw.contains("/")) {
        for (String token : raw.split("/")) {
          Double parsed = tryParseDouble(token);
          if (parsed != null) {
            return parsed;
          }
        }
        return null;
      }
      return tryParseDouble(raw);
    }
    return null;
  }

  private Double tryParseDouble(String raw) {
    if (raw == null) {
      return null;
    }
    String trimmed = raw.trim();
    if (trimmed.isEmpty()) {
      return null;
    }
    try {
      return Double.valueOf(trimmed);
    } catch (NumberFormatException ex) {
      return null;
    }
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

  private Integer flag(Double value, double threshold) {
    return value != null && value >= threshold ? 1 : 0;
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
