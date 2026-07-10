package com.predictionmarkets.weather.pilot.source.ncei;

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
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class ThreadexAcisFetcher {
  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final ObjectMapper objectMapper;
  private final PilotIngestionProperties properties;

  public ThreadexAcisFetcher(SourceHttpClient httpClient,
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

  public int ingestRange(String jobId,
                         String runId,
                         StationConfig station,
                         String startDate,
                         String endDate,
                         JobMetricsAccumulator metricsAccumulator) {
    String sid = aliasValue(station, "ghcnh_or_ghcnd");
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("sid", sid);
    payload.put("sdate", startDate);
    payload.put("edate", endDate);
    payload.put("elems", new Object[] {"maxt", "mint", "pcpn", "snow"});
    String url = "https://data.rcc-acis.org/StnData";
    HttpResponseData response = httpClient.postJson("acis", url, payload, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, "threadex_acis", "acis", station.getStationKey(), "fetch_threadex_range",
        url, "POST", response.statusCode(), null, null, response.durationMs(), response.body().length,
        null, response.retryCount(),
        response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String checksum = rawStorageService.storeText(
        runId, station.getStationKey(), "threadex_acis", "acis", sid + "::" + startDate + "::" + endDate,
        url, response.statusCode(), new String(response.body(), StandardCharsets.UTF_8),
        "RAW_STORED", 0);
    try {
      JsonNode root = objectMapper.readTree(response.body());
      JsonNode meta = root.path("meta");
      String threadexId = meta.hasNonNull("uid") ? "uid:" + meta.path("uid").asText() : sid;
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          "threadex_acis",
          "acis",
          "metadata",
          threadexId,
          null,
          null,
          "DISCOVERED",
          meta.toString(),
          Instant.now().toString(),
          Instant.now().toString()));
      int rows = 0;
      JsonNode data = root.path("data");
      for (JsonNode row : data) {
        String dateLocal = row.path(0).asText();
        Double maxt = parseDouble(row.path(1).asText(null));
        Double mint = parseDouble(row.path(2).asText(null));
        Double pcpn = parseDouble(row.path(3).asText(null));
        Double snow = parseDouble(row.path(4).asText(null));
        catalogService.execute("""
            INSERT INTO threadex_daily_aux (
              station_key, threadex_id, date_local, maxt_f, mint_f, pcpn_in, snow_in,
              auxiliary_only, not_label_of_truth, metadata_json, source_name, source_family,
              source_identifier, request_url_or_bucket_key, issue_time_utc, ingested_at_utc,
              raw_object_sha256, parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_key, threadex_id, date_local) DO UPDATE SET
              maxt_f=excluded.maxt_f,
              mint_f=excluded.mint_f,
              pcpn_in=excluded.pcpn_in,
              snow_in=excluded.snow_in,
              metadata_json=excluded.metadata_json,
              ingested_at_utc=excluded.ingested_at_utc,
              raw_object_sha256=excluded.raw_object_sha256
            """,
            station.getStationKey(),
            threadexId,
            dateLocal,
            maxt,
            mint,
            pcpn,
            snow,
            1,
            1,
            meta.toString(),
            "threadex_acis",
            "acis",
            sid + "::" + dateLocal,
            url,
            null,
            Instant.now().toString(),
            checksum,
            properties.getParserVersion());
        rows++;
      }
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows, "SUCCESS");
      manifestService.recordNormalizedPartition(runId, "threadex_daily_aux", station.getStationKey(),
          startDate, rows, checksum, catalogService.toJson(Map.of("startDate", startDate, "endDate", endDate)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "threadex_acis",
          "normalize_threadex_range", "SUCCESS", Map.of("rows_parsed", rows));
      return rows;
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("threadex_acis");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse ACIS ThreadEx payload", ex);
    }
  }

  private Double parseDouble(String raw) {
    if (raw == null || raw.isBlank() || "M".equalsIgnoreCase(raw) || "T".equalsIgnoreCase(raw)) {
      return null;
    }
    return Double.valueOf(raw);
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
