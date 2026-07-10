package com.predictionmarkets.weather.pilot.jobs;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import com.predictionmarkets.weather.pilot.config.PilotConfigLoader;
import com.predictionmarkets.weather.pilot.config.StationAlias;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.manifest.HttpRequestLogRecord;
import com.predictionmarkets.weather.pilot.manifest.ManifestService;
import com.predictionmarkets.weather.pilot.metrics.JobMetricsAccumulator;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import com.predictionmarkets.weather.pilot.source.HttpResponseData;
import com.predictionmarkets.weather.pilot.source.SourceHttpClient;
import com.predictionmarkets.weather.pilot.storage.RawStorageService;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

@Service
public class BootstrapKnycStationJob {
  private static final Pattern IEM_TIMEZONE = Pattern.compile("Time Zone:</th><td>([^<]+)");
  private static final Pattern IEM_ARCHIVE_BEGIN = Pattern.compile("Archive Begin:</th><td>([^<]+)");
  private static final Pattern IEM_RESET = Pattern.compile("METAR_RESET_MINUTE</td><td>([^<]+)");
  private static final Pattern IEM_GHCNH = Pattern.compile("GHCNH_ID</td><td>([^<]+)");
  private static final Pattern NCEI_LAT_LON = Pattern.compile(
      "Latitude/Longitude</td>\\s*<td class=\"val\">\\s*([0-9\\.-]+).*?([0-9\\.-]+)",
      Pattern.DOTALL);

  private final PilotConfigLoader configLoader;
  private final SqliteCatalogService catalogService;
  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final JobRunService jobRunService;
  private final MetricsService metricsService;
  private final ObjectMapper objectMapper;

  public BootstrapKnycStationJob(PilotConfigLoader configLoader,
                                 SqliteCatalogService catalogService,
                                 SourceHttpClient httpClient,
                                 ManifestService manifestService,
                                 RawStorageService rawStorageService,
                                 JobRunService jobRunService,
                                 MetricsService metricsService,
                                 ObjectMapper objectMapper) {
    this.configLoader = configLoader;
    this.catalogService = catalogService;
    this.httpClient = httpClient;
    this.manifestService = manifestService;
    this.rawStorageService = rawStorageService;
    this.jobRunService = jobRunService;
    this.metricsService = metricsService;
    this.objectMapper = objectMapper;
  }

  public String run() {
    StationConfig station = configLoader.requireDefaultStation();
    configLoader.loadSources().forEach(catalogService::upsertSource);
    String runId = jobRunService.startRun("bootstrapKnycStationJob", station.getStationKey());
    JobMetricsAccumulator metrics = metricsService.newAccumulator();
    try {
      String iemStation = aliasValue(station, "iem_asos_station");
      String iemNetwork = aliasValue(station, "iem_asos_network");
      String iemUrl = "https://mesonet.agron.iastate.edu/sites/site.php?network=" + iemNetwork
          + "&station=" + iemStation;
      HttpResponseData iemResponse = httpClient.get("iem", iemUrl, Map.of());
      String iemHtml = new String(iemResponse.body(), StandardCharsets.UTF_8);
      logRequest(runId, "bootstrapKnycStationJob", station.getStationKey(), "iem_station_metadata",
          "fetch_iem_station", iemUrl, "GET", iemResponse);
      rawStorageService.storeText(runId, station.getStationKey(), "iem_station_metadata", "iem",
          station.getStationKey() + "::iem_station", iemUrl, iemResponse.statusCode(), iemHtml, "RAW_STORED", 1);
      metrics.recordRequest(iemResponse.durationMs(), iemResponse.body().length, 1, "SUCCESS");

      String nceiUrl = "https://www.ncei.noaa.gov/cdo-web/datasets/LCD/stations/WBAN:94728/detail";
      String nceiHtml = null;
      Map<String, Object> nceiMetadata = new LinkedHashMap<>();
      try {
        HttpResponseData nceiResponse = httpClient.get("ncei", nceiUrl, Map.of());
        nceiHtml = new String(nceiResponse.body(), StandardCharsets.UTF_8);
        logRequest(runId, "bootstrapKnycStationJob", station.getStationKey(), "ncei_lcd_station_detail",
            "fetch_ncei_station", nceiUrl, "GET", nceiResponse);
        rawStorageService.storeText(runId, station.getStationKey(), "ncei_lcd_station_detail", "ncei",
            station.getStationKey() + "::ncei_station", nceiUrl, nceiResponse.statusCode(), nceiHtml, "RAW_STORED", 1);
        metrics.recordRequest(nceiResponse.durationMs(), nceiResponse.body().length, 1, "SUCCESS");
        nceiMetadata.putAll(parseNceiMetadata(nceiHtml));
        nceiMetadata.put("fetchStatus", "SUCCESS");
      } catch (Exception ex) {
        metrics.recordRequest(0.0d, 0, 0, "FAILED");
        metrics.recordParserFailure("ncei_lcd_station_detail");
        nceiMetadata.put("fetchStatus", "FAILED");
        nceiMetadata.put("exceptionMessage", ex.getMessage());
        jobRunService.logStructuredEvent("bootstrapKnycStationJob", runId, station.getStationKey(),
            "ncei_lcd_station_detail", "metadata_fetch_failed", "FAILED",
            Map.of("request_url_or_key", nceiUrl, "message", ex.getMessage()));
      }

      Map<String, Object> acisPayload = Map.of(
          "sid", aliasValue(station, "ghcnh_or_ghcnd"),
          "sdate", "2025-01-01",
          "edate", "2025-01-02",
          "elems", new Object[] {"maxt", "mint"});
      String acisUrl = "https://data.rcc-acis.org/StnData";
      String acisJson = null;
      JsonNode acisMeta = null;
      try {
        HttpResponseData acisResponse = httpClient.postJson("acis", acisUrl, acisPayload, Map.of());
        acisJson = new String(acisResponse.body(), StandardCharsets.UTF_8);
        logRequest(runId, "bootstrapKnycStationJob", station.getStationKey(), "acis_threadex",
            "fetch_acis_meta", acisUrl, "POST", acisResponse);
        rawStorageService.storeText(runId, station.getStationKey(), "acis_threadex", "acis",
            station.getStationKey() + "::acis_meta", acisUrl, acisResponse.statusCode(), acisJson, "RAW_STORED", 1);
        metrics.recordRequest(acisResponse.durationMs(), acisResponse.body().length, 1, "SUCCESS");
        acisMeta = objectMapper.readTree(acisJson).path("meta");
      } catch (Exception ex) {
        metrics.recordRequest(0.0d, 0, 0, "FAILED");
        metrics.recordParserFailure("acis_threadex");
        jobRunService.logStructuredEvent("bootstrapKnycStationJob", runId, station.getStationKey(),
            "acis_threadex", "metadata_fetch_failed", "FAILED",
            Map.of("request_url_or_key", acisUrl, "message", ex.getMessage()));
      }

      Map<String, Object> metadata = new LinkedHashMap<>();
      metadata.put("iem", parseIemMetadata(iemHtml));
      metadata.put("ncei", nceiMetadata);
      metadata.put("acis", acisMeta == null ? Map.of("fetchStatus", "FAILED_OR_SKIPPED") : acisMeta);
      metadata.put("verification", verificationMap(station, metadata));
      catalogService.upsertStation(station, metadata);
      jobRunService.logStructuredEvent("bootstrapKnycStationJob", runId, station.getStationKey(),
          "station_registry", "station_verified", "SUCCESS",
          Map.of("verification", metadata.get("verification")));
      jobRunService.completeRun(runId, "bootstrapKnycStationJob", station.getStationKey(), "COMPLETE", metrics);
      return runId;
    } catch (Exception ex) {
      metrics.recordParserFailure("bootstrapKnycStationJob");
      jobRunService.completeRun(runId, "bootstrapKnycStationJob", station.getStationKey(), "FAILED", metrics);
      throw new IllegalStateException("Failed bootstrapKnycStationJob", ex);
    }
  }

  private void logRequest(String runId,
                          String jobId,
                          String stationKey,
                          String sourceName,
                          String action,
                          String url,
                          String method,
                          HttpResponseData response) {
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, sourceName, sourceFamily(sourceName), stationKey, action, url, method,
        response.statusCode(), null, null, response.durationMs(), response.body().length, null,
        response.retryCount(), response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
  }

  private Map<String, Object> parseIemMetadata(String html) {
    Map<String, Object> parsed = new LinkedHashMap<>();
    parsed.put("timezone", firstGroup(html, IEM_TIMEZONE));
    parsed.put("archiveBegin", firstGroup(html, IEM_ARCHIVE_BEGIN));
    parsed.put("metarResetMinute", firstGroup(html, IEM_RESET));
    parsed.put("ghcnhId", firstGroup(html, IEM_GHCNH));
    return parsed;
  }

  private Map<String, Object> parseNceiMetadata(String html) {
    Map<String, Object> parsed = new LinkedHashMap<>();
    Matcher matcher = NCEI_LAT_LON.matcher(html);
    if (matcher.find()) {
      parsed.put("latitude", Double.parseDouble(matcher.group(1)));
      parsed.put("longitude", Double.parseDouble(matcher.group(2)));
    }
    return parsed;
  }

  private Map<String, Object> verificationMap(StationConfig station, Map<String, Object> metadata) {
    Map<String, Object> verification = new LinkedHashMap<>();
    Map<?, ?> iem = (Map<?, ?>) metadata.get("iem");
    Map<?, ?> ncei = (Map<?, ?>) metadata.get("ncei");
    verification.put("timezoneMatches", station.getTimezone().equals(iem.get("timezone")));
    verification.put("archiveBeginPresent", iem.get("archiveBegin") != null);
    if (ncei.get("latitude") instanceof Number lat && ncei.get("longitude") instanceof Number lon) {
      verification.put("latitudeWithinTolerance", Math.abs(station.getLatitude() - lat.doubleValue()) < 0.01d);
      verification.put("longitudeWithinTolerance", Math.abs(station.getLongitude() - lon.doubleValue()) < 0.01d);
    }
    return verification;
  }

  private String firstGroup(String value, Pattern pattern) {
    Matcher matcher = pattern.matcher(value);
    return matcher.find() ? matcher.group(1).trim() : null;
  }

  private String sourceFamily(String sourceName) {
    if (sourceName.startsWith("iem")) {
      return "iem";
    }
    if (sourceName.startsWith("ncei")) {
      return "ncei";
    }
    if (sourceName.startsWith("acis")) {
      return "acis";
    }
    return "unknown";
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
