package com.predictionmarkets.weather.pilot.source.nws;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.catalog.SourceInventoryRecord;
import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.manifest.HttpRequestLogRecord;
import com.predictionmarkets.weather.pilot.manifest.ManifestService;
import com.predictionmarkets.weather.pilot.metrics.JobMetricsAccumulator;
import com.predictionmarkets.weather.pilot.source.HttpResponseData;
import com.predictionmarkets.weather.pilot.source.SourceHttpClient;
import com.predictionmarkets.weather.pilot.storage.RawStorageService;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.xml.parsers.DocumentBuilderFactory;
import org.springframework.stereotype.Service;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

@Service
public class NdfdLivePointFetcher {
  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final ObjectMapper objectMapper;
  private final PilotIngestionProperties properties;

  public NdfdLivePointFetcher(SourceHttpClient httpClient,
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

  public int ingestLiveWindow(String jobId,
                              String runId,
                              StationConfig station,
                              String beginIso,
                              String endIso,
                              JobMetricsAccumulator metricsAccumulator) {
    String xmlUrl = "https://digital.weather.gov/xml/sample_products/browser_interface/ndfdXMLclient.php?lat="
        + station.getLatitude() + "&lon=" + station.getLongitude()
        + "&product=time-series&begin=" + beginIso + "&end=" + endIso
        + "&Unit=e&temp=temp&maxt=maxt&mint=mint&sky=sky&wspd=wspd&wdir=wdir&qpf=qpf&pop12=pop12";
    HttpResponseData xmlResponse = httpClient.get("nws", xmlUrl, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, "ndfd_live_point", "nws", station.getStationKey(), "fetch_ndfd_xml",
        xmlUrl, "GET", xmlResponse.statusCode(), null, null, xmlResponse.durationMs(), xmlResponse.body().length,
        null, xmlResponse.retryCount(),
        xmlResponse.statusCode() >= 200 && xmlResponse.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String xmlText = new String(xmlResponse.body(), StandardCharsets.UTF_8);
    String xmlChecksum = rawStorageService.storeText(
        runId, station.getStationKey(), "ndfd_live_point", "nws", station.getStationKey() + "::xml",
        xmlUrl, xmlResponse.statusCode(), xmlText, "RAW_STORED", 0);

    String pointsUrl = "https://api.weather.gov/points/" + station.getLatitude() + "," + station.getLongitude();
    HttpResponseData pointsResponse = httpClient.get("nws", pointsUrl,
        Map.of("Accept", "application/geo+json"));
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, "ndfd_live_point", "nws", station.getStationKey(), "fetch_points_api",
        pointsUrl, "GET", pointsResponse.statusCode(), null, null, pointsResponse.durationMs(), pointsResponse.body().length,
        null, pointsResponse.retryCount(),
        pointsResponse.statusCode() >= 200 && pointsResponse.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String pointsText = new String(pointsResponse.body(), StandardCharsets.UTF_8);
    rawStorageService.storeText(
        runId, station.getStationKey(), "ndfd_live_point", "nws", station.getStationKey() + "::points",
        pointsUrl, pointsResponse.statusCode(), pointsText, "RAW_STORED", 0);
    try {
      JsonNode pointsJson = objectMapper.readTree(pointsText);
      String forecastGridData = pointsJson.path("properties").path("forecastGridData").asText(null);
      manifestService.upsertSourceInventory(new SourceInventoryRecord(
          UUID.randomUUID().toString(),
          station.getStationKey(),
          "ndfd_live_point",
          "nws",
          "forecastGridData",
          station.getStationKey(),
          null,
          null,
          "DISCOVERED",
          catalogService.toJson(Map.of("forecastGridData", forecastGridData)),
          Instant.now().toString(),
          Instant.now().toString()));
      List<NdfdRow> rows = parseDwml(xmlText);
      for (NdfdRow row : rows) {
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
            row.issueTimeUtc(),
            row.validTimeUtc(),
            "live",
            row.tempF(),
            row.maxtF(),
            row.mintF(),
            row.skyPct(),
            row.windSpeed(),
            row.windDir(),
            row.qpf(),
            row.pop12(),
            row.impliedDailyMaxF(),
            "ndfd_live_point",
            "nws",
            station.getStationKey() + "::" + row.validTimeUtc(),
            xmlUrl,
            Instant.now().toString(),
            xmlChecksum,
            properties.getParserVersion());
      }
      metricsAccumulator.recordRequest(xmlResponse.durationMs() + pointsResponse.durationMs(),
          xmlResponse.body().length + pointsResponse.body().length, rows.size(), "SUCCESS");
      manifestService.recordNormalizedPartition(runId, "ndfd_point_forecast", station.getStationKey(),
          beginIso, rows.size(), xmlChecksum, catalogService.toJson(Map.of("beginIso", beginIso, "endIso", endIso)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "ndfd_live_point",
          "normalize_ndfd_live", "SUCCESS", Map.of("rows_parsed", rows.size()));
      return rows.size();
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("ndfd_live_point");
      metricsAccumulator.recordRequest(xmlResponse.durationMs() + pointsResponse.durationMs(),
          xmlResponse.body().length + pointsResponse.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse NDFD live point response", ex);
    }
  }

  private List<NdfdRow> parseDwml(String xmlText) throws Exception {
    DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
    factory.setNamespaceAware(false);
    Document document = factory.newDocumentBuilder().parse(new ByteArrayInputStream(xmlText.getBytes(StandardCharsets.UTF_8)));
    document.getDocumentElement().normalize();
    String issueTimeUtc = firstText(document, "creation-date");
    Map<String, List<String>> layouts = new LinkedHashMap<>();
    NodeList layoutNodes = document.getElementsByTagName("time-layout");
    for (int i = 0; i < layoutNodes.getLength(); i++) {
      Element layout = (Element) layoutNodes.item(i);
      String key = firstChildText(layout, "layout-key");
      List<String> times = new ArrayList<>();
      NodeList startTimes = layout.getElementsByTagName("start-valid-time");
      for (int j = 0; j < startTimes.getLength(); j++) {
        times.add(startTimes.item(j).getTextContent().trim());
      }
      layouts.put(key, times);
    }
    Map<String, NdfdRowBuilder> builders = new LinkedHashMap<>();
    NodeList parametersNodes = document.getElementsByTagName("parameters");
    if (parametersNodes.getLength() > 0) {
      Element parameters = (Element) parametersNodes.item(0);
      parseValues(parameters, "temperature", "hourly", layouts, builders, (builder, value) -> builder.tempF = value);
      parseValues(parameters, "temperature", "maximum", layouts, builders, (builder, value) -> builder.maxtF = value);
      parseValues(parameters, "temperature", "minimum", layouts, builders, (builder, value) -> builder.mintF = value);
      parseValues(parameters, "cloud-amount", "total", layouts, builders, (builder, value) -> builder.skyPct = value);
      parseValues(parameters, "wind-speed", "sustained", layouts, builders, (builder, value) -> builder.windSpeed = value);
      parseValues(parameters, "direction", "wind", layouts, builders, (builder, value) -> builder.windDir = value);
      parseValues(parameters, "precipitation", "liquid", layouts, builders, (builder, value) -> builder.qpf = value);
      parseValues(parameters, "probability-of-precipitation", "12 hour", layouts, builders, (builder, value) -> builder.pop12 = value);
    }
    return builders.values().stream()
        .sorted((left, right) -> left.validTimeUtc.compareTo(right.validTimeUtc))
        .map(builder -> builder.build(issueTimeUtc))
        .toList();
  }

  private void parseValues(Element parameters,
                           String tagName,
                           String type,
                           Map<String, List<String>> layouts,
                           Map<String, NdfdRowBuilder> builders,
                           java.util.function.BiConsumer<NdfdRowBuilder, Double> setter) {
    NodeList nodes = parameters.getElementsByTagName(tagName);
    for (int i = 0; i < nodes.getLength(); i++) {
      Element element = (Element) nodes.item(i);
      if (!type.equalsIgnoreCase(element.getAttribute("type"))) {
        continue;
      }
      String layoutKey = element.getAttribute("time-layout");
      List<String> times = layouts.getOrDefault(layoutKey, List.of());
      NodeList values = element.getElementsByTagName("value");
      for (int j = 0; j < Math.min(values.getLength(), times.size()); j++) {
        Double parsed = parseDouble(values.item(j).getTextContent());
        String validTime = OffsetDateTime.parse(times.get(j)).toInstant().toString();
        NdfdRowBuilder builder = builders.computeIfAbsent(validTime, NdfdRowBuilder::new);
        setter.accept(builder, parsed);
      }
    }
  }

  private Double parseDouble(String raw) {
    if (raw == null) {
      return null;
    }
    String text = raw.trim();
    if (text.isEmpty()) {
      return null;
    }
    return Double.valueOf(text);
  }

  private String firstText(Document document, String tagName) {
    NodeList nodes = document.getElementsByTagName(tagName);
    return nodes.getLength() == 0 ? null : nodes.item(0).getTextContent().trim();
  }

  private String firstChildText(Element element, String tagName) {
    NodeList nodes = element.getElementsByTagName(tagName);
    return nodes.getLength() == 0 ? null : nodes.item(0).getTextContent().trim();
  }

  private static final class NdfdRowBuilder {
    private final String validTimeUtc;
    private Double tempF;
    private Double maxtF;
    private Double mintF;
    private Double skyPct;
    private Double windSpeed;
    private Double windDir;
    private Double qpf;
    private Double pop12;

    private NdfdRowBuilder(String validTimeUtc) {
      this.validTimeUtc = validTimeUtc;
    }

    private NdfdRow build(String issueTimeUtc) {
      Double impliedDailyMax = maxtF != null ? maxtF : tempF;
      return new NdfdRow(issueTimeUtc, validTimeUtc, tempF, maxtF, mintF, skyPct, windSpeed, windDir, qpf, pop12,
          impliedDailyMax);
    }
  }

  private record NdfdRow(
      String issueTimeUtc,
      String validTimeUtc,
      Double tempF,
      Double maxtF,
      Double mintF,
      Double skyPct,
      Double windSpeed,
      Double windDir,
      Double qpf,
      Double pop12,
      Double impliedDailyMaxF) {
  }
}
