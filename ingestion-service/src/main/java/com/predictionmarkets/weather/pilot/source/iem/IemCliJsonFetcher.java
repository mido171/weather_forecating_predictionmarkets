package com.predictionmarkets.weather.pilot.source.iem;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.pilot.catalog.JobRunService;
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
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

@Service
public class IemCliJsonFetcher {
  private static final Pattern ISSUE_TIMESTAMP = Pattern.compile("(\\d{12})");
  private static final DateTimeFormatter ISSUE_FORMATTER =
      DateTimeFormatter.ofPattern("yyyyMMddHHmm");

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final ObjectMapper objectMapper;
  private final PilotIngestionProperties properties;

  public IemCliJsonFetcher(SourceHttpClient httpClient,
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

  public int ingestYear(String jobId,
                        String runId,
                        StationConfig station,
                        int year,
                        JobMetricsAccumulator metricsAccumulator) {
    String icao = aliasValue(station, "icao");
    String sourceName = "iem_cli_json";
    String url = "https://mesonet.agron.iastate.edu/json/cli.py?station=" + icao
        + "&year=" + year + "&fmt=json";
    HttpResponseData response = httpClient.get("iem", url, Map.of());
    String responseText = new String(response.body(), StandardCharsets.UTF_8);
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId,
        jobId,
        sourceName,
        "iem",
        station.getStationKey(),
        "fetch_cli_year",
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
    String checksum = rawStorageService.storeText(
        runId,
        station.getStationKey(),
        sourceName,
        "iem",
        station.getStationKey() + "::" + year,
        url,
        response.statusCode(),
        responseText,
        "RAW_STORED",
        0);
    try {
      JsonNode root = objectMapper.readTree(responseText);
      JsonNode results = root.path("results");
      int[] rowCount = {0};
      catalogService.inTransaction(connection -> {
        java.sql.PreparedStatement statement = connection.prepareStatement("""
            INSERT INTO cli_daily_label (
              station_key, target_date_local, tmax_f, tmin_f, tavg_f,
              max_obs_time_local, min_obs_time_local, precip_in, snow_in, sky_cover,
              source_issue_time_utc, source_payload_year, source_name, source_family,
              source_identifier, request_url_or_bucket_key, issue_time_utc, valid_time_utc,
              ingested_at_utc, raw_object_sha256, parser_version, label_disagreement_flag,
              raw_cli_tmax_f, raw_cli_tmin_f
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_key, target_date_local) DO UPDATE SET
              tmax_f=excluded.tmax_f,
              tmin_f=excluded.tmin_f,
              tavg_f=excluded.tavg_f,
              max_obs_time_local=excluded.max_obs_time_local,
              min_obs_time_local=excluded.min_obs_time_local,
              precip_in=excluded.precip_in,
              snow_in=excluded.snow_in,
              sky_cover=excluded.sky_cover,
              source_issue_time_utc=excluded.source_issue_time_utc,
              source_payload_year=excluded.source_payload_year,
              request_url_or_bucket_key=excluded.request_url_or_bucket_key,
              ingested_at_utc=excluded.ingested_at_utc,
              raw_object_sha256=excluded.raw_object_sha256,
              parser_version=excluded.parser_version
            """);
        try (statement) {
          for (JsonNode item : results) {
            String valid = item.path("valid").asText();
            LocalDate targetDate = LocalDate.parse(valid);
            String product = item.path("product").asText(null);
            String issueTimeUtc = parseIssueTime(product, item.path("link").asText(null));
            bindBigDecimal(statement, 3, decimalOrNull(item.get("high")));
            bindBigDecimal(statement, 4, decimalOrNull(item.get("low")));
            statement.setObject(5, null);
            statement.setString(1, station.getStationKey());
            statement.setString(2, targetDate.toString());
            statement.setString(6, textOrNull(item.get("high_time")));
            statement.setString(7, textOrNull(item.get("low_time")));
            bindBigDecimal(statement, 8, decimalOrNull(item.get("precip")));
            bindBigDecimal(statement, 9, decimalOrNull(item.get("snow")));
            statement.setString(10, textOrNull(item.get("average_sky_cover")));
            statement.setString(11, issueTimeUtc);
            statement.setInt(12, year);
            statement.setString(13, sourceName);
            statement.setString(14, "iem");
            statement.setString(15, station.getStationKey() + "::" + year);
            statement.setString(16, url);
            statement.setString(17, issueTimeUtc);
            statement.setObject(18, null);
            statement.setString(19, Instant.now().toString());
            statement.setString(20, checksum);
            statement.setString(21, properties.getParserVersion());
            statement.setInt(22, 0);
            statement.setObject(23, null);
            statement.setObject(24, null);
            statement.addBatch();
            rowCount[0]++;
          }
          statement.executeBatch();
        }
      });
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rowCount[0], "SUCCESS");
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), sourceName,
          "normalize_cli_year", "SUCCESS", Map.of("year", year, "rows_parsed", rowCount[0]));
      manifestService.recordNormalizedPartition(runId, "cli_daily_label", station.getStationKey(),
          year + "-01-01", rowCount[0], checksum, catalogService.toJson(Map.of("year", year)));
      return rowCount[0];
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure(sourceName);
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse CLI JSON for year " + year, ex);
    }
  }

  private void bindBigDecimal(java.sql.PreparedStatement statement, int index, BigDecimal value)
      throws java.sql.SQLException {
    if (value == null) {
      statement.setObject(index, null);
    } else {
      statement.setBigDecimal(index, value);
    }
  }

  private BigDecimal decimalOrNull(JsonNode node) {
    if (node == null || node.isNull()) {
      return null;
    }
    if (node.isNumber()) {
      return node.decimalValue();
    }
    if (node.isTextual()) {
      String raw = node.asText().trim();
      if (raw.isEmpty() || "M".equalsIgnoreCase(raw)) {
        return null;
      }
      if ("T".equalsIgnoreCase(raw)) {
        return BigDecimal.ZERO;
      }
      try {
        return new BigDecimal(raw);
      } catch (NumberFormatException ex) {
        return null;
      }
    }
    return null;
  }

  private String textOrNull(JsonNode node) {
    if (node == null || node.isNull()) {
      return null;
    }
    String text = node.asText().trim();
    return text.isEmpty() ? null : text;
  }

  private String parseIssueTime(String product, String link) {
    String token = product != null && !product.isBlank() ? product : link;
    if (token == null) {
      return null;
    }
    Matcher matcher = ISSUE_TIMESTAMP.matcher(token);
    if (!matcher.find()) {
      return null;
    }
    LocalDateTime localDateTime = LocalDateTime.parse(matcher.group(1), ISSUE_FORMATTER);
    return localDateTime.toInstant(ZoneOffset.UTC).toString();
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
