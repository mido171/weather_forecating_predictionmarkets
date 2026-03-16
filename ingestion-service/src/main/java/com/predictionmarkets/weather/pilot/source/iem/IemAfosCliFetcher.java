package com.predictionmarkets.weather.pilot.source.iem;

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
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.springframework.stereotype.Service;

@Service
public class IemAfosCliFetcher {
  private static final Pattern SUMMARY_DATE_PATTERN = Pattern.compile(
      "SUMMARY FOR ([A-Z]+\\s+\\d{1,2}\\s+\\d{4})", Pattern.CASE_INSENSITIVE);
  private static final Pattern ISSUED_PATTERN = Pattern.compile(
      "(\\d{1,4})\\s+(AM|PM)\\s+(EST|EDT)\\s+\\w{3}\\s+([A-Z]{3})\\s+(\\d{1,2})\\s+(\\d{4})",
      Pattern.CASE_INSENSITIVE);
  private static final Pattern MAX_PATTERN = Pattern.compile(
      "MAXIMUM\\s+(\\d+)\\s+(\\d{1,4}\\s+[AP]M)", Pattern.CASE_INSENSITIVE);
  private static final Pattern MIN_PATTERN = Pattern.compile(
      "MINIMUM\\s+(\\d+)\\s+(\\d{1,4}\\s+[AP]M)", Pattern.CASE_INSENSITIVE);
  private static final DateTimeFormatter SUMMARY_FORMATTER =
      new DateTimeFormatterBuilder()
          .parseCaseInsensitive()
          .appendPattern("MMMM d yyyy")
          .toFormatter(Locale.US);
  private static final DateTimeFormatter ISSUED_DATE_FORMATTER =
      new DateTimeFormatterBuilder()
          .parseCaseInsensitive()
          .appendPattern("MMM d yyyy")
          .toFormatter(Locale.US);

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public IemAfosCliFetcher(SourceHttpClient httpClient,
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

  public int ingestYear(String jobId,
                        String runId,
                        StationConfig station,
                        int year,
                        JobMetricsAccumulator metricsAccumulator) {
    String pil = aliasValue(station, "raw_cli_pil");
    String url = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=" + pil
        + "&fmt=zip&sdate=" + year + "-01-01T00:00Z&edate=" + year
        + "-12-31T23:59Z&limit=9999";
    HttpResponseData response = httpClient.get("iem", url, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, "iem_afos_cli", "iem", station.getStationKey(), "fetch_afos_year",
        url, "GET", response.statusCode(), null, null, response.durationMs(), response.body().length,
        null, response.retryCount(),
        response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String zipChecksum = rawStorageService.storeBytes(
        runId, station.getStationKey(), "iem_afos_cli", "iem",
        station.getStationKey() + "::" + year, url, response.statusCode(), response.body(),
        "application/zip", "ZIP_STORED", 0);
    int rows = 0;
    try (ZipInputStream zipInputStream = new ZipInputStream(new ByteArrayInputStream(response.body()))) {
      ZipEntry entry;
      while ((entry = zipInputStream.getNextEntry()) != null) {
        byte[] payload = zipInputStream.readAllBytes();
        String text = new String(payload, StandardCharsets.UTF_8);
        String entryChecksum = rawStorageService.storeText(
            runId, station.getStationKey(), "iem_afos_cli", "iem", entry.getName(),
            entry.getName(), 200, text, "TEXT_STORED", 1);
        try {
          ParsedAfosCli parsed = parseText(text, station.getTimezone(), pil, entry.getName());
          upsertParsedEntry(station, entry.getName(), text, parsed, entryChecksum);
          rows++;
        } catch (Exception ex) {
          metricsAccumulator.recordParserFailure("iem_afos_cli");
          jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "iem_afos_cli",
              "parse_afos_entry_failed", "FAILED",
              Map.of("source_identifier", entry.getName(), "message", ex.getMessage()));
        }
      }
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows, "SUCCESS");
      manifestService.recordNormalizedPartition(runId, "cli_raw_text", station.getStationKey(),
          year + "-01-01", rows, zipChecksum, catalogService.toJson(Map.of("year", year)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "iem_afos_cli",
          "normalize_afos_year", "SUCCESS", Map.of("year", year, "rows_parsed", rows));
      return rows;
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("iem_afos_cli");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse AFOS CLI archive for year " + year, ex);
    }
  }

  private void upsertParsedEntry(StationConfig station,
                                 String sourceIdentifier,
                                 String text,
                                 ParsedAfosCli parsed,
                                 String checksum) {
    catalogService.execute("""
        INSERT INTO cli_raw_text (
          station_key, pil, product_timestamp_utc, issued_by_center, local_report_date,
          max_temp_f, max_temp_local_time, min_temp_f, min_temp_local_time, text_body,
          source_name, source_family, source_identifier, request_url_or_bucket_key, issue_time_utc,
          valid_time_utc, ingested_at_utc, raw_object_sha256, parser_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_key, pil, product_timestamp_utc) DO UPDATE SET
          issued_by_center=excluded.issued_by_center,
          local_report_date=excluded.local_report_date,
          max_temp_f=excluded.max_temp_f,
          max_temp_local_time=excluded.max_temp_local_time,
          min_temp_f=excluded.min_temp_f,
          min_temp_local_time=excluded.min_temp_local_time,
          text_body=excluded.text_body,
          ingested_at_utc=excluded.ingested_at_utc,
          raw_object_sha256=excluded.raw_object_sha256,
          parser_version=excluded.parser_version
        """,
        station.getStationKey(),
        parsed.pil(),
        parsed.productTimestampUtc(),
        parsed.issuedByCenter(),
        parsed.localReportDate(),
        parsed.maxTempF(),
        parsed.maxTempLocalTime(),
        parsed.minTempF(),
        parsed.minTempLocalTime(),
        text,
        "iem_afos_cli",
        "iem",
        sourceIdentifier,
        sourceIdentifier,
        parsed.productTimestampUtc(),
        null,
        Instant.now().toString(),
        checksum,
        properties.getParserVersion());
    if (parsed.localReportDate() != null) {
      Map<String, Object> existing = catalogService.querySingle("""
          SELECT tmax_f, tmin_f
          FROM cli_daily_label
          WHERE station_key = ? AND target_date_local = ?
          """, station.getStationKey(), parsed.localReportDate());
      boolean disagreement = false;
      Object existingTmax = existing.get("tmax_f");
      Object existingTmin = existing.get("tmin_f");
      if (existingTmax instanceof Number number && parsed.maxTempF() != null) {
        disagreement |= Double.compare(number.doubleValue(), parsed.maxTempF()) != 0;
      }
      if (existingTmin instanceof Number number && parsed.minTempF() != null) {
        disagreement |= Double.compare(number.doubleValue(), parsed.minTempF()) != 0;
      }
      catalogService.execute("""
          UPDATE cli_daily_label
          SET raw_cli_tmax_f = ?, raw_cli_tmin_f = ?, label_disagreement_flag = ?
          WHERE station_key = ? AND target_date_local = ?
          """,
          parsed.maxTempF(),
          parsed.minTempF(),
          disagreement ? 1 : 0,
          station.getStationKey(),
          parsed.localReportDate());
    }
  }

  private ParsedAfosCli parseText(String text, String timezone, String pil, String sourceIdentifier) {
    String issuedByCenter = null;
    String[] lines = text.split("\\R");
    if (lines.length > 1) {
      String[] tokens = lines[1].trim().split("\\s+");
      if (tokens.length >= 2) {
        issuedByCenter = tokens[1];
      }
    }
    String localReportDate = null;
    Matcher summaryMatcher = SUMMARY_DATE_PATTERN.matcher(text);
    if (summaryMatcher.find()) {
      localReportDate = parseSummaryDate(summaryMatcher.group(1));
    }
    String productTimestamp = null;
    Matcher issuedMatcher = ISSUED_PATTERN.matcher(text);
    if (issuedMatcher.find()) {
      String hourMinute = issuedMatcher.group(1);
      String amPm = issuedMatcher.group(2).toUpperCase(Locale.ROOT);
      String zoneAbbrev = issuedMatcher.group(3).toUpperCase(Locale.ROOT);
      String month = issuedMatcher.group(4).toUpperCase(Locale.ROOT);
      String day = issuedMatcher.group(5);
      String year = issuedMatcher.group(6);
      LocalDate issuedDate = LocalDate.parse(month + " " + day + " " + year, ISSUED_DATE_FORMATTER);
      int numericTime = Integer.parseInt(hourMinute);
      int hour = numericTime / 100;
      int minute = numericTime % 100;
      if (hour == 12) {
        hour = 0;
      }
      if ("PM".equals(amPm)) {
        hour += 12;
      }
      ZoneOffset zoneOffset = "EDT".equals(zoneAbbrev) ? ZoneOffset.ofHours(-4) : ZoneOffset.ofHours(-5);
      productTimestamp = issuedDate.atTime(hour, minute).atOffset(zoneOffset).toInstant().toString();
    } else if (sourceIdentifier != null && sourceIdentifier.contains("_")) {
      String token = sourceIdentifier.substring(sourceIdentifier.indexOf('_') + 1, sourceIdentifier.indexOf('.'));
      productTimestamp = LocalDateTime.parse(token, DateTimeFormatter.ofPattern("yyyyMMddHHmm"))
          .toInstant(ZoneOffset.UTC).toString();
    }
    Double maxTempF = extractTemperature(text, MAX_PATTERN);
    String maxTime = extractLocalTime(text, MAX_PATTERN);
    Double minTempF = extractTemperature(text, MIN_PATTERN);
    String minTime = extractLocalTime(text, MIN_PATTERN);
    return new ParsedAfosCli(pil, productTimestamp, issuedByCenter, localReportDate,
        maxTempF, maxTime, minTempF, minTime);
  }

  private Double extractTemperature(String text, Pattern pattern) {
    Matcher matcher = pattern.matcher(text);
    return matcher.find() ? Double.valueOf(matcher.group(1)) : null;
  }

  private String extractLocalTime(String text, Pattern pattern) {
    Matcher matcher = pattern.matcher(text);
    return matcher.find() ? matcher.group(2) : null;
  }

  private String parseSummaryDate(String raw) {
    try {
      return LocalDate.parse(raw.trim(), SUMMARY_FORMATTER).toString();
    } catch (Exception ex) {
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

  private record ParsedAfosCli(
      String pil,
      String productTimestampUtc,
      String issuedByCenter,
      String localReportDate,
      Double maxTempF,
      String maxTempLocalTime,
      Double minTempF,
      String minTempLocalTime) {
  }
}
