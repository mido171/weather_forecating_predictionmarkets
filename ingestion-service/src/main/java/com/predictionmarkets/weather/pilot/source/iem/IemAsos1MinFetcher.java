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
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class IemAsos1MinFetcher {
  private static final DateTimeFormatter VALID_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public IemAsos1MinFetcher(SourceHttpClient httpClient,
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
                         Instant startUtc,
                         Instant endUtc,
                         JobMetricsAccumulator metricsAccumulator) {
    String stationCode = aliasValue(station, "iem_asos_station");
    String url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station=" + stationCode
        + "&vars=tmpf,dwpf,drct,sknt&sts=" + startUtc.toString().replace(":00Z", "Z")
        + "&ets=" + endUtc.toString().replace(":00Z", "Z")
        + "&sample=1min&what=download&tz=UTC&delim=comma";
    HttpResponseData response = httpClient.get("iem", url, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, "iem_asos_1min", "iem", station.getStationKey(), "fetch_asos_1min_range",
        url, "GET", response.statusCode(), null, null, response.durationMs(), response.body().length,
        null, response.retryCount(),
        response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String text = new String(response.body(), StandardCharsets.UTF_8);
    String checksum = rawStorageService.storeText(
        runId, station.getStationKey(), "iem_asos_1min", "iem", stationCode + "::" + startUtc + "::" + endUtc,
        url, response.statusCode(), text, "RAW_STORED", 0);
    try {
      List<MinuteRow> rows = new ArrayList<>();
      try (CSVParser parser = CSVParser.parse(new StringReader(text), CSVFormat.DEFAULT.builder()
          .setHeader()
          .setSkipHeaderRecord(true)
          .build())) {
        for (CSVRecord record : parser) {
          Instant valid = LocalDateTime.parse(record.get("valid(UTC)"), VALID_FORMATTER).toInstant(ZoneOffset.UTC);
          rows.add(new MinuteRow(
              valid,
              parseDouble(record.get("tmpf")),
              parseDouble(record.get("dwpf")),
              parseDouble(record.get("drct")),
              parseDouble(record.get("sknt"))));
        }
      }
      Deque<MinuteRow> fiveMinuteWindow = new ArrayDeque<>();
      Deque<MinuteRow> fifteenMinuteWindow = new ArrayDeque<>();
      for (MinuteRow row : rows) {
        advanceWindow(fiveMinuteWindow, row.validTimeUtc(), Duration.ofMinutes(5));
        fiveMinuteWindow.addLast(row);
        advanceWindow(fifteenMinuteWindow, row.validTimeUtc(), Duration.ofMinutes(15));
        fifteenMinuteWindow.addLast(row);
        catalogService.execute("""
            INSERT INTO asos_1min_obs (
              station_key, valid_time_utc, tmpf, dwpf, drct, sknt,
              aggregate_5m_json, aggregate_15m_json, source_name, source_family,
              source_identifier, request_url_or_bucket_key, issue_time_utc, ingested_at_utc,
              raw_object_sha256, parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_key, valid_time_utc) DO UPDATE SET
              tmpf=excluded.tmpf,
              dwpf=excluded.dwpf,
              drct=excluded.drct,
              sknt=excluded.sknt,
              aggregate_5m_json=excluded.aggregate_5m_json,
              aggregate_15m_json=excluded.aggregate_15m_json,
              ingested_at_utc=excluded.ingested_at_utc,
              raw_object_sha256=excluded.raw_object_sha256
            """,
            station.getStationKey(),
            row.validTimeUtc().toString(),
            row.tmpf(),
            row.dwpf(),
            row.drct(),
            row.sknt(),
            aggregateJson(fiveMinuteWindow),
            aggregateJson(fifteenMinuteWindow),
            "iem_asos_1min",
            "iem",
            stationCode + "::" + row.validTimeUtc(),
            url,
            null,
            Instant.now().toString(),
            checksum,
            properties.getParserVersion());
      }
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows.size(), "SUCCESS");
      manifestService.recordNormalizedPartition(runId, "asos_1min_obs", station.getStationKey(),
          startUtc.toString(), rows.size(), checksum,
          catalogService.toJson(Map.of("startUtc", startUtc, "endUtc", endUtc)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "iem_asos_1min",
          "normalize_asos_1min_range", "SUCCESS", Map.of("rows_parsed", rows.size()));
      return rows.size();
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("iem_asos_1min");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse ASOS 1-minute response", ex);
    }
  }

  private void advanceWindow(Deque<MinuteRow> window, Instant current, Duration duration) {
    while (!window.isEmpty() && window.peekFirst().validTimeUtc().isBefore(current.minus(duration).plusSeconds(60))) {
      window.removeFirst();
    }
  }

  private String aggregateJson(Deque<MinuteRow> window) {
    double tmpfSum = 0.0;
    double dwpfSum = 0.0;
    double drctSum = 0.0;
    double skntSum = 0.0;
    int tmpfCount = 0;
    int dwpfCount = 0;
    int drctCount = 0;
    int skntCount = 0;
    for (MinuteRow row : window) {
      if (row.tmpf() != null) {
        tmpfSum += row.tmpf();
        tmpfCount++;
      }
      if (row.dwpf() != null) {
        dwpfSum += row.dwpf();
        dwpfCount++;
      }
      if (row.drct() != null) {
        drctSum += row.drct();
        drctCount++;
      }
      if (row.sknt() != null) {
        skntSum += row.sknt();
        skntCount++;
      }
    }
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("count", window.size());
    payload.put("avgTmpf", tmpfCount == 0 ? null : tmpfSum / tmpfCount);
    payload.put("avgDwpf", dwpfCount == 0 ? null : dwpfSum / dwpfCount);
    payload.put("avgDrct", drctCount == 0 ? null : drctSum / drctCount);
    payload.put("avgSknt", skntCount == 0 ? null : skntSum / skntCount);
    return catalogService.toJson(payload);
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

  private record MinuteRow(Instant validTimeUtc, Double tmpf, Double dwpf, Double drct, Double sknt) {
  }
}
