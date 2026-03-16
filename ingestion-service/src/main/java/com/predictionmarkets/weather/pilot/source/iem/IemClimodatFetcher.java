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
import java.time.Instant;
import java.util.Locale;
import java.util.Map;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class IemClimodatFetcher {
  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public IemClimodatFetcher(SourceHttpClient httpClient,
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
    String climodatStation = aliasValue(station, "climodat_station");
    String climodatNetwork = aliasValue(station, "climodat_network");
    String[] startParts = startDate.split("-");
    String[] endParts = endDate.split("-");
    String url = "https://mesonet.agron.iastate.edu/cgi-bin/request/coop.py?station=" + climodatStation
        + "&network=" + climodatNetwork
        + "&year1=" + startParts[0] + "&month1=" + Integer.parseInt(startParts[1])
        + "&day1=" + Integer.parseInt(startParts[2])
        + "&year2=" + endParts[0] + "&month2=" + Integer.parseInt(endParts[1])
        + "&day2=" + Integer.parseInt(endParts[2])
        + "&vars=high,low,precip,snow,snowd&what=download&delim=comma";
    HttpResponseData response = httpClient.get("iem", url, Map.of());
    manifestService.recordHttpRequest(new HttpRequestLogRecord(
        runId, jobId, "iem_climodat", "iem", station.getStationKey(), "fetch_climodat_range",
        url, "GET", response.statusCode(), null, null, response.durationMs(), response.body().length,
        null, response.retryCount(),
        response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
        null, null, Instant.now().toString()));
    String text = new String(response.body(), StandardCharsets.UTF_8);
    String checksum = rawStorageService.storeText(
        runId, station.getStationKey(), "iem_climodat", "iem", climodatStation + "::" + startDate + "::" + endDate,
        url, response.statusCode(), text, "RAW_STORED", 0);
    try {
      String csv = text.lines()
          .filter(line -> !line.startsWith("#") && !line.isBlank())
          .reduce("", (left, right) -> left + right + System.lineSeparator());
      int rows = 0;
      try (CSVParser parser = CSVParser.parse(new StringReader(csv), CSVFormat.DEFAULT.builder()
          .setHeader()
          .setSkipHeaderRecord(true)
          .build())) {
        for (CSVRecord record : parser) {
          String dateLocal = record.get("day").replace('/', '-');
          catalogService.execute("""
              INSERT INTO climodat_daily_aux (
                station_key, date_localish, high_f, low_f, precip_in, snow_in, snowdepth_in,
                estimated_flags_json, auxiliary_only, not_label_of_truth, source_name,
                source_family, source_identifier, request_url_or_bucket_key, issue_time_utc,
                ingested_at_utc, raw_object_sha256, parser_version
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              ON CONFLICT(station_key, date_localish) DO UPDATE SET
                high_f=excluded.high_f,
                low_f=excluded.low_f,
                precip_in=excluded.precip_in,
                snow_in=excluded.snow_in,
                snowdepth_in=excluded.snowdepth_in,
                ingested_at_utc=excluded.ingested_at_utc,
                raw_object_sha256=excluded.raw_object_sha256
              """,
              station.getStationKey(),
              dateLocal,
              parseDouble(record.get("high")),
              parseDouble(record.get("low")),
              parseDouble(record.get("precip")),
              parseDouble(record.get("snow")),
              parseDouble(record.get("snowd")),
              null,
              1,
              1,
              "iem_climodat",
              "iem",
              climodatStation + "::" + dateLocal,
              url,
              null,
              Instant.now().toString(),
              checksum,
              properties.getParserVersion());
          rows++;
        }
      }
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rows, "SUCCESS");
      manifestService.recordNormalizedPartition(runId, "climodat_daily_aux", station.getStationKey(),
          startDate, rows, checksum, catalogService.toJson(Map.of("startDate", startDate, "endDate", endDate)));
      jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "iem_climodat",
          "normalize_climodat_range", "SUCCESS", Map.of("rows_parsed", rows));
      return rows;
    } catch (Exception ex) {
      metricsAccumulator.recordParserFailure("iem_climodat");
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, 0, "FAILED");
      throw new IllegalStateException("Failed to parse IEM Climodat response", ex);
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
