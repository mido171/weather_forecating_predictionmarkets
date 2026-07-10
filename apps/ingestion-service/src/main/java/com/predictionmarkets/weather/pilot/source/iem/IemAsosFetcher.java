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
import com.predictionmarkets.weather.pilot.source.SettlementDayUtil;
import com.predictionmarkets.weather.pilot.source.SourceHttpClient;
import com.predictionmarkets.weather.pilot.storage.RawStorageService;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class IemAsosFetcher {
  private static final DateTimeFormatter VALID_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

  private final SourceHttpClient httpClient;
  private final ManifestService manifestService;
  private final RawStorageService rawStorageService;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public IemAsosFetcher(SourceHttpClient httpClient,
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
                         LocalDate startDate,
                         LocalDate endDate,
                         JobMetricsAccumulator metricsAccumulator) {
    String network = aliasValue(station, "iem_asos_network");
    String stationCode = aliasValue(station, "iem_asos_station");
    List<HourlyObsRow> allRows = new ArrayList<>();
    ZoneId zoneId = ZoneId.of(station.getTimezone());
    LocalDate cursor = startDate;
    while (!cursor.isAfter(endDate)) {
      LocalDate nextDay = cursor.plusDays(1);
      String url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=" + network
          + "&station=" + stationCode
          + "&data=all&year1=" + cursor.getYear()
          + "&month1=" + cursor.getMonthValue()
          + "&day1=" + cursor.getDayOfMonth()
          + "&year2=" + nextDay.getYear()
          + "&month2=" + nextDay.getMonthValue()
          + "&day2=" + nextDay.getDayOfMonth()
          + "&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no";
      HttpResponseData response = httpClient.get("iem", url, Map.of());
      manifestService.recordHttpRequest(new HttpRequestLogRecord(
          runId, jobId, "iem_asos_hourly", "iem", station.getStationKey(), "fetch_asos_day",
          url, "GET", response.statusCode(), null, null, response.durationMs(), response.body().length,
          null, response.retryCount(),
          response.statusCode() >= 200 && response.statusCode() < 300 ? "SUCCESS" : "FAILED",
          null, null, Instant.now().toString()));
      String text = new String(response.body(), StandardCharsets.UTF_8);
      String checksum = rawStorageService.storeText(
          runId, station.getStationKey(), "iem_asos_hourly", "iem", stationCode + "::" + cursor,
          url, response.statusCode(), text, "RAW_STORED", 0);
      int rowsForDay = parseDayCsv(station, text, checksum, allRows, zoneId);
      metricsAccumulator.recordRequest(response.durationMs(), response.body().length, rowsForDay, "SUCCESS");
      cursor = cursor.plusDays(1);
    }
    allRows.sort(Comparator.comparing(HourlyObsRow::validTimeUtc));
    deriveObsMetrics(allRows, zoneId);
    for (HourlyObsRow row : allRows) {
      catalogService.execute("""
          INSERT INTO asos_hourly_obs (
            station_key, valid_time_utc, tmpf, dwpf, relh, drct, sknt, p01i, alti, mslp, vsby,
            gust, skyc1, skyc2, skyc3, skyc4, skyl1, skyl2, skyl3, skyl4, wxcodes, feel,
            metar_raw, snowdepth, obs_day_lstd, max_so_far_tmpf_in_obs_day,
            min_so_far_tmpf_in_obs_day, temp_trend_1h, temp_trend_3h, temp_trend_6h,
            dewpoint_trend_1h, wind_trend_1h, source_name, source_family, source_identifier,
            request_url_or_bucket_key, issue_time_utc, ingested_at_utc, raw_object_sha256,
            parser_version
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(station_key, valid_time_utc) DO UPDATE SET
            tmpf=excluded.tmpf,
            dwpf=excluded.dwpf,
            relh=excluded.relh,
            drct=excluded.drct,
            sknt=excluded.sknt,
            p01i=excluded.p01i,
            alti=excluded.alti,
            mslp=excluded.mslp,
            vsby=excluded.vsby,
            gust=excluded.gust,
            skyc1=excluded.skyc1,
            skyc2=excluded.skyc2,
            skyc3=excluded.skyc3,
            skyc4=excluded.skyc4,
            skyl1=excluded.skyl1,
            skyl2=excluded.skyl2,
            skyl3=excluded.skyl3,
            skyl4=excluded.skyl4,
            wxcodes=excluded.wxcodes,
            feel=excluded.feel,
            metar_raw=excluded.metar_raw,
            snowdepth=excluded.snowdepth,
            obs_day_lstd=excluded.obs_day_lstd,
            max_so_far_tmpf_in_obs_day=excluded.max_so_far_tmpf_in_obs_day,
            min_so_far_tmpf_in_obs_day=excluded.min_so_far_tmpf_in_obs_day,
            temp_trend_1h=excluded.temp_trend_1h,
            temp_trend_3h=excluded.temp_trend_3h,
            temp_trend_6h=excluded.temp_trend_6h,
            dewpoint_trend_1h=excluded.dewpoint_trend_1h,
            wind_trend_1h=excluded.wind_trend_1h,
            ingested_at_utc=excluded.ingested_at_utc,
            raw_object_sha256=excluded.raw_object_sha256
          """,
          station.getStationKey(),
          row.validTimeUtc().toString(),
          row.tmpf(),
          row.dwpf(),
          row.relh(),
          row.drct(),
          row.sknt(),
          row.p01i(),
          row.alti(),
          row.mslp(),
          row.vsby(),
          row.gust(),
          row.skyc1(),
          row.skyc2(),
          row.skyc3(),
          row.skyc4(),
          row.skyl1(),
          row.skyl2(),
          row.skyl3(),
          row.skyl4(),
          row.wxcodes(),
          row.feel(),
          row.metarRaw(),
          row.snowdepth(),
          row.obsDayLstd() == null ? null : row.obsDayLstd().toString(),
          row.maxSoFarTmpfInObsDay(),
          row.minSoFarTmpfInObsDay(),
          row.tempTrend1h(),
          row.tempTrend3h(),
          row.tempTrend6h(),
          row.dewpointTrend1h(),
          row.windTrend1h(),
          "iem_asos_hourly",
          "iem",
          stationCode + "::" + row.validTimeUtc(),
          row.requestUrl(),
          null,
          Instant.now().toString(),
          row.rawChecksum(),
          properties.getParserVersion());
    }
    manifestService.recordNormalizedPartition(runId, "asos_hourly_obs", station.getStationKey(),
        startDate.toString(), allRows.size(), null,
        catalogService.toJson(Map.of("startDate", startDate, "endDate", endDate)));
    jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(), "iem_asos_hourly",
        "normalize_asos_range", "SUCCESS", Map.of("rows_parsed", allRows.size()));
    return allRows.size();
  }

  private int parseDayCsv(StationConfig station,
                          String text,
                          String checksum,
                          List<HourlyObsRow> out,
                          ZoneId zoneId) {
    try (CSVParser parser = CSVParser.parse(new StringReader(text), CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .build())) {
      int rows = 0;
      for (CSVRecord record : parser) {
        Instant valid = LocalDateTime.parse(record.get("valid"), VALID_FORMATTER).toInstant(ZoneOffset.UTC);
        out.add(new HourlyObsRow(
            valid,
            parseDouble(record.get("tmpf")),
            parseDouble(record.get("dwpf")),
            parseDouble(record.get("relh")),
            parseDouble(record.get("drct")),
            parseDouble(record.get("sknt")),
            parseDouble(record.get("p01i")),
            parseDouble(record.get("alti")),
            parseDouble(record.get("mslp")),
            parseDouble(record.get("vsby")),
            parseDouble(record.get("gust")),
            textOrNull(record.get("skyc1")),
            textOrNull(record.get("skyc2")),
            textOrNull(record.get("skyc3")),
            textOrNull(record.get("skyc4")),
            parseDouble(record.get("skyl1")),
            parseDouble(record.get("skyl2")),
            parseDouble(record.get("skyl3")),
            parseDouble(record.get("skyl4")),
            textOrNull(record.get("wxcodes")),
            parseDouble(record.get("feel")),
            textOrNull(record.get("metar")),
            parseDouble(record.get("snowdepth")),
            SettlementDayUtil.localStandardDay(zoneId, valid),
            null, null, null, null, null, null, null,
            checksum,
            "hourly-day-fetch"));
        rows++;
      }
      return rows;
    } catch (Exception ex) {
      throw new IllegalStateException("Failed to parse IEM ASOS hourly CSV", ex);
    }
  }

  private void deriveObsMetrics(List<HourlyObsRow> rows, ZoneId zoneId) {
    NavigableMap<Instant, HourlyObsRow> byInstant = new TreeMap<>();
    for (HourlyObsRow row : rows) {
      byInstant.put(row.validTimeUtc(), row);
    }
    TreeMap<LocalDate, Double> maxByDay = new TreeMap<>();
    TreeMap<LocalDate, Double> minByDay = new TreeMap<>();
    for (int index = 0; index < rows.size(); index++) {
      HourlyObsRow row = rows.get(index);
      LocalDate day = row.obsDayLstd();
      if (row.tmpf() != null) {
        maxByDay.merge(day, row.tmpf(), Math::max);
        minByDay.merge(day, row.tmpf(), Math::min);
      }
      rows.set(index, row.withDerived(
          maxByDay.get(day),
          minByDay.get(day),
          trend(row, byInstant, Duration.ofHours(1), HourlyObsRow::tmpf),
          trend(row, byInstant, Duration.ofHours(3), HourlyObsRow::tmpf),
          trend(row, byInstant, Duration.ofHours(6), HourlyObsRow::tmpf),
          trend(row, byInstant, Duration.ofHours(1), HourlyObsRow::dwpf),
          trend(row, byInstant, Duration.ofHours(1), HourlyObsRow::sknt)));
    }
  }

  private Double trend(HourlyObsRow row,
                       NavigableMap<Instant, HourlyObsRow> byInstant,
                       Duration lookback,
                       java.util.function.Function<HourlyObsRow, Double> extractor) {
    Double current = extractor.apply(row);
    if (current == null) {
      return null;
    }
    Map.Entry<Instant, HourlyObsRow> previousEntry = byInstant.floorEntry(row.validTimeUtc().minus(lookback));
    if (previousEntry == null) {
      return null;
    }
    Double previous = extractor.apply(previousEntry.getValue());
    return previous == null ? null : current - previous;
  }

  private Double parseDouble(String raw) {
    if (raw == null || raw.isBlank() || "M".equalsIgnoreCase(raw) || "T".equalsIgnoreCase(raw)) {
      return null;
    }
    return Double.valueOf(raw);
  }

  private String textOrNull(String raw) {
    if (raw == null) {
      return null;
    }
    String text = raw.trim();
    return text.isEmpty() || "M".equalsIgnoreCase(text) ? null : text;
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

  private record HourlyObsRow(
      Instant validTimeUtc,
      Double tmpf,
      Double dwpf,
      Double relh,
      Double drct,
      Double sknt,
      Double p01i,
      Double alti,
      Double mslp,
      Double vsby,
      Double gust,
      String skyc1,
      String skyc2,
      String skyc3,
      String skyc4,
      Double skyl1,
      Double skyl2,
      Double skyl3,
      Double skyl4,
      String wxcodes,
      Double feel,
      String metarRaw,
      Double snowdepth,
      LocalDate obsDayLstd,
      Double maxSoFarTmpfInObsDay,
      Double minSoFarTmpfInObsDay,
      Double tempTrend1h,
      Double tempTrend3h,
      Double tempTrend6h,
      Double dewpointTrend1h,
      Double windTrend1h,
      String rawChecksum,
      String requestUrl) {
    private HourlyObsRow withDerived(Double maxSoFar,
                                     Double minSoFar,
                                     Double temp1h,
                                     Double temp3h,
                                     Double temp6h,
                                     Double dew1h,
                                     Double wind1h) {
      return new HourlyObsRow(
          validTimeUtc, tmpf, dwpf, relh, drct, sknt, p01i, alti, mslp, vsby, gust,
          skyc1, skyc2, skyc3, skyc4, skyl1, skyl2, skyl3, skyl4, wxcodes, feel,
          metarRaw, snowdepth, obsDayLstd, maxSoFar, minSoFar, temp1h, temp3h, temp6h,
          dew1h, wind1h, rawChecksum, requestUrl);
    }
  }
}
