package com.predictionmarkets.weather.kalshiapi.backtest;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.kalshiapi.config.BacktestGridProperties;
import com.predictionmarkets.weather.kalshiapi.config.BacktestGridProperties.StationSpec;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class MosBacktestGridService {

  private static final Logger log = LoggerFactory.getLogger(MosBacktestGridService.class);
  private static final ObjectMapper STATIC_MAPPER = new ObjectMapper();

  private static final int SUPPORT_LO = -20;
  private static final int SUPPORT_HI = 130;

  private static final Pattern FILE_DATE_PATTERN = Pattern.compile("_(\\d{8})\\.csv$");
  private static final Pattern LT_PATTERN = Pattern.compile("<\\s*(\\d+)");
  private static final Pattern LE_PATTERN = Pattern.compile("<=\\s*(\\d+)");
  private static final Pattern GT_PATTERN = Pattern.compile(">\\s*(\\d+)");
  private static final Pattern GE_PATTERN = Pattern.compile(">=\\s*(\\d+)");
  private static final Pattern NUMBER_PATTERN = Pattern.compile("(\\d+)");

  private static final Comparator<Candidate> CANDIDATE_ORDER =
      Comparator.comparingDouble(Candidate::modelWinProb).reversed()
          .thenComparingDouble(Candidate::ev).reversed()
          .thenComparingDouble(Candidate::marketPrice)
          .thenComparing(Candidate::stationId)
          .thenComparing(Candidate::bucketRaw)
          .thenComparing(Candidate::side);

  private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ISO_LOCAL_DATE;

  private final BacktestGridProperties properties;
  private final ObjectMapper objectMapper;

  public MosBacktestGridService(BacktestGridProperties properties, ObjectMapper objectMapper) {
    this.properties = properties;
    this.objectMapper = objectMapper;
  }

  public void run() throws Exception {
    String runId = OffsetDateTime.now(ZoneOffset.UTC).format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'"));
    Path sqlitePath = Path.of(properties.getSqlitePath());
    Path outRoot = Path.of(properties.getOutDir());
    Path runDir = outRoot.resolve(properties.getRunPrefix() + "_" + runId);
    Files.createDirectories(outRoot);
    Files.createDirectories(runDir);
    Files.createDirectories(sqlitePath.getParent());
    if (properties.isOverwriteSqlite() && Files.exists(sqlitePath)) {
      Files.delete(sqlitePath);
    }

    List<ResolvedStation> stations = resolveStations();
    exportForecastsToSqlite(runId, sqlitePath, stations);
    ForecastStore forecastStore = loadForecastStoreFromSqlite(sqlitePath, stations);
    Map<MarketKey, MarketDay> marketStore = preloadMarketStore(stations, forecastStore);
    runBacktestGrid(runId, sqlitePath, runDir, stations, forecastStore, marketStore);
  }

  private List<ResolvedStation> resolveStations() {
    List<ResolvedStation> stations = new ArrayList<>();
    for (StationSpec spec : properties.getStations()) {
      stations.add(new ResolvedStation(
          spec.getStationId().trim().toUpperCase(),
          Path.of(spec.getTruthCsvPath()),
          Path.of(spec.getKalshiRoot()),
          spec.getFilePrefix().trim().toUpperCase()
      ));
    }
    return stations;
  }

  private record ResolvedStation(String stationId, Path truthCsvPath, Path kalshiRoot, String filePrefix) {
  }

  private record ForecastStore(
      Map<String, Map<LocalDate, StationPrediction>> predictionsByStation,
      Map<String, Integer> predictionCountsByStation,
      int forecastSuccessDayCount,
      int forecastFailedDayCount) {
  }

  private record PredictionKey(String stationId, LocalDate targetDate) {
  }

  private record MarketKey(String stationId, LocalDate targetDate) {
  }

  private record StationPrediction(
      String stationId,
      LocalDate targetDate,
      double yTmax,
      double q05,
      double q10,
      double q25,
      double q50,
      double q75,
      double q90,
      double q95,
      double[] pmf) {
  }

  private record MarketColumn(String rawLabel, Bucket bucket) {
  }

  private record MarketRow(Instant timestamp, double[] values) {
  }

  private record MarketDay(
      String stationId,
      LocalDate targetDate,
      Path marketFile,
      Instant marketOpenUtc,
      List<MarketColumn> columns,
      NavigableMap<Instant, MarketRow> rowsByTimestamp) {
  }

  private record DayContext(
      String stationId,
      StationPrediction prediction,
      MarketDay marketDay,
      Instant gateCutoffUtc,
      Instant effectiveCutoffUtc,
      NavigableMap<Instant, MarketRow> eligibleRows) {
  }

  private record Candidate(
      String targetDateLocal,
      String marketFileDateLocal,
      String stationId,
      Instant entryTimestampUtc,
      Instant marketOpenUtc,
      Instant gateCutoffUtc,
      Instant effectiveCutoffUtc,
      Path marketFile,
      String bucketRaw,
      String bucket,
      String side,
      double marketPrice,
      double modelWinProb,
      double ev,
      int yTmax,
      int win,
      Instant firstEligibleTimestampUtc,
      int eligibleCountAtEntryTimestamp) {
  }

  private record TradeRow(
      Candidate candidate,
      double stake,
      double shares,
      double pnl,
      double balanceBefore,
      double balanceAfter,
      double drawdown,
      String result) {
  }

  private record BaseRow(
      String runId,
      double evMin,
      double winMin,
      int trades,
      int wins,
      int losses,
      double winRate,
      double finalBalance,
      double maxDrawdown,
      int daysWithoutTradeCandidate,
      String stationCountsJson,
      String sideCountsJson,
      String tradesCsvPath,
      String summaryJsonPath,
      String sanityJsonPath,
      String dayDebugJsonPath,
      boolean sanityPassesAllChecks,
      int sanityCheckedTrades) {
  }

  private record ComboRow(
      String runId,
      String sizingMode,
      double evMin,
      double winMin,
      Double riskFraction,
      Double kellyFraction,
      double stakeCapUsd,
      String entryRule,
      String sidePriceRule,
      String stakeRule,
      String summaryJsonPath,
      String sanityJsonPath,
      int trades,
      int wins,
      int losses,
      double winRate,
      Double profitFactor,
      double finalBalance,
      double totalPnl,
      double maxDrawdown,
      double avgEvAtTrade,
      double medianEvAtTrade,
      String stationCountsJson,
      String sideCountsJson,
      double riskFractionUsedAvg,
      double riskFractionUsedMin,
      double riskFractionUsedMax,
      int stakeCapBreachCount) {
  }

  private record RankedComboRow(
      int rankPosition,
      double compositeScorePfWinLowdd,
      ComboRow comboRow) {
  }

  private record BaseStreamResult(BaseRow baseRow, List<ComboRow> comboRows) {
  }

  private record Bucket(String labelRaw, Integer lo, Integer hi, String mode) {
    boolean contains(int tempF) {
      return switch (mode) {
        case "or_below" -> hi != null && tempF <= hi;
        case "or_above" -> lo != null && tempF >= lo;
        case "range" -> lo != null && hi != null && tempF >= lo && tempF <= hi;
        default -> false;
      };
    }

    String canonicalLabel() {
      return switch (mode) {
        case "or_below" -> hi + "F or below";
        case "or_above" -> lo + "F or above";
        case "range" -> lo + "F to " + hi + "F";
        default -> labelRaw;
      };
    }
  }

  private void exportForecastsToSqlite(String runId, Path sqlitePath, List<ResolvedStation> stations) throws Exception {
    createForecastTables(sqlitePath);
    Map<String, Map<LocalDate, Double>> truthMaps = loadTruthMaps(stations);
    List<LocalDate> days = enumerateDays();

    try (var conn = openConnection(sqlitePath)) {
      conn.setAutoCommit(false);
      try (var dayStmt = conn.prepareStatement(
          "INSERT INTO forecast_target_days (" +
              "run_id,target_date_local,target_ymd,status,report_path,report_sha256,runtime_gate_failure_path," +
              "runtime_gate_failure_json,error_text,live_run_id,quote_asof_utc,requested_station_count,available_station_count," +
              "global_guardrail_counters_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
           var stationStmt = conn.prepareStatement(
               "INSERT INTO forecast_station_predictions (" +
                   "run_id,target_date_local,station_id,status,report_path,report_sha256,truth_available,y_tmax,runtime_utc," +
                   "runtime_expected_policy_utc,runtime_equals_expected_policy_runtime,runtime_lte_quote_asof,quantiles_monotonic," +
                   "prediction_point_tmax_f,q_0_05,q_0_10,q_0_25,q_0_50,q_0_75,q_0_90,q_0_95,bundle_dir,station_block_json,station_evidence_json" +
                   ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")) {

        int done = 0;
        for (LocalDate day : days) {
          ForecastDayExport export = ensureForecastReport(day, stations);
          bindForecastDay(dayStmt, runId, export, stations.size());
          dayStmt.executeUpdate();
          for (ResolvedStation station : stations) {
            bindForecastStationRow(stationStmt, runId, day, station, export, truthMaps.get(station.stationId()));
            stationStmt.executeUpdate();
          }
          conn.commit();
          done++;
          if (done % 25 == 0 || done == days.size()) {
            log.info("Forecast export progress {}/{}", done, days.size());
          }
        }
      }
    }
    log.info("Forecast export committed to {}", sqlitePath);
  }

  private ForecastStore loadForecastStoreFromSqlite(Path sqlitePath, List<ResolvedStation> stations) throws Exception {
    Map<String, Map<LocalDate, StationPrediction>> byStation = new LinkedHashMap<>();
    Map<String, Integer> counts = new LinkedHashMap<>();
    for (ResolvedStation station : stations) {
      byStation.put(station.stationId(), new LinkedHashMap<>());
      counts.put(station.stationId(), 0);
    }

    int successDays = 0;
    int failedDays = 0;

    try (var conn = openConnection(sqlitePath)) {
      try (var rs = conn.createStatement().executeQuery(
          "SELECT status, COUNT(*) AS c FROM forecast_target_days GROUP BY status")) {
        while (rs.next()) {
          String status = rs.getString("status");
          int c = rs.getInt("c");
          if ("ok".equals(status)) {
            successDays += c;
          } else {
            failedDays += c;
          }
        }
      }

      try (var stmt = conn.prepareStatement(
          "SELECT station_id,target_date_local,y_tmax,q_0_05,q_0_10,q_0_25,q_0_50,q_0_75,q_0_90,q_0_95 " +
              "FROM forecast_station_predictions WHERE status='ok' AND truth_available=1");
           var rs = stmt.executeQuery()) {
        while (rs.next()) {
          String stationId = rs.getString("station_id");
          LocalDate day = LocalDate.parse(rs.getString("target_date_local"));
          StationPrediction prediction = new StationPrediction(
              stationId,
              day,
              rs.getDouble("y_tmax"),
              rs.getDouble("q_0_05"),
              rs.getDouble("q_0_10"),
              rs.getDouble("q_0_25"),
              rs.getDouble("q_0_50"),
              rs.getDouble("q_0_75"),
              rs.getDouble("q_0_90"),
              rs.getDouble("q_0_95"),
              pmfFromQuantiles(
                  rs.getDouble("q_0_05"),
                  rs.getDouble("q_0_10"),
                  rs.getDouble("q_0_25"),
                  rs.getDouble("q_0_50"),
                  rs.getDouble("q_0_75"),
                  rs.getDouble("q_0_90"),
                  rs.getDouble("q_0_95"))
          );
          byStation.computeIfAbsent(stationId, ignored -> new LinkedHashMap<>()).put(day, prediction);
          counts.put(stationId, counts.getOrDefault(stationId, 0) + 1);
        }
      }
    }

    return new ForecastStore(byStation, counts, successDays, failedDays);
  }

  private void createForecastTables(Path sqlitePath) throws Exception {
    try (var conn = openConnection(sqlitePath); var stmt = conn.createStatement()) {
      stmt.executeUpdate(
          "CREATE TABLE IF NOT EXISTS forecast_target_days (" +
              "run_id TEXT NOT NULL," +
              "target_date_local TEXT NOT NULL," +
              "target_ymd TEXT NOT NULL," +
              "status TEXT NOT NULL," +
              "report_path TEXT," +
              "report_sha256 TEXT," +
              "runtime_gate_failure_path TEXT," +
              "runtime_gate_failure_json TEXT," +
              "error_text TEXT," +
              "live_run_id TEXT," +
              "quote_asof_utc TEXT," +
              "requested_station_count INTEGER NOT NULL," +
              "available_station_count INTEGER NOT NULL," +
              "global_guardrail_counters_json TEXT NOT NULL" +
              ")"
      );
      stmt.executeUpdate(
          "CREATE TABLE IF NOT EXISTS forecast_station_predictions (" +
              "run_id TEXT NOT NULL," +
              "target_date_local TEXT NOT NULL," +
              "station_id TEXT NOT NULL," +
              "status TEXT NOT NULL," +
              "report_path TEXT," +
              "report_sha256 TEXT," +
              "truth_available INTEGER NOT NULL," +
              "y_tmax REAL," +
              "runtime_utc TEXT," +
              "runtime_expected_policy_utc TEXT," +
              "runtime_equals_expected_policy_runtime INTEGER," +
              "runtime_lte_quote_asof INTEGER," +
              "quantiles_monotonic INTEGER," +
              "prediction_point_tmax_f REAL," +
              "q_0_05 REAL," +
              "q_0_10 REAL," +
              "q_0_25 REAL," +
              "q_0_50 REAL," +
              "q_0_75 REAL," +
              "q_0_90 REAL," +
              "q_0_95 REAL," +
              "bundle_dir TEXT," +
              "station_block_json TEXT NOT NULL," +
              "station_evidence_json TEXT NOT NULL" +
              ")"
      );
      stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_forecast_target_days_date ON forecast_target_days(target_date_local)");
      stmt.executeUpdate("CREATE INDEX IF NOT EXISTS idx_forecast_station_predictions_date_station ON forecast_station_predictions(target_date_local, station_id)");
    }
  }

  private List<LocalDate> enumerateDays() {
    LocalDate start = LocalDate.parse(properties.getStartDate());
    LocalDate end = LocalDate.parse(properties.getEndDate());
    List<LocalDate> days = new ArrayList<>();
    for (LocalDate day = start; !day.isAfter(end); day = day.plusDays(1)) {
      days.add(day);
    }
    return days;
  }

  private Map<String, Map<LocalDate, Double>> loadTruthMaps(List<ResolvedStation> stations) throws Exception {
    Map<String, Map<LocalDate, Double>> out = new LinkedHashMap<>();
    for (ResolvedStation station : stations) {
      out.put(station.stationId(), loadTruthMap(station.truthCsvPath()));
    }
    return out;
  }

  private Map<LocalDate, Double> loadTruthMap(Path path) throws Exception {
    try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
         CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).setTrim(true).build().parse(reader)) {
      List<String> headers = parser.getHeaderNames();
      String dateColumn = headers.contains("date") ? "date" : (headers.contains("target_date_local") ? "target_date_local" : null);
      String valueColumn = headers.contains("settled_tmax") ? "settled_tmax" : (headers.contains("y_tmax") ? "y_tmax" : null);
      if (dateColumn == null || valueColumn == null) {
        throw new IllegalStateException("Truth CSV missing required columns in " + path);
      }
      Map<LocalDate, Double> out = new LinkedHashMap<>();
      for (CSVRecord record : parser) {
        String rawDate = record.get(dateColumn);
        String rawValue = record.get(valueColumn);
        if (rawDate == null || rawDate.isBlank() || rawValue == null || rawValue.isBlank()) {
          continue;
        }
        try {
          out.put(LocalDate.parse(rawDate.trim()), Double.parseDouble(rawValue.trim()));
        } catch (RuntimeException ignored) {
        }
      }
      return out;
    }
  }

  private record ForecastDayExport(
      LocalDate day,
      String status,
      Path reportPath,
      String reportSha256,
      Path runtimeGateFailurePath,
      String runtimeGateFailureJson,
      String errorText,
      JsonNode report) {
  }

  private ForecastDayExport ensureForecastReport(LocalDate day, List<ResolvedStation> stations) throws Exception {
    String ymd = day.format(DateTimeFormatter.BASIC_ISO_DATE);
    Path outDir = Path.of(properties.getLiveInferenceRoot()).resolve("target_" + ymd);
    Path reportPath = outDir.resolve("inference_report.json");
    Path failurePath = outDir.resolve("runtime_gate_failure.json");
    Set<String> stationIds = new LinkedHashSet<>();
    for (ResolvedStation station : stations) {
      stationIds.add(station.stationId());
    }

    boolean needsRun = !Files.exists(reportPath);
    JsonNode report = null;
    if (!needsRun) {
      report = readJson(reportPath);
      needsRun = !missingStations(report, stationIds).isEmpty();
    }

    if (needsRun) {
      Files.createDirectories(outDir);
      List<String> command = List.of(
          properties.getLiveScriptPython(),
          properties.getLiveScriptPath(),
          "--target-date",
          day.format(DATE_FORMAT),
          "--out-dir",
          outDir.toString(),
          "--log-level",
          properties.getLiveScriptLogLevel(),
          "--stdout-json",
          "summary"
      );
      ProcessBuilder pb = new ProcessBuilder(command);
      pb.directory(Path.of(".").toAbsolutePath().normalize().toFile());
      Process process = pb.start();
      String stdout = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
      String stderr = new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
      int code = process.waitFor();
      if (code != 0) {
        String failureJson = Files.exists(failurePath) ? Files.readString(failurePath, StandardCharsets.UTF_8) : null;
        return new ForecastDayExport(
            day,
            "live_inference_failed",
            reportPath,
            sha256IfExists(reportPath),
            Files.exists(failurePath) ? failurePath : null,
            failureJson,
            "code=" + code + " stderr_tail=" + tail(stderr, 2000) + " stdout_tail=" + tail(stdout, 2000),
            null
        );
      }
    }

    if (!Files.exists(reportPath)) {
      return new ForecastDayExport(day, "live_inference_failed", reportPath, null, null, null,
          "report_missing_after_run", null);
    }
    report = readJson(reportPath);
    Set<String> missing = missingStations(report, stationIds);
    if (!missing.isEmpty()) {
      return new ForecastDayExport(
          day,
          "missing_station_block",
          reportPath,
          sha256IfExists(reportPath),
          Files.exists(failurePath) ? failurePath : null,
          Files.exists(failurePath) ? Files.readString(failurePath, StandardCharsets.UTF_8) : null,
          "missing_stations=" + String.join(",", missing),
          report
      );
    }
    return new ForecastDayExport(
        day,
        "ok",
        reportPath,
        sha256IfExists(reportPath),
        Files.exists(failurePath) ? failurePath : null,
        Files.exists(failurePath) ? Files.readString(failurePath, StandardCharsets.UTF_8) : null,
        null,
        report
    );
  }

  private void bindForecastDay(java.sql.PreparedStatement stmt, String runId, ForecastDayExport export, int stationCount)
      throws Exception {
    JsonNode report = export.report();
    JsonNode counters = report == null ? null : report.path("leakage_proof").path("global_guardrail_counters");
    stmt.setString(1, runId);
    stmt.setString(2, export.day().format(DATE_FORMAT));
    stmt.setString(3, export.day().format(DateTimeFormatter.BASIC_ISO_DATE));
    stmt.setString(4, export.status());
    stmt.setString(5, export.reportPath().toString());
    stmt.setString(6, export.reportSha256());
    stmt.setString(7, export.runtimeGateFailurePath() == null ? null : export.runtimeGateFailurePath().toString());
    stmt.setString(8, export.runtimeGateFailureJson());
    stmt.setString(9, export.errorText());
    stmt.setString(10, report == null ? null : textOrNull(report.path("run_id")));
    stmt.setString(11, report == null ? null : textOrNull(report.path("quote_asof_utc")));
    stmt.setInt(12, stationCount);
    stmt.setInt(13, report == null ? 0 : report.path("inference_by_station").size());
    stmt.setString(14, jsonString(counters == null || counters.isMissingNode() ? Map.of() : counters));
  }

  private void bindForecastStationRow(java.sql.PreparedStatement stmt,
                                      String runId,
                                      LocalDate day,
                                      ResolvedStation station,
                                      ForecastDayExport export,
                                      Map<LocalDate, Double> truthMap) throws Exception {
    JsonNode report = export.report();
    JsonNode block = extractStationBlock(report, station.stationId());
    JsonNode quantiles = block == null ? null : block.path("quantiles");
    JsonNode evidence = report == null ? null : report.path("leakage_proof").path("per_station_evidence").path(station.stationId());
    Double truth = truthMap.get(day);
    String status = block != null && !quantiles.isMissingNode() && quantiles.isObject() && quantiles.size() > 0
        ? "ok"
        : export.status();

    stmt.setString(1, runId);
    stmt.setString(2, day.format(DATE_FORMAT));
    stmt.setString(3, station.stationId());
    stmt.setString(4, status);
    stmt.setString(5, export.reportPath().toString());
    stmt.setString(6, export.reportSha256());
    stmt.setInt(7, truth == null ? 0 : 1);
    if (truth == null) {
      stmt.setNull(8, java.sql.Types.REAL);
    } else {
      stmt.setDouble(8, truth);
    }
    stmt.setString(9, block == null ? null : textOrNull(block.path("runtime_utc")));
    stmt.setString(10, evidence == null ? null : textOrNull(evidence.path("runtime_expected_from_policy_utc")));
    bindBoolean(stmt, 11, evidence == null ? null : booleanOrNull(evidence.path("runtime_equals_expected_policy_runtime")));
    bindBoolean(stmt, 12, evidence == null ? null : booleanOrNull(evidence.path("runtime_lte_quote_asof")));
    bindBoolean(stmt, 13, evidence == null ? null : booleanOrNull(evidence.path("inference_quantiles_monotonic")));
    bindDouble(stmt, 14, block == null ? null : doubleOrNull(block.path("prediction_point_tmax_f")));
    bindDouble(stmt, 15, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.05")));
    bindDouble(stmt, 16, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.10")));
    bindDouble(stmt, 17, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.25")));
    bindDouble(stmt, 18, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.50")));
    bindDouble(stmt, 19, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.75")));
    bindDouble(stmt, 20, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.90")));
    bindDouble(stmt, 21, quantiles == null ? null : doubleOrNull(quantiles.path("q_0.95")));
    stmt.setString(22, evidence == null ? null : textOrNull(evidence.path("bundle_dir")));
    stmt.setString(23, jsonString(block == null ? Map.of() : block));
    stmt.setString(24, jsonString(evidence == null || evidence.isMissingNode() ? Map.of() : evidence));
  }

  private Set<String> missingStations(JsonNode report, Set<String> stationIds) {
    Set<String> missing = new LinkedHashSet<>();
    for (String stationId : stationIds) {
      JsonNode block = extractStationBlock(report, stationId);
      JsonNode quantiles = block == null ? null : block.path("quantiles");
      if (block == null || quantiles == null || quantiles.isMissingNode() || !quantiles.isObject() || quantiles.size() == 0) {
        missing.add(stationId);
      }
    }
    return missing;
  }

  private JsonNode extractStationBlock(JsonNode report, String stationId) {
    if (report == null || report.isMissingNode()) {
      return null;
    }
    JsonNode grouped = report.path("inference_by_station").path(stationId);
    if (!grouped.isMissingNode() && !grouped.isNull()) {
      return grouped;
    }
    JsonNode legacy = report.path("inference_" + stationId.toLowerCase());
    return legacy.isMissingNode() || legacy.isNull() ? null : legacy;
  }

  private String sha256IfExists(Path path) throws Exception {
    if (!Files.exists(path)) {
      return null;
    }
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    digest.update(Files.readAllBytes(path));
    byte[] bytes = digest.digest();
    StringBuilder sb = new StringBuilder();
    for (byte b : bytes) {
      sb.append(String.format("%02x", b));
    }
    return sb.toString();
  }

  private JsonNode readJson(Path path) throws Exception {
    return objectMapper.readTree(Files.readString(path, StandardCharsets.UTF_8));
  }

  private String jsonString(Object value) throws Exception {
    if (value instanceof JsonNode node) {
      return objectMapper.writeValueAsString(node);
    }
    return objectMapper.writeValueAsString(value);
  }

  private String textOrNull(JsonNode node) {
    return node == null || node.isMissingNode() || node.isNull() ? null : node.asText();
  }

  private Double doubleOrNull(JsonNode node) {
    return node == null || node.isMissingNode() || node.isNull() ? null : node.asDouble();
  }

  private Boolean booleanOrNull(JsonNode node) {
    return node == null || node.isMissingNode() || node.isNull() ? null : node.asBoolean();
  }

  private void bindDouble(java.sql.PreparedStatement stmt, int index, Double value) throws Exception {
    if (value == null || !Double.isFinite(value)) {
      stmt.setNull(index, java.sql.Types.REAL);
    } else {
      stmt.setDouble(index, value);
    }
  }

  private void bindBoolean(java.sql.PreparedStatement stmt, int index, Boolean value) throws Exception {
    if (value == null) {
      stmt.setNull(index, java.sql.Types.INTEGER);
    } else {
      stmt.setInt(index, value ? 1 : 0);
    }
  }

  private java.sql.Connection openConnection(Path sqlitePath) throws Exception {
    return java.sql.DriverManager.getConnection("jdbc:sqlite:" + sqlitePath);
  }

  private String tail(String value, int maxChars) {
    if (value == null || value.length() <= maxChars) {
      return value;
    }
    return value.substring(value.length() - maxChars);
  }

  private Map<MarketKey, MarketDay> preloadMarketStore(List<ResolvedStation> stations, ForecastStore forecastStore) throws Exception {
    Map<String, Map<LocalDate, Path>> indexByStation = new LinkedHashMap<>();
    for (ResolvedStation station : stations) {
      indexByStation.put(station.stationId(), buildMarketIndex(station.kalshiRoot(), station.filePrefix()));
    }

    List<Callable<MarketDay>> tasks = new ArrayList<>();
    Set<MarketKey> seen = new HashSet<>();
    for (ResolvedStation station : stations) {
      Map<LocalDate, StationPrediction> preds = forecastStore.predictionsByStation().getOrDefault(station.stationId(), Map.of());
      for (LocalDate day : preds.keySet()) {
        if (!seen.add(new MarketKey(station.stationId(), day))) {
          continue;
        }
        Path path = indexByStation.getOrDefault(station.stationId(), Map.of()).get(day);
        if (path == null || !Files.exists(path)) {
          continue;
        }
        tasks.add(() -> parseMarketFile(station.stationId(), day, path));
      }
    }

    Map<MarketKey, MarketDay> out = new LinkedHashMap<>();
    ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, Math.min(properties.getThreadCount(), 16)));
    try {
      CompletionService<MarketDay> completionService = new ExecutorCompletionService<>(pool);
      for (Callable<MarketDay> task : tasks) {
        completionService.submit(task);
      }
      for (int i = 0; i < tasks.size(); i++) {
        MarketDay marketDay = completionService.take().get();
        if (marketDay != null) {
          out.put(new MarketKey(marketDay.stationId(), marketDay.targetDate()), marketDay);
        }
      }
    } finally {
      pool.shutdownNow();
    }
    log.info("Preloaded {} market day files", out.size());
    return out;
  }

  private Map<LocalDate, Path> buildMarketIndex(Path root, String filePrefix) throws Exception {
    Map<LocalDate, Path> out = new LinkedHashMap<>();
    try (var stream = Files.walk(root)) {
      stream.filter(Files::isRegularFile)
          .filter(path -> path.getFileName().toString().startsWith(filePrefix + "_"))
          .filter(path -> path.getFileName().toString().endsWith(".csv"))
          .forEach(path -> {
            LocalDate date = marketFileDateFromPath(path);
            if (date != null) {
              out.put(date, path);
            }
          });
    }
    return out;
  }

  private MarketDay parseMarketFile(String stationId, LocalDate targetDate, Path path) throws Exception {
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
         CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).setTrim(true).build().parse(reader)) {
      List<String> headerNames = parser.getHeaderNames();
      if (!headerNames.contains("timestamp")) {
        return null;
      }

      List<MarketColumn> columns = new ArrayList<>();
      for (String header : headerNames) {
        if ("timestamp".equals(header)) {
          continue;
        }
        columns.add(new MarketColumn(header, parseBucketLabel(header)));
      }

      NavigableMap<Instant, MarketRow> rows = new TreeMap<>();
      for (CSVRecord record : parser) {
        Instant timestamp = parseTimestamp(record.get("timestamp"));
        if (timestamp == null) {
          continue;
        }
        double[] values = new double[columns.size()];
        for (int i = 0; i < columns.size(); i++) {
          String raw = record.get(columns.get(i).rawLabel());
          values[i] = parseNullableDouble(raw);
        }
        rows.put(timestamp, new MarketRow(timestamp, values));
      }
      if (rows.isEmpty()) {
        return null;
      }
      return new MarketDay(stationId, targetDate, path, rows.firstKey(), columns, rows);
    }
  }

  private LocalDate marketFileDateFromPath(Path path) {
    Matcher matcher = FILE_DATE_PATTERN.matcher(path.getFileName().toString());
    if (!matcher.find()) {
      return null;
    }
    String ymd = matcher.group(1);
    return LocalDate.parse(ymd, DateTimeFormatter.BASIC_ISO_DATE);
  }

  private Instant parseTimestamp(String raw) {
    if (raw == null || raw.isBlank()) {
      return null;
    }
    List<String> candidates = List.of(raw.trim(), raw.trim().replace(" ", "T"));
    for (String candidate : candidates) {
      try {
        return Instant.parse(candidate);
      } catch (DateTimeParseException ignored) {
      }
      try {
        return OffsetDateTime.parse(candidate).toInstant();
      } catch (DateTimeParseException ignored) {
      }
      try {
        return LocalDateTime.parse(candidate, DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss")).toInstant(ZoneOffset.UTC);
      } catch (DateTimeParseException ignored) {
      }
      try {
        return LocalDateTime.parse(candidate, DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS")).toInstant(ZoneOffset.UTC);
      } catch (DateTimeParseException ignored) {
      }
    }
    return null;
  }

  private double parseNullableDouble(String raw) {
    if (raw == null || raw.isBlank()) {
      return Double.NaN;
    }
    try {
      return Double.parseDouble(raw.trim());
    } catch (NumberFormatException ex) {
      return Double.NaN;
    }
  }

  private double[] pmfFromQuantiles(double q05, double q10, double q25, double q50, double q75, double q90, double q95) {
    double[] taus = {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95};
    double[] qvals = {q05, q10, q25, q50, q75, q90, q95};
    for (int i = 1; i < qvals.length; i++) {
      qvals[i] = Math.max(qvals[i], qvals[i - 1]);
    }
    double[] out = new double[SUPPORT_HI - SUPPORT_LO + 1];
    double total = 0.0;
    for (int t = SUPPORT_LO; t <= SUPPORT_HI; t++) {
      double p = cdfFromQuantiles(taus, qvals, t + 0.5) - cdfFromQuantiles(taus, qvals, t - 0.5);
      double clipped = Math.max(0.0, p);
      out[t - SUPPORT_LO] = clipped;
      total += clipped;
    }
    if (total <= 0.0) {
      Arrays.fill(out, 1.0 / out.length);
      return out;
    }
    for (int i = 0; i < out.length; i++) {
      out[i] = out[i] / total;
    }
    return out;
  }

  private double cdfFromQuantiles(double[] taus, double[] qvals, double x) {
    if (x <= qvals[0]) {
      return 0.0;
    }
    if (x >= qvals[qvals.length - 1]) {
      return 1.0;
    }
    for (int i = 1; i < qvals.length; i++) {
      if (x <= qvals[i]) {
        double qLo = qvals[i - 1];
        double qHi = qvals[i];
        double tLo = taus[i - 1];
        double tHi = taus[i];
        if (qHi <= qLo) {
          return tHi;
        }
        double frac = (x - qLo) / (qHi - qLo);
        return tLo + frac * (tHi - tLo);
      }
    }
    return 1.0;
  }

  private Bucket parseBucketLabel(String rawLabel) {
    String normalized = rawLabel == null ? "" : rawLabel.trim().toLowerCase().replace(" to ", "-").replaceAll("\\s+", " ");
    Matcher lt = LT_PATTERN.matcher(normalized);
    if (lt.find()) {
      return new Bucket(rawLabel, null, Integer.parseInt(lt.group(1)) - 1, "or_below");
    }
    Matcher le = LE_PATTERN.matcher(normalized);
    if (le.find()) {
      return new Bucket(rawLabel, null, Integer.parseInt(le.group(1)), "or_below");
    }
    Matcher gt = GT_PATTERN.matcher(normalized);
    if (gt.find()) {
      return new Bucket(rawLabel, Integer.parseInt(gt.group(1)) + 1, null, "or_above");
    }
    Matcher ge = GE_PATTERN.matcher(normalized);
    if (ge.find()) {
      return new Bucket(rawLabel, Integer.parseInt(ge.group(1)), null, "or_above");
    }
    List<Integer> nums = new ArrayList<>();
    Matcher matcher = NUMBER_PATTERN.matcher(normalized);
    while (matcher.find()) {
      nums.add(Integer.parseInt(matcher.group(1)));
    }
    if ((normalized.contains("or below") || normalized.contains("or less")) && !nums.isEmpty()) {
      return new Bucket(rawLabel, null, nums.get(0), "or_below");
    }
    if ((normalized.contains("or above") || normalized.contains("or higher")) && !nums.isEmpty()) {
      return new Bucket(rawLabel, nums.get(0), null, "or_above");
    }
    if (nums.size() >= 2) {
      int lo = Math.min(nums.get(0), nums.get(1));
      int hi = Math.max(nums.get(0), nums.get(1));
      return new Bucket(rawLabel, lo, hi, "range");
    }
    return null;
  }

  private double normalizePrice(double value) {
    if (!Double.isFinite(value) || value < 0.0) {
      return Double.NaN;
    }
    double normalized = value > 1.0 ? value / 100.0 : value;
    return Math.max(0.0, Math.min(1.0, normalized));
  }

  private double bucketProbability(double[] pmf, Bucket bucket) {
    double sum = 0.0;
    for (int temp = SUPPORT_LO; temp <= SUPPORT_HI; temp++) {
      boolean include = switch (bucket.mode()) {
        case "or_below" -> bucket.hi() != null && temp <= bucket.hi();
        case "or_above" -> bucket.lo() != null && temp >= bucket.lo();
        case "range" -> bucket.lo() != null && bucket.hi() != null && temp >= bucket.lo() && temp <= bucket.hi();
        default -> false;
      };
      if (include) {
        sum += pmf[temp - SUPPORT_LO];
      }
    }
    return sum;
  }

  private void runBacktestGrid(String runId,
                               Path sqlitePath,
                               Path runDir,
                               List<ResolvedStation> stations,
                               ForecastStore forecastStore,
                               Map<MarketKey, MarketDay> marketStore) throws Exception {
    createResultTables(sqlitePath);

    List<Double> evValues = buildGrid(properties.getEvStart(), properties.getEvEnd(), properties.getEvStep());
    List<Double> winValues = buildGrid(properties.getWinStart(), properties.getWinEnd(), properties.getWinStep());
    List<Double> fixedRiskValues = buildGrid(properties.getFixedRiskStart(), properties.getFixedRiskEnd(), properties.getFixedRiskStep());
    List<Double> kellyValues = buildGrid(properties.getKellyStart(), properties.getKellyEnd(), properties.getKellyStep());

    String entryRule = "entry_timestamp_utc >= max(T-1 " +
        String.format("%02d:%02dZ", properties.getEntryHourZ(), properties.getEntryMinuteZ()) +
        ", market_open_utc + " + properties.getMinEntryMinutesAfterOpen() + "m); first eligible global timestamp; " +
        "tie-break: model_win_prob desc, ev desc, market_price asc, station_id asc.";
    String sidePriceRule = "YES side uses normalized bucket price; NO side uses 1-YES.";
    String fixedStakeRule = "stake=min(balance_before*risk_fraction, stake_cap_usd)";
    String kellyStakeRule = "full_kelly=clamp((q-p)/(1-p),0,1); risk_fraction_used=kelly_fraction*full_kelly; stake=min(balance_before*risk_fraction_used, stake_cap_usd)";

    ExecutorService pool = Executors.newFixedThreadPool(properties.getThreadCount());
    CompletionService<BaseStreamResult> completionService = new ExecutorCompletionService<>(pool);
    int taskCount = 0;
    for (double evMin : evValues) {
      for (double winMin : winValues) {
        double taskEv = evMin;
        double taskWin = winMin;
        completionService.submit(() -> runBaseStream(
            runId, runDir, stations, forecastStore, marketStore, fixedRiskValues, kellyValues,
            entryRule, sidePriceRule, fixedStakeRule, kellyStakeRule, taskEv, taskWin));
        taskCount++;
      }
    }

    List<BaseRow> baseRows = new ArrayList<>();
    List<ComboRow> comboRows = new ArrayList<>();
    try (var conn = openConnection(sqlitePath)) {
      conn.setAutoCommit(false);
      try (var baseStmt = conn.prepareStatement(
          "INSERT INTO backtest_base_streams (" +
              "run_id,ev_min,win_min,trades,wins,losses,win_rate,final_balance,max_drawdown,days_without_trade_candidate," +
              "station_counts_json,side_counts_json,trades_csv_path,summary_json_path,sanity_json_path,day_debug_json_path," +
              "sanity_passes_all_checks,sanity_checked_trades) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
           var comboStmt = conn.prepareStatement(
               "INSERT INTO backtest_combo_results (" +
                   "run_id,sizing_mode,ev_min,win_min,risk_fraction,kelly_fraction,stake_cap_usd,entry_rule,side_price_rule,stake_rule," +
                   "summary_json_path,sanity_json_path,trades,wins,losses,win_rate,profit_factor,final_balance,total_pnl,max_drawdown," +
                   "avg_ev_at_trade,median_ev_at_trade,station_counts_json,side_counts_json,risk_fraction_used_avg,risk_fraction_used_min," +
                   "risk_fraction_used_max,stake_cap_breach_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")) {
        for (int done = 1; done <= taskCount; done++) {
          Future<BaseStreamResult> future = completionService.take();
          BaseStreamResult result = future.get();
          insertBaseRow(baseStmt, result.baseRow());
          baseStmt.executeUpdate();
          for (ComboRow comboRow : result.comboRows()) {
            insertComboRow(comboStmt, comboRow);
            comboStmt.executeUpdate();
          }
          conn.commit();
          baseRows.add(result.baseRow());
          comboRows.addAll(result.comboRows());
          log.info("Completed base stream {}/{} ev_min={} win_min={} trades={}",
              done, taskCount, result.baseRow().evMin(), result.baseRow().winMin(), result.baseRow().trades());
        }
      }
    } finally {
      pool.shutdownNow();
    }

    writeRankingsAndRunMeta(sqlitePath, runId, runDir, entryRule, sidePriceRule, fixedStakeRule, kellyStakeRule,
        evValues, winValues, fixedRiskValues, kellyValues, forecastStore, baseRows, comboRows);
  }

  private BaseStreamResult runBaseStream(String runId,
                                         Path runDir,
                                         List<ResolvedStation> stations,
                                         ForecastStore forecastStore,
                                         Map<MarketKey, MarketDay> marketStore,
                                         List<Double> fixedRiskValues,
                                         List<Double> kellyValues,
                                         String entryRule,
                                         String sidePriceRule,
                                         String fixedStakeRule,
                                         String kellyStakeRule,
                                         double evMin,
                                         double winMin) throws Exception {
    List<Candidate> chosen = new ArrayList<>();
    Map<String, Object> dayDebug = new LinkedHashMap<>();
    Map<String, Object> counts = new LinkedHashMap<>();
    Map<String, Integer> daysWithPredictionByStation = new LinkedHashMap<>();
    Map<String, Integer> daysWithMarketFileByStation = new LinkedHashMap<>();
    for (ResolvedStation station : stations) {
      daysWithPredictionByStation.put(station.stationId(), 0);
      daysWithMarketFileByStation.put(station.stationId(), 0);
    }
    int totalDays = 0;
    int daysWithAnyPrediction = 0;
    int daysWithAnyMarketFile = 0;
    int daysWithAnyStationContext = 0;
    int daysWithoutTradeCandidate = 0;

    for (LocalDate day : enumerateDays()) {
      totalDays++;
      boolean hasAnyPrediction = false;
      boolean hasAnyMarket = false;
      for (ResolvedStation station : stations) {
        boolean hasPrediction = forecastStore.predictionsByStation()
            .getOrDefault(station.stationId(), Map.of())
            .containsKey(day);
        boolean hasMarket = marketStore.containsKey(new MarketKey(station.stationId(), day));
        daysWithPredictionByStation.compute(station.stationId(), (k, v) -> v + (hasPrediction ? 1 : 0));
        daysWithMarketFileByStation.compute(station.stationId(), (k, v) -> v + (hasMarket ? 1 : 0));
        hasAnyPrediction = hasAnyPrediction || hasPrediction;
        hasAnyMarket = hasAnyMarket || hasMarket;
      }
      if (hasAnyPrediction) {
        daysWithAnyPrediction++;
      }
      if (hasAnyMarket) {
        daysWithAnyMarketFile++;
      }

      SelectionResult selection = selectTradeForDay(day, stations, forecastStore, marketStore, evMin, winMin, properties.getMinMarketPrice());
      dayDebug.put(day.format(DATE_FORMAT), selection.status());
      if (Boolean.TRUE.equals(selection.status().get("has_any_station_context"))) {
        daysWithAnyStationContext++;
      }
      if (selection.candidate() == null) {
        daysWithoutTradeCandidate++;
      } else {
        chosen.add(selection.candidate());
      }
    }

    List<TradeRow> trades = applyFixedRiskBaseBankroll(chosen, properties.getSelectionRiskFraction(), properties.getStakeCapUsd(), properties.getStartBalance());
    Map<String, Object> summary = buildBaseSummary(totalDays, daysWithPredictionByStation, daysWithAnyPrediction,
        daysWithMarketFileByStation, daysWithAnyMarketFile, daysWithAnyStationContext, daysWithoutTradeCandidate,
        evMin, winMin, trades);
    Map<String, Object> sanity = runSanityAudit(trades, stations, forecastStore, marketStore, evMin, winMin, properties.getMinMarketPrice());

    String evTag = tag(evMin);
    String winTag = tag(winMin);
    Path tradesPath = runDir.resolve("trades_base_ev" + evTag + "_win" + winTag + ".csv");
    Path summaryPath = runDir.resolve("summary_base_ev" + evTag + "_win" + winTag + ".json");
    Path sanityPath = runDir.resolve("sanity_base_ev" + evTag + "_win" + winTag + ".json");
    Path debugPath = runDir.resolve("day_debug_base_ev" + evTag + "_win" + winTag + ".json");
    writeTradesCsv(tradesPath, trades);
    writeJson(summaryPath, summary);
    writeJson(sanityPath, sanity);
    writeJson(debugPath, dayDebug);

    BaseRow baseRow = new BaseRow(
        runId,
        evMin,
        winMin,
        trades.size(),
        intValue(summary.get("wins")),
        intValue(summary.get("losses")),
        doubleValue(summary.get("win_rate")),
        doubleValue(summary.get("final_balance")),
        doubleValue(summary.get("max_drawdown")),
        intValue(summary.get("days_without_trade_candidate")),
        jsonString(summary.get("station_counts")),
        jsonString(summary.get("side_counts")),
        tradesPath.toString(),
        summaryPath.toString(),
        sanityPath.toString(),
        debugPath.toString(),
        Boolean.TRUE.equals(sanity.get("passes_all_checks")),
        intValue(sanity.get("checked_trades"))
    );

    List<ComboRow> comboRows = new ArrayList<>();
    for (double riskFraction : fixedRiskValues) {
      ComboMetrics metrics = simulateBankroll(trades, "fixed_risk", riskFraction, null);
      comboRows.add(metrics.toComboRow(runId, evMin, winMin, properties.getStakeCapUsd(), entryRule, sidePriceRule, fixedStakeRule,
          summaryPath.toString(), sanityPath.toString()));
    }
    for (double kellyFraction : kellyValues) {
      ComboMetrics metrics = simulateBankroll(trades, "fractional_kelly", null, kellyFraction);
      comboRows.add(metrics.toComboRow(runId, evMin, winMin, properties.getStakeCapUsd(), entryRule, sidePriceRule, kellyStakeRule,
          summaryPath.toString(), sanityPath.toString()));
    }

    return new BaseStreamResult(baseRow, comboRows);
  }

  private void createResultTables(Path sqlitePath) throws Exception {
    try (var conn = openConnection(sqlitePath); var stmt = conn.createStatement()) {
      stmt.executeUpdate(
          "CREATE TABLE IF NOT EXISTS backtest_base_streams (" +
              "run_id TEXT NOT NULL," +
              "ev_min REAL NOT NULL," +
              "win_min REAL NOT NULL," +
              "trades INTEGER NOT NULL," +
              "wins INTEGER NOT NULL," +
              "losses INTEGER NOT NULL," +
              "win_rate REAL NOT NULL," +
              "final_balance REAL NOT NULL," +
              "max_drawdown REAL NOT NULL," +
              "days_without_trade_candidate INTEGER NOT NULL," +
              "station_counts_json TEXT NOT NULL," +
              "side_counts_json TEXT NOT NULL," +
              "trades_csv_path TEXT NOT NULL," +
              "summary_json_path TEXT NOT NULL," +
              "sanity_json_path TEXT NOT NULL," +
              "day_debug_json_path TEXT NOT NULL," +
              "sanity_passes_all_checks INTEGER NOT NULL," +
              "sanity_checked_trades INTEGER NOT NULL" +
              ")"
      );
      stmt.executeUpdate(
          "CREATE TABLE IF NOT EXISTS backtest_combo_results (" +
              "run_id TEXT NOT NULL," +
              "sizing_mode TEXT NOT NULL," +
              "ev_min REAL NOT NULL," +
              "win_min REAL NOT NULL," +
              "risk_fraction REAL," +
              "kelly_fraction REAL," +
              "stake_cap_usd REAL NOT NULL," +
              "entry_rule TEXT NOT NULL," +
              "side_price_rule TEXT NOT NULL," +
              "stake_rule TEXT NOT NULL," +
              "summary_json_path TEXT NOT NULL," +
              "sanity_json_path TEXT NOT NULL," +
              "trades INTEGER NOT NULL," +
              "wins INTEGER NOT NULL," +
              "losses INTEGER NOT NULL," +
              "win_rate REAL NOT NULL," +
              "profit_factor REAL," +
              "final_balance REAL NOT NULL," +
              "total_pnl REAL NOT NULL," +
              "max_drawdown REAL NOT NULL," +
              "avg_ev_at_trade REAL NOT NULL," +
              "median_ev_at_trade REAL NOT NULL," +
              "station_counts_json TEXT NOT NULL," +
              "side_counts_json TEXT NOT NULL," +
              "risk_fraction_used_avg REAL NOT NULL," +
              "risk_fraction_used_min REAL NOT NULL," +
              "risk_fraction_used_max REAL NOT NULL," +
              "stake_cap_breach_count INTEGER NOT NULL" +
              ")"
      );
      stmt.executeUpdate(
          "CREATE TABLE IF NOT EXISTS backtest_ranked_scores (" +
              "rank_position INTEGER NOT NULL," +
              "composite_score_pf_win_lowdd REAL NOT NULL," +
              "run_id TEXT NOT NULL," +
              "sizing_mode TEXT NOT NULL," +
              "ev_min REAL NOT NULL," +
              "win_min REAL NOT NULL," +
              "risk_fraction REAL," +
              "kelly_fraction REAL," +
              "stake_cap_usd REAL NOT NULL," +
              "entry_rule TEXT NOT NULL," +
              "side_price_rule TEXT NOT NULL," +
              "stake_rule TEXT NOT NULL," +
              "summary_json_path TEXT NOT NULL," +
              "sanity_json_path TEXT NOT NULL," +
              "trades INTEGER NOT NULL," +
              "wins INTEGER NOT NULL," +
              "losses INTEGER NOT NULL," +
              "win_rate REAL NOT NULL," +
              "profit_factor REAL," +
              "final_balance REAL NOT NULL," +
              "total_pnl REAL NOT NULL," +
              "max_drawdown REAL NOT NULL," +
              "avg_ev_at_trade REAL NOT NULL," +
              "median_ev_at_trade REAL NOT NULL," +
              "station_counts_json TEXT NOT NULL," +
              "side_counts_json TEXT NOT NULL," +
              "risk_fraction_used_avg REAL NOT NULL," +
              "risk_fraction_used_min REAL NOT NULL," +
              "risk_fraction_used_max REAL NOT NULL," +
              "stake_cap_breach_count INTEGER NOT NULL" +
              ")"
      );
      stmt.executeUpdate(
          "CREATE TABLE IF NOT EXISTS backtest_run_meta (" +
              "run_id TEXT NOT NULL," +
              "created_at_utc TEXT NOT NULL," +
              "start_date TEXT NOT NULL," +
              "end_date TEXT NOT NULL," +
              "stations_json TEXT NOT NULL," +
              "forecast_live_root TEXT NOT NULL," +
              "thread_count INTEGER NOT NULL," +
              "entry_rule TEXT NOT NULL," +
              "side_price_rule TEXT NOT NULL," +
              "stake_rule_fixed TEXT NOT NULL," +
              "stake_rule_fractional_kelly TEXT NOT NULL," +
              "requested_day_count INTEGER NOT NULL," +
              "forecast_success_day_count INTEGER NOT NULL," +
              "forecast_failed_day_count INTEGER NOT NULL," +
              "base_stream_count INTEGER NOT NULL," +
              "combo_count INTEGER NOT NULL," +
              "out_dir TEXT NOT NULL," +
              "sqlite_path TEXT NOT NULL" +
              ")"
      );
    }
  }

  private void insertBaseRow(java.sql.PreparedStatement stmt, BaseRow row) throws Exception {
    stmt.setString(1, row.runId());
    stmt.setDouble(2, row.evMin());
    stmt.setDouble(3, row.winMin());
    stmt.setInt(4, row.trades());
    stmt.setInt(5, row.wins());
    stmt.setInt(6, row.losses());
    stmt.setDouble(7, row.winRate());
    stmt.setDouble(8, row.finalBalance());
    stmt.setDouble(9, row.maxDrawdown());
    stmt.setInt(10, row.daysWithoutTradeCandidate());
    stmt.setString(11, row.stationCountsJson());
    stmt.setString(12, row.sideCountsJson());
    stmt.setString(13, row.tradesCsvPath());
    stmt.setString(14, row.summaryJsonPath());
    stmt.setString(15, row.sanityJsonPath());
    stmt.setString(16, row.dayDebugJsonPath());
    stmt.setInt(17, row.sanityPassesAllChecks() ? 1 : 0);
    stmt.setInt(18, row.sanityCheckedTrades());
  }

  private void insertComboRow(java.sql.PreparedStatement stmt, ComboRow row) throws Exception {
    insertComboRow(stmt, row, 0);
  }

  private void insertComboRow(java.sql.PreparedStatement stmt, ComboRow row, int indexOffset) throws Exception {
    stmt.setString(1 + indexOffset, row.runId());
    stmt.setString(2 + indexOffset, row.sizingMode());
    stmt.setDouble(3 + indexOffset, row.evMin());
    stmt.setDouble(4 + indexOffset, row.winMin());
    bindDouble(stmt, 5 + indexOffset, row.riskFraction());
    bindDouble(stmt, 6 + indexOffset, row.kellyFraction());
    stmt.setDouble(7 + indexOffset, row.stakeCapUsd());
    stmt.setString(8 + indexOffset, row.entryRule());
    stmt.setString(9 + indexOffset, row.sidePriceRule());
    stmt.setString(10 + indexOffset, row.stakeRule());
    stmt.setString(11 + indexOffset, row.summaryJsonPath());
    stmt.setString(12 + indexOffset, row.sanityJsonPath());
    stmt.setInt(13 + indexOffset, row.trades());
    stmt.setInt(14 + indexOffset, row.wins());
    stmt.setInt(15 + indexOffset, row.losses());
    stmt.setDouble(16 + indexOffset, row.winRate());
    bindDouble(stmt, 17 + indexOffset, row.profitFactor());
    stmt.setDouble(18 + indexOffset, row.finalBalance());
    stmt.setDouble(19 + indexOffset, row.totalPnl());
    stmt.setDouble(20 + indexOffset, row.maxDrawdown());
    stmt.setDouble(21 + indexOffset, row.avgEvAtTrade());
    stmt.setDouble(22 + indexOffset, row.medianEvAtTrade());
    stmt.setString(23 + indexOffset, row.stationCountsJson());
    stmt.setString(24 + indexOffset, row.sideCountsJson());
    stmt.setDouble(25 + indexOffset, row.riskFractionUsedAvg());
    stmt.setDouble(26 + indexOffset, row.riskFractionUsedMin());
    stmt.setDouble(27 + indexOffset, row.riskFractionUsedMax());
    stmt.setInt(28 + indexOffset, row.stakeCapBreachCount());
  }

  private record SelectionResult(Candidate candidate, Map<String, Object> status) {
  }

  private SelectionResult selectTradeForDay(LocalDate day,
                                            List<ResolvedStation> stations,
                                            ForecastStore forecastStore,
                                            Map<MarketKey, MarketDay> marketStore,
                                            double evMin,
                                            double winMin,
                                            double minMarketPrice) {
    Instant gate = computeEntryCutoff(day);
    Map<String, Object> status = new LinkedHashMap<>();
    status.put("day", day.format(DATE_FORMAT));
    status.put("gate_cutoff_utc", toUtcString(gate));
    Map<String, Object> stationStatus = new LinkedHashMap<>();
    status.put("station_status", stationStatus);

    List<DayContext> contexts = new ArrayList<>();
    for (ResolvedStation station : stations) {
      Map<String, Object> st = new LinkedHashMap<>();
      st.put("has_prediction", false);
      st.put("has_market_file", false);
      st.put("has_rows_after_gate", false);
      st.put("has_rows_after_effective_cutoff", false);
      st.put("market_file", null);
      st.put("market_open_utc", null);
      st.put("effective_cutoff_utc", null);

      StationPrediction prediction = forecastStore.predictionsByStation()
          .getOrDefault(station.stationId(), Map.of())
          .get(day);
      if (prediction == null) {
        stationStatus.put(station.stationId(), st);
        continue;
      }
      st.put("has_prediction", true);

      MarketDay marketDay = marketStore.get(new MarketKey(station.stationId(), day));
      if (marketDay == null) {
        stationStatus.put(station.stationId(), st);
        continue;
      }
      st.put("has_market_file", true);
      st.put("market_file", marketDay.marketFile().toString());
      st.put("market_open_utc", toUtcString(marketDay.marketOpenUtc()));

      NavigableMap<Instant, MarketRow> rowsAfterGate = marketDay.rowsByTimestamp().tailMap(gate, true);
      if (rowsAfterGate.isEmpty()) {
        stationStatus.put(station.stationId(), st);
        continue;
      }
      st.put("has_rows_after_gate", true);

      Instant openDelayCutoff = marketDay.marketOpenUtc().plusSeconds(properties.getMinEntryMinutesAfterOpen() * 60L);
      Instant effectiveCutoff = gate.isAfter(openDelayCutoff) ? gate : openDelayCutoff;
      st.put("effective_cutoff_utc", toUtcString(effectiveCutoff));
      NavigableMap<Instant, MarketRow> rowsAfterEffectiveCutoff = rowsAfterGate.tailMap(effectiveCutoff, true);
      if (rowsAfterEffectiveCutoff.isEmpty()) {
        stationStatus.put(station.stationId(), st);
        continue;
      }
      st.put("has_rows_after_effective_cutoff", true);
      stationStatus.put(station.stationId(), st);
      contexts.add(new DayContext(station.stationId(), prediction, marketDay, gate, effectiveCutoff, rowsAfterEffectiveCutoff));
    }

    status.put("has_any_station_context", !contexts.isEmpty());
    if (contexts.isEmpty()) {
      status.put("first_eligible_timestamp_utc", null);
      status.put("chosen_trade_key", null);
      return new SelectionResult(null, status);
    }

    Set<Instant> timestamps = new LinkedHashSet<>();
    for (DayContext ctx : contexts) {
      timestamps.addAll(ctx.eligibleRows().keySet());
    }
    List<Instant> ordered = new ArrayList<>(timestamps);
    ordered.sort(Comparator.naturalOrder());
    for (Instant timestamp : ordered) {
      List<Candidate> candidates = candidatesAtTimestamp(timestamp, contexts, evMin, winMin, minMarketPrice);
      if (candidates.isEmpty()) {
        continue;
      }
      candidates.sort(CANDIDATE_ORDER);
      Candidate chosen = candidates.get(0);
      status.put("first_eligible_timestamp_utc", toUtcString(timestamp));
      status.put("chosen_trade_key", Map.of(
          "station_id", chosen.stationId(),
          "bucket_raw", chosen.bucketRaw(),
          "side", chosen.side(),
          "entry_timestamp_utc", toUtcString(chosen.entryTimestampUtc())
      ));
      return new SelectionResult(chosen, status);
    }

    status.put("first_eligible_timestamp_utc", null);
    status.put("chosen_trade_key", null);
    return new SelectionResult(null, status);
  }

  private List<Candidate> candidatesAtTimestamp(Instant timestamp,
                                                List<DayContext> contexts,
                                                double evMin,
                                                double winMin,
                                                double minMarketPrice) {
    List<Candidate> out = new ArrayList<>();
    for (DayContext ctx : contexts) {
      MarketRow row = ctx.eligibleRows().get(timestamp);
      if (row == null) {
        continue;
      }
      List<MarketColumn> columns = ctx.marketDay().columns();
      for (int i = 0; i < columns.size(); i++) {
        MarketColumn column = columns.get(i);
        if (column.bucket() == null) {
          continue;
        }
        double pYesMarket = normalizePrice(row.values()[i]);
        if (!Double.isFinite(pYesMarket)) {
          continue;
        }
        double pYesModel = bucketProbability(ctx.prediction().pmf(), column.bucket());
        double pNoModel = 1.0 - pYesModel;
        double pNoMarket = 1.0 - pYesMarket;

        double evYes = pYesModel - pYesMarket;
        if (pYesMarket >= minMarketPrice && pYesModel >= winMin && evYes >= evMin) {
          out.add(new Candidate(
              ctx.prediction().targetDate().format(DATE_FORMAT),
              marketFileDateFromPath(ctx.marketDay().marketFile()).format(DATE_FORMAT),
              ctx.stationId(),
              timestamp,
              ctx.marketDay().marketOpenUtc(),
              ctx.gateCutoffUtc(),
              ctx.effectiveCutoffUtc(),
              ctx.marketDay().marketFile(),
              column.rawLabel(),
              column.bucket().canonicalLabel(),
              "YES",
              pYesMarket,
              pYesModel,
              evYes,
              (int) Math.round(ctx.prediction().yTmax()),
              column.bucket().contains((int) Math.round(ctx.prediction().yTmax())) ? 1 : 0,
              timestamp,
              0
          ));
        }

        double evNo = pNoModel - pNoMarket;
        if (pNoMarket >= minMarketPrice && pNoModel >= winMin && evNo >= evMin) {
          out.add(new Candidate(
              ctx.prediction().targetDate().format(DATE_FORMAT),
              marketFileDateFromPath(ctx.marketDay().marketFile()).format(DATE_FORMAT),
              ctx.stationId(),
              timestamp,
              ctx.marketDay().marketOpenUtc(),
              ctx.gateCutoffUtc(),
              ctx.effectiveCutoffUtc(),
              ctx.marketDay().marketFile(),
              column.rawLabel(),
              column.bucket().canonicalLabel(),
              "NO",
              pNoMarket,
              pNoModel,
              evNo,
              (int) Math.round(ctx.prediction().yTmax()),
              column.bucket().contains((int) Math.round(ctx.prediction().yTmax())) ? 0 : 1,
              timestamp,
              0
          ));
        }
      }
    }
    return out;
  }

  private Instant computeEntryCutoff(LocalDate day) {
    return day.minusDays(1)
        .atTime(LocalTime.of(properties.getEntryHourZ(), properties.getEntryMinuteZ()))
        .toInstant(ZoneOffset.UTC);
  }

  private String toUtcString(Instant instant) {
    return instant.toString();
  }

  private List<TradeRow> applyFixedRiskBaseBankroll(List<Candidate> chosen,
                                                    double riskFraction,
                                                    double stakeCapUsd,
                                                    double startBalance) {
    List<Candidate> ordered = new ArrayList<>(chosen);
    ordered.sort(Comparator.comparing(Candidate::targetDateLocal).thenComparing(Candidate::entryTimestampUtc));
    List<TradeRow> out = new ArrayList<>();
    double balance = startBalance;
    double peak = startBalance;
    for (Candidate candidate : ordered) {
      double stake = Math.min(balance * riskFraction, stakeCapUsd);
      double shares = candidate.marketPrice() > 0.0 ? stake / candidate.marketPrice() : 0.0;
      double pnl = candidate.win() == 1 ? shares * (1.0 - candidate.marketPrice()) : -stake;
      double balanceBefore = balance;
      balance += pnl;
      peak = Math.max(peak, balance);
      double drawdown = peak <= 0.0 ? 0.0 : (peak - balance) / peak;
      out.add(new TradeRow(candidate, stake, shares, pnl, balanceBefore, balance, drawdown, candidate.win() == 1 ? "W" : "L"));
    }
    return out;
  }

  private Map<String, Object> buildBaseSummary(int totalDays,
                                               Map<String, Integer> daysWithPredictionByStation,
                                               int daysWithAnyPrediction,
                                               Map<String, Integer> daysWithMarketFileByStation,
                                               int daysWithAnyMarketFile,
                                               int daysWithAnyStationContext,
                                               int daysWithoutTradeCandidate,
                                               double evMin,
                                               double winMin,
                                               List<TradeRow> trades) {
    Map<String, Integer> stationCounts = new LinkedHashMap<>();
    Map<String, Integer> sideCounts = new LinkedHashMap<>();
    int wins = 0;
    int losses = 0;
    double grossProfit = 0.0;
    double grossLoss = 0.0;
    double avgEv = 0.0;
    double medianEv = 0.0;
    double maxDrawdown = 0.0;
    double finalBalance = properties.getStartBalance();
    if (!trades.isEmpty()) {
      List<Double> evs = new ArrayList<>();
      for (TradeRow trade : trades) {
        stationCounts.merge(trade.candidate().stationId(), 1, Integer::sum);
        sideCounts.merge(trade.candidate().side(), 1, Integer::sum);
        if (trade.candidate().win() == 1) {
          wins++;
          grossProfit += trade.pnl();
        } else {
          losses++;
          grossLoss += -trade.pnl();
        }
        evs.add(trade.candidate().ev());
        maxDrawdown = Math.max(maxDrawdown, trade.drawdown());
        finalBalance = trade.balanceAfter();
      }
      avgEv = evs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      evs.sort(Comparator.naturalOrder());
      medianEv = evs.get(evs.size() / 2);
      if (evs.size() % 2 == 0) {
        medianEv = (evs.get(evs.size() / 2 - 1) + evs.get(evs.size() / 2)) / 2.0;
      }
    }

    Map<String, Object> summary = new LinkedHashMap<>();
    summary.put("period_start", properties.getStartDate());
    summary.put("period_end", properties.getEndDate());
    summary.put("entry_hour_z", properties.getEntryHourZ());
    summary.put("entry_minute_z", properties.getEntryMinuteZ());
    summary.put("min_entry_minutes_after_open", properties.getMinEntryMinutesAfterOpen());
    summary.put("ev_min", evMin);
    summary.put("win_min", winMin);
    summary.put("min_market_price", properties.getMinMarketPrice());
    summary.put("risk_fraction", properties.getSelectionRiskFraction());
    summary.put("stake_cap_usd", properties.getStakeCapUsd());
    summary.put("total_days", totalDays);
    summary.put("days_with_prediction_by_station", daysWithPredictionByStation);
    summary.put("days_with_any_prediction", daysWithAnyPrediction);
    summary.put("days_with_market_file_by_station", daysWithMarketFileByStation);
    summary.put("days_with_any_market_file", daysWithAnyMarketFile);
    summary.put("days_with_any_station_context", daysWithAnyStationContext);
    summary.put("days_without_trade_candidate", daysWithoutTradeCandidate);
    summary.put("trades", trades.size());
    summary.put("wins", wins);
    summary.put("losses", losses);
    summary.put("win_rate", trades.isEmpty() ? 0.0 : (double) wins / trades.size());
    summary.put("profit_factor", grossLoss > 0.0 ? grossProfit / grossLoss : null);
    summary.put("start_balance", properties.getStartBalance());
    summary.put("final_balance", finalBalance);
    summary.put("total_pnl", finalBalance - properties.getStartBalance());
    summary.put("avg_ev_at_trade", avgEv);
    summary.put("median_ev_at_trade", medianEv);
    summary.put("max_drawdown", maxDrawdown);
    summary.put("station_counts", stationCounts);
    summary.put("side_counts", sideCounts);
    return summary;
  }

  private Map<String, Object> runSanityAudit(List<TradeRow> trades,
                                             List<ResolvedStation> stations,
                                             ForecastStore forecastStore,
                                             Map<MarketKey, MarketDay> marketStore,
                                             double evMin,
                                             double winMin,
                                             double minMarketPrice) throws Exception {
    Map<String, Integer> failures = new LinkedHashMap<>();
    for (String key : List.of(
        "more_than_one_trade_per_day_global",
        "entry_before_gate",
        "entry_before_effective_cutoff",
        "entry_not_first_eligible_timestamp_globally",
        "tie_break_policy_violation",
        "market_file_missing",
        "market_file_date_mismatch_target_date",
        "bucket_not_found",
        "bucket_unparseable",
        "entry_bucket_price_missing_at_timestamp",
        "market_price_mismatch",
        "market_price_below_min_market_price",
        "model_prob_mismatch",
        "ev_mismatch",
        "win_label_mismatch",
        "stake_cap_breach",
        "pnl_mismatch")) {
      failures.put(key, 0);
    }

    Set<String> seenDays = new HashSet<>();
    int duplicates = 0;
    for (TradeRow trade : trades) {
      if (!seenDays.add(trade.candidate().targetDateLocal())) {
        duplicates++;
      }
    }
    failures.put("more_than_one_trade_per_day_global", duplicates);

    int checked = 0;
    for (TradeRow trade : trades) {
      checked++;
      LocalDate day = LocalDate.parse(trade.candidate().targetDateLocal());
      Instant gate = computeEntryCutoff(day);
      if (trade.candidate().entryTimestampUtc().isBefore(gate)) {
        failures.compute("entry_before_gate", (k, v) -> v + 1);
      }
      if (trade.candidate().entryTimestampUtc().isBefore(trade.candidate().effectiveCutoffUtc())) {
        failures.compute("entry_before_effective_cutoff", (k, v) -> v + 1);
      }

      SelectionResult chosenAgain = selectTradeForDay(day, stations, forecastStore, marketStore, evMin, winMin, minMarketPrice);
      String firstEligible = Objects.toString(chosenAgain.status().get("first_eligible_timestamp_utc"), null);
      if (!Objects.equals(firstEligible, toUtcString(trade.candidate().entryTimestampUtc()))) {
        failures.compute("entry_not_first_eligible_timestamp_globally", (k, v) -> v + 1);
      }
      Candidate expected = chosenAgain.candidate();
      if (expected == null
          || !Objects.equals(expected.stationId(), trade.candidate().stationId())
          || !Objects.equals(expected.bucketRaw(), trade.candidate().bucketRaw())
          || !Objects.equals(expected.side(), trade.candidate().side())
          || !Objects.equals(expected.entryTimestampUtc(), trade.candidate().entryTimestampUtc())) {
        failures.compute("tie_break_policy_violation", (k, v) -> v + 1);
      }

      MarketDay marketDay = marketStore.get(new MarketKey(trade.candidate().stationId(), day));
      if (marketDay == null) {
        failures.compute("market_file_missing", (k, v) -> v + 1);
        continue;
      }
      LocalDate marketFileDate = marketFileDateFromPath(marketDay.marketFile());
      if (!Objects.equals(marketFileDate, day)) {
        failures.compute("market_file_date_mismatch_target_date", (k, v) -> v + 1);
      }
      MarketRow row = marketDay.rowsByTimestamp().get(trade.candidate().entryTimestampUtc());
      if (row == null) {
        failures.compute("market_file_missing", (k, v) -> v + 1);
        continue;
      }
      int columnIndex = -1;
      Bucket bucket = null;
      for (int i = 0; i < marketDay.columns().size(); i++) {
        if (Objects.equals(marketDay.columns().get(i).rawLabel(), trade.candidate().bucketRaw())) {
          columnIndex = i;
          bucket = marketDay.columns().get(i).bucket();
          break;
        }
      }
      if (columnIndex < 0) {
        failures.compute("bucket_not_found", (k, v) -> v + 1);
        continue;
      }
      if (bucket == null) {
        failures.compute("bucket_unparseable", (k, v) -> v + 1);
        continue;
      }
      double pYesRaw = row.values()[columnIndex];
      if (!Double.isFinite(pYesRaw)) {
        failures.compute("entry_bucket_price_missing_at_timestamp", (k, v) -> v + 1);
        continue;
      }
      double pYesMarket = normalizePrice(pYesRaw);
      double expectedMarketPrice = "YES".equals(trade.candidate().side()) ? pYesMarket : (1.0 - pYesMarket);
      if (!eq(trade.candidate().marketPrice(), expectedMarketPrice, 1e-8)) {
        failures.compute("market_price_mismatch", (k, v) -> v + 1);
      }
      if (trade.candidate().marketPrice() + 1e-12 < minMarketPrice) {
        failures.compute("market_price_below_min_market_price", (k, v) -> v + 1);
      }

      StationPrediction prediction = forecastStore.predictionsByStation()
          .getOrDefault(trade.candidate().stationId(), Map.of())
          .get(day);
      if (prediction == null) {
        failures.compute("model_prob_mismatch", (k, v) -> v + 1);
        failures.compute("ev_mismatch", (k, v) -> v + 1);
        continue;
      }
      double pYesModel = bucketProbability(prediction.pmf(), bucket);
      double expectedModelProb = "YES".equals(trade.candidate().side()) ? pYesModel : (1.0 - pYesModel);
      if (!eq(trade.candidate().modelWinProb(), expectedModelProb, 1e-8)) {
        failures.compute("model_prob_mismatch", (k, v) -> v + 1);
      }
      if (!eq(trade.candidate().ev(), expectedModelProb - expectedMarketPrice, 1e-8)) {
        failures.compute("ev_mismatch", (k, v) -> v + 1);
      }
      int y = (int) Math.round(trade.candidate().yTmax());
      int expectedWin = ("YES".equals(trade.candidate().side()) && bucket.contains(y))
          || ("NO".equals(trade.candidate().side()) && !bucket.contains(y)) ? 1 : 0;
      if (trade.candidate().win() != expectedWin) {
        failures.compute("win_label_mismatch", (k, v) -> v + 1);
      }
      if (trade.stake() > properties.getStakeCapUsd() + 1e-9) {
        failures.compute("stake_cap_breach", (k, v) -> v + 1);
      }
      double expectedPnl = trade.candidate().win() == 1
          ? trade.shares() * (1.0 - trade.candidate().marketPrice())
          : -trade.stake();
      if (!eq(trade.pnl(), expectedPnl, 1e-6)) {
        failures.compute("pnl_mismatch", (k, v) -> v + 1);
      }
    }

    boolean passesAllChecks = failures.values().stream().allMatch(v -> v == 0);
    Map<String, Object> out = new LinkedHashMap<>();
    out.put("checked_trades", checked);
    out.put("passes_all_checks", passesAllChecks);
    out.put("failures", failures);
    return out;
  }

  private boolean eq(double a, double b, double tol) {
    return Double.isFinite(a) && Double.isFinite(b) && Math.abs(a - b) <= tol;
  }

  private record ComboMetrics(
      String sizingMode,
      Double riskFraction,
      Double kellyFraction,
      int trades,
      int wins,
      int losses,
      double winRate,
      Double profitFactor,
      double finalBalance,
      double totalPnl,
      double maxDrawdown,
      double avgEvAtTrade,
      double medianEvAtTrade,
      Map<String, Integer> stationCounts,
      Map<String, Integer> sideCounts,
      double riskFractionUsedAvg,
      double riskFractionUsedMin,
      double riskFractionUsedMax,
      int stakeCapBreachCount) {

    ComboRow toComboRow(String runId,
                        double evMin,
                        double winMin,
                        double stakeCapUsd,
                        String entryRule,
                        String sidePriceRule,
                        String stakeRule,
                        String summaryJsonPath,
                        String sanityJsonPath) throws Exception {
      return new ComboRow(
          runId, sizingMode, evMin, winMin, riskFraction, kellyFraction, stakeCapUsd,
          entryRule, sidePriceRule, stakeRule, summaryJsonPath, sanityJsonPath,
          trades, wins, losses, winRate, profitFactor, finalBalance, totalPnl, maxDrawdown,
          avgEvAtTrade, medianEvAtTrade,
          STATIC_MAPPER.writeValueAsString(stationCounts),
          STATIC_MAPPER.writeValueAsString(sideCounts),
          riskFractionUsedAvg, riskFractionUsedMin, riskFractionUsedMax, stakeCapBreachCount);
    }
  }

  private ComboMetrics simulateBankroll(List<TradeRow> baseTrades, String sizingMode, Double riskFraction, Double kellyFraction) {
    double balance = properties.getStartBalance();
    double peak = balance;
    double maxDrawdown = 0.0;
    int wins = 0;
    int losses = 0;
    double grossProfit = 0.0;
    double grossLoss = 0.0;
    List<Double> evs = new ArrayList<>();
    List<Double> usedFractions = new ArrayList<>();
    Map<String, Integer> stationCounts = new LinkedHashMap<>();
    Map<String, Integer> sideCounts = new LinkedHashMap<>();
    int stakeCapBreaches = 0;

    for (TradeRow trade : baseTrades) {
      double price = trade.candidate().marketPrice();
      double modelWinProb = trade.candidate().modelWinProb();
      double usedRisk;
      if ("fixed_risk".equals(sizingMode)) {
        usedRisk = riskFraction == null ? 0.0 : riskFraction;
      } else {
        double fullKelly = (price <= 0.0 || price >= 1.0) ? 0.0 : (modelWinProb - price) / (1.0 - price);
        fullKelly = Math.max(0.0, Math.min(1.0, fullKelly));
        usedRisk = (kellyFraction == null ? 0.0 : kellyFraction) * fullKelly;
      }
      double stake = Math.min(balance * usedRisk, properties.getStakeCapUsd());
      if (stake > properties.getStakeCapUsd() + 1e-9) {
        stakeCapBreaches++;
      }
      double shares = price > 0.0 ? stake / price : 0.0;
      double pnl = trade.candidate().win() == 1 ? shares * (1.0 - price) : -stake;
      balance += pnl;
      peak = Math.max(peak, balance);
      double drawdown = peak <= 0.0 ? 0.0 : (peak - balance) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
      if (trade.candidate().win() == 1) {
        wins++;
        grossProfit += pnl;
      } else {
        losses++;
        grossLoss += -pnl;
      }
      evs.add(trade.candidate().ev());
      usedFractions.add(usedRisk);
      stationCounts.merge(trade.candidate().stationId(), 1, Integer::sum);
      sideCounts.merge(trade.candidate().side(), 1, Integer::sum);
    }

    List<Double> sortedEvs = new ArrayList<>(evs);
    sortedEvs.sort(Comparator.naturalOrder());
    double medianEv = sortedEvs.isEmpty() ? 0.0 : sortedEvs.get(sortedEvs.size() / 2);
    if (!sortedEvs.isEmpty() && sortedEvs.size() % 2 == 0) {
      medianEv = (sortedEvs.get(sortedEvs.size() / 2 - 1) + sortedEvs.get(sortedEvs.size() / 2)) / 2.0;
    }
    return new ComboMetrics(
        sizingMode,
        riskFraction,
        kellyFraction,
        baseTrades.size(),
        wins,
        losses,
        baseTrades.isEmpty() ? 0.0 : (double) wins / baseTrades.size(),
        grossLoss > 0.0 ? grossProfit / grossLoss : null,
        balance,
        balance - properties.getStartBalance(),
        maxDrawdown,
        evs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0),
        medianEv,
        stationCounts,
        sideCounts,
        usedFractions.stream().mapToDouble(Double::doubleValue).average().orElse(0.0),
        usedFractions.stream().mapToDouble(Double::doubleValue).min().orElse(0.0),
        usedFractions.stream().mapToDouble(Double::doubleValue).max().orElse(0.0),
        stakeCapBreaches
    );
  }

  private void writeRankingsAndRunMeta(Path sqlitePath,
                                       String runId,
                                       Path runDir,
                                       String entryRule,
                                       String sidePriceRule,
                                       String fixedStakeRule,
                                       String kellyStakeRule,
                                       List<Double> evValues,
                                       List<Double> winValues,
                                       List<Double> fixedRiskValues,
                                       List<Double> kellyValues,
                                       ForecastStore forecastStore,
                                       List<BaseRow> baseRows,
                                       List<ComboRow> comboRows) throws Exception {
    List<Map<String, Object>> scoreRows = new ArrayList<>();
    for (ComboRow row : comboRows) {
      double pfComponent = row.profitFactor() == null ? 0.0 : Math.log1p(row.profitFactor());
      Map<String, Object> scoreRow = new LinkedHashMap<>();
      scoreRow.put("combo_row", row);
      scoreRow.put("pf_component_raw", pfComponent);
      scoreRow.put("win_rate", row.winRate());
      scoreRow.put("max_drawdown", row.maxDrawdown());
      scoreRows.add(scoreRow);
    }
    double maxPfLog = scoreRows.stream().mapToDouble(r -> ((Number) r.get("pf_component_raw")).doubleValue()).max().orElse(1.0);
    if (maxPfLog <= 0.0) {
      maxPfLog = 1.0;
    }
    for (Map<String, Object> row : scoreRows) {
      double pfComponent = ((Number) row.get("pf_component_raw")).doubleValue() / maxPfLog;
      double winComponent = ((Number) row.get("win_rate")).doubleValue();
      double drawdownComponent = Math.max(0.0, 1.0 - ((Number) row.get("max_drawdown")).doubleValue());
      row.put("composite_score_pf_win_lowdd", Math.cbrt(pfComponent * winComponent * drawdownComponent));
    }
    scoreRows.sort(
        Comparator
            .comparingDouble((Map<String, Object> row) -> -doubleValue(row.get("composite_score_pf_win_lowdd")))
            .thenComparingDouble(row -> -((ComboRow) row.get("combo_row")).finalBalance())
            .thenComparingDouble(row -> ((ComboRow) row.get("combo_row")).maxDrawdown())
            .thenComparing(row -> ((ComboRow) row.get("combo_row")).sizingMode())
    );
    List<RankedComboRow> rankedRows = new ArrayList<>();
    for (int i = 0; i < scoreRows.size(); i++) {
      rankedRows.add(new RankedComboRow(
          i + 1,
          doubleValue(scoreRows.get(i).get("composite_score_pf_win_lowdd")),
          (ComboRow) scoreRows.get(i).get("combo_row")));
    }

    try (var conn = openConnection(sqlitePath)) {
      conn.setAutoCommit(false);
      try (var deleteRank = conn.createStatement()) {
        deleteRank.executeUpdate("DELETE FROM backtest_ranked_scores");
        deleteRank.executeUpdate("DELETE FROM backtest_run_meta");
      }
      try (var rankStmt = conn.prepareStatement(
          "INSERT INTO backtest_ranked_scores (" +
              "rank_position,composite_score_pf_win_lowdd," +
              "run_id,sizing_mode,ev_min,win_min,risk_fraction,kelly_fraction,stake_cap_usd,entry_rule,side_price_rule,stake_rule," +
              "summary_json_path,sanity_json_path,trades,wins,losses,win_rate,profit_factor,final_balance,total_pnl,max_drawdown," +
              "avg_ev_at_trade,median_ev_at_trade,station_counts_json,side_counts_json,risk_fraction_used_avg,risk_fraction_used_min," +
              "risk_fraction_used_max,stake_cap_breach_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")) {
        for (RankedComboRow rankedRow : rankedRows) {
          rankStmt.setInt(1, rankedRow.rankPosition());
          rankStmt.setDouble(2, rankedRow.compositeScorePfWinLowdd());
          insertComboRow(rankStmt, rankedRow.comboRow(), 2);
          rankStmt.executeUpdate();
        }
      }
      try (var metaStmt = conn.prepareStatement(
          "INSERT INTO backtest_run_meta (" +
              "run_id,created_at_utc,start_date,end_date,stations_json,forecast_live_root,thread_count,entry_rule,side_price_rule," +
              "stake_rule_fixed,stake_rule_fractional_kelly,requested_day_count,forecast_success_day_count,forecast_failed_day_count," +
              "base_stream_count,combo_count,out_dir,sqlite_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")) {
        metaStmt.setString(1, runId);
        metaStmt.setString(2, OffsetDateTime.now(ZoneOffset.UTC).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME));
        metaStmt.setString(3, properties.getStartDate());
        metaStmt.setString(4, properties.getEndDate());
        metaStmt.setString(5, objectMapper.writeValueAsString(stationIds()));
        metaStmt.setString(6, properties.getLiveInferenceRoot());
        metaStmt.setInt(7, properties.getThreadCount());
        metaStmt.setString(8, entryRule);
        metaStmt.setString(9, sidePriceRule);
        metaStmt.setString(10, fixedStakeRule);
        metaStmt.setString(11, kellyStakeRule);
        metaStmt.setInt(12, enumerateDays().size());
        metaStmt.setInt(13, forecastStore.forecastSuccessDayCount());
        metaStmt.setInt(14, forecastStore.forecastFailedDayCount());
        metaStmt.setInt(15, baseRows.size());
        metaStmt.setInt(16, comboRows.size());
        metaStmt.setString(17, runDir.toString());
        metaStmt.setString(18, sqlitePath.toString());
        metaStmt.executeUpdate();
      }
      conn.commit();
    }

    Map<String, Object> runSummary = new LinkedHashMap<>();
    runSummary.put("run_id", runId);
    runSummary.put("sqlite_path", sqlitePath.toString());
    runSummary.put("out_dir", runDir.toString());
    runSummary.put("period_start", properties.getStartDate());
    runSummary.put("period_end", properties.getEndDate());
    runSummary.put("stations", stationIds());
    runSummary.put("forecast_success_day_count", forecastStore.forecastSuccessDayCount());
    runSummary.put("forecast_failed_day_count", forecastStore.forecastFailedDayCount());
    runSummary.put("base_stream_count", baseRows.size());
    runSummary.put("combo_count", comboRows.size());
    writeJson(runDir.resolve("run_summary.json"), runSummary);
    writeJson(runDir.resolve("run_sanity.json"), Map.of(
        "run_id", runId,
        "passes_all_checks", baseRows.stream().allMatch(BaseRow::sanityPassesAllChecks)
            && comboRows.stream().mapToInt(ComboRow::stakeCapBreachCount).sum() == 0,
        "forecast_success_day_count", forecastStore.forecastSuccessDayCount(),
        "forecast_failed_day_count", forecastStore.forecastFailedDayCount(),
        "base_stream_count", baseRows.size(),
        "combo_count", comboRows.size(),
        "sqlite_path", sqlitePath.toString()
    ));
  }

  private List<Double> buildGrid(double start, double end, double step) {
    List<Double> out = new ArrayList<>();
    BigDecimal cur = BigDecimal.valueOf(start);
    BigDecimal stop = BigDecimal.valueOf(end);
    BigDecimal inc = BigDecimal.valueOf(step);
    while (cur.compareTo(stop.add(BigDecimal.valueOf(1e-12))) <= 0) {
      out.add(cur.setScale(6, RoundingMode.HALF_UP).doubleValue());
      cur = cur.add(inc);
    }
    return out;
  }

  private String tag(double value) {
    String text = BigDecimal.valueOf(value).stripTrailingZeros().toPlainString();
    return text.replace(".", "p");
  }

  private void writeJson(Path path, Object payload) throws Exception {
    Files.createDirectories(path.getParent());
    try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
      objectMapper.writerWithDefaultPrettyPrinter().writeValue(writer, payload);
    }
  }

  private void writeTradesCsv(Path path, List<TradeRow> trades) throws Exception {
    Files.createDirectories(path.getParent());
    try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
         CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader(
             "target_date_local", "market_file_date_local", "station_id", "entry_timestamp_utc", "market_open_utc",
             "gate_cutoff_utc", "effective_cutoff_utc", "market_file", "bucket", "bucket_raw", "side",
             "market_price", "model_win_prob", "ev", "y_tmax", "win", "stake", "shares", "pnl",
             "balance_before", "balance_after", "drawdown", "result"))) {
      for (TradeRow trade : trades) {
        printer.printRecord(
            trade.candidate().targetDateLocal(),
            trade.candidate().marketFileDateLocal(),
            trade.candidate().stationId(),
            toUtcString(trade.candidate().entryTimestampUtc()),
            toUtcString(trade.candidate().marketOpenUtc()),
            toUtcString(trade.candidate().gateCutoffUtc()),
            toUtcString(trade.candidate().effectiveCutoffUtc()),
            trade.candidate().marketFile().toString(),
            trade.candidate().bucket(),
            trade.candidate().bucketRaw(),
            trade.candidate().side(),
            trade.candidate().marketPrice(),
            trade.candidate().modelWinProb(),
            trade.candidate().ev(),
            trade.candidate().yTmax(),
            trade.candidate().win(),
            trade.stake(),
            trade.shares(),
            trade.pnl(),
            trade.balanceBefore(),
            trade.balanceAfter(),
            trade.drawdown(),
            trade.result()
        );
      }
    }
  }

  private List<String> stationIds() {
    List<String> ids = new ArrayList<>();
    for (StationSpec station : properties.getStations()) {
      ids.add(station.getStationId().trim().toUpperCase());
    }
    return ids;
  }

  private int intValue(Object value) {
    if (value instanceof Number number) {
      return number.intValue();
    }
    return Integer.parseInt(String.valueOf(value));
  }

  private double doubleValue(Object value) {
    if (value instanceof Number number) {
      return number.doubleValue();
    }
    return Double.parseDouble(String.valueOf(value));
  }

}
