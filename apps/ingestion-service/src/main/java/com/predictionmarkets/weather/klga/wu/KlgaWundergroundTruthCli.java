package com.predictionmarkets.weather.klga.wu;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class KlgaWundergroundTruthCli {
  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper()
      .findAndRegisterModules()
      .enable(SerializationFeature.INDENT_OUTPUT);

  private KlgaWundergroundTruthCli() {
  }

  public static void main(String[] args) throws Exception {
    WuTruthConfig config = WuTruthConfig.fromArgs(args);
    KlgaWundergroundTruthCli cli = new KlgaWundergroundTruthCli();
    ObjectNode result = switch (config.command()) {
      case "rebuild" -> cli.rebuild(config);
      case "audit-day" -> cli.auditDay(config);
      case "validate-sample" -> cli.validateSample(config);
      default -> throw new IllegalArgumentException("Unknown --command: " + config.command());
    };
    System.out.println(OBJECT_MAPPER.writeValueAsString(result));
  }

  private ObjectNode rebuild(WuTruthConfig config) throws Exception {
    config.requireApiKey();
    List<WuTruthStation> stations = WuTruthStation.select(config.stationSelection());
    List<FetchTask> fullTasks = buildTasks(stations, config.startDate(), config.endDate(), config.chunkDays());
    WuTruthFetchClient client = new WuTruthFetchClient(config);
    WuTruthParser parser = new WuTruthParser(OBJECT_MAPPER);
    ResumePlan resumePlan;

    ObjectNode summary = OBJECT_MAPPER.createObjectNode();
    summary.put("ok", true);
    summary.put("command", "rebuild");
    summary.put("start_date", config.startDate().toString());
    summary.put("end_date", config.endDate().toString());
    summary.put("stations", stations.size());
    summary.put("workers", config.workers());
    summary.put("chunk_days", config.chunkDays());
    summary.put("rate_limit_per_minute", config.rateLimitPerMinute());
    summary.put("resume", config.resume());
    summary.put("tasks_total_without_resume", fullTasks.size());

    int rowsUpserted = 0;
    int windowsSucceeded = 0;
    int windowsNoData = 0;
    int windowsFailed = 0;
    int noDataRows = 0;
    int acceptedRows = 0;
    int suspectRows = 0;

    ExecutorService executor = Executors.newFixedThreadPool(config.workers());
    List<Future<TaskResult>> futures = new ArrayList<>();
    try (WuTruthRepository repository = new WuTruthRepository(config, OBJECT_MAPPER)) {
      repository.ensureTableExists();
      resumePlan = config.resume()
          ? buildResumePlan(stations, config.startDate(), config.endDate(), config.chunkDays(), repository)
          : new ResumePlan(fullTasks, 0, stationDays(stations, config.startDate(), config.endDate()));
      List<FetchTask> tasks = resumePlan.tasks();
      summary.put("tasks_planned", tasks.size());
      summary.put("station_days_already_complete", resumePlan.completedStationDays());
      summary.put("station_days_to_fetch", resumePlan.stationDaysToFetch());
      for (FetchTask task : tasks) {
        futures.add(executor.submit(new FetchCallable(task, client, parser)));
      }
      for (Future<TaskResult> future : futures) {
        TaskResult result = future.get();
        rowsUpserted += repository.upsertRows(result.rows());
        boolean hasFetchFailedRow = false;
        boolean hasNonNoDataRow = false;
        if (result.fetchSucceeded()) {
          windowsSucceeded++;
        }
        for (WuTruthDailyRow row : result.rows()) {
          if ("no_data".equals(row.validationStatus())) {
            noDataRows++;
          } else if ("accepted".equals(row.validationStatus()) || "manual_confirmed".equals(row.validationStatus())) {
            acceptedRows++;
            hasNonNoDataRow = true;
          } else if ("suspect".equals(row.validationStatus())) {
            suspectRows++;
            hasNonNoDataRow = true;
          } else if ("fetch_failed".equals(row.validationStatus())) {
            hasFetchFailedRow = true;
            hasNonNoDataRow = true;
          }
        }
        if (!result.fetchSucceeded()) {
          if (!hasFetchFailedRow && !hasNonNoDataRow) {
            windowsNoData++;
          } else {
            windowsFailed++;
          }
        }
      }
      summary.set("coverage_summary", repository.coverageSummary());
    } finally {
      executor.shutdownNow();
    }

    summary.put("rows_upserted", rowsUpserted);
    summary.put("windows_succeeded", windowsSucceeded);
    summary.put("windows_no_data", windowsNoData);
    summary.put("windows_failed", windowsFailed);
    summary.put("accepted_rows_in_run", acceptedRows);
    summary.put("suspect_rows_in_run", suspectRows);
    summary.put("no_data_rows_in_run", noDataRows);
    summary.put("ok", windowsFailed == 0);
    return summary;
  }

  private ObjectNode auditDay(WuTruthConfig config) throws Exception {
    config.requireApiKey();
    WuTruthStation station = WuTruthStation.byStationId(config.stationId());
    WuTruthFetchClient client = new WuTruthFetchClient(config);
    WuTruthParser parser = new WuTruthParser(OBJECT_MAPPER);
    WuTruthFetchResult fetched = client.fetch(station, config.auditDate(), config.auditDate());
    List<WuTruthDailyRow> rows = parser.parse(fetched);
    try (WuTruthRepository repository = new WuTruthRepository(config, OBJECT_MAPPER)) {
      repository.ensureTableExists();
      repository.upsertRows(rows);
      ObjectNode row = repository.loadDay(station.stationId(), config.auditDate());
      ObjectNode root = OBJECT_MAPPER.createObjectNode();
      root.put("ok", fetched.success() && row.path("tmax_f").asInt(-999) != -999);
      root.put("command", "audit-day");
      root.set("row", row);
      root.put("weathercom_fetch_success", fetched.success());
      root.put("http_status", fetched.httpStatus());
      root.put("wu_page_url", station.pageUrl(config.auditDate()));
      root.put("known_klga_2026_05_21_canary_passed",
          "KLGA".equals(station.stationId())
              && LocalDate.of(2026, 5, 21).equals(config.auditDate())
              && row.path("tmax_f").asInt(-999) == 66
              && row.path("tmin_f").asInt(-999) == 56
              && "hourly_temp_max".equals(row.path("daily_high_source").asText()));
      return root;
    }
  }

  private ObjectNode validateSample(WuTruthConfig config) throws Exception {
    WuTruthFetchClient client = new WuTruthFetchClient(config);
    ArrayNode validations = OBJECT_MAPPER.createArrayNode();
    int passed = 0;
    int failed = 0;
    try (WuTruthRepository repository = new WuTruthRepository(config, OBJECT_MAPPER)) {
      repository.ensureTableExists();
      List<ObjectNode> rows = repository.loadValidationSample(config.sampleSize(), config.seed());
      for (ObjectNode row : rows) {
        ObjectNode validation = validateStoredRow(row, client);
        if (validation.path("pass").asBoolean(false)) {
          passed++;
        } else {
          failed++;
        }
        validations.add(validation);
      }
    }

    Path reportPath = writeValidationReport(config, validations, passed, failed);
    ObjectNode root = OBJECT_MAPPER.createObjectNode();
    root.put("ok", failed == 0);
    root.put("command", "validate-sample");
    root.put("sample_size_requested", config.sampleSize());
    root.put("sample_rows_checked", passed + failed);
    root.put("passed", passed);
    root.put("failed", failed);
    root.put("report_path", reportPath.toAbsolutePath().toString());
    return root;
  }

  private ObjectNode validateStoredRow(ObjectNode row, WuTruthFetchClient client) {
    ObjectNode validation = OBJECT_MAPPER.createObjectNode();
    validation.put("station_id", row.path("station_id").asText());
    validation.put("local_date", row.path("local_date").asText());
    validation.put("wu_page_url", row.path("wu_page_url").asText());
    validation.put("db_tmax_f", row.path("tmax_f").isNull() ? null : row.path("tmax_f").asInt());
    validation.put("db_tmin_f", row.path("tmin_f").isNull() ? null : row.path("tmin_f").asInt());
    validation.put("daily_high_source", row.path("daily_high_source").asText());

    ArrayNode hourly = (ArrayNode) row.path("hourly_observations_json");
    Integer maxHourly = null;
    Integer minHourly = null;
    boolean sane = true;
    for (int i = 0; i < hourly.size(); i++) {
      var tempNode = hourly.get(i).path("temp_f");
      if (tempNode.isMissingNode() || tempNode.isNull()) {
        continue;
      }
      int temp = (int) Math.round(tempNode.asDouble());
      if (temp < -40 || temp > 130) {
        sane = false;
      }
      maxHourly = maxHourly == null ? temp : Math.max(maxHourly, temp);
      minHourly = minHourly == null ? temp : Math.min(minHourly, temp);
    }
    boolean maxMatches = row.path("tmax_f").isNull() ? maxHourly == null : row.path("tmax_f").asInt() == maxHourly;
    boolean minMatches = row.path("tmin_f").isNull() ? minHourly == null : row.path("tmin_f").asInt() == minHourly;
    boolean sourceMatches = "hourly_temp_max".equals(row.path("daily_high_source").asText());
    boolean canaryPass = true;
    if ("KLGA".equals(row.path("station_id").asText()) && "2026-05-21".equals(row.path("local_date").asText())) {
      canaryPass = row.path("tmax_f").asInt(-999) == 66 && row.path("tmin_f").asInt(-999) == 56;
    }
    validation.put("max_hourly_temp_f", maxHourly);
    validation.put("min_hourly_temp_f", minHourly);
    validation.put("max_matches", maxMatches);
    validation.put("min_matches", minMatches);
    validation.put("temps_sane", sane);
    validation.put("source_matches_hourly_temp_max", sourceMatches);
    validation.put("known_klga_2026_05_21_canary_passed", canaryPass);
    validation.put("wu_static_page_fetch_evidence", client.fetchPage(row.path("wu_page_url").asText()));
    validation.put("pass", maxMatches && minMatches && sane && sourceMatches && canaryPass);
    return validation;
  }

  private Path writeValidationReport(WuTruthConfig config, ArrayNode validations, int passed, int failed) throws Exception {
    Path directory = config.artifactRoot().resolve("wunderground").resolve("validation");
    Files.createDirectories(directory);
    String timestamp = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'")
        .withZone(ZoneOffset.UTC)
        .format(Instant.now());
    Path reportPath = directory.resolve("wu_validation_" + timestamp + ".json");
    ObjectNode report = OBJECT_MAPPER.createObjectNode();
    report.put("generated_at_utc", Instant.now().toString());
    report.put("sample_size_requested", config.sampleSize());
    report.put("seed", config.seed());
    report.put("passed", passed);
    report.put("failed", failed);
    report.set("validations", validations);
    OBJECT_MAPPER.writeValue(reportPath.toFile(), report);
    return reportPath;
  }

  private List<FetchTask> buildTasks(
      List<WuTruthStation> stations,
      LocalDate startDate,
      LocalDate endDate,
      int chunkDays) {
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("end date must be >= start date");
    }
    List<FetchTask> tasks = new ArrayList<>();
    for (WuTruthStation station : stations) {
      LocalDate cursor = startDate;
      while (!cursor.isAfter(endDate)) {
        LocalDate chunkEnd = cursor.plusDays(chunkDays - 1L);
        if (chunkEnd.isAfter(endDate)) {
          chunkEnd = endDate;
        }
        tasks.add(new FetchTask(station, cursor, chunkEnd));
        cursor = chunkEnd.plusDays(1);
      }
    }
    return tasks;
  }

  private ResumePlan buildResumePlan(
      List<WuTruthStation> stations,
      LocalDate startDate,
      LocalDate endDate,
      int chunkDays,
      WuTruthRepository repository) throws Exception {
    List<FetchTask> tasks = new ArrayList<>();
    int completedStationDays = 0;
    int stationDaysToFetch = 0;
    for (WuTruthStation station : stations) {
      var completedDates = repository.loadCompletedDates(station.stationId(), startDate, endDate);
      completedStationDays += completedDates.size();
      LocalDate cursor = startDate;
      while (!cursor.isAfter(endDate)) {
        if (completedDates.contains(cursor)) {
          cursor = cursor.plusDays(1);
          continue;
        }
        LocalDate taskStart = cursor;
        int daysInTask = 0;
        while (!cursor.isAfter(endDate) && !completedDates.contains(cursor) && daysInTask < chunkDays) {
          cursor = cursor.plusDays(1);
          daysInTask++;
        }
        tasks.add(new FetchTask(station, taskStart, cursor.minusDays(1)));
        stationDaysToFetch += daysInTask;
      }
    }
    return new ResumePlan(tasks, completedStationDays, stationDaysToFetch);
  }

  private int stationDays(List<WuTruthStation> stations, LocalDate startDate, LocalDate endDate) {
    return Math.toIntExact(((endDate.toEpochDay() - startDate.toEpochDay()) + 1L) * stations.size());
  }

  private record FetchCallable(
      FetchTask task,
      WuTruthFetchClient client,
      WuTruthParser parser
  ) implements Callable<TaskResult> {
    @Override
    public TaskResult call() {
      WuTruthFetchResult fetched = client.fetch(task.station(), task.startDate(), task.endDate());
      return new TaskResult(fetched.success(), parser.parse(fetched));
    }
  }

  private record FetchTask(WuTruthStation station, LocalDate startDate, LocalDate endDate) {
  }

  private record TaskResult(boolean fetchSucceeded, List<WuTruthDailyRow> rows) {
  }

  private record ResumePlan(List<FetchTask> tasks, int completedStationDays, int stationDaysToFetch) {
  }
}
