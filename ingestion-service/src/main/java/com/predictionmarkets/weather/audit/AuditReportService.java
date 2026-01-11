package com.predictionmarkets.weather.audit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.audit.AuditReport.AlignmentResult;
import com.predictionmarkets.weather.audit.AuditReport.AlignmentViolation;
import com.predictionmarkets.weather.audit.AuditReport.CliRevisionResult;
import com.predictionmarkets.weather.audit.AuditReport.FeatureCoverage;
import com.predictionmarkets.weather.audit.AuditReport.FeatureCoverageResult;
import com.predictionmarkets.weather.audit.AuditReport.LeakageSample;
import com.predictionmarkets.weather.audit.AuditReport.MissingReasonCount;
import com.predictionmarkets.weather.audit.AuditReport.MosAvailability;
import com.predictionmarkets.weather.audit.AuditReport.MosAvailabilityResult;
import com.predictionmarkets.weather.audit.AuditReport.NoLeakageResult;
import com.predictionmarkets.weather.audit.AuditReport.StationCoverage;
import com.predictionmarkets.weather.audit.AuditReport.StationCoverageResult;
import com.predictionmarkets.weather.audit.AuditReport.Status;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.repository.AsofPolicyRepository;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Date;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class AuditReportService {
  private static final String SCHEMA_VERSION = "1";
  private static final String MISSING_REASON_FALLBACK = "UNSPECIFIED";
  private static final String CLI_REVISION_NOTE =
      "CLI revision history is not stored; revision counts are unavailable.";

  private static final String LEAKAGE_COUNT_SQL = """
      SELECT COUNT(*) AS violation_count
        FROM mos_asof_feature
       WHERE station_id IN (:stationIds)
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
         AND chosen_runtime_utc IS NOT NULL
         AND chosen_runtime_utc > asof_utc
      """;

  private static final String LEAKAGE_SAMPLE_SQL = """
      SELECT station_id,
             target_date_local,
             model,
             asof_utc,
             chosen_runtime_utc
        FROM mos_asof_feature
       WHERE station_id IN (:stationIds)
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
         AND chosen_runtime_utc IS NOT NULL
         AND chosen_runtime_utc > asof_utc
       ORDER BY chosen_runtime_utc DESC
       LIMIT :sampleLimit
      """;

  private static final String CLI_DATES_SQL = """
      SELECT target_date_local
        FROM cli_daily
       WHERE station_id = :stationId
         AND target_date_local BETWEEN :startDate AND :endDate
      """;

  private static final String MOS_RUNTIME_DATES_SQL = """
      SELECT DISTINCT CAST(runtime_utc AS DATE) AS runtime_date
        FROM mos_run
       WHERE station_id = :stationId
         AND model = :model
         AND runtime_utc >= :startUtc
         AND runtime_utc < :endUtc
      """;

  private static final String MOS_RUNTIME_STATS_SQL = """
      SELECT MIN(runtime_utc) AS min_runtime,
             MAX(runtime_utc) AS max_runtime,
             COUNT(*) AS run_count
        FROM mos_run
       WHERE station_id = :stationId
         AND model = :model
         AND runtime_utc >= :startUtc
         AND runtime_utc < :endUtc
      """;

  private static final String FEATURE_STATS_SQL = """
      SELECT station_id,
             model,
             COUNT(*) AS total_count,
             SUM(CASE WHEN tmax_f IS NULL THEN 1 ELSE 0 END) AS missing_count
        FROM mos_asof_feature
       WHERE station_id IN (:stationIds)
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
       GROUP BY station_id, model
      """;

  private static final String FEATURE_REASON_SQL = """
      SELECT station_id,
             model,
             COALESCE(missing_reason, :fallbackReason) AS reason,
             COUNT(*) AS missing_count
        FROM mos_asof_feature
       WHERE station_id IN (:stationIds)
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
         AND tmax_f IS NULL
       GROUP BY station_id, model, COALESCE(missing_reason, :fallbackReason)
       ORDER BY station_id, model, missing_count DESC
      """;

  private static final String ALIGNMENT_SQL = """
      SELECT station_id,
             model,
             target_date_local,
             chosen_runtime_utc,
             station_zoneid,
             tmax_f
        FROM mos_asof_feature
       WHERE station_id IN (:stationIds)
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;
  private final ObjectMapper objectMapper;
  private final AsofPolicyRepository asofPolicyRepository;

  public AuditReportService(NamedParameterJdbcTemplate jdbcTemplate,
                            ObjectMapper objectMapper,
                            AsofPolicyRepository asofPolicyRepository) {
    this.jdbcTemplate = jdbcTemplate;
    this.objectMapper = objectMapper;
    this.asofPolicyRepository = asofPolicyRepository;
  }

  public AuditReportArtifacts generateAndWriteReport(AuditRequest request, Path outputDir) {
    AuditReport report = generateReport(request);
    Path resolvedOutput = resolveOutputDir(outputDir);
    String baseName = buildBaseName(report.stationIds(), report.startDateLocal(),
        report.endDateLocal(), report.asofPolicyId());
    Path markdownPath = resolvedOutput.resolve(baseName + ".md");
    Path jsonPath = resolvedOutput.resolve(baseName + ".json");

    writeMarkdown(report, markdownPath);
    writeJson(report, jsonPath);
    return new AuditReportArtifacts(report, markdownPath, jsonPath);
  }

  public AuditReport generateReport(AuditRequest request) {
    AuditRequest normalized = normalizeRequest(request);
    List<String> stationIds = normalized.stationIds();
    LocalDate startDate = normalized.startDateLocal();
    LocalDate endDate = normalized.endDateLocal();
    Long asofPolicyId = normalized.asofPolicyId();
    List<MosModel> models = normalized.models();
    int sampleLimit = normalized.sampleLimit();
    int maxForecastDays = normalized.maxForecastDays();

    List<LocalDate> expectedDates = buildExpectedDates(startDate, endDate);
    long expectedDays = expectedDates.size();

    NoLeakageResult noLeakage = buildNoLeakageResult(stationIds, startDate, endDate,
        asofPolicyId, models, sampleLimit);
    StationCoverageResult stationCoverage = buildStationCoverageResult(
        stationIds, expectedDates, startDate, endDate, sampleLimit);
    MosAvailabilityResult mosAvailability = buildMosAvailabilityResult(
        stationIds, models, expectedDates, startDate, endDate, sampleLimit);
    FeatureCoverageResult featureCoverage = buildFeatureCoverageResult(
        stationIds, models, expectedDays, startDate, endDate, asofPolicyId, sampleLimit);
    AlignmentResult alignment = buildAlignmentResult(
        stationIds, models, startDate, endDate, asofPolicyId, maxForecastDays, sampleLimit);
    CliRevisionResult cliRevisions = new CliRevisionResult(Status.SKIPPED, CLI_REVISION_NOTE, null);

    Status overallStatus = resolveOverallStatus(List.of(
        noLeakage.status(),
        stationCoverage.status(),
        mosAvailability.status(),
        featureCoverage.status(),
        alignment.status()));

    return new AuditReport(
        SCHEMA_VERSION,
        Instant.now(),
        startDate,
        endDate,
        asofPolicyId,
        stationIds,
        models,
        overallStatus,
        noLeakage,
        stationCoverage,
        mosAvailability,
        featureCoverage,
        alignment,
        cliRevisions);
  }

  private AuditRequest normalizeRequest(AuditRequest request) {
    Objects.requireNonNull(request, "request is required");
    if (request.stationIds() == null || request.stationIds().isEmpty()) {
      throw new IllegalArgumentException("stationIds are required");
    }
    LocalDate startDate = Objects.requireNonNull(request.startDateLocal(), "startDateLocal is required");
    LocalDate endDate = Objects.requireNonNull(request.endDateLocal(), "endDateLocal is required");
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("endDateLocal must be >= startDateLocal");
    }
    Long asofPolicyId = Objects.requireNonNull(request.asofPolicyId(), "asofPolicyId is required");
    if (!asofPolicyRepository.existsById(asofPolicyId)) {
      throw new IllegalArgumentException("asofPolicyId not found: " + asofPolicyId);
    }
    List<MosModel> models = normalizeModels(request.models());
    int maxForecastDays = request.maxForecastDays();
    if (maxForecastDays < 0) {
      throw new IllegalArgumentException("maxForecastDays must be >= 0");
    }
    int sampleLimit = request.sampleLimit() > 0 ? request.sampleLimit() : 10;
    List<String> stationIds = normalizeStationIds(request.stationIds());
    return new AuditRequest(stationIds, startDate, endDate, asofPolicyId, models,
        maxForecastDays, sampleLimit);
  }

  private List<String> normalizeStationIds(List<String> stationIds) {
    Set<String> unique = new LinkedHashSet<>();
    for (String stationId : stationIds) {
      if (stationId == null || stationId.isBlank()) {
        throw new IllegalArgumentException("stationId is required");
      }
      unique.add(stationId.trim().toUpperCase(Locale.ROOT));
    }
    return List.copyOf(unique);
  }

  private List<MosModel> normalizeModels(List<MosModel> models) {
    if (models == null || models.isEmpty()) {
      throw new IllegalArgumentException("models are required");
    }
    Set<MosModel> unique = new LinkedHashSet<>();
    for (MosModel model : models) {
      if (model == null) {
        throw new IllegalArgumentException("model is required");
      }
      unique.add(model);
    }
    return List.copyOf(unique);
  }

  private NoLeakageResult buildNoLeakageResult(List<String> stationIds,
                                               LocalDate startDate,
                                               LocalDate endDate,
                                               Long asofPolicyId,
                                               List<MosModel> models,
                                               int sampleLimit) {
    MapSqlParameterSource params = new MapSqlParameterSource()
        .addValue("stationIds", stationIds)
        .addValue("asofPolicyId", asofPolicyId)
        .addValue("startDate", startDate)
        .addValue("endDate", endDate)
        .addValue("models", toModelNames(models));
    long violations = jdbcTemplate.queryForObject(LEAKAGE_COUNT_SQL, params, Long.class);
    List<LeakageSample> samples = Collections.emptyList();
    if (violations > 0) {
      params.addValue("sampleLimit", sampleLimit);
      samples = jdbcTemplate.query(LEAKAGE_SAMPLE_SQL, params, leakageRowMapper());
    }
    Status status = violations == 0 ? Status.PASS : Status.FAIL;
    return new NoLeakageResult(status, violations, samples);
  }

  private StationCoverageResult buildStationCoverageResult(List<String> stationIds,
                                                           List<LocalDate> expectedDates,
                                                           LocalDate startDate,
                                                           LocalDate endDate,
                                                           int sampleLimit) {
    List<StationCoverage> results = new ArrayList<>();
    boolean hasFailures = false;
    for (String stationId : stationIds) {
      MapSqlParameterSource params = new MapSqlParameterSource()
          .addValue("stationId", stationId)
          .addValue("startDate", startDate)
          .addValue("endDate", endDate);
      List<LocalDate> actualDates = jdbcTemplate.query(
          CLI_DATES_SQL, params, (rs, rowNum) -> toLocalDate(rs, "target_date_local"));
      Set<LocalDate> actualSet = new HashSet<>(actualDates);
      List<LocalDate> missing = new ArrayList<>();
      for (LocalDate expected : expectedDates) {
        if (!actualSet.contains(expected)) {
          missing.add(expected);
        }
      }
      long expectedDays = expectedDates.size();
      long missingDays = missing.size();
      long actualDays = actualDates.size();
      if (missingDays > 0 || actualDays != expectedDays) {
        hasFailures = true;
      }
      List<LocalDate> sampleMissing = limitList(missing, sampleLimit);
      results.add(new StationCoverage(stationId, expectedDays, actualDays, missingDays, sampleMissing));
    }
    Status status = hasFailures ? Status.FAIL : Status.PASS;
    return new StationCoverageResult(status, results);
  }

  private MosAvailabilityResult buildMosAvailabilityResult(List<String> stationIds,
                                                           List<MosModel> models,
                                                           List<LocalDate> expectedDates,
                                                           LocalDate startDate,
                                                           LocalDate endDate,
                                                           int sampleLimit) {
    List<MosAvailability> results = new ArrayList<>();
    boolean hasFailures = false;
    Instant startUtc = startDate.atStartOfDay(ZoneOffset.UTC).toInstant();
    Instant endUtc = endDate.plusDays(1).atStartOfDay(ZoneOffset.UTC).toInstant();
    long expectedDays = expectedDates.size();
    for (String stationId : stationIds) {
      for (MosModel model : models) {
        MapSqlParameterSource params = new MapSqlParameterSource()
            .addValue("stationId", stationId)
            .addValue("model", model.name())
            .addValue("startUtc", startUtc)
            .addValue("endUtc", endUtc);
        List<LocalDate> runtimeDates = jdbcTemplate.query(
            MOS_RUNTIME_DATES_SQL, params, (rs, rowNum) -> toLocalDate(rs, "runtime_date"));
        RuntimeStats stats = jdbcTemplate.queryForObject(
            MOS_RUNTIME_STATS_SQL, params, runtimeStatsRowMapper());
        Set<LocalDate> runtimeSet = new HashSet<>(runtimeDates);
        List<LocalDate> missing = new ArrayList<>();
        for (LocalDate expected : expectedDates) {
          if (!runtimeSet.contains(expected)) {
            missing.add(expected);
          }
        }
        long runtimeDays = runtimeDates.size();
        long missingDays = missing.size();
        if (missingDays > 0) {
          hasFailures = true;
        }
        results.add(new MosAvailability(
            stationId,
            model,
            expectedDays,
            runtimeDays,
            missingDays,
            stats.minRuntimeUtc(),
            stats.maxRuntimeUtc(),
            limitList(missing, sampleLimit)));
      }
    }
    Status status = hasFailures ? Status.FAIL : Status.PASS;
    return new MosAvailabilityResult(status, results);
  }

  private FeatureCoverageResult buildFeatureCoverageResult(List<String> stationIds,
                                                           List<MosModel> models,
                                                           long expectedDays,
                                                           LocalDate startDate,
                                                           LocalDate endDate,
                                                           Long asofPolicyId,
                                                           int sampleLimit) {
    MapSqlParameterSource params = new MapSqlParameterSource()
        .addValue("stationIds", stationIds)
        .addValue("asofPolicyId", asofPolicyId)
        .addValue("startDate", startDate)
        .addValue("endDate", endDate)
        .addValue("models", toModelNames(models))
        .addValue("fallbackReason", MISSING_REASON_FALLBACK);

    Map<StationModelKey, MissingStats> statsMap = new HashMap<>();
    List<MissingStats> stats = jdbcTemplate.query(FEATURE_STATS_SQL, params, missingStatsRowMapper());
    for (MissingStats stat : stats) {
      statsMap.put(new StationModelKey(stat.stationId(), stat.model()), stat);
    }

    Map<StationModelKey, List<MissingReasonCount>> reasonsMap = new HashMap<>();
    List<MissingReasonRow> reasons = jdbcTemplate.query(FEATURE_REASON_SQL, params, reasonsRowMapper());
    for (MissingReasonRow reason : reasons) {
      StationModelKey key = new StationModelKey(reason.stationId(), reason.model());
      reasonsMap.computeIfAbsent(key, ignored -> new ArrayList<>())
          .add(new MissingReasonCount(reason.reason(), reason.count()));
    }

    List<FeatureCoverage> results = new ArrayList<>();
    boolean hasFailures = false;
    for (String stationId : stationIds) {
      for (MosModel model : models) {
        StationModelKey key = new StationModelKey(stationId, model);
        MissingStats stat = statsMap.get(key);
        long totalCount = stat != null ? stat.totalCount() : 0L;
        long missingCount = stat != null ? stat.missingCount() : 0L;
        long missingRows = expectedDays - totalCount;
        if (missingRows < 0) {
          missingRows = 0;
        }
        double missingPercent = totalCount == 0
            ? 100.0
            : (100.0 * missingCount / totalCount);
        if (missingCount > 0 || missingRows > 0) {
          hasFailures = true;
        }
        List<MissingReasonCount> reasonsForKey = reasonsMap.getOrDefault(key, List.of());
        results.add(new FeatureCoverage(
            stationId,
            model,
            expectedDays,
            totalCount,
            missingRows,
            missingCount,
            missingPercent,
            limitList(reasonsForKey, sampleLimit)));
      }
    }
    Status status = hasFailures ? Status.FAIL : Status.PASS;
    return new FeatureCoverageResult(status, results);
  }

  private AlignmentResult buildAlignmentResult(List<String> stationIds,
                                               List<MosModel> models,
                                               LocalDate startDate,
                                               LocalDate endDate,
                                               Long asofPolicyId,
                                               int maxForecastDays,
                                               int sampleLimit) {
    MapSqlParameterSource params = new MapSqlParameterSource()
        .addValue("stationIds", stationIds)
        .addValue("asofPolicyId", asofPolicyId)
        .addValue("startDate", startDate)
        .addValue("endDate", endDate)
        .addValue("models", toModelNames(models));
    List<AlignmentRow> rows = jdbcTemplate.query(ALIGNMENT_SQL, params, alignmentRowMapper());
    List<AlignmentViolation> violations = new ArrayList<>();
    long checked = 0;
    for (AlignmentRow row : rows) {
      if (row.chosenRuntimeUtc() == null) {
        if (row.tmaxFPresent()) {
          violations.add(new AlignmentViolation(
              row.stationId(),
              row.model(),
              row.targetDateLocal(),
              null,
              null,
              0,
              "tmax_f present without chosen runtime"));
        }
        continue;
      }
      checked++;
      LocalDate runtimeLocalDate;
      try {
        ZoneId zoneId = ZoneId.of(row.stationZoneid());
        runtimeLocalDate = row.chosenRuntimeUtc().atZone(zoneId).toLocalDate();
      } catch (RuntimeException ex) {
        violations.add(new AlignmentViolation(
            row.stationId(),
            row.model(),
            row.targetDateLocal(),
            row.chosenRuntimeUtc(),
            null,
            0,
            "invalid station_zoneid: " + row.stationZoneid()));
        continue;
      }
      long daysAhead = ChronoUnit.DAYS.between(runtimeLocalDate, row.targetDateLocal());
      if (daysAhead < 0) {
        violations.add(new AlignmentViolation(
            row.stationId(),
            row.model(),
            row.targetDateLocal(),
            row.chosenRuntimeUtc(),
            runtimeLocalDate,
            daysAhead,
            "target date before runtime local date"));
      } else if (daysAhead > maxForecastDays) {
        violations.add(new AlignmentViolation(
            row.stationId(),
            row.model(),
            row.targetDateLocal(),
            row.chosenRuntimeUtc(),
            runtimeLocalDate,
            daysAhead,
            "target date exceeds max forecast horizon"));
      }
    }
    Status status = violations.isEmpty() ? Status.PASS : Status.FAIL;
    return new AlignmentResult(status, maxForecastDays, checked, limitList(violations, sampleLimit));
  }

  private Status resolveOverallStatus(List<Status> statuses) {
    for (Status status : statuses) {
      if (status == Status.FAIL) {
        return Status.FAIL;
      }
    }
    return Status.PASS;
  }

  private List<LocalDate> buildExpectedDates(LocalDate startDate, LocalDate endDate) {
    List<LocalDate> dates = new ArrayList<>();
    LocalDate current = startDate;
    while (!current.isAfter(endDate)) {
      dates.add(current);
      current = current.plusDays(1);
    }
    return Collections.unmodifiableList(dates);
  }

  private List<String> toModelNames(List<MosModel> models) {
    return models.stream()
        .filter(Objects::nonNull)
        .map(model -> model.name().toUpperCase(Locale.ROOT))
        .distinct()
        .collect(Collectors.toList());
  }

  private Path resolveOutputDir(Path outputDir) {
    Path resolved = outputDir != null ? outputDir : Path.of("reports");
    try {
      Files.createDirectories(resolved);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to create report output dir: " + resolved, ex);
    }
    return resolved.toAbsolutePath().normalize();
  }

  private String buildBaseName(List<String> stationIds,
                               LocalDate startDate,
                               LocalDate endDate,
                               Long asofPolicyId) {
    String stationToken = stationIds.size() == 1 ? stationIds.get(0) : "multi";
    return String.format(Locale.ROOT, "audit-report-%s-%s-%s-asof-%d",
        stationToken, startDate, endDate, asofPolicyId);
  }

  private void writeMarkdown(AuditReport report, Path markdownPath) {
    try {
      Files.writeString(markdownPath, renderMarkdown(report), StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write markdown report: " + markdownPath, ex);
    }
  }

  private void writeJson(AuditReport report, Path jsonPath) {
    try {
      objectMapper.writerWithDefaultPrettyPrinter().writeValue(jsonPath.toFile(), report);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write JSON report: " + jsonPath, ex);
    }
  }

  private String renderMarkdown(AuditReport report) {
    StringBuilder builder = new StringBuilder();
    builder.append("# Data Quality Audit Report\n\n");
    builder.append("Generated at (UTC): ").append(report.generatedAtUtc()).append('\n');
    builder.append("Date range: ").append(report.startDateLocal())
        .append("..").append(report.endDateLocal()).append('\n');
    builder.append("As-of policy id: ").append(report.asofPolicyId()).append('\n');
    builder.append("Stations: ").append(String.join(", ", report.stationIds())).append('\n');
    builder.append("Models: ").append(report.models().stream()
        .map(MosModel::name)
        .collect(Collectors.joining(", "))).append("\n\n");

    builder.append("## Summary\n");
    builder.append("- Overall status: ").append(report.overallStatus()).append('\n');
    builder.append("- No leakage: ").append(report.noLeakage().status())
        .append(" (violations: ").append(report.noLeakage().violationCount()).append(")\n");
    builder.append("- Station coverage: ").append(report.stationCoverage().status()).append('\n');
    builder.append("- MOS availability: ").append(report.mosAvailability().status()).append('\n');
    builder.append("- Feature coverage: ").append(report.featureCoverage().status()).append('\n');
    builder.append("- Alignment: ").append(report.alignment().status()).append('\n');
    builder.append("- CLI revisions: ").append(report.cliRevisions().status()).append("\n\n");

    appendNoLeakageSection(builder, report.noLeakage());
    appendStationCoverageSection(builder, report.stationCoverage());
    appendMosAvailabilitySection(builder, report.mosAvailability());
    appendFeatureCoverageSection(builder, report.featureCoverage());
    appendAlignmentSection(builder, report.alignment());
    appendCliRevisionSection(builder, report.cliRevisions());

    return builder.toString();
  }

  private void appendNoLeakageSection(StringBuilder builder, NoLeakageResult result) {
    builder.append("## No leakage\n");
    builder.append("Status: ").append(result.status()).append('\n');
    builder.append("Violations: ").append(result.violationCount()).append('\n');
    if (!result.samples().isEmpty()) {
      builder.append("Sample violations:\n");
      for (LeakageSample sample : result.samples()) {
        builder.append("- station=").append(sample.stationId())
            .append(" target_date=").append(sample.targetDateLocal())
            .append(" model=").append(sample.model().name())
            .append(" asof_utc=").append(sample.asofUtc())
            .append(" chosen_runtime_utc=").append(sample.chosenRuntimeUtc())
            .append('\n');
      }
    }
    builder.append('\n');
  }

  private void appendStationCoverageSection(StringBuilder builder, StationCoverageResult result) {
    builder.append("## Station coverage (CLI)\n");
    builder.append("Status: ").append(result.status()).append('\n');
    builder.append("| Station | Expected days | Actual days | Missing days | Sample missing dates |\n");
    builder.append("| --- | --- | --- | --- | --- |\n");
    for (StationCoverage coverage : result.stations()) {
      builder.append("| ").append(coverage.stationId()).append(" | ")
          .append(coverage.expectedDays()).append(" | ")
          .append(coverage.actualDays()).append(" | ")
          .append(coverage.missingDays()).append(" | ")
          .append(formatSampleDates(coverage.sampleMissingDates())).append(" |\n");
    }
    builder.append('\n');
  }

  private void appendMosAvailabilitySection(StringBuilder builder, MosAvailabilityResult result) {
    builder.append("## MOS availability (mos_run coverage)\n");
    builder.append("Status: ").append(result.status()).append('\n');
    builder.append("| Station | Model | Expected days | Runtime days | Missing days | Min runtime | Max runtime | Sample missing days |\n");
    builder.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (MosAvailability availability : result.stations()) {
      builder.append("| ").append(availability.stationId()).append(" | ")
          .append(availability.model().name()).append(" | ")
          .append(availability.expectedDays()).append(" | ")
          .append(availability.runtimeDays()).append(" | ")
          .append(availability.missingRuntimeDays()).append(" | ")
          .append(formatInstant(availability.minRuntimeUtc())).append(" | ")
          .append(formatInstant(availability.maxRuntimeUtc())).append(" | ")
          .append(formatSampleDates(availability.sampleMissingRuntimeDates())).append(" |\n");
    }
    builder.append('\n');
  }

  private void appendFeatureCoverageSection(StringBuilder builder, FeatureCoverageResult result) {
    builder.append("## Feature coverage (mos_asof_feature)\n");
    builder.append("Status: ").append(result.status()).append('\n');
    builder.append("| Station | Model | Expected rows | Actual rows | Missing rows | Missing tmax | Missing % | Top missing reasons |\n");
    builder.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (FeatureCoverage coverage : result.stations()) {
      builder.append("| ").append(coverage.stationId()).append(" | ")
          .append(coverage.model().name()).append(" | ")
          .append(coverage.expectedRows()).append(" | ")
          .append(coverage.actualRows()).append(" | ")
          .append(coverage.missingRows()).append(" | ")
          .append(coverage.missingTmaxCount()).append(" | ")
          .append(formatPercent(coverage.missingTmaxPercent())).append(" | ")
          .append(formatReasons(coverage.topMissingReasons())).append(" |\n");
    }
    builder.append('\n');
  }

  private void appendAlignmentSection(StringBuilder builder, AlignmentResult result) {
    builder.append("## Alignment (target date vs runtime horizon)\n");
    builder.append("Status: ").append(result.status()).append('\n');
    builder.append("Max forecast days: ").append(result.maxForecastDays()).append('\n');
    builder.append("Checked rows: ").append(result.checkedRows()).append('\n');
    if (!result.samples().isEmpty()) {
      builder.append("Sample violations:\n");
      for (AlignmentViolation violation : result.samples()) {
        builder.append("- station=").append(violation.stationId())
            .append(" model=").append(violation.model().name())
            .append(" target_date=").append(violation.targetDateLocal())
            .append(" runtime_utc=").append(violation.chosenRuntimeUtc())
            .append(" runtime_local_date=").append(violation.runtimeLocalDate())
            .append(" days_ahead=").append(violation.daysAhead())
            .append(" message=").append(violation.message())
            .append('\n');
      }
    }
    builder.append('\n');
  }

  private void appendCliRevisionSection(StringBuilder builder, CliRevisionResult result) {
    builder.append("## CLI revisions\n");
    builder.append("Status: ").append(result.status()).append('\n');
    if (result.revisionCount() != null) {
      builder.append("Revision count: ").append(result.revisionCount()).append('\n');
    }
    builder.append("Note: ").append(result.note()).append("\n\n");
  }

  private String formatSampleDates(List<LocalDate> dates) {
    if (dates == null || dates.isEmpty()) {
      return "-";
    }
    return dates.stream()
        .map(LocalDate::toString)
        .collect(Collectors.joining(", "));
  }

  private String formatReasons(List<MissingReasonCount> reasons) {
    if (reasons == null || reasons.isEmpty()) {
      return "-";
    }
    return reasons.stream()
        .map(reason -> reason.reason() + "=" + reason.count())
        .collect(Collectors.joining(", "));
  }

  private String formatPercent(double value) {
    return String.format(Locale.ROOT, "%.1f%%", value);
  }

  private String formatInstant(Instant instant) {
    return instant == null ? "-" : instant.toString();
  }

  private <T> List<T> limitList(List<T> values, int limit) {
    if (values == null || values.isEmpty()) {
      return List.of();
    }
    if (values.size() <= limit) {
      return values;
    }
    return List.copyOf(values.subList(0, limit));
  }

  private RowMapper<LeakageSample> leakageRowMapper() {
    return (rs, rowNum) -> new LeakageSample(
        rs.getString("station_id"),
        toLocalDate(rs, "target_date_local"),
        MosModel.valueOf(rs.getString("model")),
        toInstant(rs, "asof_utc"),
        toInstant(rs, "chosen_runtime_utc"));
  }

  private RowMapper<RuntimeStats> runtimeStatsRowMapper() {
    return (rs, rowNum) -> new RuntimeStats(
        toInstant(rs, "min_runtime"),
        toInstant(rs, "max_runtime"),
        rs.getLong("run_count"));
  }

  private RowMapper<MissingStats> missingStatsRowMapper() {
    return (rs, rowNum) -> new MissingStats(
        rs.getString("station_id"),
        MosModel.valueOf(rs.getString("model")),
        rs.getLong("total_count"),
        rs.getLong("missing_count"));
  }

  private RowMapper<MissingReasonRow> reasonsRowMapper() {
    return (rs, rowNum) -> new MissingReasonRow(
        rs.getString("station_id"),
        MosModel.valueOf(rs.getString("model")),
        rs.getString("reason"),
        rs.getLong("missing_count"));
  }

  private RowMapper<AlignmentRow> alignmentRowMapper() {
    return (rs, rowNum) -> new AlignmentRow(
        rs.getString("station_id"),
        MosModel.valueOf(rs.getString("model")),
        toLocalDate(rs, "target_date_local"),
        toInstant(rs, "chosen_runtime_utc"),
        rs.getString("station_zoneid"),
        rs.getObject("tmax_f") != null);
  }

  private Instant toInstant(ResultSet rs, String column) throws SQLException {
    Timestamp timestamp = rs.getTimestamp(column);
    return timestamp != null ? timestamp.toInstant() : null;
  }

  private LocalDate toLocalDate(ResultSet rs, String column) throws SQLException {
    Date date = rs.getDate(column);
    return date != null ? date.toLocalDate() : null;
  }

  private record RuntimeStats(Instant minRuntimeUtc, Instant maxRuntimeUtc, long runCount) {
  }

  private record MissingStats(String stationId, MosModel model, long totalCount, long missingCount) {
  }

  private record MissingReasonRow(String stationId, MosModel model, String reason, long count) {
  }

  private record AlignmentRow(String stationId,
                              MosModel model,
                              LocalDate targetDateLocal,
                              Instant chosenRuntimeUtc,
                              String stationZoneid,
                              boolean tmaxFPresent) {
  }

  private record StationModelKey(String stationId, MosModel model) {
  }
}
