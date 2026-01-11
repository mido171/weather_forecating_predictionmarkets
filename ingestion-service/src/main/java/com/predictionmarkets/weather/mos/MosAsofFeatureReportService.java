package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class MosAsofFeatureReportService {
  private static final Logger logger = LoggerFactory.getLogger(MosAsofFeatureReportService.class);
  private static final String MISSING_REASON_FALLBACK = "UNSPECIFIED";

  private static final String MISSING_STATS_SQL = """
      SELECT model,
             COUNT(*) AS total_count,
             SUM(CASE WHEN tmax_f IS NULL THEN 1 ELSE 0 END) AS missing_count
        FROM mos_asof_feature
       WHERE station_id = :stationId
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
       GROUP BY model
      """;

  private static final String MISSING_REASONS_SQL = """
      SELECT model,
             COALESCE(missing_reason, :fallbackReason) AS reason,
             COUNT(*) AS missing_count
        FROM mos_asof_feature
       WHERE station_id = :stationId
         AND asof_policy_id = :asofPolicyId
         AND target_date_local BETWEEN :startDate AND :endDate
         AND model IN (:models)
         AND tmax_f IS NULL
       GROUP BY model, COALESCE(missing_reason, :fallbackReason)
       ORDER BY missing_count DESC
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public MosAsofFeatureReportService(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public void logCompletenessReport(String stationId,
                                    LocalDate startDate,
                                    LocalDate endDate,
                                    Long asofPolicyId,
                                    List<MosModel> models) {
    String normalizedStation = normalizeStationId(stationId);
    Objects.requireNonNull(startDate, "startDate is required");
    Objects.requireNonNull(endDate, "endDate is required");
    Objects.requireNonNull(asofPolicyId, "asofPolicyId is required");
    if (models == null || models.isEmpty()) {
      throw new IllegalArgumentException("models are required");
    }

    List<String> modelNames = models.stream()
        .filter(Objects::nonNull)
        .map(model -> model.name().toUpperCase(Locale.ROOT))
        .distinct()
        .toList();

    MapSqlParameterSource params = new MapSqlParameterSource()
        .addValue("stationId", normalizedStation)
        .addValue("asofPolicyId", asofPolicyId)
        .addValue("startDate", startDate)
        .addValue("endDate", endDate)
        .addValue("models", modelNames)
        .addValue("fallbackReason", MISSING_REASON_FALLBACK);

    List<MissingStats> stats = jdbcTemplate.query(MISSING_STATS_SQL, params, statsRowMapper());
    Map<MosModel, MissingStats> statsByModel = new EnumMap<>(MosModel.class);
    for (MissingStats stat : stats) {
      statsByModel.put(stat.model(), stat);
    }

    Map<MosModel, List<MissingReasonCount>> reasonsByModel = new EnumMap<>(MosModel.class);
    List<MissingReasonCount> reasons = jdbcTemplate.query(
        MISSING_REASONS_SQL, params, reasonsRowMapper());
    for (MissingReasonCount reason : reasons) {
      reasonsByModel.computeIfAbsent(reason.model(), ignored -> new ArrayList<>())
          .add(reason);
    }

    logger.info("Feature completeness report station={} range={}..{} asofPolicyId={}",
        normalizedStation, startDate, endDate, asofPolicyId);
    for (MosModel model : models) {
      if (model == null) {
        continue;
      }
      MissingStats stat = statsByModel.get(model);
      if (stat == null) {
        logger.info("Model {}: no rows for range.", model.name());
        continue;
      }
      double missingPct = stat.totalCount() == 0
          ? 0.0
          : (100.0 * stat.missingCount() / stat.totalCount());
      logger.info("Model {}: missing {}/{} ({})",
          model.name(),
          stat.missingCount(),
          stat.totalCount(),
          formatPercent(missingPct));

      List<MissingReasonCount> modelReasons = reasonsByModel.get(model);
      if (modelReasons == null || modelReasons.isEmpty()) {
        continue;
      }
      String reasonsText = formatReasons(modelReasons, 3);
      logger.info("Model {} missing reasons: {}", model.name(), reasonsText);
    }
  }

  private RowMapper<MissingStats> statsRowMapper() {
    return (rs, rowNum) -> new MissingStats(
        MosModel.valueOf(rs.getString("model")),
        rs.getLong("total_count"),
        rs.getLong("missing_count"));
  }

  private RowMapper<MissingReasonCount> reasonsRowMapper() {
    return (rs, rowNum) -> new MissingReasonCount(
        MosModel.valueOf(rs.getString("model")),
        rs.getString("reason"),
        rs.getLong("missing_count"));
  }

  private String formatReasons(List<MissingReasonCount> reasons, int limit) {
    StringBuilder builder = new StringBuilder();
    int count = 0;
    for (MissingReasonCount reason : reasons) {
      if (count >= limit) {
        break;
      }
      if (count > 0) {
        builder.append(", ");
      }
      builder.append(reason.reason()).append('=').append(reason.count());
      count++;
    }
    return builder.toString();
  }

  private String formatPercent(double value) {
    return String.format(Locale.ROOT, "%.1f%%", value);
  }

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private record MissingStats(MosModel model, long totalCount, long missingCount) {
  }

  private record MissingReasonCount(MosModel model, String reason, long count) {
  }
}
