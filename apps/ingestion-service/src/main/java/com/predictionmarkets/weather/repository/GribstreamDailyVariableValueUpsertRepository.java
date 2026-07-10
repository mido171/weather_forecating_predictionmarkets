package com.predictionmarkets.weather.repository;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class GribstreamDailyVariableValueUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO gribstream_daily_variable_value (
        station_id,
        zone_id,
        model_code,
        target_date_local,
        asof_utc,
        member_stat,
        member,
        variable_name,
        variable_level,
        variable_info,
        variable_alias,
        reducer,
        value_num,
        value_text,
        sample_count,
        expected_count,
        window_start_utc,
        window_end_utc,
        request_json,
        request_sha256,
        response_sha256,
        retrieved_at_utc,
        notes
      ) VALUES (
        :stationId,
        :zoneId,
        :modelCode,
        :targetDateLocal,
        :asofUtc,
        :memberStat,
        :member,
        :variableName,
        :variableLevel,
        :variableInfo,
        :variableAlias,
        :reducer,
        :valueNum,
        :valueText,
        :sampleCount,
        :expectedCount,
        :windowStartUtc,
        :windowEndUtc,
        :requestJson,
        :requestSha256,
        :responseSha256,
        :retrievedAtUtc,
        :notes
      ) AS new
      ON DUPLICATE KEY UPDATE
        zone_id = new.zone_id,
        variable_name = new.variable_name,
        variable_level = new.variable_level,
        variable_info = new.variable_info,
        value_num = new.value_num,
        value_text = new.value_text,
        sample_count = new.sample_count,
        expected_count = new.expected_count,
        window_start_utc = new.window_start_utc,
        window_end_utc = new.window_end_utc,
        request_json = new.request_json,
        request_sha256 = new.request_sha256,
        response_sha256 = new.response_sha256,
        retrieved_at_utc = new.retrieved_at_utc,
        notes = new.notes
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public GribstreamDailyVariableValueUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int[] upsertAll(List<UpsertRow> rows) {
    SqlParameterSource[] batch = rows.stream()
        .map(GribstreamDailyVariableValueUpsertRepository::toParams)
        .toArray(SqlParameterSource[]::new);
    return jdbcTemplate.batchUpdate(UPSERT_SQL, batch);
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("zoneId", row.zoneId)
        .addValue("modelCode", row.modelCode)
        .addValue("targetDateLocal", row.targetDateLocal)
        .addValue("asofUtc", row.asofUtc)
        .addValue("memberStat", row.memberStat)
        .addValue("member", row.member)
        .addValue("variableName", row.variableName)
        .addValue("variableLevel", row.variableLevel)
        .addValue("variableInfo", row.variableInfo)
        .addValue("variableAlias", row.variableAlias)
        .addValue("reducer", row.reducer)
        .addValue("valueNum", row.valueNum)
        .addValue("valueText", row.valueText)
        .addValue("sampleCount", row.sampleCount)
        .addValue("expectedCount", row.expectedCount)
        .addValue("windowStartUtc", row.windowStartUtc)
        .addValue("windowEndUtc", row.windowEndUtc)
        .addValue("requestJson", row.requestJson)
        .addValue("requestSha256", row.requestSha256)
        .addValue("responseSha256", row.responseSha256)
        .addValue("retrievedAtUtc", row.retrievedAtUtc)
        .addValue("notes", row.notes);
  }

  public static final class UpsertRow {
    private final String stationId;
    private final String zoneId;
    private final String modelCode;
    private final LocalDate targetDateLocal;
    private final Instant asofUtc;
    private final String memberStat;
    private final int member;
    private final String variableName;
    private final String variableLevel;
    private final String variableInfo;
    private final String variableAlias;
    private final String reducer;
    private final Double valueNum;
    private final String valueText;
    private final Integer sampleCount;
    private final Integer expectedCount;
    private final Instant windowStartUtc;
    private final Instant windowEndUtc;
    private final String requestJson;
    private final String requestSha256;
    private final String responseSha256;
    private final Instant retrievedAtUtc;
    private final String notes;

    public UpsertRow(String stationId,
                     String zoneId,
                     String modelCode,
                     LocalDate targetDateLocal,
                     Instant asofUtc,
                     String memberStat,
                     int member,
                     String variableName,
                     String variableLevel,
                     String variableInfo,
                     String variableAlias,
                     String reducer,
                     Double valueNum,
                     String valueText,
                     Integer sampleCount,
                     Integer expectedCount,
                     Instant windowStartUtc,
                     Instant windowEndUtc,
                     String requestJson,
                     String requestSha256,
                     String responseSha256,
                     Instant retrievedAtUtc,
                     String notes) {
      this.stationId = stationId;
      this.zoneId = zoneId;
      this.modelCode = modelCode;
      this.targetDateLocal = targetDateLocal;
      this.asofUtc = asofUtc;
      this.memberStat = memberStat;
      this.member = member;
      this.variableName = variableName;
      this.variableLevel = variableLevel;
      this.variableInfo = variableInfo;
      this.variableAlias = variableAlias;
      this.reducer = reducer;
      this.valueNum = valueNum;
      this.valueText = valueText;
      this.sampleCount = sampleCount;
      this.expectedCount = expectedCount;
      this.windowStartUtc = windowStartUtc;
      this.windowEndUtc = windowEndUtc;
      this.requestJson = requestJson;
      this.requestSha256 = requestSha256;
      this.responseSha256 = responseSha256;
      this.retrievedAtUtc = retrievedAtUtc;
      this.notes = notes;
    }
  }
}
