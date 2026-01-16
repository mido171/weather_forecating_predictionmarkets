package com.predictionmarkets.weather.repository;

import java.time.Instant;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class GribstreamForecastValueUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO gribstream_forecast_value (
        station_id,
        zone_id,
        model_code,
        asof_utc,
        forecasted_at_utc,
        forecasted_time_utc,
        member,
        variable_name,
        variable_level,
        variable_info,
        variable_alias,
        value_num,
        value_text,
        request_json,
        request_sha256,
        response_sha256,
        retrieved_at_utc,
        notes
      ) VALUES (
        :stationId,
        :zoneId,
        :modelCode,
        :asofUtc,
        :forecastedAtUtc,
        :forecastedTimeUtc,
        :member,
        :variableName,
        :variableLevel,
        :variableInfo,
        :variableAlias,
        :valueNum,
        :valueText,
        :requestJson,
        :requestSha256,
        :responseSha256,
        :retrievedAtUtc,
        :notes
      )
      ON DUPLICATE KEY UPDATE
        value_num = VALUES(value_num),
        value_text = VALUES(value_text),
        request_json = VALUES(request_json),
        request_sha256 = VALUES(request_sha256),
        response_sha256 = VALUES(response_sha256),
        retrieved_at_utc = VALUES(retrieved_at_utc),
        notes = VALUES(notes)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public GribstreamForecastValueUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int[] upsertAll(List<UpsertRow> rows) {
    SqlParameterSource[] batch = rows.stream()
        .map(GribstreamForecastValueUpsertRepository::toParams)
        .toArray(SqlParameterSource[]::new);
    return jdbcTemplate.batchUpdate(UPSERT_SQL, batch);
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("zoneId", row.zoneId)
        .addValue("modelCode", row.modelCode)
        .addValue("asofUtc", row.asofUtc)
        .addValue("forecastedAtUtc", row.forecastedAtUtc)
        .addValue("forecastedTimeUtc", row.forecastedTimeUtc)
        .addValue("member", row.member)
        .addValue("variableName", row.variableName)
        .addValue("variableLevel", row.variableLevel)
        .addValue("variableInfo", row.variableInfo)
        .addValue("variableAlias", row.variableAlias)
        .addValue("valueNum", row.valueNum)
        .addValue("valueText", row.valueText)
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
    private final Instant asofUtc;
    private final Instant forecastedAtUtc;
    private final Instant forecastedTimeUtc;
    private final Integer member;
    private final String variableName;
    private final String variableLevel;
    private final String variableInfo;
    private final String variableAlias;
    private final Double valueNum;
    private final String valueText;
    private final String requestJson;
    private final String requestSha256;
    private final String responseSha256;
    private final Instant retrievedAtUtc;
    private final String notes;

    public UpsertRow(String stationId,
                     String zoneId,
                     String modelCode,
                     Instant asofUtc,
                     Instant forecastedAtUtc,
                     Instant forecastedTimeUtc,
                     Integer member,
                     String variableName,
                     String variableLevel,
                     String variableInfo,
                     String variableAlias,
                     Double valueNum,
                     String valueText,
                     String requestJson,
                     String requestSha256,
                     String responseSha256,
                     Instant retrievedAtUtc,
                     String notes) {
      this.stationId = stationId;
      this.zoneId = zoneId;
      this.modelCode = modelCode;
      this.asofUtc = asofUtc;
      this.forecastedAtUtc = forecastedAtUtc;
      this.forecastedTimeUtc = forecastedTimeUtc;
      this.member = member;
      this.variableName = variableName;
      this.variableLevel = variableLevel;
      this.variableInfo = variableInfo;
      this.variableAlias = variableAlias;
      this.valueNum = valueNum;
      this.valueText = valueText;
      this.requestJson = requestJson;
      this.requestSha256 = requestSha256;
      this.responseSha256 = responseSha256;
      this.retrievedAtUtc = retrievedAtUtc;
      this.notes = notes;
    }
  }
}
