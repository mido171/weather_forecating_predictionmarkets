package com.predictionmarkets.weather.repository;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class MosForecastValueUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO mos_forecast_value (
        station_id,
        model,
        runtime_utc,
        forecast_time_utc,
        variable_code,
        value_num,
        value_text,
        value_raw,
        raw_payload_hash_ref,
        retrieved_at_utc
      ) VALUES (
        :stationId,
        :model,
        :runtimeUtc,
        :forecastTimeUtc,
        :variableCode,
        :valueNum,
        :valueText,
        :valueRaw,
        :rawPayloadHashRef,
        :retrievedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        value_num = VALUES(value_num),
        value_text = VALUES(value_text),
        value_raw = VALUES(value_raw),
        raw_payload_hash_ref = VALUES(raw_payload_hash_ref),
        retrieved_at_utc = VALUES(retrieved_at_utc)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public MosForecastValueUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int[] upsertAll(List<UpsertRow> rows) {
    SqlParameterSource[] batch = rows.stream()
        .map(MosForecastValueUpsertRepository::toParams)
        .toArray(SqlParameterSource[]::new);
    return jdbcTemplate.batchUpdate(UPSERT_SQL, batch);
  }

  public void deleteAll() {
    jdbcTemplate.update("delete from mos_forecast_value", new MapSqlParameterSource());
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("model", row.model)
        .addValue("runtimeUtc", row.runtimeUtc)
        .addValue("forecastTimeUtc", row.forecastTimeUtc)
        .addValue("variableCode", row.variableCode)
        .addValue("valueNum", row.valueNum)
        .addValue("valueText", row.valueText)
        .addValue("valueRaw", row.valueRaw)
        .addValue("rawPayloadHashRef", row.rawPayloadHashRef)
        .addValue("retrievedAtUtc", row.retrievedAtUtc);
  }

  public static final class UpsertRow {
    private final String stationId;
    private final String model;
    private final Instant runtimeUtc;
    private final Instant forecastTimeUtc;
    private final String variableCode;
    private final BigDecimal valueNum;
    private final String valueText;
    private final String valueRaw;
    private final String rawPayloadHashRef;
    private final Instant retrievedAtUtc;

    public UpsertRow(String stationId,
                     String model,
                     Instant runtimeUtc,
                     Instant forecastTimeUtc,
                     String variableCode,
                     BigDecimal valueNum,
                     String valueText,
                     String valueRaw,
                     String rawPayloadHashRef,
                     Instant retrievedAtUtc) {
      this.stationId = stationId;
      this.model = model;
      this.runtimeUtc = runtimeUtc;
      this.forecastTimeUtc = forecastTimeUtc;
      this.variableCode = variableCode;
      this.valueNum = valueNum;
      this.valueText = valueText;
      this.valueRaw = valueRaw;
      this.rawPayloadHashRef = rawPayloadHashRef;
      this.retrievedAtUtc = retrievedAtUtc;
    }

    public Instant getRuntimeUtc() {
      return runtimeUtc;
    }

    public Instant getForecastTimeUtc() {
      return forecastTimeUtc;
    }

    public String getVariableCode() {
      return variableCode;
    }
  }
}
