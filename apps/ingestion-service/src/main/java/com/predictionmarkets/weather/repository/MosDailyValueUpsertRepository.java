package com.predictionmarkets.weather.repository;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class MosDailyValueUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO mos_daily_value (
        station_id,
        station_zoneid,
        model,
        asof_utc,
        runtime_utc,
        target_date_local,
        variable_code,
        value_min,
        value_max,
        value_mean,
        value_median,
        sample_count,
        first_forecast_time_utc,
        last_forecast_time_utc,
        raw_payload_hash_ref,
        retrieved_at_utc
      ) VALUES (
        :stationId,
        :stationZoneid,
        :model,
        :asofUtc,
        :runtimeUtc,
        :targetDateLocal,
        :variableCode,
        :valueMin,
        :valueMax,
        :valueMean,
        :valueMedian,
        :sampleCount,
        :firstForecastTimeUtc,
        :lastForecastTimeUtc,
        :rawPayloadHashRef,
        :retrievedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        station_zoneid = VALUES(station_zoneid),
        asof_utc = VALUES(asof_utc),
        value_min = VALUES(value_min),
        value_max = VALUES(value_max),
        value_mean = VALUES(value_mean),
        value_median = VALUES(value_median),
        sample_count = VALUES(sample_count),
        first_forecast_time_utc = VALUES(first_forecast_time_utc),
        last_forecast_time_utc = VALUES(last_forecast_time_utc),
        raw_payload_hash_ref = VALUES(raw_payload_hash_ref),
        retrieved_at_utc = VALUES(retrieved_at_utc)
      """;
  private static final int DEFAULT_BATCH_SIZE = 200;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public MosDailyValueUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int upsertAll(List<UpsertRow> rows) {
    if (rows == null || rows.isEmpty()) {
      return 0;
    }
    int updated = 0;
    for (int start = 0; start < rows.size(); start += DEFAULT_BATCH_SIZE) {
      int end = Math.min(start + DEFAULT_BATCH_SIZE, rows.size());
      List<UpsertRow> slice = rows.subList(start, end);
      SqlParameterSource[] batch = slice.stream()
          .map(MosDailyValueUpsertRepository::toParams)
          .toArray(SqlParameterSource[]::new);
      updated += Arrays.stream(jdbcTemplate.batchUpdate(UPSERT_SQL, batch)).sum();
    }
    return updated;
  }

  public void deleteAll() {
    jdbcTemplate.update("delete from mos_daily_value", new MapSqlParameterSource());
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("stationZoneid", row.stationZoneid)
        .addValue("model", row.model)
        .addValue("asofUtc", row.asofUtc)
        .addValue("runtimeUtc", row.runtimeUtc)
        .addValue("targetDateLocal", row.targetDateLocal)
        .addValue("variableCode", row.variableCode)
        .addValue("valueMin", row.valueMin)
        .addValue("valueMax", row.valueMax)
        .addValue("valueMean", row.valueMean)
        .addValue("valueMedian", row.valueMedian)
        .addValue("sampleCount", row.sampleCount)
        .addValue("firstForecastTimeUtc", row.firstForecastTimeUtc)
        .addValue("lastForecastTimeUtc", row.lastForecastTimeUtc)
        .addValue("rawPayloadHashRef", row.rawPayloadHashRef)
        .addValue("retrievedAtUtc", row.retrievedAtUtc);
  }

  public static final class UpsertRow {
    private final String stationId;
    private final String stationZoneid;
    private final String model;
    private final Instant asofUtc;
    private final Instant runtimeUtc;
    private final LocalDate targetDateLocal;
    private final String variableCode;
    private final BigDecimal valueMin;
    private final BigDecimal valueMax;
    private final BigDecimal valueMean;
    private final BigDecimal valueMedian;
    private final int sampleCount;
    private final Instant firstForecastTimeUtc;
    private final Instant lastForecastTimeUtc;
    private final String rawPayloadHashRef;
    private final Instant retrievedAtUtc;

    public UpsertRow(String stationId,
                     String stationZoneid,
                     String model,
                     Instant asofUtc,
                     Instant runtimeUtc,
                     LocalDate targetDateLocal,
                     String variableCode,
                     BigDecimal valueMin,
                     BigDecimal valueMax,
                     BigDecimal valueMean,
                     BigDecimal valueMedian,
                     int sampleCount,
                     Instant firstForecastTimeUtc,
                     Instant lastForecastTimeUtc,
                     String rawPayloadHashRef,
                     Instant retrievedAtUtc) {
      this.stationId = stationId;
      this.stationZoneid = stationZoneid;
      this.model = model;
      this.asofUtc = asofUtc;
      this.runtimeUtc = runtimeUtc;
      this.targetDateLocal = targetDateLocal;
      this.variableCode = variableCode;
      this.valueMin = valueMin;
      this.valueMax = valueMax;
      this.valueMean = valueMean;
      this.valueMedian = valueMedian;
      this.sampleCount = sampleCount;
      this.firstForecastTimeUtc = firstForecastTimeUtc;
      this.lastForecastTimeUtc = lastForecastTimeUtc;
      this.rawPayloadHashRef = rawPayloadHashRef;
      this.retrievedAtUtc = retrievedAtUtc;
    }

    public Instant getRuntimeUtc() {
      return runtimeUtc;
    }

    public LocalDate getTargetDateLocal() {
      return targetDateLocal;
    }

    public String getVariableCode() {
      return variableCode;
    }
  }
}
