package com.predictionmarkets.weather.repository;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class WundergroundDailyMaxTemperatureUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO wunderground_ml.wunderground_station_daily_max_temperature (
        request_location_id,
        obs_id,
        station_zoneid,
        target_date_local,
        max_temp_f,
        source_valid_time_gmt,
        source_api_call_id,
        source_type,
        observation_count,
        created_at_utc,
        updated_at_utc
      ) VALUES (
        :requestLocationId,
        :obsId,
        :stationZoneid,
        :targetDateLocal,
        :maxTempF,
        :sourceValidTimeGmt,
        :sourceApiCallId,
        :sourceType,
        :observationCount,
        :createdAtUtc,
        :updatedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        station_zoneid = VALUES(station_zoneid),
        max_temp_f = VALUES(max_temp_f),
        source_valid_time_gmt = VALUES(source_valid_time_gmt),
        source_api_call_id = VALUES(source_api_call_id),
        source_type = VALUES(source_type),
        observation_count = VALUES(observation_count),
        updated_at_utc = VALUES(updated_at_utc)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public WundergroundDailyMaxTemperatureUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int upsertAll(List<UpsertRow> rows) {
    if (rows == null || rows.isEmpty()) {
      return 0;
    }
    SqlParameterSource[] batch = rows.stream()
        .map(WundergroundDailyMaxTemperatureUpsertRepository::toParams)
        .toArray(SqlParameterSource[]::new);
    int[] updated = jdbcTemplate.batchUpdate(UPSERT_SQL, batch);
    int total = 0;
    for (int value : updated) {
      total += value;
    }
    return total;
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("requestLocationId", row.requestLocationId)
        .addValue("obsId", row.obsId)
        .addValue("stationZoneid", row.stationZoneid)
        .addValue("targetDateLocal", row.targetDateLocal)
        .addValue("maxTempF", row.maxTempF)
        .addValue("sourceValidTimeGmt", row.sourceValidTimeGmt)
        .addValue("sourceApiCallId", row.sourceApiCallId)
        .addValue("sourceType", row.sourceType)
        .addValue("observationCount", row.observationCount)
        .addValue("createdAtUtc", row.createdAtUtc)
        .addValue("updatedAtUtc", row.updatedAtUtc);
  }

  public record UpsertRow(
      String requestLocationId,
      String obsId,
      String stationZoneid,
      LocalDate targetDateLocal,
      Integer maxTempF,
      Long sourceValidTimeGmt,
      Long sourceApiCallId,
      String sourceType,
      Integer observationCount,
      Instant createdAtUtc,
      Instant updatedAtUtc
  ) {
  }
}
