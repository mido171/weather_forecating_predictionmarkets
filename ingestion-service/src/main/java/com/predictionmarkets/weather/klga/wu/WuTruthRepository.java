package com.predictionmarkets.weather.klga.wu;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.sql.Connection;
import java.sql.Date;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

final class WuTruthRepository implements AutoCloseable {
  private static final String UPSERT_SQL = """
      INSERT INTO public.wunderground_daily_tmax (
        station_id,
        wunderground_station_id,
        local_date,
        timezone_name,
        tmax_f,
        tmin_f,
        observation_count,
        high_observation_times_local_json,
        hourly_observations_json,
        provider_max_temp_values_json,
        provider_min_temp_values_json,
        source_url_redacted,
        wu_page_url,
        payload_hash,
        parser_version,
        fetched_at_utc,
        settlement_available_at_utc,
        daily_high_source,
        validation_status,
        validation_notes_json,
        fetch_status,
        http_status,
        error_type,
        error_message,
        attempts,
        updated_at_utc
      ) VALUES (
        ?, ?, ?, ?, ?, ?, ?,
        ?::jsonb, ?::jsonb, ?::jsonb, ?::jsonb,
        ?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?, ?, ?, ?, ?, now()
      )
      ON CONFLICT (station_id, local_date)
      DO UPDATE SET
        wunderground_station_id = EXCLUDED.wunderground_station_id,
        timezone_name = EXCLUDED.timezone_name,
        tmax_f = EXCLUDED.tmax_f,
        tmin_f = EXCLUDED.tmin_f,
        observation_count = EXCLUDED.observation_count,
        high_observation_times_local_json = EXCLUDED.high_observation_times_local_json,
        hourly_observations_json = EXCLUDED.hourly_observations_json,
        provider_max_temp_values_json = EXCLUDED.provider_max_temp_values_json,
        provider_min_temp_values_json = EXCLUDED.provider_min_temp_values_json,
        source_url_redacted = EXCLUDED.source_url_redacted,
        wu_page_url = EXCLUDED.wu_page_url,
        payload_hash = EXCLUDED.payload_hash,
        parser_version = EXCLUDED.parser_version,
        fetched_at_utc = EXCLUDED.fetched_at_utc,
        settlement_available_at_utc = EXCLUDED.settlement_available_at_utc,
        daily_high_source = EXCLUDED.daily_high_source,
        validation_status = EXCLUDED.validation_status,
        validation_notes_json = EXCLUDED.validation_notes_json,
        fetch_status = EXCLUDED.fetch_status,
        http_status = EXCLUDED.http_status,
        error_type = EXCLUDED.error_type,
        error_message = EXCLUDED.error_message,
        attempts = EXCLUDED.attempts,
        updated_at_utc = now()
      """;

  private final Connection connection;
  private final ObjectMapper objectMapper;

  WuTruthRepository(WuTruthConfig config, ObjectMapper objectMapper) throws SQLException {
    this.objectMapper = objectMapper;
    Properties properties = new Properties();
    if (config.dbUser() != null) {
      properties.setProperty("user", config.dbUser());
    }
    if (config.dbPassword() != null) {
      properties.setProperty("password", config.dbPassword());
    }
    this.connection = DriverManager.getConnection(config.jdbcUrl(), properties);
    this.connection.setAutoCommit(false);
  }

  void ensureTableExists() throws SQLException {
    try (PreparedStatement statement = connection.prepareStatement("SELECT to_regclass('public.wunderground_daily_tmax')")) {
      try (ResultSet rs = statement.executeQuery()) {
        if (!rs.next() || rs.getString(1) == null) {
          throw new IllegalStateException("public.wunderground_daily_tmax does not exist; run klga-tmax db migrate first");
        }
      }
    }
  }

  int upsertRows(List<WuTruthDailyRow> rows) throws SQLException {
    if (rows.isEmpty()) {
      return 0;
    }
    try (PreparedStatement statement = connection.prepareStatement(UPSERT_SQL)) {
      for (WuTruthDailyRow row : rows) {
        bindRow(statement, row);
        statement.addBatch();
      }
      int[] counts = statement.executeBatch();
      connection.commit();
      int total = 0;
      for (int count : counts) {
        if (count > 0) {
          total += count;
        }
      }
      return total;
    } catch (SQLException ex) {
      connection.rollback();
      throw ex;
    }
  }

  ObjectNode loadDay(String stationId, LocalDate localDate) throws SQLException {
    String sql = """
        SELECT
          station_id,
          wunderground_station_id,
          local_date,
          timezone_name,
          tmax_f,
          tmin_f,
          observation_count,
          high_observation_times_local_json,
          hourly_observations_json,
          provider_max_temp_values_json,
          provider_min_temp_values_json,
          source_url_redacted,
          wu_page_url,
          payload_hash,
          parser_version,
          fetched_at_utc,
          settlement_available_at_utc,
          daily_high_source,
          validation_status,
          validation_notes_json,
          fetch_status,
          http_status,
          error_type,
          error_message,
          attempts
        FROM public.wunderground_daily_tmax
        WHERE station_id = ?
          AND local_date = ?
        """;
    try (PreparedStatement statement = connection.prepareStatement(sql)) {
      statement.setString(1, stationId);
      statement.setDate(2, Date.valueOf(localDate));
      try (ResultSet rs = statement.executeQuery()) {
        if (!rs.next()) {
          throw new IllegalStateException("No WU truth row for " + stationId + " " + localDate);
        }
        return rowToJson(rs);
      }
    }
  }

  List<ObjectNode> loadValidationSample(int sampleSize, long seed) throws SQLException {
    String sql = """
        SELECT
          station_id,
          wunderground_station_id,
          local_date,
          timezone_name,
          tmax_f,
          tmin_f,
          observation_count,
          high_observation_times_local_json,
          hourly_observations_json,
          provider_max_temp_values_json,
          provider_min_temp_values_json,
          source_url_redacted,
          wu_page_url,
          payload_hash,
          parser_version,
          fetched_at_utc,
          settlement_available_at_utc,
          daily_high_source,
          validation_status,
          validation_notes_json,
          fetch_status,
          http_status,
          error_type,
          error_message,
          attempts
        FROM public.wunderground_daily_tmax
        WHERE validation_status IN ('accepted','manual_confirmed')
        ORDER BY md5(station_id || ':' || local_date::text || ':' || ?::text)
        LIMIT ?
        """;
    List<ObjectNode> rows = new ArrayList<>();
    try (PreparedStatement statement = connection.prepareStatement(sql)) {
      statement.setLong(1, seed);
      statement.setInt(2, sampleSize);
      try (ResultSet rs = statement.executeQuery()) {
        while (rs.next()) {
          rows.add(rowToJson(rs));
        }
      }
    }
    return rows;
  }

  Set<LocalDate> loadCompletedDates(String stationId, LocalDate startDate, LocalDate endDate) throws SQLException {
    String sql = """
        SELECT local_date
        FROM public.wunderground_daily_tmax
        WHERE station_id = ?
          AND local_date BETWEEN ? AND ?
          AND validation_status IN ('accepted','manual_confirmed','suspect','no_data')
        """;
    Set<LocalDate> dates = new HashSet<>();
    try (PreparedStatement statement = connection.prepareStatement(sql)) {
      statement.setString(1, stationId);
      statement.setDate(2, Date.valueOf(startDate));
      statement.setDate(3, Date.valueOf(endDate));
      try (ResultSet rs = statement.executeQuery()) {
        while (rs.next()) {
          dates.add(rs.getDate("local_date").toLocalDate());
        }
      }
    }
    return dates;
  }

  ObjectNode coverageSummary() throws SQLException {
    ObjectNode root = objectMapper.createObjectNode();
    String sql = """
        SELECT
          station_id,
          count(*) AS row_count,
          min(local_date) AS min_date,
          max(local_date) AS max_date,
          count(*) FILTER (WHERE validation_status IN ('accepted','manual_confirmed')) AS accepted_rows,
          count(*) FILTER (WHERE validation_status = 'manual_confirmed') AS manual_confirmed_rows,
          count(*) FILTER (WHERE validation_status = 'suspect') AS suspect_rows,
          count(*) FILTER (WHERE validation_status = 'no_data') AS no_data_rows,
          count(*) FILTER (WHERE validation_status = 'fetch_failed') AS fetch_failed_rows
        FROM public.wunderground_daily_tmax
        GROUP BY station_id
        ORDER BY station_id
        """;
    var array = root.putArray("stations");
    try (PreparedStatement statement = connection.prepareStatement(sql);
         ResultSet rs = statement.executeQuery()) {
      while (rs.next()) {
        ObjectNode row = array.addObject();
        row.put("station_id", rs.getString("station_id"));
        row.put("row_count", rs.getLong("row_count"));
        row.put("min_date", rs.getString("min_date"));
        row.put("max_date", rs.getString("max_date"));
        row.put("accepted_rows", rs.getLong("accepted_rows"));
        row.put("manual_confirmed_rows", rs.getLong("manual_confirmed_rows"));
        row.put("suspect_rows", rs.getLong("suspect_rows"));
        row.put("no_data_rows", rs.getLong("no_data_rows"));
        row.put("fetch_failed_rows", rs.getLong("fetch_failed_rows"));
      }
    }
    return root;
  }

  @Override
  public void close() throws SQLException {
    connection.close();
  }

  private void bindRow(PreparedStatement statement, WuTruthDailyRow row) throws SQLException {
    int index = 1;
    statement.setString(index++, row.stationId());
    statement.setString(index++, row.wundergroundStationId());
    statement.setDate(index++, Date.valueOf(row.localDate()));
    statement.setString(index++, row.timezoneName());
    setInteger(statement, index++, row.tmaxF());
    setInteger(statement, index++, row.tminF());
    statement.setInt(index++, row.observationCount());
    statement.setString(index++, row.highObservationTimesLocalJson().toString());
    statement.setString(index++, row.hourlyObservationsJson().toString());
    statement.setString(index++, row.providerMaxTempValuesJson().toString());
    statement.setString(index++, row.providerMinTempValuesJson().toString());
    statement.setString(index++, row.sourceUrlRedacted());
    statement.setString(index++, row.wuPageUrl());
    statement.setString(index++, row.payloadHash());
    statement.setString(index++, row.parserVersion());
    statement.setTimestamp(index++, Timestamp.from(row.fetchedAtUtc()));
    statement.setTimestamp(index++, Timestamp.from(row.settlementAvailableAtUtc()));
    statement.setString(index++, row.dailyHighSource());
    statement.setString(index++, row.validationStatus());
    statement.setString(index++, row.validationNotesJson().toString());
    statement.setString(index++, row.fetchStatus());
    setInteger(statement, index++, row.httpStatus());
    statement.setString(index++, row.errorType());
    statement.setString(index++, row.errorMessage());
    statement.setInt(index, row.attempts());
  }

  private void setInteger(PreparedStatement statement, int index, Integer value) throws SQLException {
    if (value == null) {
      statement.setNull(index, java.sql.Types.INTEGER);
    } else {
      statement.setInt(index, value);
    }
  }

  private ObjectNode rowToJson(ResultSet rs) throws SQLException {
    ObjectNode row = objectMapper.createObjectNode();
    row.put("station_id", rs.getString("station_id"));
    row.put("wunderground_station_id", rs.getString("wunderground_station_id"));
    row.put("local_date", rs.getString("local_date"));
    row.put("timezone_name", rs.getString("timezone_name"));
    putNullableInt(row, "tmax_f", rs, "tmax_f");
    putNullableInt(row, "tmin_f", rs, "tmin_f");
    row.put("observation_count", rs.getInt("observation_count"));
    putJson(row, "high_observation_times_local_json", rs.getString("high_observation_times_local_json"));
    putJson(row, "hourly_observations_json", rs.getString("hourly_observations_json"));
    putJson(row, "provider_max_temp_values_json", rs.getString("provider_max_temp_values_json"));
    putJson(row, "provider_min_temp_values_json", rs.getString("provider_min_temp_values_json"));
    row.put("source_url_redacted", rs.getString("source_url_redacted"));
    row.put("wu_page_url", rs.getString("wu_page_url"));
    row.put("payload_hash", rs.getString("payload_hash"));
    row.put("parser_version", rs.getString("parser_version"));
    row.put("fetched_at_utc", String.valueOf(rs.getTimestamp("fetched_at_utc").toInstant()));
    row.put("settlement_available_at_utc", String.valueOf(rs.getTimestamp("settlement_available_at_utc").toInstant()));
    row.put("daily_high_source", rs.getString("daily_high_source"));
    row.put("validation_status", rs.getString("validation_status"));
    putJson(row, "validation_notes_json", rs.getString("validation_notes_json"));
    row.put("fetch_status", rs.getString("fetch_status"));
    putNullableInt(row, "http_status", rs, "http_status");
    row.put("error_type", rs.getString("error_type"));
    row.put("error_message", rs.getString("error_message"));
    row.put("attempts", rs.getInt("attempts"));
    return row;
  }

  private void putNullableInt(ObjectNode node, String field, ResultSet rs, String column) throws SQLException {
    int value = rs.getInt(column);
    if (rs.wasNull()) {
      node.putNull(field);
    } else {
      node.put(field, value);
    }
  }

  private void putJson(ObjectNode node, String field, String value) {
    try {
      JsonNode parsed = value == null ? objectMapper.nullNode() : objectMapper.readTree(value);
      node.set(field, parsed);
    } catch (Exception ex) {
      node.put(field, value);
    }
  }
}
