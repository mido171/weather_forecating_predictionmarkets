package com.predictionmarkets.weather.pilot.catalog;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.pilot.config.SourceConfig;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.storage.SqliteConnectionFactory;
import com.predictionmarkets.weather.pilot.storage.SqliteSchemaInitializer;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class SqliteCatalogService {
  private final SqliteConnectionFactory connectionFactory;
  private final SqliteSchemaInitializer schemaInitializer;
  private final ObjectMapper objectMapper;

  public SqliteCatalogService(SqliteConnectionFactory connectionFactory,
                              SqliteSchemaInitializer schemaInitializer,
                              ObjectMapper objectMapper) {
    this.connectionFactory = connectionFactory;
    this.schemaInitializer = schemaInitializer;
    this.objectMapper = objectMapper;
  }

  public void initialize() {
    schemaInitializer.initializeSchema();
  }

  public String nowUtc() {
    return Instant.now().toString();
  }

  public <T> T withConnection(SqlFunction<T> callback) {
    initialize();
    try (Connection connection = connectionFactory.openConnection()) {
      return callback.apply(connection);
    } catch (SQLException ex) {
      throw new IllegalStateException("SQLite operation failed", ex);
    }
  }

  public void inTransaction(SqlConsumer callback) {
    initialize();
    try (Connection connection = connectionFactory.openConnection()) {
      boolean previousAutoCommit = connection.getAutoCommit();
      connection.setAutoCommit(false);
      try {
        callback.accept(connection);
        connection.commit();
      } catch (Exception ex) {
        connection.rollback();
        throw ex;
      } finally {
        connection.setAutoCommit(previousAutoCommit);
      }
    } catch (SQLException ex) {
      throw new IllegalStateException("SQLite transaction failed", ex);
    }
  }

  public void upsertStation(StationConfig stationConfig, Map<String, Object> metadata) {
    String now = nowUtc();
    String metadataJson = toJson(metadata);
    String aliasesJson = toJson(stationConfig.getAliases());
    inTransaction(connection -> {
      try (PreparedStatement statement = connection.prepareStatement("""
          INSERT INTO station_registry (
            station_key, display_name, timezone, latitude, longitude, elevation_m,
            metar_reset_minute, aliases_json, metadata_json, created_at_utc, updated_at_utc
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(station_key) DO UPDATE SET
            display_name=excluded.display_name,
            timezone=excluded.timezone,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            elevation_m=excluded.elevation_m,
            metar_reset_minute=excluded.metar_reset_minute,
            aliases_json=excluded.aliases_json,
            metadata_json=excluded.metadata_json,
            updated_at_utc=excluded.updated_at_utc
          """)) {
        statement.setString(1, stationConfig.getStationKey());
        statement.setString(2, stationConfig.getDisplayName());
        statement.setString(3, stationConfig.getTimezone());
        statement.setDouble(4, stationConfig.getLatitude());
        statement.setDouble(5, stationConfig.getLongitude());
        if (stationConfig.getElevationM() == null) {
          statement.setObject(6, null);
        } else {
          statement.setDouble(6, stationConfig.getElevationM());
        }
        if (stationConfig.getMetarResetMinute() == null) {
          statement.setObject(7, null);
        } else {
          statement.setInt(7, stationConfig.getMetarResetMinute());
        }
        statement.setString(8, aliasesJson);
        statement.setString(9, metadataJson);
        statement.setString(10, now);
        statement.setString(11, now);
        statement.executeUpdate();
      }
    });
  }

  public void upsertSource(SourceConfig sourceConfig) {
    String now = nowUtc();
    inTransaction(connection -> {
      try (PreparedStatement statement = connection.prepareStatement("""
          INSERT INTO source_registry (
            source_name, source_family, enabled, base_url, config_json, created_at_utc, updated_at_utc
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(source_name) DO UPDATE SET
            source_family=excluded.source_family,
            enabled=excluded.enabled,
            base_url=excluded.base_url,
            config_json=excluded.config_json,
            updated_at_utc=excluded.updated_at_utc
          """)) {
        statement.setString(1, sourceConfig.getSourceName());
        statement.setString(2, sourceConfig.getSourceFamily());
        statement.setInt(3, sourceConfig.isEnabled() ? 1 : 0);
        statement.setString(4, sourceConfig.getBaseUrl());
        statement.setString(5, toJson(Map.of(
            "sourceName", sourceConfig.getSourceName(),
            "sourceFamily", sourceConfig.getSourceFamily(),
            "enabled", sourceConfig.isEnabled(),
            "baseUrl", sourceConfig.getBaseUrl())));
        statement.setString(6, now);
        statement.setString(7, now);
        statement.executeUpdate();
      }
    });
  }

  public List<Map<String, Object>> query(String sql, Object... params) {
    return withConnection(connection -> {
      try (PreparedStatement statement = connection.prepareStatement(sql)) {
        bind(statement, params);
        try (ResultSet resultSet = statement.executeQuery()) {
          List<Map<String, Object>> rows = new ArrayList<>();
          int count = resultSet.getMetaData().getColumnCount();
          while (resultSet.next()) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (int index = 1; index <= count; index++) {
              row.put(resultSet.getMetaData().getColumnLabel(index), resultSet.getObject(index));
            }
            rows.add(row);
          }
          return rows;
        }
      }
    });
  }

  public Map<String, Object> querySingle(String sql, Object... params) {
    List<Map<String, Object>> rows = query(sql, params);
    return rows.isEmpty() ? Map.of() : rows.get(0);
  }

  public int execute(String sql, Object... params) {
    return withConnection(connection -> {
      try (PreparedStatement statement = connection.prepareStatement(sql)) {
        bind(statement, params);
        return statement.executeUpdate();
      }
    });
  }

  public String toJson(Object value) {
    if (value == null) {
      return null;
    }
    try {
      return objectMapper.writeValueAsString(value);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize value to JSON", ex);
    }
  }

  private void bind(PreparedStatement statement, Object... params) throws SQLException {
    for (int index = 0; index < params.length; index++) {
      statement.setObject(index + 1, params[index]);
    }
  }

  @FunctionalInterface
  public interface SqlFunction<T> {
    T apply(Connection connection) throws SQLException;
  }

  @FunctionalInterface
  public interface SqlConsumer {
    void accept(Connection connection) throws SQLException;
  }
}
