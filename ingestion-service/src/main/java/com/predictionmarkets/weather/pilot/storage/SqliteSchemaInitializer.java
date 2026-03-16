package com.predictionmarkets.weather.pilot.storage;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.SQLException;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

@Component
public class SqliteSchemaInitializer {
  private final SqliteConnectionFactory connectionFactory;
  private volatile boolean initialized;

  public SqliteSchemaInitializer(SqliteConnectionFactory connectionFactory) {
    this.connectionFactory = connectionFactory;
  }

  public void initializeSchema() {
    if (initialized) {
      return;
    }
    synchronized (this) {
      if (initialized) {
        return;
      }
      try (Connection connection = connectionFactory.openConnection()) {
        String sql = new String(
            new ClassPathResource("sqlite/knyc_pilot_schema.sql").getInputStream().readAllBytes(),
            StandardCharsets.UTF_8);
        for (String statement : sql.split(";")) {
          String trimmed = statement.trim();
          if (trimmed.isEmpty()) {
            continue;
          }
          try (var sqlStatement = connection.createStatement()) {
            sqlStatement.execute(trimmed);
          }
        }
        initialized = true;
      } catch (SQLException | IOException ex) {
        throw new IllegalStateException("Failed to initialize SQLite schema", ex);
      }
    }
  }
}
