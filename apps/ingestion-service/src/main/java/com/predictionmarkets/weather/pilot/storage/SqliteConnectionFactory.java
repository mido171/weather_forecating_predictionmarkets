package com.predictionmarkets.weather.pilot.storage;

import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import org.springframework.stereotype.Component;

@Component
public class SqliteConnectionFactory {
  private final PilotIngestionProperties properties;

  public SqliteConnectionFactory(PilotIngestionProperties properties) {
    this.properties = properties;
  }

  public Path databasePath() {
    return Path.of(properties.getSqliteRoot(), properties.getSqliteFileName());
  }

  public Connection openConnection() throws SQLException {
    try {
      Files.createDirectories(databasePath().getParent());
    } catch (Exception ex) {
      throw new IllegalStateException("Failed to create SQLite parent directory", ex);
    }
    Connection connection = DriverManager.getConnection("jdbc:sqlite:" + databasePath());
    try (var statement = connection.createStatement()) {
      statement.execute("PRAGMA journal_mode=WAL");
      statement.execute("PRAGMA synchronous=NORMAL");
      statement.execute("PRAGMA foreign_keys=OFF");
      statement.execute("PRAGMA busy_timeout=10000");
    }
    return connection;
  }
}
