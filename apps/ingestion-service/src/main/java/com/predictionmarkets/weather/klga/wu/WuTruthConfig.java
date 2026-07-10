package com.predictionmarkets.weather.klga.wu;

import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

record WuTruthConfig(
    String command,
    LocalDate startDate,
    LocalDate endDate,
    String stationSelection,
    String stationId,
    LocalDate auditDate,
    int workers,
    int chunkDays,
    int sampleSize,
    long seed,
    String apiKey,
    String baseUrl,
    int timeoutMillis,
    int maxRetries,
    int rateLimitPerMinute,
    boolean resume,
    String jdbcUrl,
    String dbUser,
    String dbPassword,
    Path artifactRoot
) {
  static WuTruthConfig fromArgs(String[] args) {
    Map<String, String> parsed = parseArgs(args);
    String command = value(parsed, "command", env("KLGA_WU_COMMAND", null));
    if (command == null || command.isBlank()) {
      throw new IllegalArgumentException("--command is required");
    }
    command = command.trim().toLowerCase(Locale.ROOT);

    LocalDate latestSettled = LocalDate.now(ZoneId.of("America/New_York")).minusDays(1);
    LocalDate startDate = parseDate(value(parsed, "start-date", "1973-01-01"), "start-date");
    LocalDate endDate = parseDate(value(parsed, "end-date", "latest"), "end-date", latestSettled);
    LocalDate auditDate = parseDate(value(parsed, "date", latestSettled.toString()), "date");

    String sqlalchemyUrl = env("KLGA_DB_URL", null);
    DbParts dbParts = parseDbUrl(sqlalchemyUrl);
    String jdbcUrl = value(parsed, "jdbc-url", env("KLGA_WU_JDBC_URL", dbParts.jdbcUrl()));
    String dbUser = value(parsed, "db-user", env("KLGA_WU_DB_USER", dbParts.user()));
    String dbPassword = value(parsed, "db-password", env("KLGA_WU_DB_PASSWORD", dbParts.password()));
    if (jdbcUrl == null || jdbcUrl.isBlank()) {
      throw new IllegalArgumentException("JDBC URL is required via --jdbc-url, KLGA_WU_JDBC_URL, or KLGA_DB_URL");
    }

    String apiKey = value(parsed, "api-key", env("KLGA_WU_API_KEY", env("WUNDERGROUND_API_KEY", env("WEATHERCOM_API_KEY", null))));
    String baseUrl = value(parsed, "base-url", env("KLGA_WU_BASE_URL", env("WUNDERGROUND_API_BASE_URL", "https://api.weather.com")));
    Path artifactRoot = Path.of(value(parsed, "artifact-root", env("KLGA_ARTIFACT_ROOT", "artifacts/klga_tmax")));

    return new WuTruthConfig(
        command,
        startDate,
        endDate,
        value(parsed, "stations", "all"),
        value(parsed, "station", "KLGA").toUpperCase(Locale.ROOT),
        auditDate,
        parseInt(value(parsed, "workers", env("WUNDERGROUND_MAX_WORKERS", "20")), "workers", 1),
        parseInt(value(parsed, "chunk-days", env("WUNDERGROUND_CHUNK_DAYS", "31")), "chunk-days", 1),
        parseInt(value(parsed, "sample-size", "500"), "sample-size", 1),
        parseLong(value(parsed, "seed", "1729"), "seed"),
        apiKey,
        stripTrailingSlash(baseUrl),
        parseInt(value(parsed, "timeout-millis", env("WUNDERGROUND_API_TIMEOUT_MILLIS", "30000")), "timeout-millis", 1),
        parseInt(value(parsed, "max-retries", env("WUNDERGROUND_API_MAX_RETRIES", "5")), "max-retries", 0),
        parseInt(value(parsed, "rate-limit-per-minute", env("WUNDERGROUND_API_RATE_LIMIT_PER_MINUTE", "120")), "rate-limit-per-minute", 1),
        parseBoolean(value(parsed, "resume", "false"), "resume")
            || parseBoolean(value(parsed, "missing-only", "false"), "missing-only"),
        jdbcUrl,
        dbUser,
        dbPassword,
        artifactRoot);
  }

  void requireApiKey() {
    if (apiKey == null || apiKey.isBlank()) {
      throw new IllegalArgumentException("WEATHERCOM_API_KEY or WUNDERGROUND_API_KEY is required");
    }
  }

  private static Map<String, String> parseArgs(String[] args) {
    Map<String, String> parsed = new HashMap<>();
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      if (!arg.startsWith("--")) {
        continue;
      }
      String key = arg.substring(2);
      String value = "true";
      int equalsIndex = key.indexOf('=');
      if (equalsIndex >= 0) {
        value = key.substring(equalsIndex + 1);
        key = key.substring(0, equalsIndex);
      } else if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
        value = args[++i];
      }
      parsed.put(key, value);
    }
    return parsed;
  }

  private static String value(Map<String, String> parsed, String key, String defaultValue) {
    String value = parsed.get(key);
    if (value == null || value.isBlank()) {
      return defaultValue;
    }
    return value;
  }

  private static String env(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isBlank() ? defaultValue : value;
  }

  private static LocalDate parseDate(String value, String name) {
    return parseDate(value, name, null);
  }

  private static LocalDate parseDate(String value, String name, LocalDate latestSettled) {
    if ("latest".equalsIgnoreCase(value)) {
      if (latestSettled == null) {
        throw new IllegalArgumentException(name + " does not accept latest");
      }
      return latestSettled;
    }
    try {
      return LocalDate.parse(value);
    } catch (RuntimeException ex) {
      throw new IllegalArgumentException("--" + name + " must be YYYY-MM-DD or latest", ex);
    }
  }

  private static int parseInt(String value, String name, int minimum) {
    try {
      int parsed = Integer.parseInt(value);
      if (parsed < minimum) {
        throw new IllegalArgumentException("--" + name + " must be >= " + minimum);
      }
      return parsed;
    } catch (NumberFormatException ex) {
      throw new IllegalArgumentException("--" + name + " must be an integer", ex);
    }
  }

  private static long parseLong(String value, String name) {
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException ex) {
      throw new IllegalArgumentException("--" + name + " must be an integer", ex);
    }
  }

  private static boolean parseBoolean(String value, String name) {
    if ("true".equalsIgnoreCase(value) || "1".equals(value) || "yes".equalsIgnoreCase(value)) {
      return true;
    }
    if ("false".equalsIgnoreCase(value) || "0".equals(value) || "no".equalsIgnoreCase(value)) {
      return false;
    }
    throw new IllegalArgumentException("--" + name + " must be a boolean");
  }

  private static String stripTrailingSlash(String value) {
    if (value == null) {
      return null;
    }
    while (value.endsWith("/")) {
      value = value.substring(0, value.length() - 1);
    }
    return value;
  }

  private static DbParts parseDbUrl(String url) {
    if (url == null || url.isBlank()) {
      return new DbParts(null, null, null);
    }
    try {
      String normalized = url.replace("postgresql+psycopg://", "postgresql://");
      URI uri = URI.create(normalized);
      if (!"postgresql".equalsIgnoreCase(uri.getScheme()) && !"postgres".equalsIgnoreCase(uri.getScheme())) {
        return new DbParts(null, null, null);
      }
      String user = null;
      String password = null;
      if (uri.getUserInfo() != null) {
        String[] parts = uri.getUserInfo().split(":", 2);
        user = decode(parts[0]);
        if (parts.length > 1) {
          password = decode(parts[1]);
        }
      }
      int port = uri.getPort() > 0 ? uri.getPort() : 5432;
      String jdbc = "jdbc:postgresql://" + uri.getHost() + ":" + port + uri.getPath();
      if (uri.getQuery() != null && !uri.getQuery().isBlank()) {
        jdbc += "?" + uri.getQuery();
      }
      return new DbParts(jdbc, user, password);
    } catch (RuntimeException ignored) {
      return new DbParts(null, null, null);
    }
  }

  private static String decode(String value) {
    return URLDecoder.decode(value, StandardCharsets.UTF_8);
  }

  private record DbParts(String jdbcUrl, String user, String password) {
  }
}
