package com.predictionmarkets.weather.weathercom.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.weathercom.client.dto.WeatherComHistoricalResponse;
import com.predictionmarkets.weather.weathercom.config.WeatherComProperties;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.springframework.stereotype.Service;

@Service
public class WeatherComClient {
  private static final Set<Integer> RETRYABLE_STATUS_CODES = Set.of(429, 500, 502, 503, 504);
  private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.BASIC_ISO_DATE;
  private static final DateTimeFormatter RETRY_AFTER_HTTP_DATE = DateTimeFormatter.RFC_1123_DATE_TIME;

  private final HttpUrl baseUrl;
  private final OkHttpClient httpClient;
  private final ObjectMapper objectMapper;
  private final WeatherComProperties properties;
  private final WeatherComRateLimiter rateLimiter;

  public WeatherComClient(ObjectMapper objectMapper, WeatherComProperties properties) {
    this.objectMapper = objectMapper;
    this.properties = properties;
    HttpUrl parsedBase = HttpUrl.parse(properties.getApi().getBaseUrl());
    if (parsedBase == null) {
      throw new IllegalArgumentException("Invalid weathercom.api.base-url");
    }
    this.baseUrl = parsedBase;
    this.httpClient = new OkHttpClient.Builder()
        .connectTimeout(Math.max(1, properties.getApi().getConnectTimeoutMs()), TimeUnit.MILLISECONDS)
        .readTimeout(Math.max(1, properties.getApi().getReadTimeoutMs()), TimeUnit.MILLISECONDS)
        .writeTimeout(Math.max(1, properties.getApi().getReadTimeoutMs()), TimeUnit.MILLISECONDS)
        .callTimeout(Math.max(1, properties.getApi().getReadTimeoutMs()), TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build();
    this.rateLimiter = new WeatherComRateLimiter(
        properties.getIngestion().getRateLimit().getPermitsPerSecond());
  }

  public WeatherComClientResult fetchHistoricalObservations(String locationId,
                                                            String units,
                                                            LocalDate startDate,
                                                            LocalDate endDate) {
    String normalizedLocationId = requireLocationId(locationId);
    String normalizedUnits = normalizeUnits(units);
    if (startDate == null || endDate == null) {
      throw new IllegalArgumentException("startDate and endDate are required");
    }
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("endDate must be >= startDate");
    }
    String apiKey = properties.getApi().getApiKey();
    if (apiKey == null || apiKey.isBlank()) {
      return new WeatherComClientResult(
          false,
          0,
          null,
          null,
          "MISSING_API_KEY",
          "weathercom.api.api-key is not configured",
          Instant.now(),
          0,
          0);
    }

    int maxAttempts = Math.max(1, properties.getIngestion().getMaxRetries() + 1);
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      rateLimiter.acquire();
      Request request = buildRequest(normalizedLocationId, normalizedUnits, startDate, endDate, apiKey);
      long startNanos = System.nanoTime();
      try (Response response = httpClient.newCall(request).execute()) {
        int durationMs = elapsedMs(startNanos);
        Instant fetchedAtUtc = Instant.now();
        int status = response.code();
        String body = readResponseBody(response);
        if (status >= 200 && status < 300) {
          try {
            WeatherComHistoricalResponse payload =
                objectMapper.readValue(body, WeatherComHistoricalResponse.class);
            return new WeatherComClientResult(
                true,
                status,
                payload,
                body,
                null,
                null,
                fetchedAtUtc,
                durationMs,
                attempt);
          } catch (IOException ex) {
            return new WeatherComClientResult(
                false,
                status,
                null,
                body,
                "JSON_PARSE_ERROR",
                abbreviate(ex.getMessage(), 1000),
                fetchedAtUtc,
                durationMs,
                attempt);
          }
        }
        if (isRetryableStatus(status) && attempt < maxAttempts) {
          sleepBeforeRetry(attempt, parseRetryAfterMillis(response));
          continue;
        }
        return new WeatherComClientResult(
            false,
            status,
            null,
            body,
            "HTTP_" + status,
            abbreviate(body, 2000),
            fetchedAtUtc,
            durationMs,
            attempt);
      } catch (IOException ex) {
        int durationMs = elapsedMs(startNanos);
        if (attempt < maxAttempts) {
          sleepBeforeRetry(attempt, null);
          continue;
        }
        return new WeatherComClientResult(
            false,
            0,
            null,
            null,
            "IO_EXCEPTION",
            abbreviate(ex.getMessage(), 1000),
            Instant.now(),
            durationMs,
            attempt);
      }
    }
    return new WeatherComClientResult(
        false,
        0,
        null,
        null,
        "UNKNOWN",
        "Request attempts exhausted unexpectedly",
        Instant.now(),
        0,
        maxAttempts);
  }

  private Request buildRequest(String locationId,
                               String units,
                               LocalDate startDate,
                               LocalDate endDate,
                               String apiKey) {
    HttpUrl url = baseUrl.newBuilder()
        .addPathSegments("v1/location")
        .addPathSegment(locationId)
        .addPathSegments("observations/historical.json")
        .addQueryParameter("apiKey", apiKey)
        .addQueryParameter("units", units)
        .addQueryParameter("startDate", DATE_FORMATTER.format(startDate))
        .addQueryParameter("endDate", DATE_FORMATTER.format(endDate))
        .build();
    Request.Builder builder = new Request.Builder()
        .url(url)
        .get()
        .header("Accept", "application/json")
        .header("Accept-Encoding", "gzip");
    String userAgent = properties.getApi().getUserAgent();
    if (userAgent != null && !userAgent.isBlank()) {
      builder.header("User-Agent", userAgent.trim());
    }
    return builder.build();
  }

  private boolean isRetryableStatus(int statusCode) {
    return RETRYABLE_STATUS_CODES.contains(statusCode);
  }

  private void sleepBeforeRetry(int attempt, Long retryAfterMillis) {
    long delayMs = retryAfterMillis == null ? computeBackoffMillis(attempt) : retryAfterMillis;
    if (delayMs <= 0L) {
      return;
    }
    try {
      Thread.sleep(delayMs);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private long computeBackoffMillis(int attempt) {
    long base = Math.max(1L, properties.getIngestion().getRetryBackoffMs());
    long maxBackoff = Math.max(base, properties.getIngestion().getMaxBackoffMs());
    int exponent = Math.min(30, Math.max(0, attempt - 1));
    long exponential = base * (1L << exponent);
    long capped = Math.min(exponential, maxBackoff);
    long jitterMax = Math.max(0L, properties.getIngestion().getRetryJitterMs());
    long jitter = jitterMax == 0L ? 0L : ThreadLocalRandom.current().nextLong(0, jitterMax + 1);
    return Math.min(maxBackoff, capped + jitter);
  }

  private Long parseRetryAfterMillis(Response response) {
    if (response == null || response.code() != 429) {
      return null;
    }
    String retryAfter = response.header("Retry-After");
    if (retryAfter == null || retryAfter.isBlank()) {
      return null;
    }
    String trimmed = retryAfter.trim();
    try {
      long seconds = Long.parseLong(trimmed);
      return Math.max(0L, seconds * 1000L);
    } catch (NumberFormatException ignored) {
      // Continue to HTTP-date parse.
    }
    try {
      ZonedDateTime dateTime = ZonedDateTime.parse(trimmed, RETRY_AFTER_HTTP_DATE);
      long millis = dateTime.toInstant().toEpochMilli() - System.currentTimeMillis();
      return Math.max(0L, millis);
    } catch (Exception ignored) {
      return null;
    }
  }

  private String readResponseBody(Response response) throws IOException {
    ResponseBody body = response.body();
    if (body == null) {
      return "";
    }
    byte[] bytes = body.bytes();
    String encoding = response.header("Content-Encoding");
    if (encoding != null && "gzip".equalsIgnoreCase(encoding.trim())) {
      bytes = gunzip(bytes);
    }
    return new String(bytes, StandardCharsets.UTF_8);
  }

  private byte[] gunzip(byte[] input) throws IOException {
    try (GZIPInputStream gzipInputStream = new GZIPInputStream(new ByteArrayInputStream(input));
         ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
      byte[] buffer = new byte[4096];
      int read;
      while ((read = gzipInputStream.read(buffer)) >= 0) {
        outputStream.write(buffer, 0, read);
      }
      return outputStream.toByteArray();
    }
  }

  private int elapsedMs(long startNanos) {
    long elapsedNanos = Math.max(0L, System.nanoTime() - startNanos);
    long millis = TimeUnit.NANOSECONDS.toMillis(elapsedNanos);
    if (millis > Integer.MAX_VALUE) {
      return Integer.MAX_VALUE;
    }
    return (int) millis;
  }

  private String requireLocationId(String locationId) {
    if (locationId == null || locationId.isBlank()) {
      throw new IllegalArgumentException("locationId is required");
    }
    return locationId.trim();
  }

  private String normalizeUnits(String units) {
    if (units == null || units.isBlank()) {
      throw new IllegalArgumentException("units is required");
    }
    String normalized = units.trim().toLowerCase(Locale.ROOT);
    if (!Set.of("e", "m", "h", "s").contains(normalized)) {
      throw new IllegalArgumentException("units must be one of: e, m, h, s");
    }
    return normalized;
  }

  private String abbreviate(String value, int maxLen) {
    if (value == null || value.length() <= maxLen) {
      return value;
    }
    return value.substring(0, maxLen);
  }
}

