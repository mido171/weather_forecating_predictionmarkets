package com.predictionmarkets.weather.klga.wu;

import java.io.IOException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;

final class WuTruthFetchClient {
  private static final Set<Integer> RETRYABLE_STATUS_CODES = Set.of(429, 500, 502, 503, 504);
  private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.BASIC_ISO_DATE;
  private static final DateTimeFormatter RETRY_AFTER_HTTP_DATE = DateTimeFormatter.RFC_1123_DATE_TIME;

  private final WuTruthConfig config;
  private final OkHttpClient httpClient;
  private final SharedRateLimiter rateLimiter;

  WuTruthFetchClient(WuTruthConfig config) {
    this.config = config;
    this.httpClient = new OkHttpClient.Builder()
        .connectTimeout(config.timeoutMillis(), TimeUnit.MILLISECONDS)
        .readTimeout(config.timeoutMillis(), TimeUnit.MILLISECONDS)
        .writeTimeout(config.timeoutMillis(), TimeUnit.MILLISECONDS)
        .callTimeout(config.timeoutMillis() * 2L, TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build();
    this.rateLimiter = new SharedRateLimiter(config.rateLimitPerMinute());
  }

  WuTruthFetchResult fetch(WuTruthStation station, LocalDate startDate, LocalDate endDate) {
    config.requireApiKey();
    int maxAttempts = Math.max(1, config.maxRetries() + 1);
    String redactedUrl = buildUrl(station, startDate, endDate, "REDACTED").toString();
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      rateLimiter.acquire();
      Request request = new Request.Builder()
          .url(buildUrl(station, startDate, endDate, config.apiKey()))
          .header("Accept", "application/json")
          .header("User-Agent", "klga-wu-truth-table/1.0")
          .get()
          .build();
      try (Response response = httpClient.newCall(request).execute()) {
        int httpStatus = response.code();
        String body = responseBody(response);
        Instant fetchedAt = Instant.now();
        if (httpStatus >= 200 && httpStatus < 300) {
          return new WuTruthFetchResult(
              station, startDate, endDate, true, httpStatus, body, null, null, fetchedAt, attempt, redactedUrl);
        }
        if (RETRYABLE_STATUS_CODES.contains(httpStatus) && attempt < maxAttempts) {
          sleepBeforeRetry(attempt, parseRetryAfterMillis(response));
          continue;
        }
        return new WuTruthFetchResult(
            station,
            startDate,
            endDate,
            false,
            httpStatus,
            body,
            "HTTP_" + httpStatus,
            abbreviate(body, 2000),
            fetchedAt,
            attempt,
            redactedUrl);
      } catch (IOException ex) {
        if (attempt < maxAttempts) {
          sleepBeforeRetry(attempt, null);
          continue;
        }
        return new WuTruthFetchResult(
            station,
            startDate,
            endDate,
            false,
            0,
            null,
            "IO_EXCEPTION",
            abbreviate(ex.getMessage(), 1000),
            Instant.now(),
            attempt,
            redactedUrl);
      }
    }
    return new WuTruthFetchResult(
        station,
        startDate,
        endDate,
        false,
        0,
        null,
        "UNKNOWN",
        "Request attempts exhausted unexpectedly",
        Instant.now(),
        maxAttempts,
        redactedUrl);
  }

  String fetchPage(String url) {
    Request request = new Request.Builder()
        .url(url)
        .header("Accept", "text/html,application/xhtml+xml")
        .header("User-Agent", "klga-wu-truth-table-validation/1.0")
        .get()
        .build();
    try (Response response = httpClient.newCall(request).execute()) {
      return response.code() + ":" + abbreviate(responseBody(response), 500);
    } catch (IOException ex) {
      return "ERROR:" + abbreviate(ex.getMessage(), 500);
    }
  }

  private HttpUrl buildUrl(WuTruthStation station, LocalDate startDate, LocalDate endDate, String apiKey) {
    HttpUrl base = HttpUrl.parse(config.baseUrl());
    if (base == null) {
      throw new IllegalArgumentException("Invalid Weather.com base URL: " + config.baseUrl());
    }
    return base.newBuilder()
        .addPathSegments("v1/location")
        .addPathSegment(station.weatherComLocationId())
        .addPathSegments("observations/historical.json")
        .addQueryParameter("apiKey", apiKey)
        .addQueryParameter("units", "e")
        .addQueryParameter("startDate", DATE_FORMAT.format(startDate))
        .addQueryParameter("endDate", DATE_FORMAT.format(endDate))
        .build();
  }

  private String responseBody(Response response) throws IOException {
    ResponseBody body = response.body();
    return body == null ? "" : body.string();
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
    long base = 500L;
    long max = 10_000L;
    long exponential = base * (1L << Math.min(20, Math.max(0, attempt - 1)));
    long jitter = ThreadLocalRandom.current().nextLong(0, 251);
    return Math.min(max, exponential + jitter);
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
      return Math.max(0L, Long.parseLong(trimmed) * 1000L);
    } catch (NumberFormatException ignored) {
      // Continue to HTTP-date parse.
    }
    try {
      long millis = ZonedDateTime.parse(trimmed, RETRY_AFTER_HTTP_DATE).toInstant().toEpochMilli()
          - System.currentTimeMillis();
      return Math.max(0L, millis);
    } catch (RuntimeException ignored) {
      return null;
    }
  }

  private String abbreviate(String value, int maxLen) {
    if (value == null || value.length() <= maxLen) {
      return value;
    }
    return value.substring(0, maxLen);
  }

  private static final class SharedRateLimiter {
    private final long minSpacingNanos;
    private long nextAllowedNanos;

    SharedRateLimiter(int permitsPerMinute) {
      double perMinute = Math.max(1, permitsPerMinute);
      this.minSpacingNanos = Math.max(1L, (long) (60_000_000_000L / perMinute));
      this.nextAllowedNanos = 0L;
    }

    synchronized void acquire() {
      long now = System.nanoTime();
      long waitNanos = nextAllowedNanos - now;
      if (waitNanos > 0L) {
        try {
          TimeUnit.NANOSECONDS.sleep(waitNanos);
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          throw new IllegalStateException("Interrupted while rate limiting Weather.com calls", interrupted);
        }
        now = System.nanoTime();
      }
      nextAllowedNanos = Math.max(now, nextAllowedNanos) + minSpacingNanos;
    }
  }
}
