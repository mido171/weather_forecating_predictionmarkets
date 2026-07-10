package com.predictionmarkets.weather.common.http;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;
import java.util.function.Supplier;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class HardenedWebClient {
  private static final Logger LOGGER = LoggerFactory.getLogger(HardenedWebClient.class);

  private final ThreadLocal<OkHttpClient> httpClientByThread;
  private final HttpClientSettings settings;
  private final Sleeper sleeper;
  private final HttpUrl baseUrl;

  public HardenedWebClient(String baseUrl, HttpClientSettings settings) {
    this(baseUrl, settings, Sleeper.defaultSleeper());
  }

  HardenedWebClient(String baseUrl,
                    HttpClientSettings settings,
                    Sleeper sleeper) {
    Objects.requireNonNull(baseUrl, "baseUrl must not be null");
    this.baseUrl = parseBaseUrl(baseUrl);
    this.settings = Objects.requireNonNull(settings, "settings must not be null");
    this.sleeper = Objects.requireNonNull(sleeper, "sleeper must not be null");
    this.httpClientByThread = ThreadLocal.withInitial(() -> buildClient(settings));
  }

  public byte[] getBytes(String endpoint,
                         String correlationId,
                         Function<HttpUrl.Builder, HttpUrl> urlBuilder) {
    Objects.requireNonNull(urlBuilder, "urlBuilder must not be null");
    return executeWithRetry("GET", endpoint, correlationId, () -> fetchOnce(urlBuilder));
  }

  private HttpResult fetchOnce(Function<HttpUrl.Builder, HttpUrl> urlBuilder) {
    HttpUrl url = Objects.requireNonNull(urlBuilder.apply(baseUrl.newBuilder()),
        "urlBuilder returned null");
    Request request = new Request.Builder()
        .url(url)
        .get()
        .build();
    try (Response response = client().newCall(request).execute()) {
      ResponseBody body = response.body();
      byte[] payload = body == null ? new byte[0] : body.bytes();
      return new HttpResult(response.code(), payload);
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
  }

  private byte[] executeWithRetry(String method,
                                  String endpoint,
                                  String correlationId,
                                  Supplier<HttpResult> request) {
    HttpRetryPolicy retryPolicy = settings.retryPolicy();
    int maxAttempts = retryPolicy.maxAttempts();
    RuntimeException lastException = null;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        HttpResult result = request.get();
        if (result == null) {
          throw new IllegalStateException("HTTP response was null");
        }
        int status = result.statusCode();
        if (status >= 200 && status < 300) {
          return result.body();
        }
        if (retryPolicy.isRetryableStatus(status) && attempt < maxAttempts) {
          long backoffMillis = retryPolicy.computeDelayMillis(attempt);
          logRetry(method, endpoint, correlationId, attempt, maxAttempts, status, backoffMillis, null);
          sleepBackoff(backoffMillis);
          continue;
        }
        throw new IllegalStateException("HTTP " + method + " " + endpoint + " failed with status " + status);
      } catch (RuntimeException ex) {
        lastException = ex;
        if (!isRetryableException(ex) || attempt >= maxAttempts) {
          throw ex;
        }
        long backoffMillis = retryPolicy.computeDelayMillis(attempt);
        logRetry(method, endpoint, correlationId, attempt, maxAttempts, null, backoffMillis, ex);
        sleepBackoff(backoffMillis);
      }
    }
    throw lastException;
  }

  private void sleepBackoff(long backoffMillis) {
    if (backoffMillis <= 0) {
      return;
    }
    try {
      sleeper.sleep(backoffMillis);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("HTTP retry interrupted", ex);
    }
  }

  private boolean isRetryableException(Throwable ex) {
    if (ex instanceof TimeoutException) {
      return true;
    }
    if (ex instanceof IOException) {
      return true;
    }
    Throwable cause = ex.getCause();
    if (cause != null && cause != ex) {
      return isRetryableException(cause);
    }
    return false;
  }

  private void logRetry(String method,
                        String endpoint,
                        String correlationId,
                        int attempt,
                        int maxAttempts,
                        Integer statusCode,
                        long backoffMillis,
                        Throwable error) {
    String safeEndpoint = endpoint == null || endpoint.isBlank() ? "unknown" : endpoint;
    String safeCorrelation = correlationId == null || correlationId.isBlank() ? "n/a" : correlationId;
    String statusLabel = statusCode == null ? "n/a" : statusCode.toString();
    if (error == null) {
      LOGGER.warn(
          "HTTP retry {}/{} for {} {} (status={}, backoffMs={}, correlationId={})",
          attempt,
          maxAttempts,
          method,
          safeEndpoint,
          statusLabel,
          backoffMillis,
          safeCorrelation);
    } else {
      LOGGER.warn(
          "HTTP retry {}/{} for {} {} (status={}, backoffMs={}, correlationId={})",
          attempt,
          maxAttempts,
          method,
          safeEndpoint,
          statusLabel,
          backoffMillis,
          safeCorrelation,
          error);
    }
  }

  private static OkHttpClient buildClient(HttpClientSettings settings) {
    Duration connectTimeout = settings.connectTimeout();
    Duration readTimeout = settings.readTimeout();
    return new OkHttpClient.Builder()
        .connectTimeout(connectTimeout.toMillis(), TimeUnit.MILLISECONDS)
        .readTimeout(readTimeout.toMillis(), TimeUnit.MILLISECONDS)
        .callTimeout(readTimeout.toMillis(), TimeUnit.MILLISECONDS)
        .writeTimeout(readTimeout.toMillis(), TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build();
  }

  private OkHttpClient client() {
    return httpClientByThread.get();
  }

  private static HttpUrl parseBaseUrl(String baseUrl) {
    HttpUrl parsed = HttpUrl.parse(baseUrl);
    if (parsed == null) {
      throw new IllegalArgumentException("Invalid baseUrl: " + baseUrl);
    }
    return parsed;
  }

  private record HttpResult(int statusCode, byte[] body) {
  }
}
