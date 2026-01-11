package com.predictionmarkets.weather.common.http;

import io.netty.channel.ChannelOption;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.util.UriBuilder;
import reactor.netty.http.client.HttpClient;

public final class HardenedWebClient {
  private static final Logger LOGGER = LoggerFactory.getLogger(HardenedWebClient.class);

  private final WebClient webClient;
  private final HttpClientSettings settings;
  private final Sleeper sleeper;

  public HardenedWebClient(WebClient.Builder builder, String baseUrl, HttpClientSettings settings) {
    this(builder, baseUrl, settings, Sleeper.defaultSleeper());
  }

  HardenedWebClient(WebClient.Builder builder,
                    String baseUrl,
                    HttpClientSettings settings,
                    Sleeper sleeper) {
    Objects.requireNonNull(builder, "builder must not be null");
    Objects.requireNonNull(baseUrl, "baseUrl must not be null");
    this.settings = Objects.requireNonNull(settings, "settings must not be null");
    this.sleeper = Objects.requireNonNull(sleeper, "sleeper must not be null");
    this.webClient = buildWebClient(builder, baseUrl, settings);
  }

  public byte[] getBytes(String endpoint, String correlationId, Function<UriBuilder, URI> uriBuilder) {
    Objects.requireNonNull(uriBuilder, "uriBuilder must not be null");
    return executeWithRetry("GET", endpoint, correlationId, () -> fetchOnce(uriBuilder));
  }

  private HttpResult fetchOnce(Function<UriBuilder, URI> uriBuilder) {
    Duration readTimeout = settings.readTimeout();
    return webClient.get()
        .uri(uriBuilder)
        .exchangeToMono(response -> response.bodyToMono(byte[].class)
            .defaultIfEmpty(new byte[0])
            .map(body -> new HttpResult(response.statusCode().value(), body)))
        .timeout(readTimeout)
        .block();
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
    if (ex instanceof WebClientRequestException) {
      return true;
    }
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

  private static WebClient buildWebClient(WebClient.Builder builder,
                                          String baseUrl,
                                          HttpClientSettings settings) {
    Duration connectTimeout = settings.connectTimeout();
    Duration readTimeout = settings.readTimeout();
    HttpClient httpClient = HttpClient.create()
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, toMillisInt(connectTimeout))
        .responseTimeout(readTimeout);
    return builder.baseUrl(baseUrl)
        .clientConnector(new ReactorClientHttpConnector(httpClient))
        .build();
  }

  private static int toMillisInt(Duration duration) {
    long millis = duration.toMillis();
    if (millis > Integer.MAX_VALUE) {
      return Integer.MAX_VALUE;
    }
    return (int) millis;
  }

  private record HttpResult(int statusCode, byte[] body) {
  }
}
