package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import com.predictionmarkets.weather.common.http.HttpRetryPolicy;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * GribStream history API contract (hardcoded):
 * - URL: POST https://gribstream.com/api/v2/{model}/history
 * - Headers: Content-Type: application/json, Authorization: Bearer &lt;token&gt;,
 *   Accept: application/ndjson, Accept-Encoding: gzip (recommended)
 * - Body fields used: fromTime, untilTime, asOf, minHorizon, maxHorizon, coordinates,
 *   variables, optional members
 * - Response fields used: forecasted_at, forecasted_time, and variable alias (tmpk)
 */
@Service
public class GribstreamClient {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamClient.class);
  private static final MediaType JSON_MEDIA = MediaType.parse("application/json");
  private static final int SNIPPET_LIMIT = 500;

  private final OkHttpClient httpClient;
  private final HttpUrl baseUrl;
  private final String authHeader;
  private final String acceptHeader;
  private final boolean gzipEnabled;
  private final HttpRetryPolicy retryPolicy;
  private final ObjectMapper objectMapper;

  public GribstreamClient(GribstreamProperties properties,
                          ObjectMapper objectMapper,
                          HttpClientSettings httpClientSettings) {
    Objects.requireNonNull(properties, "properties is required");
    this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper is required");
    String apiToken = normalizeToken(properties.getApiToken());
    String authScheme = normalizeAuthScheme(properties.getAuthScheme());
    this.authHeader = buildAuthorizationHeader(apiToken, authScheme);
    this.acceptHeader = properties.getDefaultAccept();
    this.gzipEnabled = properties.isGzip();
    this.retryPolicy = Objects.requireNonNull(httpClientSettings, "httpClientSettings is required")
        .retryPolicy();
    this.baseUrl = parseBaseUrl(properties.getBaseUrl());
    this.httpClient = new OkHttpClient.Builder()
        .connectTimeout(properties.getConnectTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
        .readTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
        .callTimeout(Duration.ofMillis(properties.getReadTimeoutMillis()))
        .writeTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build();
    String tokenFingerprint = Hashing.sha256Hex(apiToken);
    logger.info("[GRIBSTREAM] Auth token loaded OK. tokenLen={} tokenSha256Prefix={}",
        apiToken.length(),
        tokenFingerprint.substring(0, 12));
    logger.debug("Gribstream client configured baseUrl={} tokenSha256Prefix={} tokenLength={}",
        properties.getBaseUrl(),
        Hashing.sha256Hex(apiToken).substring(0, 12),
        apiToken.length());
  }

  public GribstreamClientResponse fetchHistory(String modelCode, GribstreamHistoryRequest request) {
    if (modelCode == null || modelCode.isBlank()) {
      throw new IllegalArgumentException("modelCode is required");
    }
    Objects.requireNonNull(request, "request is required");
    String requestJson = serializeRequest(request, modelCode);
    String requestSha256 = Hashing.sha256Hex(requestJson);
    byte[] responseBytes = executeRequest(modelCode, requestJson, requestSha256);
    String responseSha256 = Hashing.sha256Hex(responseBytes);
    Instant retrievedAtUtc = Instant.now();
    return new GribstreamClientResponse(
        requestJson,
        requestSha256,
        responseSha256,
        retrievedAtUtc,
        GribstreamResponseParser.parseRows(objectMapper, responseBytes, modelCode, requestSha256));
  }

  private String serializeRequest(GribstreamHistoryRequest request, String modelCode) {
    try {
      return objectMapper.writeValueAsString(request);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize Gribstream request for model " + modelCode, ex);
    }
  }

  private byte[] executeRequest(String modelCode, String requestJson, String requestSha256) {
    return executeWithRetry(modelCode, requestJson, requestSha256);
  }

  private byte[] executeWithRetry(String modelCode, String requestJson, String requestSha256) {
    int maxAttempts = retryPolicy.maxAttempts();
    RuntimeException lastException = null;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        GribstreamHttpResponse response = executeOnce(modelCode, requestJson);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          return response.bodyBytes;
        }
        if (retryPolicy.isRetryableStatus(response.statusCode) && attempt < maxAttempts) {
          long backoffMillis = retryPolicy.computeDelayMillis(attempt);
          logRetry(modelCode, attempt, maxAttempts, response.statusCode, backoffMillis, null);
          sleepBackoff(backoffMillis);
          continue;
        }
        throw new GribstreamResponseException("Gribstream HTTP status " + response.statusCode
            + " model=" + modelCode
            + " requestSha256=" + requestSha256
            + " bodySnippet=" + response.snippet());
      } catch (RuntimeException ex) {
        lastException = ex;
        if (!isRetryableException(ex) || attempt >= maxAttempts) {
          throw ex;
        }
        long backoffMillis = retryPolicy.computeDelayMillis(attempt);
        logRetry(modelCode, attempt, maxAttempts, null, backoffMillis, ex);
        sleepBackoff(backoffMillis);
      }
    }
    throw lastException;
  }

  private GribstreamHttpResponse executeOnce(String modelCode, String requestJson) {
    if (authHeader == null || authHeader.isBlank() || !hasAuthScheme(authHeader)) {
      throw new IllegalStateException(
          "GribStream authorization header is invalid; refusing to send request.");
    }
    HttpUrl url = baseUrl.newBuilder()
        .addPathSegments("api/v2")
        .addPathSegment(modelCode)
        .addPathSegment("history")
        .build();
    RequestBody body = RequestBody.create(requestJson, JSON_MEDIA);
    Request.Builder builder = new Request.Builder()
        .url(url)
        .post(body)
        .header("Authorization", authHeader)
        .header("Accept", acceptHeader)
        .header("Content-Type", "application/json");
    if (gzipEnabled) {
      builder.header("Accept-Encoding", "gzip");
    }
    Request request = builder.build();
    try (Response response = httpClient.newCall(request).execute()) {
      ResponseBody responseBody = response.body();
      byte[] payload = responseBody == null ? new byte[0] : responseBody.bytes();
      return new GribstreamHttpResponse(response.code(), payload);
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
  }

  private static String normalizeToken(String apiToken) {
    if (apiToken == null) {
      throw new IllegalArgumentException("gribstream.apiToken is required");
    }
    String trimmed = apiToken.trim();
    if (trimmed.isEmpty()) {
      throw new IllegalArgumentException("gribstream.apiToken is required");
    }
    if (trimmed.equalsIgnoreCase("<PUT_TOKEN_HERE>")) {
      throw new IllegalArgumentException("gribstream.apiToken must be set (placeholder found)");
    }
    return trimmed;
  }

  private static String buildAuthorizationHeader(String apiToken, String authScheme) {
    String token = apiToken.trim();
    if (token.isEmpty()) {
      throw new IllegalArgumentException("gribstream.apiToken is required");
    }
    if (hasAuthScheme(token)) {
      return token;
    }
    String scheme = normalizeAuthScheme(authScheme);
    return scheme + " " + token;
  }

  private static String normalizeAuthScheme(String authScheme) {
    if (authScheme == null || authScheme.isBlank()) {
      return "Bearer";
    }
    return authScheme.trim();
  }

  private static boolean hasAuthScheme(String headerValue) {
    int spaceIndex = headerValue.indexOf(' ');
    return spaceIndex > 0 && spaceIndex < headerValue.length() - 1;
  }

  private static HttpUrl parseBaseUrl(String baseUrl) {
    HttpUrl parsed = HttpUrl.parse(baseUrl);
    if (parsed == null) {
      throw new IllegalArgumentException("Invalid gribstream.baseUrl: " + baseUrl);
    }
    return parsed;
  }

  private boolean isRetryableException(Throwable ex) {
    if (ex instanceof IOException || ex instanceof UncheckedIOException) {
      return true;
    }
    Throwable cause = ex.getCause();
    if (cause != null && cause != ex) {
      return isRetryableException(cause);
    }
    return false;
  }

  private void logRetry(String modelCode,
                        int attempt,
                        int maxAttempts,
                        Integer statusCode,
                        long backoffMillis,
                        Throwable error) {
    String statusLabel = statusCode == null ? "n/a" : statusCode.toString();
    if (error == null) {
      logger.warn(
          "Gribstream retry {}/{} for model={} (status={}, backoffMs={})",
          attempt,
          maxAttempts,
          modelCode,
          statusLabel,
          backoffMillis);
    } else {
      logger.warn(
          "Gribstream retry {}/{} for model={} (status={}, backoffMs={})",
          attempt,
          maxAttempts,
          modelCode,
          statusLabel,
          backoffMillis,
          error);
    }
  }

  private void sleepBackoff(long backoffMillis) {
    if (backoffMillis <= 0) {
      return;
    }
    try {
      Thread.sleep(backoffMillis);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Gribstream retry interrupted", ex);
    }
  }

  private static final class GribstreamHttpResponse {
    private final int statusCode;
    private final byte[] bodyBytes;

    private GribstreamHttpResponse(int statusCode, byte[] bodyBytes) {
      this.statusCode = statusCode;
      this.bodyBytes = bodyBytes == null ? new byte[0] : bodyBytes;
    }

    private String snippet() {
      String text = new String(bodyBytes, java.nio.charset.StandardCharsets.UTF_8);
      if (text.length() <= SNIPPET_LIMIT) {
        return text;
      }
      return text.substring(0, SNIPPET_LIMIT);
    }
  }
}
