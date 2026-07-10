package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.config.EvomiProxyProperties;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.zip.GZIPInputStream;
import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringBootVersion;
import org.springframework.stereotype.Service;

/**
 * GribStream forecasts API contract:
 * - URL: POST https://gribstream.com/api/v2/{model}/forecasts
 * - Headers: Content-Type: application/json, Authorization: Bearer &lt;token&gt;,
 *   Accept: text/csv, Accept-Encoding: gzip
 * - Body fields used: forecastedFrom, forecastedUntil, minHorizon, maxHorizon, coordinates,
 *   variables, optional members
 * - Docs: https://gribstream.com/docs (history + forecasts), https://gribstream.com/quickstart
 */
@Service
public class GribstreamForecastClient {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamForecastClient.class);
  private static final MediaType JSON_MEDIA = MediaType.parse("application/json");
  private static final int SNIPPET_LIMIT = 500;
  private static final int MAX_ATTEMPTS = 2;
  private static final Set<Integer> RETRYABLE_STATUS = Set.of(408, 429, 500, 502, 503, 504);

  private final OkHttpClient httpClient;
  private final HttpUrl baseUrl;
  private final String authHeader;
  private final boolean credentialsConfigured;
  private final boolean gzipEnabled;
  private final boolean logHttp;
  private final int logBodyLimit;
  private final int connectTimeoutMillis;
  private final int readTimeoutMillis;
  private final int callTimeoutMillis;
  private final ObjectMapper objectMapper;
  private final boolean evomiProxyEnabled;
  private final String userAgent;

  public GribstreamForecastClient(GribstreamProperties properties,
                                  EvomiProxyProperties evomiProxyProperties,
                                  ObjectMapper objectMapper) {
    Objects.requireNonNull(properties, "properties is required");
    this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper is required");
    String apiToken = normalizeOptionalToken(properties.getApiToken());
    String authScheme = normalizeAuthScheme(properties.getAuthScheme());
    this.authHeader = buildAuthorizationHeader(apiToken, authScheme);
    this.credentialsConfigured = !apiToken.isBlank();
    // Forecasts endpoint always requests gzip per spec.
    this.gzipEnabled = true;
    this.logHttp = properties.isLogHttp();
    this.logBodyLimit = Math.max(0, properties.getLogBodyLimit());
    this.connectTimeoutMillis = properties.getConnectTimeoutMillis();
    this.readTimeoutMillis = properties.getReadTimeoutMillis();
    this.callTimeoutMillis = this.readTimeoutMillis;
    this.baseUrl = parseBaseUrl(properties.getBaseUrl());
    this.evomiProxyEnabled = evomiProxyProperties != null && evomiProxyProperties.isEnabled();
    this.userAgent = buildUserAgent("gribstream-forecast", gzipEnabled, evomiProxyEnabled);
    OkHttpClient.Builder builder = new OkHttpClient.Builder()
        .connectTimeout(connectTimeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)
        .readTimeout(readTimeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)
        .callTimeout(Duration.ofMillis(callTimeoutMillis))
        .writeTimeout(readTimeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        ;
    EvomiProxySupport.applyIfEnabled(builder, evomiProxyProperties, logger);
    this.httpClient = builder.build();
    logger.info("[GRIBSTREAM] Forecast client credentials configured={}", credentialsConfigured);
    logger.debug("Gribstream forecast client configured baseUrl={}", properties.getBaseUrl());
  }

  public GribstreamForecastRawResponse fetchForecastsRaw(String modelCode,
                                                         GribstreamForecastRequest request) {
    requireCredentials();
    if (modelCode == null || modelCode.isBlank()) {
      throw new IllegalArgumentException("modelCode is required");
    }
    Objects.requireNonNull(request, "request is required");
    String requestJson = serializeRequest(request, modelCode);
    String requestSha256 = Hashing.sha256Hex(requestJson);
    GribstreamHttpResponse response = executeWithRetry(modelCode, requestJson, requestSha256, true);
    byte[] bodyBytes = response.bodyBytes;
    String responseSha256 = Hashing.sha256Hex(bodyBytes);
    Instant retrievedAtUtc = Instant.now();
    return new GribstreamForecastRawResponse(
        requestJson,
        requestSha256,
        responseSha256,
        retrievedAtUtc,
        response.statusCode,
        bodyBytes);
  }

  private String serializeRequest(GribstreamForecastRequest request, String modelCode) {
    try {
      return objectMapper.writeValueAsString(request);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize Gribstream request for model " + modelCode, ex);
    }
  }

  private GribstreamHttpResponse executeWithRetry(String modelCode,
                                                  String requestJson,
                                                  String requestSha256,
                                                  boolean allowNon2xx) {
    HttpUrl url = buildUrl(modelCode);
    if (logHttp) {
      logRequest(modelCode, url, requestJson, requestSha256);
    }
    RuntimeException lastException = null;
    for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        GribstreamHttpResponse response =
            executeOnce(modelCode, url, requestJson, requestSha256, attempt);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          if (isBlankPayload(response.bodyBytes)) {
            if (attempt < MAX_ATTEMPTS) {
              long backoffMillis = computeBackoffMillis(attempt);
              logRetry(modelCode, attempt, response.statusCode, backoffMillis, "empty_body", null);
              sleepBackoff(backoffMillis);
              continue;
            }
            throw new GribstreamEmptyResponseException("Gribstream empty response body"
                + " status=" + response.statusCode
                + " model=" + modelCode
                + " requestSha256=" + requestSha256);
          }
          return response;
        }
        if (response.statusCode == 401) {
          logUnauthorized(modelCode, requestSha256);
          if (allowNon2xx) {
            return response;
          }
          throw new IllegalStateException("Gribstream unauthorized (401). Check gribstream.apiToken.");
        }
        if (RETRYABLE_STATUS.contains(response.statusCode) && attempt < MAX_ATTEMPTS) {
          long backoffMillis = computeRetryBackoffMillis(attempt, response);
          logRetry(modelCode, attempt, response.statusCode, backoffMillis, "retryable_status", null);
          sleepBackoff(backoffMillis);
          continue;
        }
        if (allowNon2xx) {
          logFailure(modelCode, requestSha256, response.statusCode);
          return response;
        }
        logFailure(modelCode, requestSha256, response.statusCode);
        throw new GribstreamResponseException("Gribstream HTTP status " + response.statusCode
            + " model=" + modelCode
            + " requestSha256=" + requestSha256
            + " bodySnippet=" + response.snippet());
      } catch (RuntimeException ex) {
        lastException = ex;
        if (!isRetryableException(ex) || attempt >= MAX_ATTEMPTS) {
          logRequestException(modelCode, url, requestSha256, requestJson, attempt, ex);
          throw ex;
        }
        long backoffMillis = computeBackoffMillis(attempt);
        logRetry(modelCode, attempt, null, backoffMillis, ex.getClass().getSimpleName(), ex);
        sleepBackoff(backoffMillis);
      }
    }
    throw lastException;
  }

  private GribstreamHttpResponse executeOnce(String modelCode,
                                             HttpUrl url,
                                             String requestJson,
                                             String requestSha256,
                                             int attempt) {
    if (authHeader == null || authHeader.isBlank() || !hasAuthScheme(authHeader)) {
      throw new IllegalStateException(
          "GribStream authorization header is invalid; refusing to send request.");
    }
    RequestBody body = RequestBody.create(requestJson, JSON_MEDIA);
    Request.Builder builder = new Request.Builder()
        .url(url)
        .post(body)
        .header("Authorization", authHeader)
        .header("User-Agent", userAgent)
        .header("Accept", "text/csv")
        .header("Content-Type", "application/json");
    if (gzipEnabled) {
      builder.header("Accept-Encoding", "gzip");
    } else {
      builder.header("Accept-Encoding", "identity");
    }
    Request request = builder.build();
    try (Response response = httpClient.newCall(request).execute()) {
      ResponseBody responseBody = response.body();
      byte[] rawPayload = responseBody == null ? new byte[0] : responseBody.bytes();
      byte[] decodedPayload = rawPayload;
      if (isGzipEncoded(response) && rawPayload.length > 0) {
        decodedPayload = decompressGzip(rawPayload);
      }
      if (logHttp) {
        logResponse(modelCode, url, requestSha256, attempt, response,
            rawPayload.length, decodedPayload);
      }
      Long retryAfterMillis = parseRetryAfterMillis(response.header("Retry-After"));
      return new GribstreamHttpResponse(response.code(), decodedPayload, retryAfterMillis);
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
  }

  private static String normalizeOptionalToken(String apiToken) {
    if (apiToken == null) {
      return "";
    }
    String trimmed = apiToken.trim();
    if (trimmed.isEmpty()) {
      return "";
    }
    if (trimmed.equalsIgnoreCase("<PUT_TOKEN_HERE>")) {
      throw new IllegalArgumentException("gribstream.apiToken must be set (placeholder found)");
    }
    return trimmed;
  }

  private static String buildAuthorizationHeader(String apiToken, String authScheme) {
    String token = apiToken.trim();
    if (token.isEmpty()) {
      return "";
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

  private void logUnauthorized(String modelCode, String requestSha256) {
    logger.error(
        "[GRIBSTREAM] Unauthorized response model={} requestSha256={} evomiProxyEnabled={}",
        modelCode,
        requestSha256,
        evomiProxyEnabled);
  }

  private void logFailure(String modelCode, String requestSha256, int statusCode) {
    logger.error(
        "[GRIBSTREAM] Forecast request failed model={} status={} requestSha256={} evomiProxyEnabled={}",
        modelCode,
        statusCode,
        requestSha256,
        evomiProxyEnabled);
  }

  private void requireCredentials() {
    if (!credentialsConfigured) {
      throw new IllegalStateException(
          "GRIBSTREAM_API_TOKEN is required before a GribStream network request");
    }
  }

  private void logRequestException(String modelCode,
                                   HttpUrl url,
                                   String requestSha256,
                                   String requestJson,
                                   int attempt,
                                   Throwable error) {
    Throwable root = rootCause(error);
    String payload = limitBody(requestJson);
    logger.error(
        "[GRIBSTREAM] Forecast request error model={} url={} attempt={}/{} requestSha256={} "
            + "connectTimeoutMs={} readTimeoutMs={} callTimeoutMs={} "
            + "rootCause={} message={} payload={}",
        modelCode,
        url,
        attempt,
        MAX_ATTEMPTS,
        requestSha256,
        connectTimeoutMillis,
        readTimeoutMillis,
        callTimeoutMillis,
        root.getClass().getSimpleName(),
        root.getMessage(),
        payload,
        error);
  }

  private static String buildUserAgent(String clientName,
                                       boolean gzipEnabled,
                                       boolean evomiProxyEnabled) {
    String appName = "weather-forecasting-predictionmarkets";
    String appVersion = resolveImplementationVersion(GribstreamForecastClient.class, "dev");
    String profiles = normalizeProperty(System.getProperty("spring.profiles.active"), "default");
    String javaVersion = normalizeProperty(System.getProperty("java.version"), "unknown");
    String javaRuntime = normalizeProperty(System.getProperty("java.runtime.name"), "unknown");
    String javaRuntimeVersion = normalizeProperty(System.getProperty("java.runtime.version"), "");
    String javaVendor = normalizeProperty(System.getProperty("java.vendor"), "unknown");
    String vmName = normalizeProperty(System.getProperty("java.vm.name"), "unknown");
    String vmVersion = normalizeProperty(System.getProperty("java.vm.version"), "unknown");
    String vmVendor = normalizeProperty(System.getProperty("java.vm.vendor"), "unknown");
    String osName = normalizeProperty(System.getProperty("os.name"), "unknown");
    String osVersion = normalizeProperty(System.getProperty("os.version"), "unknown");
    String osArch = normalizeProperty(System.getProperty("os.arch"), "unknown");
    String timezone = normalizeProperty(System.getProperty("user.timezone"), "unknown");
    String language = normalizeProperty(System.getProperty("user.language"), "unknown");
    String country = normalizeProperty(System.getProperty("user.country"), "");
    String springBoot = normalizeProperty(SpringBootVersion.getVersion(), "unknown");
    String okhttpVersion = resolveImplementationVersion(OkHttpClient.class, "unknown");

    StringBuilder ua = new StringBuilder(256);
    ua.append(appName).append('/').append(appVersion)
        .append(" (module=ingestion-service; client=").append(clientName)
        .append("; profiles=").append(profiles)
        .append("; gzip=").append(gzipEnabled ? "on" : "off")
        .append("; proxy=evomi:").append(evomiProxyEnabled ? "on" : "off")
        .append(") ");

    ua.append("Java/").append(javaVersion)
        .append(" (").append(javaRuntime);
    if (!javaRuntimeVersion.isBlank()) {
      ua.append(' ').append(javaRuntimeVersion);
    }
    ua.append("; ").append(javaVendor).append(") ");

    ua.append("VM/").append(vmName).append(' ').append(vmVersion)
        .append(" (").append(vmVendor).append(") ");

    ua.append("OS/").append(osName).append(' ').append(osVersion)
        .append(" (").append(osArch).append(") ");

    ua.append("TZ/").append(timezone)
        .append(" Locale/").append(language);
    if (!country.isBlank()) {
      ua.append('-').append(country);
    }

    ua.append(" SpringBoot/").append(springBoot)
        .append(" OkHttp/").append(okhttpVersion);
    return ua.toString();
  }

  private static String resolveImplementationVersion(Class<?> type, String fallback) {
    if (type == null) {
      return fallback;
    }
    Package pkg = type.getPackage();
    String version = pkg == null ? null : pkg.getImplementationVersion();
    if (version == null || version.isBlank()) {
      return fallback;
    }
    return version;
  }

  private static String normalizeProperty(String value, String fallback) {
    if (value == null) {
      return fallback;
    }
    String trimmed = value.trim();
    if (trimmed.isEmpty()) {
      return fallback;
    }
    return trimmed.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ');
  }

  private static HttpUrl parseBaseUrl(String baseUrl) {
    HttpUrl parsed = HttpUrl.parse(baseUrl);
    if (parsed == null) {
      throw new IllegalArgumentException("Invalid gribstream.baseUrl: " + baseUrl);
    }
    return parsed;
  }

  private HttpUrl buildUrl(String modelCode) {
    return baseUrl.newBuilder()
        .addPathSegments("api/v2")
        .addPathSegment(modelCode)
        .addPathSegment("forecasts")
        .build();
  }

  private void logRequest(String modelCode,
                          HttpUrl url,
                          String requestJson,
                          String requestSha256) {
    byte[] requestBytes = requestJson.getBytes(StandardCharsets.UTF_8);
    String payload = limitBody(requestJson);
    logger.info(
        "[GRIBSTREAM-HTTP] request model={} url={} attempt=1/{} accept={} contentType={} "
            + "acceptEncoding={} auth={} bodyBytes={} requestSha256={} bodyJson={}",
        modelCode,
        url,
        MAX_ATTEMPTS,
        "text/csv",
        JSON_MEDIA,
        gzipEnabled ? "gzip" : "identity",
        maskAuthHeader(authHeader),
        requestBytes.length,
        requestSha256,
        payload);
  }

  private void logResponse(String modelCode,
                           HttpUrl url,
                           String requestSha256,
                           int attempt,
                           Response response,
                           int rawBytesLength,
                           byte[] decodedPayload) {
    String bodySnippet = limitBody(decodedPayload);
    String contentType = response.header("Content-Type");
    String contentEncoding = response.header("Content-Encoding");
    String contentLength = response.header("Content-Length");
    String transferEncoding = response.header("Transfer-Encoding");
    String responseSha = decodedPayload == null ? "null" : Hashing.sha256Hex(decodedPayload);
    int decodedLength = decodedPayload == null ? 0 : decodedPayload.length;
    logger.info(
        "[GRIBSTREAM-HTTP] response model={} url={} attempt={}/{} status={} contentType={} "
            + "contentEncoding={} contentLength={} transferEncoding={} rawBytes={} decodedBytes={} "
            + "requestSha256={} responseSha256={} bodySnippet={}",
        modelCode,
        url,
        attempt,
        MAX_ATTEMPTS,
        response.code(),
        contentType,
        contentEncoding,
        contentLength,
        transferEncoding,
        rawBytesLength,
        decodedLength,
        requestSha256,
        responseSha,
        bodySnippet);
  }

  private String limitBody(String body) {
    if (body == null || body.isEmpty()) {
      return "";
    }
    if (logBodyLimit <= 0 || body.length() <= logBodyLimit) {
      return body;
    }
    return body.substring(0, logBodyLimit) + "...(truncated)";
  }

  private String limitBody(byte[] payload) {
    if (payload == null || payload.length == 0) {
      return "";
    }
    String text = new String(payload, StandardCharsets.UTF_8);
    return limitBody(text);
  }

  private static String maskAuthHeader(String authHeader) {
    if (authHeader == null || authHeader.isBlank()) {
      return "<empty>";
    }
    int spaceIndex = authHeader.indexOf(' ');
    if (spaceIndex < 0 || spaceIndex == authHeader.length() - 1) {
      return "<redacted>";
    }
    String scheme = authHeader.substring(0, spaceIndex).trim();
    String token = authHeader.substring(spaceIndex + 1).trim();
    if (token.length() <= 8) {
      return scheme + " ****";
    }
    return scheme + " " + token.substring(0, 4) + "..." + token.substring(token.length() - 4);
  }

  private static boolean isGzipEncoded(Response response) {
    String encoding = response.header("Content-Encoding");
    if (encoding == null || encoding.isBlank()) {
      return false;
    }
    String[] parts = encoding.split(",");
    for (String part : parts) {
      if ("gzip".equalsIgnoreCase(part.trim())) {
        return true;
      }
    }
    return false;
  }

  private static boolean isBlankPayload(byte[] payload) {
    if (payload == null || payload.length == 0) {
      return true;
    }
    for (byte value : payload) {
      if (!isWhitespace(value)) {
        return false;
      }
    }
    return true;
  }

  private static boolean isWhitespace(byte value) {
    return value == ' ' || value == '\n' || value == '\r' || value == '\t';
  }

  private static byte[] decompressGzip(byte[] payload) {
    try (ByteArrayInputStream input = new ByteArrayInputStream(payload);
         GZIPInputStream gzip = new GZIPInputStream(input);
         ByteArrayOutputStream output = new ByteArrayOutputStream()) {
      gzip.transferTo(output);
      return output.toByteArray();
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
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
                        Integer statusCode,
                        long backoffMillis,
                        String reason,
                        Throwable error) {
    String statusLabel = statusCode == null ? "n/a" : statusCode.toString();
    if (error == null) {
      logger.warn(
          "Gribstream forecast retry {}/{} for model={} (status={}, backoffMs={}, reason={})",
          attempt,
          MAX_ATTEMPTS,
          modelCode,
          statusLabel,
          backoffMillis,
          reason);
      return;
    }
    logger.warn(
        "Gribstream forecast retry {}/{} for model={} (status={}, backoffMs={}, reason={})",
        attempt,
        MAX_ATTEMPTS,
        modelCode,
        statusLabel,
        backoffMillis,
        reason,
        error);
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

  private long computeBackoffMillis(int attempt) {
    int exponent = Math.min(attempt - 1, 4);
    long seconds = 1L << exponent;
    return seconds * 1000L;
  }

  private long computeRetryBackoffMillis(int attempt, GribstreamHttpResponse response) {
    long backoffMillis = computeBackoffMillis(attempt);
    if (response != null && response.retryAfterMillis() != null) {
      backoffMillis = Math.max(backoffMillis, response.retryAfterMillis());
    }
    return addJitter(backoffMillis);
  }

  private long addJitter(long backoffMillis) {
    if (backoffMillis <= 0) {
      return 0;
    }
    long jitter = ThreadLocalRandom.current().nextLong(0, 250);
    return backoffMillis + jitter;
  }

  private static Long parseRetryAfterMillis(String headerValue) {
    if (headerValue == null || headerValue.isBlank()) {
      return null;
    }
    String trimmed = headerValue.trim();
    try {
      long seconds = Long.parseLong(trimmed);
      return Math.max(0L, seconds) * 1000L;
    } catch (NumberFormatException ignored) {
      // Fall through to RFC 1123 date parsing.
    }
    try {
      ZonedDateTime dateTime = ZonedDateTime.parse(trimmed, DateTimeFormatter.RFC_1123_DATE_TIME);
      long millis = Duration.between(Instant.now(), dateTime.toInstant()).toMillis();
      return Math.max(0L, millis);
    } catch (Exception ignored) {
      return null;
    }
  }

  private static Throwable rootCause(Throwable error) {
    if (error == null) {
      return new IllegalStateException("unknown");
    }
    Throwable current = error;
    while (current.getCause() != null && current.getCause() != current) {
      current = current.getCause();
    }
    return current;
  }

  private static final class GribstreamHttpResponse {
    private final int statusCode;
    private final byte[] bodyBytes;
    private final Long retryAfterMillis;

    private GribstreamHttpResponse(int statusCode, byte[] bodyBytes, Long retryAfterMillis) {
      this.statusCode = statusCode;
      this.bodyBytes = bodyBytes == null ? new byte[0] : bodyBytes;
      this.retryAfterMillis = retryAfterMillis;
    }

    private String snippet() {
      String text = new String(bodyBytes, StandardCharsets.UTF_8);
      if (text.length() <= SNIPPET_LIMIT) {
        return text;
      }
      return text.substring(0, SNIPPET_LIMIT);
    }

    private Long retryAfterMillis() {
      return retryAfterMillis;
    }
  }
}
