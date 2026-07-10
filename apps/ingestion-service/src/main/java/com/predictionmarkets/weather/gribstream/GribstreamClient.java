package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import com.predictionmarkets.weather.common.http.HttpRetryPolicy;
import com.predictionmarkets.weather.config.EvomiProxyProperties;
import com.predictionmarkets.weather.gribstream.GribstreamRawResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
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
  private final boolean credentialsConfigured;
  private final String acceptHeader;
  private final boolean gzipEnabled;
  private final boolean logHttp;
  private final int logBodyLimit;
  private final HttpRetryPolicy retryPolicy;
  private final ObjectMapper objectMapper;
  private final boolean evomiProxyEnabled;
  private final String userAgent;

  public GribstreamClient(GribstreamProperties properties,
                          EvomiProxyProperties evomiProxyProperties,
                          ObjectMapper objectMapper,
                          HttpClientSettings httpClientSettings) {
    Objects.requireNonNull(properties, "properties is required");
    this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper is required");
    String apiToken = normalizeOptionalToken(properties.getApiToken());
    String authScheme = normalizeAuthScheme(properties.getAuthScheme());
    this.authHeader = buildAuthorizationHeader(apiToken, authScheme);
    this.credentialsConfigured = !apiToken.isBlank();
    this.acceptHeader = properties.getDefaultAccept();
    this.gzipEnabled = properties.isGzip();
    this.logHttp = properties.isLogHttp();
    this.logBodyLimit = Math.max(0, properties.getLogBodyLimit());
    this.retryPolicy = Objects.requireNonNull(httpClientSettings, "httpClientSettings is required")
        .retryPolicy();
    this.baseUrl = parseBaseUrl(properties.getBaseUrl());
    this.evomiProxyEnabled = evomiProxyProperties != null && evomiProxyProperties.isEnabled();
    this.userAgent = buildUserAgent("gribstream-history", gzipEnabled, evomiProxyEnabled);
    OkHttpClient.Builder builder = new OkHttpClient.Builder()
        .connectTimeout(properties.getConnectTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
        .readTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
        .callTimeout(Duration.ofMillis(properties.getReadTimeoutMillis()))
        .writeTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        ;
    EvomiProxySupport.applyIfEnabled(builder, evomiProxyProperties, logger);
    this.httpClient = builder.build();
    logger.info("[GRIBSTREAM] History client credentials configured={}", credentialsConfigured);
    logger.debug("Gribstream client configured baseUrl={}", properties.getBaseUrl());
  }

  public GribstreamClientResponse fetchHistory(String modelCode, GribstreamHistoryRequest request) {
    requireCredentials();
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

  public GribstreamRawResponse fetchHistoryRaw(String modelCode, GribstreamHistoryRequest request) {
    requireCredentials();
    if (modelCode == null || modelCode.isBlank()) {
      throw new IllegalArgumentException("modelCode is required");
    }
    Objects.requireNonNull(request, "request is required");
    String requestJson = serializeRequest(request, modelCode);
    String requestSha256 = Hashing.sha256Hex(requestJson);
    byte[] responseBytes = executeRequest(modelCode, requestJson, requestSha256);
    String responseSha256 = Hashing.sha256Hex(responseBytes);
    Instant retrievedAtUtc = Instant.now();
    return new GribstreamRawResponse(
        requestJson,
        requestSha256,
        responseSha256,
        retrievedAtUtc,
        responseBytes);
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
    HttpUrl url = buildUrl(modelCode);
    if (logHttp) {
      logRequest(modelCode, url, requestJson, requestSha256, maxAttempts);
    }
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        GribstreamHttpResponse response =
            executeOnce(modelCode, url, requestJson, requestSha256, attempt, maxAttempts);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          if (isBlankPayload(response.bodyBytes)) {
            if (attempt < maxAttempts) {
              long backoffMillis = retryPolicy.computeDelayMillis(attempt);
              logger.warn(
                  "Gribstream retry {}/{} for model={} (status={}, backoffMs={}, reason=empty_body)",
                  attempt,
                  maxAttempts,
                  modelCode,
                  response.statusCode,
                  backoffMillis);
              sleepBackoff(backoffMillis);
              continue;
            }
            throw new GribstreamEmptyResponseException("Gribstream empty response body"
                + " status=" + response.statusCode
                + " model=" + modelCode
                + " requestSha256=" + requestSha256);
          }
          return response.bodyBytes;
        }
        if (response.statusCode == 401) {
          logUnauthorized(modelCode, requestSha256);
        }
        if (retryPolicy.isRetryableStatus(response.statusCode) && attempt < maxAttempts) {
          long backoffMillis = retryPolicy.computeDelayMillis(attempt);
          logRetry(modelCode, attempt, maxAttempts, response.statusCode, backoffMillis, null);
          sleepBackoff(backoffMillis);
          continue;
        }
        if (response.statusCode != 401) {
          logFailure(modelCode, requestSha256, response.statusCode);
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

  private GribstreamHttpResponse executeOnce(String modelCode,
                                             HttpUrl url,
                                             String requestJson,
                                             String requestSha256,
                                             int attempt,
                                             int maxAttempts) {
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
        .header("Accept", acceptHeader)
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
        logResponse(modelCode, url, requestSha256, attempt, maxAttempts, response,
            rawPayload.length, decodedPayload);
      }
      return new GribstreamHttpResponse(response.code(), decodedPayload);
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
        "[GRIBSTREAM] History request failed model={} status={} requestSha256={} evomiProxyEnabled={}",
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

  private static String buildUserAgent(String clientName,
                                       boolean gzipEnabled,
                                       boolean evomiProxyEnabled) {
    String appName = "weather-forecasting-predictionmarkets";
    String appVersion = resolveImplementationVersion(GribstreamClient.class, "dev");
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
        .addPathSegment("history")
        .build();
  }

  private void logRequest(String modelCode,
                          HttpUrl url,
                          String requestJson,
                          String requestSha256,
                          int maxAttempts) {
    byte[] requestBytes = requestJson.getBytes(StandardCharsets.UTF_8);
    String payload = limitBody(requestJson);
    logger.info(
        "[GRIBSTREAM-HTTP] request model={} url={} attempt=1/{} accept={} contentType={} "
            + "acceptEncoding={} auth={} bodyBytes={} requestSha256={} bodyJson={}",
        modelCode,
        url,
        maxAttempts,
        acceptHeader,
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
                           int maxAttempts,
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
        maxAttempts,
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
