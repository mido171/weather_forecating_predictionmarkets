package com.predictionmarkets.weather.pilot.source;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import okhttp3.MediaType;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.stereotype.Service;

@Service
public class SourceHttpClient {
  private static final MediaType JSON = MediaType.parse("application/json");

  private final HttpClientFactory httpClientFactory;
  private final RetryPolicyFactory retryPolicyFactory;
  private final RateLimiterRegistry rateLimiterRegistry;
  private final ObjectMapper objectMapper;

  public SourceHttpClient(HttpClientFactory httpClientFactory,
                          RetryPolicyFactory retryPolicyFactory,
                          RateLimiterRegistry rateLimiterRegistry,
                          ObjectMapper objectMapper) {
    this.httpClientFactory = httpClientFactory;
    this.retryPolicyFactory = retryPolicyFactory;
    this.rateLimiterRegistry = rateLimiterRegistry;
    this.objectMapper = objectMapper;
  }

  public HttpResponseData get(String sourceFamily, String url, Map<String, String> headers) {
    return execute(sourceFamily, new Request.Builder().url(url), null, headers);
  }

  public HttpResponseData postJson(String sourceFamily,
                                   String url,
                                   Object payload,
                                   Map<String, String> headers) {
    try {
      String json = objectMapper.writeValueAsString(payload);
      RequestBody body = RequestBody.create(json.getBytes(StandardCharsets.UTF_8), JSON);
      return execute(sourceFamily, new Request.Builder().url(url), body, headers);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to serialize request body for " + url, ex);
    }
  }

  private HttpResponseData execute(String sourceFamily,
                                   Request.Builder builder,
                                   RequestBody body,
                                   Map<String, String> headers) {
    RetryPolicyFactory.RetryPolicy retryPolicy = retryPolicyFactory.forSourceFamily(sourceFamily);
    int attempt = 0;
    while (true) {
      attempt += 1;
      try {
        int currentAttempt = attempt;
        return rateLimiterRegistry.withPermit(sourceFamily,
            () -> executeOnce(builder, body, headers, currentAttempt));
      } catch (Exception ex) {
        if (attempt >= retryPolicy.maxAttempts()) {
          throw new IllegalStateException("HTTP request failed after retries at " + Instant.now(), ex);
        }
        sleep(retryPolicy.computeDelayMs(attempt));
      }
    }
  }

  private HttpResponseData executeOnce(Request.Builder builder,
                                       RequestBody body,
                                       Map<String, String> headers,
                                       int attempt) throws IOException {
    builder.header("User-Agent", "weather-forecasting-predictionmarkets/1.0");
    if (headers != null) {
      for (Map.Entry<String, String> entry : headers.entrySet()) {
        builder.header(entry.getKey(), entry.getValue());
      }
    }
    Request request = body == null ? builder.get().build() : builder.post(body).build();
    long startedNanos = System.nanoTime();
    try (Response response = httpClientFactory.client().newCall(request).execute()) {
      byte[] bytes = response.body() == null ? new byte[0] : response.body().bytes();
      double durationMs = (System.nanoTime() - startedNanos) / 1_000_000.0d;
      Map<String, String> responseHeaders = new LinkedHashMap<>();
      response.headers().forEach(pair -> responseHeaders.put(pair.getFirst(), pair.getSecond()));
      int statusCode = response.code();
      if (statusCode >= 500 || statusCode == 429) {
        throw new IOException("Retryable HTTP status " + statusCode + " for " + request.url());
      }
      return new HttpResponseData(
          statusCode,
          bytes,
          durationMs,
          request.url().toString(),
          responseHeaders,
          Math.max(0, attempt - 1));
    }
  }

  private void sleep(long millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted during HTTP retry backoff", ex);
    }
  }
}
