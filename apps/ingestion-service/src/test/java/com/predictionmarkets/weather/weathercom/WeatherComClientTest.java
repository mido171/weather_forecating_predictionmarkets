package com.predictionmarkets.weather.weathercom;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.weathercom.client.WeatherComClient;
import com.predictionmarkets.weather.weathercom.client.WeatherComClientResult;
import com.predictionmarkets.weather.weathercom.config.WeatherComProperties;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.core.io.ClassPathResource;

class WeatherComClientTest {
  private static final MockWebServer SERVER = new MockWebServer();

  @BeforeAll
  static void startServer() throws IOException {
    SERVER.start();
  }

  @AfterAll
  static void shutdownServer() throws IOException {
    SERVER.shutdown();
  }

  @Test
  void sendsExpectedQueryParamsAndGzipHeader() throws Exception {
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(readFixture()));

    WeatherComClient client = buildClient(2);
    WeatherComClientResult result = client.fetchHistoricalObservations(
        "KNYC:9:US",
        "e",
        LocalDate.of(2026, 2, 17),
        LocalDate.of(2026, 2, 17));

    assertThat(result.success()).isTrue();
    RecordedRequest request = SERVER.takeRequest();
    assertThat(request.getRequestUrl().pathSegments())
        .containsExactly("v1", "location", "KNYC:9:US", "observations", "historical.json");
    assertThat(request.getRequestUrl().queryParameter("apiKey")).isEqualTo("test-weathercom-key");
    assertThat(request.getRequestUrl().queryParameter("units")).isEqualTo("e");
    assertThat(request.getRequestUrl().queryParameter("startDate")).isEqualTo("20260217");
    assertThat(request.getRequestUrl().queryParameter("endDate")).isEqualTo("20260217");
    assertThat(request.getHeader("Accept-Encoding")).isEqualTo("gzip");
  }

  @Test
  void retriesOn500ThenSucceeds() {
    int initialRequests = SERVER.getRequestCount();
    SERVER.enqueue(new MockResponse().setResponseCode(500).setBody("{\"error\":\"server\"}"));
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(readFixture()));

    WeatherComClient client = buildClient(2);
    WeatherComClientResult result = client.fetchHistoricalObservations(
        "KNYC:9:US",
        "e",
        LocalDate.of(2026, 2, 17),
        LocalDate.of(2026, 2, 17));

    assertThat(result.success()).isTrue();
    assertThat(SERVER.getRequestCount()).isEqualTo(initialRequests + 2);
  }

  @Test
  void retriesOn429ThenSucceeds() {
    int initialRequests = SERVER.getRequestCount();
    SERVER.enqueue(new MockResponse()
        .setResponseCode(429)
        .setHeader("Retry-After", "0")
        .setBody("{\"error\":\"rate_limited\"}"));
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(readFixture()));

    WeatherComClient client = buildClient(2);
    WeatherComClientResult result = client.fetchHistoricalObservations(
        "KNYC:9:US",
        "e",
        LocalDate.of(2026, 2, 17),
        LocalDate.of(2026, 2, 17));

    assertThat(result.success()).isTrue();
    assertThat(SERVER.getRequestCount()).isEqualTo(initialRequests + 2);
  }

  @ParameterizedTest
  @ValueSource(ints = {401, 403, 404})
  void doesNotRetryOnNonRetryableAuthOrNotFoundCodes(int statusCode) {
    int initialRequests = SERVER.getRequestCount();
    SERVER.enqueue(new MockResponse().setResponseCode(statusCode).setBody("{\"error\":\"nope\"}"));

    WeatherComClient client = buildClient(4);
    WeatherComClientResult result = client.fetchHistoricalObservations(
        "KNYC:9:US",
        "e",
        LocalDate.of(2026, 2, 17),
        LocalDate.of(2026, 2, 17));

    assertThat(result.success()).isFalse();
    assertThat(result.httpStatus()).isEqualTo(statusCode);
    assertThat(SERVER.getRequestCount()).isEqualTo(initialRequests + 1);
  }

  private WeatherComClient buildClient(int maxRetries) {
    WeatherComProperties properties = new WeatherComProperties();
    properties.getApi().setBaseUrl(SERVER.url("/").toString());
    properties.getApi().setApiKey("test-weathercom-key");
    properties.getApi().setConnectTimeoutMs(1000);
    properties.getApi().setReadTimeoutMs(1000);
    properties.getApi().setUserAgent("weathercom-test");
    properties.getIngestion().setMaxRetries(maxRetries);
    properties.getIngestion().setRetryBackoffMs(1);
    properties.getIngestion().setMaxBackoffMs(5);
    properties.getIngestion().setRetryJitterMs(0);
    properties.getIngestion().getRateLimit().setPermitsPerSecond(0.0d);
    return new WeatherComClient(new ObjectMapper(), properties);
  }

  private static String readFixture() {
    try {
      return new String(
          new ClassPathResource("weathercom/historical_observations_fixture.json")
              .getInputStream()
              .readAllBytes(),
          StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }
}
