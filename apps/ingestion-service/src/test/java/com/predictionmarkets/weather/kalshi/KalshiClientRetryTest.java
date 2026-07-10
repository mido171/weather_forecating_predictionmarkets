package com.predictionmarkets.weather.kalshi;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import com.predictionmarkets.weather.common.http.HttpRetryPolicy;
import java.io.IOException;
import java.time.Duration;
import java.util.Set;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class KalshiClientRetryTest {
  private static final MockWebServer SERVER = new MockWebServer();

  @BeforeAll
  static void startServer() throws IOException {
    SERVER.start();
  }

  @AfterAll
  static void shutdown() throws IOException {
    SERVER.shutdown();
  }

  @Test
  void retriesOn503ThenSucceeds() {
    SERVER.enqueue(new MockResponse().setResponseCode(503));
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(seriesPayload("KXHIGHMIA")));

    KalshiProperties properties = new KalshiProperties();
    properties.setBaseUrl(SERVER.url("/").toString());
    HttpRetryPolicy retryPolicy = new HttpRetryPolicy(
        3,
        Duration.ofMillis(1),
        Duration.ofMillis(5),
        Duration.ZERO,
        Set.of(503));
    HttpClientSettings settings = new HttpClientSettings(
        Duration.ofSeconds(1),
        Duration.ofSeconds(1),
        retryPolicy);
    KalshiClient client = new KalshiClient(properties, new ObjectMapper(), settings);

    KalshiSeriesPayload payload = client.fetchSeries("KXHIGHMIA");

    assertThat(payload.seriesTicker()).isEqualTo("KXHIGHMIA");
    assertThat(SERVER.getRequestCount()).isEqualTo(2);
  }

  private static String seriesPayload(String ticker) {
    return "{\"series\":{\"ticker\":\"" + ticker + "\",\"title\":\"Test Series\",\"category\":\"weather\","
        + "\"settlement_sources\":[{\"name\":\"NWS CLI\",\"url\":\"https://example.test/cli\"}]}}";
  }
}
