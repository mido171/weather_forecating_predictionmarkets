package com.predictionmarkets.weather.kalshi;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Duration;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class KalshiClient {
  private final WebClient webClient;
  private final ObjectMapper objectMapper;

  public KalshiClient(WebClient.Builder builder, KalshiProperties properties, ObjectMapper objectMapper) {
    this.webClient = builder.baseUrl(properties.getBaseUrl()).build();
    this.objectMapper = objectMapper;
  }

  public KalshiSeriesPayload fetchSeries(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      throw new IllegalArgumentException("seriesTicker is required");
    }
    String rawJson = webClient.get()
        .uri("/series/{seriesTicker}", seriesTicker)
        .retrieve()
        .bodyToMono(String.class)
        .timeout(Duration.ofSeconds(30))
        .block();
    if (rawJson == null || rawJson.isBlank()) {
      throw new IllegalStateException("Kalshi series payload was empty");
    }
    KalshiSeriesPayload payload = KalshiSeriesPayload.parse(objectMapper, rawJson);
    String expectedTicker = seriesTicker.trim().toUpperCase(Locale.ROOT);
    if (!payload.seriesTicker().equalsIgnoreCase(expectedTicker)) {
      throw new IllegalArgumentException("Series ticker mismatch: expected " + expectedTicker
          + " but got " + payload.seriesTicker());
    }
    return payload;
  }
}
