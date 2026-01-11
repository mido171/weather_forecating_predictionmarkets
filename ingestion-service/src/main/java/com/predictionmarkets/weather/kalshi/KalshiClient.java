package com.predictionmarkets.weather.kalshi;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.http.HardenedWebClient;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class KalshiClient {
  private final HardenedWebClient httpClient;
  private final ObjectMapper objectMapper;

  public KalshiClient(WebClient.Builder builder,
                      KalshiProperties properties,
                      ObjectMapper objectMapper,
                      HttpClientSettings httpClientSettings) {
    this.httpClient = new HardenedWebClient(builder, properties.getBaseUrl(), httpClientSettings);
    this.objectMapper = objectMapper;
  }

  public KalshiSeriesPayload fetchSeries(String seriesTicker) {
    if (seriesTicker == null || seriesTicker.isBlank()) {
      throw new IllegalArgumentException("seriesTicker is required");
    }
    String normalizedTicker = seriesTicker.trim().toUpperCase(Locale.ROOT);
    String endpoint = "/series/" + normalizedTicker;
    String correlationId = "kalshi-series-" + normalizedTicker;
    byte[] rawBytes = httpClient.getBytes(endpoint, correlationId,
        uriBuilder -> uriBuilder.path("/series/{seriesTicker}").build(normalizedTicker));
    KalshiSeriesPayload payload = KalshiSeriesPayload.parse(objectMapper, rawBytes);
    String expectedTicker = seriesTicker.trim().toUpperCase(Locale.ROOT);
    if (!payload.seriesTicker().equalsIgnoreCase(expectedTicker)) {
      throw new IllegalArgumentException("Series ticker mismatch: expected " + expectedTicker
          + " but got " + payload.seriesTicker());
    }
    return payload;
  }
}
