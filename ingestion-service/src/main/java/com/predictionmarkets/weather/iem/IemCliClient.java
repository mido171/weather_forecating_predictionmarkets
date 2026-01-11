package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Duration;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class IemCliClient {
  private final WebClient webClient;
  private final ObjectMapper objectMapper;

  public IemCliClient(WebClient.Builder builder, IemProperties properties, ObjectMapper objectMapper) {
    this.webClient = builder.baseUrl(properties.getBaseUrl()).build();
    this.objectMapper = objectMapper;
  }

  public IemCliPayload fetchYear(String stationId, int year) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    if (year < 1800 || year > 2500) {
      throw new IllegalArgumentException("year out of range: " + year);
    }
    String normalizedStation = stationId.trim().toUpperCase(Locale.ROOT);
    String rawJson = webClient.get()
        .uri(uriBuilder -> uriBuilder.path("/json/cli.py")
            .queryParam("station", normalizedStation)
            .queryParam("year", year)
            .queryParam("fmt", "json")
            .build())
        .retrieve()
        .bodyToMono(String.class)
        .timeout(Duration.ofSeconds(30))
        .block();
    if (rawJson == null || rawJson.isBlank()) {
      throw new IllegalStateException("IEM CLI payload was empty");
    }
    return IemCliPayload.parse(objectMapper, rawJson, normalizedStation);
  }
}
