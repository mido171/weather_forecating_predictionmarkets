package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.http.HardenedWebClient;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class IemCliClient {
  private final HardenedWebClient httpClient;
  private final ObjectMapper objectMapper;

  public IemCliClient(WebClient.Builder builder,
                      IemProperties properties,
                      ObjectMapper objectMapper,
                      HttpClientSettings httpClientSettings) {
    this.httpClient = new HardenedWebClient(builder, properties.getBaseUrl(), httpClientSettings);
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
    String endpoint = "/json/cli.py?station=" + normalizedStation + "&year=" + year + "&fmt=json";
    String correlationId = "iem-cli-" + normalizedStation + "-" + year;
    byte[] rawBytes = httpClient.getBytes(endpoint, correlationId, uriBuilder -> uriBuilder.path("/json/cli.py")
        .queryParam("station", normalizedStation)
        .queryParam("year", year)
        .queryParam("fmt", "json")
        .build());
    return IemCliPayload.parse(objectMapper, rawBytes, normalizedStation);
  }
}
