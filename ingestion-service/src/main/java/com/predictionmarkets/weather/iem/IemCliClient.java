package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.http.HardenedWebClient;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import java.util.Locale;
import org.springframework.stereotype.Service;

@Service
public class IemCliClient {
  private final HardenedWebClient httpClient;
  private final ObjectMapper objectMapper;

  public IemCliClient(IemProperties properties,
                      ObjectMapper objectMapper,
                      HttpClientSettings httpClientSettings) {
    this.httpClient = new HardenedWebClient(properties.getBaseUrl(), httpClientSettings);
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
    byte[] rawBytes = httpClient.getBytes(endpoint, correlationId, urlBuilder -> urlBuilder
        .addPathSegments("json/cli.py")
        .addQueryParameter("station", normalizedStation)
        .addQueryParameter("year", String.valueOf(year))
        .addQueryParameter("fmt", "json")
        .build());
    return IemCliPayload.parse(objectMapper, rawBytes, normalizedStation);
  }
}
