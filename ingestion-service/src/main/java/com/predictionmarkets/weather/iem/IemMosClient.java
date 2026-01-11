package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.http.HardenedWebClient;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import com.predictionmarkets.weather.models.MosModel;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
public class IemMosClient {
  private static final DateTimeFormatter WINDOW_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm'Z'").withZone(ZoneOffset.UTC);

  private final HardenedWebClient httpClient;
  private final ObjectMapper objectMapper;

  public IemMosClient(WebClient.Builder builder,
                      IemProperties properties,
                      ObjectMapper objectMapper,
                      HttpClientSettings httpClientSettings) {
    this.httpClient = new HardenedWebClient(builder, properties.getBaseUrl(), httpClientSettings);
    this.objectMapper = objectMapper;
  }

  public IemMosPayload fetchWindow(String stationId, MosModel model, Instant startUtc, Instant endUtc) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    if (model == null) {
      throw new IllegalArgumentException("model is required");
    }
    if (startUtc == null || endUtc == null) {
      throw new IllegalArgumentException("startUtc and endUtc are required");
    }
    if (!endUtc.isAfter(startUtc)) {
      throw new IllegalArgumentException("endUtc must be after startUtc");
    }
    String normalizedStation = stationId.trim().toUpperCase(Locale.ROOT);
    String startParam = WINDOW_FORMATTER.format(startUtc);
    String endParam = WINDOW_FORMATTER.format(endUtc);
    String endpoint = "/cgi-bin/request/mos.py?station=" + normalizedStation
        + "&model=" + model.name()
        + "&sts=" + startParam
        + "&ets=" + endParam
        + "&format=json";
    String correlationId = "iem-mos-" + normalizedStation + "-" + model.name()
        + "-" + startParam + "-" + endParam;
    byte[] rawBytes = httpClient.getBytes(endpoint, correlationId, uriBuilder -> uriBuilder.path("/cgi-bin/request/mos.py")
        .queryParam("station", normalizedStation)
        .queryParam("model", model.name())
        .queryParam("sts", startParam)
        .queryParam("ets", endParam)
        .queryParam("format", "json")
        .build());
    return IemMosPayload.parse(objectMapper, rawBytes, normalizedStation, model);
  }
}
