package com.predictionmarkets.weather.iem;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.models.MosModel;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.concurrent.ThreadLocalRandom;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientResponseException;

@Service
public class IemMosClient {
  private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(30);
  private static final int MAX_ATTEMPTS = 3;
  private static final long BASE_BACKOFF_MILLIS = 250;
  private static final long MAX_BACKOFF_MILLIS = 2000;
  private static final DateTimeFormatter WINDOW_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm'Z'").withZone(ZoneOffset.UTC);

  private final WebClient webClient;
  private final ObjectMapper objectMapper;

  public IemMosClient(WebClient.Builder builder, IemProperties properties, ObjectMapper objectMapper) {
    this.webClient = builder.baseUrl(properties.getBaseUrl()).build();
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
    String rawJson = fetchWithRetry(normalizedStation, model, startParam, endParam);
    if (rawJson == null || rawJson.isBlank()) {
      throw new IllegalStateException("IEM MOS payload was empty");
    }
    return IemMosPayload.parse(objectMapper, rawJson, normalizedStation, model);
  }

  private String fetchWithRetry(String stationId, MosModel model, String startParam, String endParam) {
    RuntimeException lastError = null;
    for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        return fetchOnce(stationId, model, startParam, endParam);
      } catch (RuntimeException ex) {
        lastError = ex;
        if (!isRetryable(ex) || attempt == MAX_ATTEMPTS) {
          throw ex;
        }
        sleepBackoff(attempt);
      }
    }
    throw lastError;
  }

  private String fetchOnce(String stationId, MosModel model, String startParam, String endParam) {
    return webClient.get()
        .uri(uriBuilder -> uriBuilder.path("/cgi-bin/request/mos.py")
            .queryParam("station", stationId)
            .queryParam("model", model.name())
            .queryParam("sts", startParam)
            .queryParam("ets", endParam)
            .queryParam("format", "json")
            .build())
        .retrieve()
        .bodyToMono(String.class)
        .timeout(REQUEST_TIMEOUT)
        .block();
  }

  private boolean isRetryable(RuntimeException ex) {
    if (ex instanceof WebClientResponseException responseException) {
      int status = responseException.getStatusCode().value();
      return status == 429 || status == 502 || status == 503 || status == 504;
    }
    if (ex instanceof WebClientRequestException) {
      return true;
    }
    return false;
  }

  private void sleepBackoff(int attempt) {
    long baseDelay = BASE_BACKOFF_MILLIS * (1L << (attempt - 1));
    long jitter = ThreadLocalRandom.current().nextLong(0, BASE_BACKOFF_MILLIS);
    long delay = Math.min(baseDelay + jitter, MAX_BACKOFF_MILLIS);
    try {
      Thread.sleep(delay);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("MOS retry interrupted", ex);
    }
  }
}
