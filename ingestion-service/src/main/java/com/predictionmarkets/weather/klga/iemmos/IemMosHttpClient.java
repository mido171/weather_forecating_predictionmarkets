package com.predictionmarkets.weather.klga.iemmos;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.springframework.stereotype.Component;

@Component
public class IemMosHttpClient {
  private static final DateTimeFormatter IEM_TIME =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm'Z'").withZone(ZoneOffset.UTC);

  private final OkHttpClient client;
  private final String baseUrl;

  public IemMosHttpClient(com.predictionmarkets.weather.iem.IemProperties properties) {
    this.baseUrl = Objects.requireNonNull(properties.getBaseUrl(), "iem.base-url is required");
    this.client = new OkHttpClient.Builder()
        .connectTimeout(Duration.ofSeconds(10).toMillis(), TimeUnit.MILLISECONDS)
        .readTimeout(Duration.ofMinutes(3).toMillis(), TimeUnit.MILLISECONDS)
        .callTimeout(Duration.ofMinutes(4).toMillis(), TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build();
  }

  public IemMosFetchResult fetch(IemMosChunk chunk) throws IOException {
    HttpUrl base = HttpUrl.parse(baseUrl);
    if (base == null) {
      throw new IllegalArgumentException("Invalid IEM base URL: " + baseUrl);
    }
    HttpUrl url = base.newBuilder()
        .addPathSegments("cgi-bin/request/mos.py")
        .addQueryParameter("station", chunk.station().stationId())
        .addQueryParameter("model", chunk.product().endpointModel().name())
        .addQueryParameter("sts", IEM_TIME.format(chunk.windowStartUtc()))
        .addQueryParameter("ets", IEM_TIME.format(chunk.windowEndUtc()))
        .addQueryParameter("format", "json")
        .build();
    Request request = new Request.Builder().url(url).get().build();
    Instant retrievedAt = Instant.now();
    try (Response response = client.newCall(request).execute()) {
      ResponseBody body = response.body();
      byte[] bytes = body == null ? new byte[0] : body.bytes();
      String contentType = body == null || body.contentType() == null
          ? null
          : body.contentType().toString();
      return new IemMosFetchResult(
          response.code(),
          bytes,
          contentType,
          retrievedAt,
          url.toString());
    }
  }
}
