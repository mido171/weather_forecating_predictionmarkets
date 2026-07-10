package com.predictionmarkets.weather.klga.iemmos;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.iem.IemProperties;
import java.time.Instant;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.Test;

class IemMosHttpClientTest {
  @Test
  void fetchUsesStructuredMosEndpointAndCanonicalStationId() throws Exception {
    try (MockWebServer server = new MockWebServer()) {
      server.enqueue(new MockResponse()
          .setResponseCode(200)
          .setBody("[]")
          .setHeader("Content-Type", "application/json"));
      server.start();

      IemProperties properties = new IemProperties();
      properties.setBaseUrl(server.url("/").toString());
      IemMosHttpClient client = new IemMosHttpClient(properties);
      IemMosChunk chunk = new IemMosChunk(
          "chunk",
          "job",
          new IemMosStation("KLGA", "LGA"),
          IemMosProduct.MAV,
          "T_1245UTC",
          Instant.parse("2026-06-01T00:00:00Z"),
          Instant.parse("2026-06-02T00:00:00Z"),
          "requesthash",
          "{}");

      IemMosFetchResult result = client.fetch(chunk);
      RecordedRequest request = server.takeRequest();

      assertThat(result.httpStatus()).isEqualTo(200);
      assertThat(request.getPath()).contains("/cgi-bin/request/mos.py");
      assertThat(request.getRequestUrl().queryParameter("station")).isEqualTo("KLGA");
      assertThat(request.getRequestUrl().queryParameter("model")).isEqualTo("GFS");
      assertThat(request.getRequestUrl().queryParameter("sts")).isEqualTo("2026-06-01T00:00Z");
      assertThat(request.getRequestUrl().queryParameter("ets")).isEqualTo("2026-06-02T00:00Z");
      assertThat(request.getRequestUrl().queryParameter("format")).isEqualTo("json");
    }
  }
}
