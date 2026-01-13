package com.predictionmarkets.weather.gribstream;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.http.HttpClientSettings;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okio.Buffer;
import org.junit.jupiter.api.Test;

class GribstreamClientParsingTest {
  @Test
  void parsesNdjsonArrayResponses() throws Exception {
    MockWebServer server = startServer();
    try {
    String gfsBody = loadFixture("gribstream/gfs_history.json");
    String gefsBody = loadFixture("gribstream/gefs_history.json");
    for (int i = 0; i < 5; i++) {
      server.enqueue(gzipResponse(gfsBody));
    }
    server.enqueue(gzipResponse(gefsBody));

    GribstreamProperties properties = new GribstreamProperties();
    properties.setBaseUrl(server.url("/").toString());
    properties.setApiToken("test-token");
    GribstreamClient client = new GribstreamClient(
        properties,
        new ObjectMapper(),
        HttpClientSettings.defaultSettings());

    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        "2026-01-10T05:00:00Z",
        "2026-01-11T05:00:00Z",
        "2026-01-09T17:00:00Z",
        0,
        48,
        List.of(new GribstreamCoordinate(40.77898, -73.96925, "KNYC")),
        List.of(new GribstreamVariable("TMP", "2 m above ground", "", "tmpk")),
        null);

    List<String> models = List.of("gfs", "hrrr", "nbm", "rap", "gefsatmosmean", "gefsatmos");
    List<GribstreamClientResponse> responses = new ArrayList<>();
    for (String model : models) {
      GribstreamHistoryRequest modelRequest = request;
      if (model.equals("gefsatmos")) {
        modelRequest = new GribstreamHistoryRequest(
            request.fromTime(),
            request.untilTime(),
            request.asOf(),
            request.minHorizon(),
            request.maxHorizon(),
            request.coordinates(),
            request.variables(),
            List.of(0, 1));
      }
      responses.add(client.fetchHistory(model, modelRequest));
    }

    GribstreamClientResponse gfsResponse = responses.get(0);
    assertThat(gfsResponse.rows()).hasSize(2);
    assertThat(gfsResponse.rows().get(0).forecastedAt())
        .isEqualTo(Instant.parse("2026-01-09T12:00:00Z"));
    assertThat(gfsResponse.rows().get(1).tmpk()).isEqualTo(282.0);

    GribstreamClientResponse gefsResponse = responses.get(5);
    assertThat(gefsResponse.rows()).hasSize(2);
    assertThat(gefsResponse.rows().get(0).member()).isEqualTo(0);

    List<String> paths = new ArrayList<>();
    for (int i = 0; i < models.size(); i++) {
      RecordedRequest recorded = server.takeRequest();
      paths.add(recorded.getPath());
    }
    assertThat(paths).containsExactly(
        "/api/v2/gfs/history",
        "/api/v2/hrrr/history",
        "/api/v2/nbm/history",
        "/api/v2/rap/history",
        "/api/v2/gefsatmosmean/history",
        "/api/v2/gefsatmos/history");
    } finally {
      server.shutdown();
    }
  }

  @Test
  void parsesNdjsonLineResponses() throws Exception {
    MockWebServer server = startServer();
    try {
    String ndjsonBody = String.join("\n",
        "{\"forecasted_at\":\"2026-01-09T12:00:00Z\",\"forecasted_time\":\"2026-01-10T12:00:00Z\","
            + "\"lat\":40.77898,\"lon\":-73.96925,\"name\":\"KNYC\",\"tmpk\":280.0}",
        "{\"forecasted_at\":\"2026-01-09T12:00:00Z\",\"forecasted_time\":\"2026-01-10T18:00:00Z\","
            + "\"lat\":40.77898,\"lon\":-73.96925,\"name\":\"KNYC\",\"tmpk\":282.0}");
    server.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/ndjson")
        .setBody(ndjsonBody));

    GribstreamProperties properties = new GribstreamProperties();
    properties.setBaseUrl(server.url("/").toString());
    properties.setApiToken("test-token");
    GribstreamClient client = new GribstreamClient(
        properties,
        new ObjectMapper(),
        HttpClientSettings.defaultSettings());

    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        "2026-01-10T05:00:00Z",
        "2026-01-11T05:00:00Z",
        "2026-01-09T17:00:00Z",
        0,
        48,
        List.of(new GribstreamCoordinate(40.77898, -73.96925, "KNYC")),
        List.of(new GribstreamVariable("TMP", "2 m above ground", "", "tmpk")),
        null);

    GribstreamClientResponse response = client.fetchHistory("hrrr", request);
    assertThat(response.rows()).hasSize(2);
    assertThat(response.rows().get(0).forecastedAt())
        .isEqualTo(Instant.parse("2026-01-09T12:00:00Z"));
    } finally {
      server.shutdown();
    }
  }

  @Test
  void skipsNullTmpkRows() throws Exception {
    MockWebServer server = startServer();
    try {
    String ndjsonBody = String.join("\n",
        "{\"forecasted_at\":\"2026-01-09T12:00:00Z\",\"forecasted_time\":\"2026-01-10T12:00:00Z\","
            + "\"lat\":40.77898,\"lon\":-73.96925,\"name\":\"KNYC\",\"tmpk\":null}",
        "{\"forecasted_at\":\"2026-01-09T12:00:00Z\",\"forecasted_time\":\"2026-01-10T18:00:00Z\","
            + "\"lat\":40.77898,\"lon\":-73.96925,\"name\":\"KNYC\",\"tmpk\":282.0}");
    server.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/ndjson")
        .setBody(ndjsonBody));

    GribstreamProperties properties = new GribstreamProperties();
    properties.setBaseUrl(server.url("/").toString());
    properties.setApiToken("test-token");
    GribstreamClient client = new GribstreamClient(
        properties,
        new ObjectMapper(),
        HttpClientSettings.defaultSettings());

    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        "2026-01-10T05:00:00Z",
        "2026-01-11T05:00:00Z",
        "2026-01-09T17:00:00Z",
        0,
        48,
        List.of(new GribstreamCoordinate(40.77898, -73.96925, "KNYC")),
        List.of(new GribstreamVariable("TMP", "2 m above ground", "", "tmpk")),
        null);

    GribstreamClientResponse response = client.fetchHistory("rap", request);
    assertThat(response.rows()).hasSize(1);
    assertThat(response.rows().get(0).tmpk()).isEqualTo(282.0);
    } finally {
      server.shutdown();
    }
  }

  private static String loadFixture(String path) throws IOException {
    try (InputStream input = GribstreamClientParsingTest.class.getClassLoader().getResourceAsStream(path)) {
      if (input == null) {
        throw new IllegalArgumentException("Fixture not found: " + path);
      }
      return new String(input.readAllBytes(), StandardCharsets.UTF_8);
    }
  }

  private static MockResponse gzipResponse(String body) throws IOException {
    Buffer buffer = new Buffer();
    buffer.write(gzipBytes(body));
    return new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/ndjson")
        .setHeader("Content-Encoding", "gzip")
        .setBody(buffer);
  }

  private static byte[] gzipBytes(String body) throws IOException {
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    try (GZIPOutputStream gzip = new GZIPOutputStream(output)) {
      gzip.write(body.getBytes(StandardCharsets.UTF_8));
    }
    return output.toByteArray();
  }

  private static MockWebServer startServer() throws IOException {
    MockWebServer server = new MockWebServer();
    server.start();
    return server;
  }
}
