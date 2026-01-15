package com.predictionmarkets.weather.executors;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.gribstream.GribstreamProperties;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

public final class GribstreamDiscoveryExecutor {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamDiscoveryExecutor.class);
  private static final DateTimeFormatter RUN_ID_FORMAT =
      DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").withZone(ZoneOffset.UTC);

  private GribstreamDiscoveryExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamProperties properties = context.getBean(GribstreamProperties.class);
      ObjectMapper mapper = context.getBean(ObjectMapper.class);
      DiscoveryRunner runner = new DiscoveryRunner(properties, mapper);
      runner.run();
    }
  }

  private static final class DiscoveryRunner {
    private final GribstreamProperties properties;
    private final ObjectMapper objectMapper;
    private final OkHttpClient httpClient;
    private final String authHeader;

    private DiscoveryRunner(GribstreamProperties properties, ObjectMapper objectMapper) {
      this.properties = Objects.requireNonNull(properties, "properties is required");
      this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper is required");
      this.httpClient = new OkHttpClient.Builder()
          .connectTimeout(properties.getConnectTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
          .readTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
          .callTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
          .writeTimeout(properties.getReadTimeoutMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
          .retryOnConnectionFailure(true)
          .build();
      this.authHeader = buildAuthorizationHeader(properties.getApiToken(), properties.getAuthScheme());
    }

    private void run() {
      String runId = RUN_ID_FORMAT.format(Instant.now());
      String baseUrl = normalizeBaseUrl(properties.getBaseUrl());
      List<Map<String, Object>> manifest = new ArrayList<>();

      snapshot("Starting Gribstream discovery (console output only)");

      List<DiscoveryTarget> targets = buildTargets(baseUrl, resolveModels());
      for (DiscoveryTarget target : targets) {
        Map<String, Object> result = fetchAndDump(target);
        manifest.add(result);
      }

      Map<String, Object> summary = new LinkedHashMap<>();
      summary.put("run_id", runId);
      summary.put("base_url", baseUrl);
      summary.put("timestamp_utc", Instant.now().toString());
      summary.put("targets", manifest);

      snapshot("Gribstream discovery complete. targets=" + manifest.size());
      String summaryJson = toJson(summary);
      System.out.println("===== GRIBSTREAM DISCOVERY SUMMARY =====");
      System.out.println(summaryJson);
      System.out.println("===== END GRIBSTREAM DISCOVERY SUMMARY =====");
    }

    private Map<String, Object> fetchAndDump(DiscoveryTarget target) {
      Map<String, Object> entry = new LinkedHashMap<>();
      entry.put("label", target.label());
      entry.put("url", target.url());
      entry.put("method", "GET");
      entry.put("timestamp_utc", Instant.now().toString());
      Request request = new Request.Builder()
          .url(target.url())
          .get()
          .header("Authorization", authHeader)
          .header("Accept", target.accept())
          .header("Accept-Encoding", "gzip")
          .build();

      try (Response response = httpClient.newCall(request).execute()) {
        ResponseBody body = response.body();
        byte[] raw = body == null ? new byte[0] : body.bytes();
        byte[] payload = maybeDecodeGzip(response, raw);
        String contentType = response.header("Content-Type");
        int status = response.code();

        String textPayload = new String(payload, StandardCharsets.UTF_8);

        entry.put("status", status);
        entry.put("content_type", contentType);
        entry.put("bytes", payload.length);
        entry.put("sha256", Hashing.sha256Hex(payload));
        entry.put("output_path", null);

        snapshot("Fetched " + target.label() + " status=" + status + " bytes=" + payload.length);
        System.out.println("===== GRIBSTREAM DISCOVERY START " + target.label() + " =====");
        System.out.println("URL: " + target.url());
        System.out.println("STATUS: " + status + " CONTENT_TYPE: " + contentType);
        System.out.println(textPayload);
        System.out.println("===== GRIBSTREAM DISCOVERY END " + target.label() + " =====");
        return entry;
      } catch (IOException ex) {
        String errorMessage = ex.getClass().getSimpleName() + ": " + ex.getMessage();
        entry.put("status", "error");
        entry.put("error", errorMessage);
        snapshot("Failed " + target.label() + " error=" + errorMessage);
        System.out.println("===== GRIBSTREAM DISCOVERY ERROR " + target.label() + " =====");
        System.out.println(errorMessage);
        System.out.println("===== END GRIBSTREAM DISCOVERY ERROR " + target.label() + " =====");
        return entry;
      }
    }

    private List<DiscoveryTarget> buildTargets(String baseUrl, List<String> models) {
      List<DiscoveryTarget> targets = new ArrayList<>();
      targets.add(new DiscoveryTarget("models_index", baseUrl + "/models", "text/html",
          "models_index.html"));
      for (String suffix : List.of("openapi.json", "openapi.yaml", "openapi", "swagger.json")) {
        targets.add(new DiscoveryTarget("openapi_" + suffix.replace('.', '_'),
            baseUrl + "/" + suffix,
            "application/json",
            "openapi_" + suffix.replace('.', '_') + ".txt"));
      }
      for (String endpoint : List.of("/api/v2/models", "/api/v2/catalog", "/api/v2/variables")) {
        targets.add(new DiscoveryTarget("catalog_" + sanitize(endpoint),
            baseUrl + endpoint,
            "application/json",
            "catalog_" + sanitize(endpoint) + ".txt"));
      }
      for (String model : models) {
        targets.add(new DiscoveryTarget("model_page_" + model,
            baseUrl + "/models/" + model,
            "text/html",
            "model_" + model + ".html"));
        for (String suffix : List.of("variables", "inventory", "metadata")) {
          String endpoint = "/api/v2/" + model + "/" + suffix;
          targets.add(new DiscoveryTarget("model_" + model + "_" + suffix,
              baseUrl + endpoint,
              "application/json",
              "model_" + model + "_" + suffix + ".txt"));
        }
      }
      return targets;
    }

    private List<String> resolveModels() {
      Set<String> models = new LinkedHashSet<>();
      if (properties.getModels() != null) {
        models.addAll(properties.getModels().keySet());
      }
      models.add("gfs");
      models.add("nam");
      models.add("hrrr");
      models.add("rap");
      models.add("nbm");
      models.add("gefsatmos");
      models.add("gefsatmosmean");
      List<String> ordered = new ArrayList<>();
      for (String model : models) {
        if (model == null || model.isBlank()) {
          continue;
        }
        ordered.add(model.trim().toLowerCase(Locale.ROOT));
      }
      return ordered;
    }

    private String toJson(Object payload) {
      try {
        return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(payload);
      } catch (IOException ex) {
        throw new UncheckedIOException("Failed to serialize JSON", ex);
      }
    }
  }

  private static String sanitize(String endpoint) {
    return endpoint.replace("/", "_").replace("-", "_").replace(".", "_");
  }

  private static String normalizeBaseUrl(String baseUrl) {
    if (baseUrl == null || baseUrl.isBlank()) {
      throw new IllegalArgumentException("gribstream.baseUrl is required");
    }
    String trimmed = baseUrl.trim();
    if (trimmed.endsWith("/")) {
      return trimmed.substring(0, trimmed.length() - 1);
    }
    return trimmed;
  }

  private static String buildAuthorizationHeader(String apiToken, String authScheme) {
    if (apiToken == null || apiToken.isBlank()) {
      throw new IllegalArgumentException("gribstream.apiToken is required");
    }
    String trimmed = apiToken.trim();
    if (trimmed.contains(" ")) {
      return trimmed;
    }
    String scheme = (authScheme == null || authScheme.isBlank()) ? "Bearer" : authScheme.trim();
    return scheme + " " + trimmed;
  }

  private static byte[] maybeDecodeGzip(Response response, byte[] payload) {
    String encoding = response.header("Content-Encoding");
    if (encoding == null || encoding.isBlank()) {
      return payload;
    }
    for (String part : encoding.split(",")) {
      if ("gzip".equalsIgnoreCase(part.trim())) {
        return decompressGzip(payload);
      }
    }
    return payload;
  }

  private static byte[] decompressGzip(byte[] payload) {
    try (GZIPInputStream gzip = new GZIPInputStream(new java.io.ByteArrayInputStream(payload))) {
      return gzip.readAllBytes();
    } catch (IOException ex) {
      throw new UncheckedIOException(ex);
    }
  }

  private static void snapshot(String message) {
    String payload = "[GRIBSTREAM-DISCOVERY] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private record DiscoveryTarget(String label, String url, String accept, String fileName) {
  }
}
