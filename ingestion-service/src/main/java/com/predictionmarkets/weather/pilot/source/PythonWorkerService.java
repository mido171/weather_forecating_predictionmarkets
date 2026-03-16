package com.predictionmarkets.weather.pilot.source;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class PythonWorkerService {
  private final ObjectMapper objectMapper;
  private final JobRunService jobRunService;
  private final PilotIngestionProperties properties;

  public PythonWorkerService(ObjectMapper objectMapper,
                             JobRunService jobRunService,
                             PilotIngestionProperties properties) {
    this.objectMapper = objectMapper;
    this.jobRunService = jobRunService;
    this.properties = properties;
  }

  public JsonNode runWorker(String worker,
                            Object request,
                            String jobId,
                            String runId,
                            String stationKey) {
    Path workerDir = workerDirectory();
    Path tempDir = workerTempDirectory();
    try {
      Files.createDirectories(tempDir);
      String token = UUID.randomUUID().toString();
      Path requestPath = tempDir.resolve(worker + "_" + token + "_request.json");
      Path outputPath = tempDir.resolve(worker + "_" + token + "_output.json");
      Files.writeString(requestPath, objectMapper.writeValueAsString(request), StandardCharsets.UTF_8);
      ProcessBuilder processBuilder = new ProcessBuilder(
          "python",
          "-m",
          "app.cli",
          "--worker",
          worker,
          "--request-json",
          requestPath.toString(),
          "--output-json",
          outputPath.toString(),
          "--log-level",
          "INFO");
      processBuilder.directory(workerDir.toFile());
      processBuilder.redirectErrorStream(true);
      processBuilder.environment().put("PYTHONIOENCODING", "utf-8");
      long started = System.nanoTime();
      Process process = processBuilder.start();
      String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
      int exitCode = process.waitFor();
      double durationMs = (System.nanoTime() - started) / 1_000_000.0d;
      Map<String, Object> payload = new LinkedHashMap<>();
      payload.put("worker", worker);
      payload.put("exitCode", exitCode);
      payload.put("durationMs", durationMs);
      payload.put("stdout", truncate(output, 4000));
      payload.put("requestPath", requestPath.toString());
      payload.put("outputPath", outputPath.toString());
      payload.put("completedAtUtc", Instant.now().toString());
      jobRunService.logStructuredEvent(jobId, runId, stationKey, worker,
          "worker_process_completed", exitCode == 0 ? "SUCCESS" : "FAILED", payload);
      if (exitCode != 0) {
        throw new IllegalStateException("Worker " + worker + " failed with exit code " + exitCode);
      }
      return objectMapper.readTree(Files.readString(outputPath, StandardCharsets.UTF_8));
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to execute worker " + worker, ex);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while waiting for worker " + worker, ex);
    }
  }

  private Path workerDirectory() {
    Path ingestionServiceDir = Path.of(properties.getConfigDir()).toAbsolutePath().normalize().getParent();
    if (ingestionServiceDir == null) {
      throw new IllegalStateException("Unable to resolve ingestion-service directory from " + properties.getConfigDir());
    }
    return ingestionServiceDir.resolve("workers").resolve("python");
  }

  private Path workerTempDirectory() {
    Path ingestionServiceDir = Path.of(properties.getConfigDir()).toAbsolutePath().normalize().getParent();
    if (ingestionServiceDir == null) {
      throw new IllegalStateException("Unable to resolve ingestion-service directory from " + properties.getConfigDir());
    }
    return ingestionServiceDir.resolve("data").resolve("tmp").resolve("worker_requests");
  }

  private String truncate(String text, int maxChars) {
    if (text == null || text.length() <= maxChars) {
      return text;
    }
    return text.substring(0, maxChars) + "...";
  }
}
