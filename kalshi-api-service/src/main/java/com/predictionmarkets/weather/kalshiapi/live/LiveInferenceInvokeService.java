package com.predictionmarkets.weather.kalshiapi.live;

import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
public class LiveInferenceInvokeService {

  private static final Logger log = LoggerFactory.getLogger(LiveInferenceInvokeService.class);
  private static final Pattern REPORT_PATH_PATTERN =
      Pattern.compile("\"report_path\"\\s*:\\s*\"([^\"]+)\"");
  private static final int MAX_MESSAGE_CHARS = 2_000;

  private final LiveTradingProperties properties;
  private final AtomicBoolean running = new AtomicBoolean(false);

  public LiveInferenceInvokeService(LiveTradingProperties properties) {
    this.properties = properties;
  }

  public LiveInferenceRunResponse invokeForDate(LocalDate targetDateLocal) {
    if (targetDateLocal == null) {
      return new LiveInferenceRunResponse(null, "invalid", null, null, "targetDateLocal is required");
    }
    if (!properties.isInferenceInvokeEnabled()) {
      return new LiveInferenceRunResponse(
          targetDateLocal.toString(),
          "disabled",
          null,
          null,
          "Inference invocation is disabled by configuration");
    }
    if (!running.compareAndSet(false, true)) {
      return new LiveInferenceRunResponse(
          targetDateLocal.toString(),
          "busy",
          null,
          null,
          "Inference invocation already in progress");
    }

    try {
      Path scriptPath = resolveScriptPath();
      Path workingDir = resolveWorkingDirectory(scriptPath);
      List<String> command = buildCommand(scriptPath, targetDateLocal);
      ProcessBuilder processBuilder = new ProcessBuilder(command);
      processBuilder.directory(workingDir.toFile());
      processBuilder.redirectErrorStream(true);

      Process process = processBuilder.start();
      boolean finished = process.waitFor(
          Math.max(10, properties.getInferenceInvokeTimeoutSeconds()),
          TimeUnit.SECONDS);
      if (!finished) {
        process.destroyForcibly();
        return new LiveInferenceRunResponse(
            targetDateLocal.toString(),
            "timeout",
            null,
            null,
            "Inference script timed out");
      }

      int exitCode = process.exitValue();
      String output = readStream(process.getInputStream());
      String reportPath = extractReportPath(output);
      if (exitCode == 0) {
        log.info("Live inference invoke success targetDate={} reportPath={}", targetDateLocal, reportPath);
        return new LiveInferenceRunResponse(
            targetDateLocal.toString(),
            "success",
            exitCode,
            reportPath,
            abbreviateMessage(output));
      }
      log.warn("Live inference invoke failed targetDate={} exitCode={} output={}",
          targetDateLocal, exitCode, abbreviateMessage(output));
      return new LiveInferenceRunResponse(
          targetDateLocal.toString(),
          "failed",
          exitCode,
          reportPath,
          abbreviateMessage(output));
    } catch (Exception ex) {
      log.warn("Live inference invoke error targetDate={} error={}", targetDateLocal, ex.toString());
      return new LiveInferenceRunResponse(
          targetDateLocal.toString(),
          "error",
          null,
          null,
          ex.getMessage());
    } finally {
      running.set(false);
    }
  }

  private List<String> buildCommand(Path scriptPath, LocalDate targetDateLocal) {
    List<String> command = new ArrayList<>();
    command.add(properties.getInferenceInvokePythonExecutable().trim());
    command.add(scriptPath.toString());
    command.add("--target-date");
    command.add(targetDateLocal.toString());
    command.add("--stdout-json");
    command.add("summary");
    command.add("--log-level");
    command.add("ERROR");
    return command;
  }

  private Path resolveScriptPath() {
    String configured = properties.getInferenceInvokeScriptPath();
    if (!StringUtils.hasText(configured)) {
      throw new IllegalStateException("Inference script path is blank");
    }

    Path configuredPath = Path.of(configured.trim());
    if (configuredPath.isAbsolute()) {
      if (Files.isRegularFile(configuredPath)) {
        return configuredPath.normalize();
      }
      throw new IllegalStateException("Inference script path not found: " + configuredPath);
    }

    Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    for (int i = 0; i < 8 && current != null; i += 1) {
      Path candidate = current.resolve(configuredPath).normalize();
      if (Files.isRegularFile(candidate)) {
        return candidate;
      }
      current = current.getParent();
    }
    throw new IllegalStateException("Inference script path not found relative to working tree: " + configuredPath);
  }

  private Path resolveWorkingDirectory(Path scriptPath) {
    Path liveDir = scriptPath.getParent();
    if (liveDir == null) {
      return Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    }
    Path toolsDir = liveDir.getParent();
    if (toolsDir == null) {
      return Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    }
    Path repoRoot = toolsDir.getParent();
    if (repoRoot != null && Files.isRegularFile(repoRoot.resolve("pom.xml"))) {
      return repoRoot;
    }
    return Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
  }

  private String readStream(InputStream inputStream) throws IOException {
    byte[] bytes = inputStream.readAllBytes();
    if (bytes.length == 0) {
      return "";
    }
    return new String(bytes, StandardCharsets.UTF_8);
  }

  private String extractReportPath(String output) {
    if (!StringUtils.hasText(output)) {
      return null;
    }
    Matcher matcher = REPORT_PATH_PATTERN.matcher(output);
    String reportPath = null;
    while (matcher.find()) {
      reportPath = matcher.group(1);
    }
    return StringUtils.hasText(reportPath) ? reportPath : null;
  }

  private String abbreviateMessage(String raw) {
    if (!StringUtils.hasText(raw)) {
      return null;
    }
    String normalized = raw.replace("\r\n", "\n").trim();
    if (normalized.length() <= MAX_MESSAGE_CHARS) {
      return normalized;
    }
    return normalized.substring(0, MAX_MESSAGE_CHARS) + "...";
  }
}
