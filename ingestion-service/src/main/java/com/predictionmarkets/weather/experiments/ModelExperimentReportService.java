package com.predictionmarkets.weather.experiments;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.models.ModelExperiment;
import com.predictionmarkets.weather.repository.ModelExperimentRepository;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class ModelExperimentReportService {
  private static final Logger logger = LoggerFactory.getLogger(ModelExperimentReportService.class);

  private final ModelExperimentRepository repository;
  private final ObjectMapper objectMapper;

  public ModelExperimentReportService(ModelExperimentRepository repository, ObjectMapper objectMapper) {
    this.repository = repository;
    this.objectMapper = objectMapper;
  }

  public Path writeAggregatedReport(ModelExperimentReportProperties properties) {
    List<String> stationFilter = properties.getStationFilter();
    List<String> normalizedStations = normalizeStations(stationFilter);
    List<ModelExperiment> experiments = repository.findAll().stream()
        .filter(exp -> normalizedStations.isEmpty()
            || (exp.getStationId() != null
                && normalizedStations.contains(exp.getStationId().toUpperCase(Locale.ROOT))))
        .sorted(Comparator
            .comparing(ModelExperiment::getTestMae, Comparator.nullsLast(Double::compareTo))
            .thenComparing(exp -> safeLower(exp.getExperimentKey())))
        .collect(Collectors.toList());

    int limit = properties.getLimit();
    if (limit > 0 && experiments.size() > limit) {
      experiments = experiments.subList(0, limit);
    }

    String reportText = renderReport(experiments, normalizedStations);
    Path outputDir = resolveOutputDir(properties.getOutputDir());
    try {
      Files.createDirectories(outputDir);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to create report output dir: " + outputDir, ex);
    }

    String timestamp = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'")
        .withZone(java.time.ZoneOffset.UTC)
        .format(Instant.now());
    String stationToken = normalizedStations.isEmpty()
        ? "all"
        : String.join("-", normalizedStations);
    Path outputPath = outputDir.resolve("model-experiments-" + stationToken + "-" + timestamp + ".txt");
    try {
      Files.writeString(outputPath, reportText, StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write experiment report: " + outputPath, ex);
    }

    logger.info("Model experiment report written: {}", outputPath);
    return outputPath;
  }

  private Path resolveOutputDir(String outputDir) {
    if (outputDir == null || outputDir.isBlank()) {
      return resolveRepoRoot().resolve("artifacts/experiment_reports").toAbsolutePath().normalize();
    }
    Path path = Path.of(outputDir);
    if (path.isAbsolute()) {
      return path.normalize();
    }
    return resolveRepoRoot().resolve(path).toAbsolutePath().normalize();
  }

  private Path resolveRepoRoot() {
    Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    Path candidate = current;
    for (int i = 0; i < 8; i++) {
      if (Files.exists(candidate.resolve("pom.xml"))
          && Files.exists(candidate.resolve("models"))
          && Files.exists(candidate.resolve("ingestion-service"))) {
        return candidate;
      }
      Path parent = candidate.getParent();
      if (parent == null) {
        break;
      }
      candidate = parent;
    }
    return current;
  }

  private String renderReport(List<ModelExperiment> experiments, List<String> normalizedStations) {
    StringBuilder builder = new StringBuilder();
    builder.append("MODEL_EXPERIMENT_REPORT\n");
    builder.append("generated_at_utc: ").append(Instant.now()).append("\n");
    if (normalizedStations != null && !normalizedStations.isEmpty()) {
      builder.append("station_filter: ")
          .append(String.join(",", normalizedStations))
          .append("\n");
    } else {
      builder.append("station_filter: none\n");
    }
    builder.append("total_records: ").append(experiments.size()).append("\n\n");

    for (ModelExperiment experiment : experiments) {
      builder.append("EXPERIMENT_START\n");
      builder.append("id: ").append(experiment.getId()).append("\n");
      builder.append("experiment_key: ").append(experiment.getExperimentKey()).append("\n");
      builder.append("experiment_name: ").append(nullToDash(experiment.getExperimentName())).append("\n");
      builder.append("station_id: ").append(nullToDash(experiment.getStationId())).append("\n");
      builder.append("model_name: ").append(nullToDash(experiment.getModelName())).append("\n");
      builder.append("model_family: ").append(nullToDash(experiment.getModelFamily())).append("\n");
      builder.append("source_path: ").append(nullToDash(experiment.getSourcePath())).append("\n");
      builder.append("train_mae: ").append(nullToDash(experiment.getTrainMae())).append("\n");
      builder.append("train_rmse: ").append(nullToDash(experiment.getTrainRmse())).append("\n");
      builder.append("train_bias: ").append(nullToDash(experiment.getTrainBias())).append("\n");
      builder.append("train_median_ae: ").append(nullToDash(experiment.getTrainMedianAe())).append("\n");
      builder.append("train_max_ae: ").append(nullToDash(experiment.getTrainMaxAe())).append("\n");
      builder.append("train_corr: ").append(nullToDash(experiment.getTrainCorr())).append("\n");
      builder.append("train_n: ").append(nullToDash(experiment.getTrainN())).append("\n");
      builder.append("validation_mae: ").append(nullToDash(experiment.getValidationMae())).append("\n");
      builder.append("validation_rmse: ").append(nullToDash(experiment.getValidationRmse())).append("\n");
      builder.append("validation_bias: ").append(nullToDash(experiment.getValidationBias())).append("\n");
      builder.append("validation_median_ae: ").append(nullToDash(experiment.getValidationMedianAe())).append("\n");
      builder.append("validation_max_ae: ").append(nullToDash(experiment.getValidationMaxAe())).append("\n");
      builder.append("validation_corr: ").append(nullToDash(experiment.getValidationCorr())).append("\n");
      builder.append("validation_n: ").append(nullToDash(experiment.getValidationN())).append("\n");
      builder.append("test_mae: ").append(nullToDash(experiment.getTestMae())).append("\n");
      builder.append("test_rmse: ").append(nullToDash(experiment.getTestRmse())).append("\n");
      builder.append("test_bias: ").append(nullToDash(experiment.getTestBias())).append("\n");
      builder.append("test_median_ae: ").append(nullToDash(experiment.getTestMedianAe())).append("\n");
      builder.append("test_max_ae: ").append(nullToDash(experiment.getTestMaxAe())).append("\n");
      builder.append("test_corr: ").append(nullToDash(experiment.getTestCorr())).append("\n");
      builder.append("test_n: ").append(nullToDash(experiment.getTestN())).append("\n");
      builder.append("description_text: ").append(nullToDash(experiment.getDescriptionText())).append("\n");
      builder.append("raw_payload_hash: ").append(nullToDash(experiment.getRawPayloadHash())).append("\n");
      builder.append("retrieved_at_utc: ").append(nullToDash(experiment.getRetrievedAtUtc())).append("\n");
      builder.append("created_at_utc: ").append(nullToDash(experiment.getCreatedAtUtc())).append("\n");
      builder.append("updated_at_utc: ").append(nullToDash(experiment.getUpdatedAtUtc())).append("\n");
      builder.append("metrics_train_json: ").append(formatJson(experiment.getMetricsTrainJson())).append("\n");
      builder.append("metrics_validation_json: ").append(formatJson(experiment.getMetricsValidationJson())).append("\n");
      builder.append("metrics_test_json: ").append(formatJson(experiment.getMetricsTestJson())).append("\n");
      builder.append("metadata_json:\n").append(formatJsonBlock(experiment.getMetadataJson())).append("\n");
      builder.append("EXPERIMENT_END\n\n");
    }

    return builder.toString();
  }

  private String formatJson(String json) {
    if (json == null || json.isBlank()) {
      return "-";
    }
    return json.replace("\n", " ");
  }

  private String formatJsonBlock(String json) {
    if (json == null || json.isBlank()) {
      return "-";
    }
    try {
      JsonNode node = objectMapper.readTree(json);
      return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(node);
    } catch (JsonProcessingException ex) {
      return json;
    }
  }

  private String nullToDash(Object value) {
    if (value == null) {
      return "-";
    }
    if (value instanceof Double) {
      return String.format(Locale.ROOT, "%.4f", (Double) value);
    }
    return value.toString();
  }

  private String safeLower(String value) {
    return value == null ? "" : value.toLowerCase(Locale.ROOT);
  }

  private List<String> normalizeStations(List<String> stations) {
    if (stations == null || stations.isEmpty()) {
      return List.of();
    }
    return stations.stream()
        .filter(station -> station != null && !station.isBlank())
        .map(station -> station.trim().toUpperCase(Locale.ROOT))
        .distinct()
        .collect(Collectors.toList());
  }
}
