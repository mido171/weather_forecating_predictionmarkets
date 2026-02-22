package com.predictionmarkets.weather.experiments;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.ModelExperiment;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class ModelExperimentIngestService {
  private static final Logger logger = LoggerFactory.getLogger(ModelExperimentIngestService.class);
  private static final String METADATA_SCHEMA_VERSION = "model_experiment_v1";

  private final ObjectMapper objectMapper;
  private final ModelExperimentService modelExperimentService;
  private final ModelExperimentIngestProperties properties;
  private final ModelExperimentDescriptionService descriptionService;

  public ModelExperimentIngestService(ObjectMapper objectMapper,
                                      ModelExperimentService modelExperimentService,
                                      ModelExperimentIngestProperties properties,
                                      ModelExperimentDescriptionService descriptionService) {
    this.objectMapper = objectMapper;
    this.modelExperimentService = modelExperimentService;
    this.properties = properties;
    this.descriptionService = descriptionService;
  }

  public ModelExperimentIngestReport ingestAll() {
    Path repoRoot = resolveRepoRoot();
    List<Path> scanRoots = resolveScanRoots(repoRoot, properties.getScanRoots());
    int candidateFiles = 0;
    int ingested = 0;
    int skipped = 0;
    int errors = 0;

    for (Path root : scanRoots) {
      if (!Files.exists(root)) {
        logger.warn("Experiment ingest scan root missing: {}", root);
        continue;
      }
      try (Stream<Path> stream = Files.walk(root)) {
        for (Path file : stream.filter(Files::isRegularFile).toList()) {
          if (!isCandidateFile(file)) {
            continue;
          }
          candidateFiles++;
          try {
            List<ModelExperiment> experiments = parseFile(file, repoRoot);
            if (experiments.isEmpty()) {
              skipped++;
              continue;
            }
            for (ModelExperiment experiment : experiments) {
              modelExperimentService.upsert(experiment);
              ingested++;
            }
          } catch (RuntimeException ex) {
            errors++;
            logger.warn("Failed to ingest experiment file: {}", file, ex);
          }
        }
      } catch (IOException ex) {
        errors++;
        logger.warn("Failed to scan experiment root: {}", root, ex);
      }
    }

    ModelExperimentIngestReport report = new ModelExperimentIngestReport(
        repoRoot.toString(),
        scanRoots.stream().map(Path::toString).toList(),
        candidateFiles,
        ingested,
        skipped,
        errors);
    logger.info("Model experiment ingest done: {}", report);
    return report;
  }

  private List<ModelExperiment> parseFile(Path file, Path repoRoot) {
    String fileName = file.getFileName().toString();
    String raw;
    try {
      raw = Files.readString(file, StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read experiment file: " + file, ex);
    }

    JsonNode root;
    try {
      root = objectMapper.readTree(raw);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Invalid JSON in experiment file: " + file, ex);
    }

    String lowerName = fileName.toLowerCase(Locale.ROOT);
    if (root.isArray()) {
      return parseResultsArray(file, root, raw, repoRoot);
    }

    if ("summary.json".equals(lowerName) && hasResultsSibling(file)) {
      return List.of();
    }

    if ("summary.json".equals(lowerName)) {
      return parseSummaryBest(file, root, raw, repoRoot);
    }

    return parseMetricsOrReport(file, root, raw, repoRoot);
  }

  private List<ModelExperiment> parseResultsArray(Path file,
                                                  JsonNode root,
                                                  String raw,
                                                  Path repoRoot) {
    List<ModelExperiment> experiments = new ArrayList<>();
    String stationId = resolveStationIdWithExtras(file, raw);
    if (!shouldIncludeStation(stationId, raw, file)) {
      return List.of();
    }
    JsonNode summaryNode = loadSummarySibling(file);

    int index = 0;
    for (JsonNode item : root) {
      String name = readText(item, "name");
      String kind = readText(item, "kind");
      String experimentKey = buildExperimentKey(repoRoot, file, name, index);
      MetricsNodes metrics = extractMetrics(item, stationId);
      ExperimentMetadata metadata = buildMetadata(file, repoRoot, "results", item, summaryNode);

      ModelExperiment experiment = buildExperiment(
          experimentKey,
          name,
          stationId,
          deriveModelName(item, kind),
          deriveModelFamily(item, kind),
          metadata,
          metrics,
          file,
          repoRoot);
      experiments.add(experiment);
      index++;
    }
    return experiments;
  }

  private List<ModelExperiment> parseSummaryBest(Path file,
                                                 JsonNode root,
                                                 String raw,
                                                 Path repoRoot) {
    JsonNode best = root.path("best");
    if (best.isMissingNode() || best.isNull()) {
      return List.of();
    }
    String stationId = resolveStationIdWithExtras(file, raw);
    if (!shouldIncludeStation(stationId, raw, file)) {
      return List.of();
    }
    String name = readText(best, "name");
    String kind = readText(best, "kind");
    String experimentKey = buildExperimentKey(repoRoot, file, name, null);
    MetricsNodes metrics = extractMetrics(best, stationId);
    ExperimentMetadata metadata = buildMetadata(file, repoRoot, "summary", best, root);

    ModelExperiment experiment = buildExperiment(
        experimentKey,
        name == null ? "summary-best" : name,
        stationId,
        deriveModelName(best, kind),
        deriveModelFamily(best, kind),
        metadata,
        metrics,
        file,
        repoRoot);
    return List.of(experiment);
  }

  private List<ModelExperiment> parseMetricsOrReport(Path file,
                                                     JsonNode root,
                                                     String raw,
                                                     Path repoRoot) {
    String stationId = resolveStationIdWithExtras(file, raw);
    if (!shouldIncludeStation(stationId, raw, file)) {
      return List.of();
    }

    JsonNode metricsRoot = root.has("metrics") ? root.path("metrics") : root;
    MetricsNodes metrics = extractMetrics(metricsRoot, stationId);
    if (metrics.isEmpty()) {
      return List.of();
    }

    String name = deriveExperimentName(file, root);
    String modelName = deriveModelName(root, null);
    String modelFamily = deriveModelFamily(root, null);
    String experimentKey = buildExperimentKey(repoRoot, file, name, null);
    ExperimentMetadata metadata = buildMetadata(file, repoRoot, "metrics", root, null);

    ModelExperiment experiment = buildExperiment(
        experimentKey,
        name,
        stationId,
        modelName,
        modelFamily,
        metadata,
        metrics,
        file,
        repoRoot);
    return List.of(experiment);
  }

  private ExperimentMetadata buildMetadata(Path file,
                                           Path repoRoot,
                                           String type,
                                           JsonNode primaryNode,
                                           JsonNode extraNode) {
    Map<String, Object> raw = new LinkedHashMap<>();
    raw.put("primary", primaryNode);
    if (extraNode != null) {
      raw.put("extra", extraNode);
    }

    Map<String, Object> extras = new LinkedHashMap<>();
    for (Path extraFile : findExtraFiles(file)) {
      Object value = readExtraFile(extraFile);
      if (value != null) {
        extras.put(relativePath(repoRoot, extraFile), value);
      }
    }

    Map<String, Object> metadata = new LinkedHashMap<>();
    metadata.put("schema_version", METADATA_SCHEMA_VERSION);
    metadata.put("primary_type", type);
    metadata.put("primary_path", relativePath(repoRoot, file));
    metadata.put("raw", raw);
    if (!extras.isEmpty()) {
      metadata.put("extras", extras);
    }

    String metadataJson;
    try {
      metadataJson = objectMapper.writeValueAsString(metadata);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize metadata for " + file, ex);
    }

    return new ExperimentMetadata(metadataJson, metadata);
  }

  private ModelExperiment buildExperiment(String experimentKey,
                                          String experimentName,
                                          String stationId,
                                          String modelName,
                                          String modelFamily,
                                          ExperimentMetadata metadata,
                                          MetricsNodes metrics,
                                          Path file,
                                          Path repoRoot) {
    ModelExperiment experiment = new ModelExperiment();
    experiment.setExperimentKey(experimentKey);
    experiment.setExperimentName(experimentName);
    experiment.setStationId(stationId);
    experiment.setModelName(modelName);
    experiment.setModelFamily(modelFamily);
    experiment.setSourcePath(relativePath(repoRoot, file));
    experiment.setMetadataJson(metadata.metadataJson());

    if (metrics.trainNode() != null) {
      experiment.setMetricsTrainJson(toJson(metrics.trainNode()));
      MetricsValues values = readMetrics(metrics.trainNode());
      applyTrainMetrics(experiment, values);
    }
    if (metrics.validationNode() != null) {
      experiment.setMetricsValidationJson(toJson(metrics.validationNode()));
      MetricsValues values = readMetrics(metrics.validationNode());
      applyValidationMetrics(experiment, values);
    }
    if (metrics.testNode() != null) {
      experiment.setMetricsTestJson(toJson(metrics.testNode()));
      MetricsValues values = readMetrics(metrics.testNode());
      applyTestMetrics(experiment, values);
    }

    experiment.setDescriptionText(descriptionService.buildHighLevelDescription(experiment));
    experiment.setRawPayloadHash(Hashing.sha256Hex(metadata.metadataJson()));
    experiment.setRetrievedAtUtc(Instant.now());
    return experiment;
  }

  private void applyTrainMetrics(ModelExperiment experiment, MetricsValues values) {
    experiment.setTrainMae(values.mae());
    experiment.setTrainRmse(values.rmse());
    experiment.setTrainBias(values.bias());
    experiment.setTrainMedianAe(values.medianAe());
    experiment.setTrainMaxAe(values.maxAe());
    experiment.setTrainCorr(values.corr());
    experiment.setTrainN(values.n());
  }

  private void applyValidationMetrics(ModelExperiment experiment, MetricsValues values) {
    experiment.setValidationMae(values.mae());
    experiment.setValidationRmse(values.rmse());
    experiment.setValidationBias(values.bias());
    experiment.setValidationMedianAe(values.medianAe());
    experiment.setValidationMaxAe(values.maxAe());
    experiment.setValidationCorr(values.corr());
    experiment.setValidationN(values.n());
  }

  private void applyTestMetrics(ModelExperiment experiment, MetricsValues values) {
    experiment.setTestMae(values.mae());
    experiment.setTestRmse(values.rmse());
    experiment.setTestBias(values.bias());
    experiment.setTestMedianAe(values.medianAe());
    experiment.setTestMaxAe(values.maxAe());
    experiment.setTestCorr(values.corr());
    experiment.setTestN(values.n());
  }


  private MetricsNodes extractMetrics(JsonNode node, String stationId) {
    JsonNode train = node.path("train");
    JsonNode validation = node.path("validation");
    if (validation.isMissingNode()) {
      validation = node.path("val");
    }
    JsonNode test = node.path("test");

    boolean hasExplicitSplits = !(train.isMissingNode() && validation.isMissingNode() && test.isMissingNode());
    if (!hasExplicitSplits && hasFlatMetrics(node)) {
      test = node;
    }

    JsonNode perStationTest = node.path("per_station_test");
    if (stationId != null && perStationTest.has(stationId)) {
      test = perStationTest.path(stationId);
    }
    JsonNode perStationTrain = node.path("per_station_train");
    if (stationId != null && perStationTrain.has(stationId)) {
      train = perStationTrain.path(stationId);
    }
    JsonNode perStationValidation = node.path("per_station_validation");
    if (stationId != null && perStationValidation.has(stationId)) {
      validation = perStationValidation.path(stationId);
    }

    return new MetricsNodes(
        train.isMissingNode() || train.isNull() ? null : train,
        validation.isMissingNode() || validation.isNull() ? null : validation,
        test.isMissingNode() || test.isNull() ? null : test);
  }

  private boolean hasFlatMetrics(JsonNode node) {
    return node.has("mae") || node.has("rmse") || node.has("bias") || node.has("medianAE")
        || node.has("medianAe") || node.has("median_ae") || node.has("p50_abs_error")
        || node.has("maxAE") || node.has("maxAe") || node.has("max_ae")
        || node.has("p95_abs_error") || node.has("corr") || node.has("correlation");
  }

  private MetricsValues readMetrics(JsonNode node) {
    return new MetricsValues(
        readDouble(node, "mae"),
        readDouble(node, "rmse"),
        readDouble(node, "bias"),
        readDouble(node, "medianAE", "medianAe", "median_ae", "p50_abs_error", "p50"),
        readDouble(node, "maxAE", "maxAe", "max_ae", "p95_abs_error", "p95"),
        readDouble(node, "corr", "correlation"),
        readInt(node, "n", "rows"));
  }

  private Double readDouble(JsonNode node, String... keys) {
    for (String key : keys) {
      JsonNode value = node.get(key);
      if (value != null && value.isNumber()) {
        return value.doubleValue();
      }
    }
    return null;
  }

  private Integer readInt(JsonNode node, String... keys) {
    for (String key : keys) {
      JsonNode value = node.get(key);
      if (value != null && value.isNumber()) {
        return value.intValue();
      }
    }
    return null;
  }

  private String toJson(JsonNode node) {
    try {
      return objectMapper.writeValueAsString(node);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize metrics JSON", ex);
    }
  }

  private String deriveExperimentName(Path file, JsonNode root) {
    String name = readText(root, "name");
    if (name != null) {
      return name;
    }
    String experimentId = readText(root, "experiment_id", "experimentId");
    if (experimentId != null) {
      return experimentId;
    }
    String description = readText(root, "description");
    if (description != null) {
      return description;
    }
    String base = stripExtension(file.getFileName().toString());
    if (base.equalsIgnoreCase("metrics") || base.equalsIgnoreCase("report")
        || base.toLowerCase(Locale.ROOT).startsWith("report_")) {
      Path parent = file.getParent();
      if (parent != null) {
        return parent.getFileName().toString();
      }
    }
    return base;
  }

  private String deriveModelName(JsonNode root, String fallback) {
    if (root.has("model")) {
      JsonNode modelNode = root.get("model");
      if (modelNode.isTextual()) {
        return modelNode.asText();
      }
      if (modelNode.has("name")) {
        return modelNode.get("name").asText();
      }
    }
    String model = readText(root, "model");
    if (model != null) {
      return model;
    }
    return fallback;
  }

  private String deriveModelFamily(JsonNode root, String fallback) {
    String kind = readText(root, "kind");
    if (kind != null) {
      return kind;
    }
    String modelName = deriveModelName(root, fallback);
    if (modelName == null) {
      return fallback;
    }
    String lower = modelName.toLowerCase(Locale.ROOT);
    if (lower.contains("xgb")) {
      return "xgb";
    }
    if (lower.contains("cat")) {
      return "catboost";
    }
    if (lower.contains("lgb")) {
      return "lightgbm";
    }
    if (lower.contains("random")) {
      return "random_forest";
    }
    return fallback;
  }

  private String readText(JsonNode node, String... keys) {
    for (String key : keys) {
      JsonNode value = node.get(key);
      if (value != null && value.isTextual()) {
        return value.asText();
      }
    }
    return null;
  }

  private String resolveStationId(Path file, String raw) {
    List<String> stations = properties.getStationFilter();
    if (stations == null || stations.isEmpty()) {
      return null;
    }
    String pathLower = file.toString().toLowerCase(Locale.ROOT);
    String rawLower = raw == null ? "" : raw.toLowerCase(Locale.ROOT);
    for (String station : stations) {
      if (station == null || station.isBlank()) {
        continue;
      }
      String stationLower = station.toLowerCase(Locale.ROOT);
      if (pathLower.contains(stationLower) || rawLower.contains(stationLower)) {
        return station.trim().toUpperCase(Locale.ROOT);
      }
    }
    return null;
  }

  private String resolveStationIdWithExtras(Path file, String raw) {
    String stationId = resolveStationId(file, raw);
    if (stationId != null) {
      return stationId;
    }
    List<String> stations = properties.getStationFilter();
    if (stations == null || stations.isEmpty()) {
      return null;
    }
    for (Path extraFile : findExtraFiles(file)) {
      String content = readExtraFileForStation(extraFile);
      if (content == null) {
        continue;
      }
      String detected = matchStationInContent(content, stations);
      if (detected != null) {
        return detected;
      }
    }
    return null;
  }

  private boolean shouldIncludeStation(String stationId, String raw, Path file) {
    List<String> stations = properties.getStationFilter();
    if (stations == null || stations.isEmpty()) {
      return true;
    }
    if (stationId != null) {
      return true;
    }
    String rawLower = raw == null ? "" : raw.toLowerCase(Locale.ROOT);
    String pathLower = file.toString().toLowerCase(Locale.ROOT);
    for (String station : stations) {
      if (station == null || station.isBlank()) {
        continue;
      }
      String stationLower = station.toLowerCase(Locale.ROOT);
      if (rawLower.contains(stationLower) || pathLower.contains(stationLower)) {
        return true;
      }
    }
    return false;
  }

  private String matchStationInContent(String content, List<String> stations) {
    String lower = content.toLowerCase(Locale.ROOT);
    for (String station : stations) {
      if (station == null || station.isBlank()) {
        continue;
      }
      String stationLower = station.toLowerCase(Locale.ROOT);
      if (lower.contains(stationLower)) {
        return station.trim().toUpperCase(Locale.ROOT);
      }
    }
    return null;
  }

  private String readExtraFileForStation(Path path) {
    try {
      if (!Files.isRegularFile(path)) {
        return null;
      }
      long maxBytes = properties.getMaxMetadataBytes();
      if (maxBytes > 0 && Files.size(path) > maxBytes) {
        return null;
      }
      return Files.readString(path, StandardCharsets.UTF_8);
    } catch (IOException ex) {
      return null;
    }
  }

  private boolean hasResultsSibling(Path file) {
    Path resultsPath = file.getParent().resolve("results.json");
    return Files.exists(resultsPath);
  }

  private JsonNode loadSummarySibling(Path file) {
    Path summaryPath = file.getParent().resolve("summary.json");
    if (!Files.exists(summaryPath)) {
      return null;
    }
    try {
      return objectMapper.readTree(Files.readString(summaryPath, StandardCharsets.UTF_8));
    } catch (IOException ex) {
      logger.warn("Failed to read summary sibling: {}", summaryPath, ex);
      return null;
    }
  }

  private List<Path> findExtraFiles(Path file) {
    Path dir = file.getParent();
    if (dir == null || !Files.exists(dir)) {
      return List.of();
    }
    Set<Path> extras = new LinkedHashSet<>();
    String[] fixedNames = {
        "experiment_meta.json",
        "experiment_feature_columns.json",
        "run_metadata.json",
        "split_info.json",
        "config_resolved.yaml",
        "dataset_id.txt",
        "summary.json"
    };
    for (String name : fixedNames) {
      Path candidate = dir.resolve(name);
      if (Files.exists(candidate) && !candidate.equals(file)) {
        extras.add(candidate);
      }
    }

    try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "feature_list*.json")) {
      for (Path candidate : stream) {
        if (!candidate.equals(file)) {
          extras.add(candidate);
        }
      }
    } catch (IOException ex) {
      logger.debug("Failed to scan feature_list extras in {}", dir, ex);
    }

    return List.copyOf(extras);
  }

  private Object readExtraFile(Path path) {
    try {
      if (!Files.isRegularFile(path)) {
        return null;
      }
      long maxBytes = properties.getMaxMetadataBytes();
      if (maxBytes > 0 && Files.size(path) > maxBytes) {
        return Map.of("path", path.toString(), "note", "skipped_large_file");
      }
      String raw = Files.readString(path, StandardCharsets.UTF_8);
      String lower = path.getFileName().toString().toLowerCase(Locale.ROOT);
      if (lower.endsWith(".json")) {
        return objectMapper.readTree(raw);
      }
      return raw;
    } catch (IOException ex) {
      logger.warn("Failed to read extra file {}", path, ex);
      return null;
    }
  }

  private boolean isCandidateFile(Path file) {
    String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
    if (!name.endsWith(".json")) {
      return false;
    }
    return name.startsWith("metrics")
        || name.startsWith("report")
        || name.equals("summary.json")
        || name.equals("results.json");
  }

  private String buildExperimentKey(Path repoRoot, Path file, String name, Integer index) {
    String relative = relativePath(repoRoot, file);
    String suffix = null;
    if (name != null && !name.isBlank()) {
      suffix = name.trim().replace(" ", "_");
    } else if (index != null) {
      suffix = "item_" + index;
    }
    if (suffix == null) {
      return relative;
    }
    return relative + "::" + suffix;
  }

  private String relativePath(Path repoRoot, Path file) {
    try {
      return repoRoot.relativize(file.toAbsolutePath().normalize()).toString().replace("\\", "/");
    } catch (IllegalArgumentException ex) {
      return file.toAbsolutePath().normalize().toString().replace("\\", "/");
    }
  }

  private Path resolveRepoRoot() {
    if (properties.getRepoRoot() != null && !properties.getRepoRoot().isBlank()) {
      return Path.of(properties.getRepoRoot()).toAbsolutePath().normalize();
    }
    Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    Path candidate = current;
    for (int i = 0; i < 6; i++) {
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

  private List<Path> resolveScanRoots(Path repoRoot, List<String> roots) {
    List<Path> resolved = new ArrayList<>();
    if (roots != null) {
      for (String root : roots) {
        if (root == null || root.isBlank()) {
          continue;
        }
        Path path = Path.of(root);
        if (!path.isAbsolute()) {
          path = repoRoot.resolve(path);
        }
        resolved.add(path.normalize());
      }
    }
    if (resolved.isEmpty()) {
      resolved.add(repoRoot.resolve("artifacts"));
    }
    return resolved;
  }

  private String stripExtension(String name) {
    int idx = name.lastIndexOf('.');
    if (idx <= 0) {
      return name;
    }
    return name.substring(0, idx);
  }

  private record ExperimentMetadata(String metadataJson, Map<String, Object> metadata) {
  }

  private record MetricsNodes(JsonNode trainNode, JsonNode validationNode, JsonNode testNode) {
    boolean isEmpty() {
      return trainNode == null && validationNode == null && testNode == null;
    }
  }

  private record MetricsValues(Double mae,
                               Double rmse,
                               Double bias,
                               Double medianAe,
                               Double maxAe,
                               Double corr,
                               Integer n) {
  }
}
