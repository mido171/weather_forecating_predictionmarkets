package com.predictionmarkets.weather.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.ModelExperiment;
import com.predictionmarkets.weather.repository.ModelExperimentRepository;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class ModelExperimentDescriptionService {
  private static final Logger logger = LoggerFactory.getLogger(ModelExperimentDescriptionService.class);
  private static final int MIN_WORDS = 50;
  private static final int MAX_WORDS = 80;
  private static final int FEATURE_SCAN_LIMIT = 500;
  private static final Pattern WHITESPACE = Pattern.compile("\\s+");
  private static final Pattern WORD = Pattern.compile("[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?");
  private static final DateTimeFormatter SNAPSHOT_TS =
      DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneOffset.UTC);

  private final ModelExperimentRepository repository;
  private final ObjectMapper objectMapper;

  public ModelExperimentDescriptionService(ModelExperimentRepository repository,
                                           ObjectMapper objectMapper) {
    this.repository = repository;
    this.objectMapper = objectMapper;
  }

  @Transactional
  public int refreshDescriptions(int limit) {
    DescriptionSnapshot snapshot = writeDescriptionsSnapshot("artifacts/experiment_descriptions", limit);
    return applyDescriptionsSnapshot(snapshot.ndjsonPath());
  }

  public String buildHighLevelDescription(ModelExperiment experiment) {
    Objects.requireNonNull(experiment, "experiment is required");
    String key = experiment.getExperimentKey() != null
        ? experiment.getExperimentKey()
        : String.valueOf(experiment.getId());
    JsonNode metadata = parseMetadata(experiment.getMetadataJson());
    JsonNode primary = extractPrimaryNode(metadata);
    List<String> features = collectFeatureNames(metadata, FEATURE_SCAN_LIMIT);

    String stationId = nullToFallback(experiment.getStationId(), "unknown station");
    String target = inferTarget(features, experiment);
    String modelPhrase = buildModelPhrase(experiment, primary, metadata, key);
    String focus = inferFocusPhrase(experiment, metadata, primary, features, key);
    String groupLabel = inferGroupLabel(experiment.getSourcePath());

    FeatureSignals signals = extractFeatureSignals(features, experiment.getSourcePath());
    List<String> sources = pickSome(signals.sources(), key, "sources", 3);
    List<String> techniques = pickSome(signals.derived(), key, "techniques", 3);
    List<String> details = buildSpecificDetails(experiment, metadata, primary, features, key);

    String description = assembleDescription(key, stationId, target, modelPhrase, focus,
        sources, techniques, details, groupLabel);
    return enforceWordRange(description, MIN_WORDS, MAX_WORDS, key, details, groupLabel,
        extractSplitInfo(metadata));
  }

  public DescriptionSnapshot writeDescriptionsSnapshot(String outputDir, int limit) {
    Objects.requireNonNull(outputDir, "outputDir is required");
    List<ModelExperiment> experiments = repository.findAll();
    if (limit > 0 && experiments.size() > limit) {
      experiments = experiments.subList(0, limit);
    }

    Instant now = Instant.now();
    String stamp = SNAPSHOT_TS.format(now);
    Path dir = resolveOutputDir(outputDir);

    try {
      Files.createDirectories(dir);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to create snapshot directory: " + dir, ex);
    }

    Path ndjsonPath = dir.resolve("model_experiment_descriptions_" + stamp + ".ndjson");
    Path markdownPath = dir.resolve("model_experiment_descriptions_" + stamp + ".md");

    List<DescriptionSnapshotRow> rows = new ArrayList<>(experiments.size());
    for (ModelExperiment experiment : experiments) {
      String description = buildHighLevelDescription(experiment);
      rows.add(new DescriptionSnapshotRow(
          experiment.getId(),
          experiment.getExperimentKey(),
          experiment.getExperimentName(),
          experiment.getStationId(),
          experiment.getModelFamily(),
          experiment.getModelName(),
          experiment.getSourcePath(),
          countWords(description),
          description,
          Hashing.sha256Hex(description)));
    }

    writeNdjsonSnapshot(ndjsonPath, rows);
    writeMarkdownSnapshot(markdownPath, now, rows);
    logger.info("Wrote description snapshot ({} rows) to {}", rows.size(), ndjsonPath);
    return new DescriptionSnapshot(ndjsonPath, markdownPath, rows.size());
  }

  @Transactional
  public int applyDescriptionsSnapshot(Path ndjsonPath) {
    Objects.requireNonNull(ndjsonPath, "ndjsonPath is required");
    Map<Long, String> idToDescription = readDescriptionsFromNdjson(ndjsonPath);
    if (idToDescription.isEmpty()) {
      logger.warn("No rows found in snapshot {}", ndjsonPath);
      return 0;
    }

    List<ModelExperiment> experiments = repository.findAllById(idToDescription.keySet());
    int updated = 0;
    Instant now = Instant.now();
    for (ModelExperiment experiment : experiments) {
      String description = idToDescription.get(experiment.getId());
      if (description == null || description.isBlank()) {
        continue;
      }
      if (!Objects.equals(description, experiment.getDescriptionText())) {
        experiment.setDescriptionText(description);
        experiment.setUpdatedAtUtc(now);
        updated++;
      }
    }
    repository.saveAll(experiments);
    logger.info("Applied snapshot {}. Updated {} rows", ndjsonPath, updated);
    return updated;
  }

  private JsonNode parseMetadata(String metadataJson) {
    if (metadataJson == null || metadataJson.isBlank()) {
      return null;
    }
    try {
      return objectMapper.readTree(metadataJson);
    } catch (Exception ex) {
      return null;
    }
  }

  private JsonNode extractPrimaryNode(JsonNode metadata) {
    if (metadata == null || metadata.isNull() || metadata.isMissingNode()) {
      return null;
    }
    JsonNode raw = metadata.path("raw");
    JsonNode primary = raw.path("primary");
    if (!primary.isMissingNode() && !primary.isNull()) {
      return primary;
    }
    JsonNode fallback = metadata.path("primary");
    if (!fallback.isMissingNode() && !fallback.isNull()) {
      return fallback;
    }
    return null;
  }

  private String resolveModelDescriptor(ModelExperiment experiment, JsonNode metadata) {
    String modelName = experiment.getModelName();
    if (modelName == null) {
      modelName = findFirstText(metadata, Set.of("model", "model_name", "modelName"));
    }
    String modelFamily = experiment.getModelFamily();
    if (modelFamily == null) {
      modelFamily = findFirstText(metadata, Set.of("kind", "model_family", "modelFamily"));
    }
    String normalizedFamily = normalizeModelFamily(modelFamily);
    if (modelName == null && normalizedFamily == null) {
      return "machine learning";
    }
    if (modelName == null) {
      return normalizedFamily;
    }
    if (normalizedFamily == null) {
      return modelName;
    }
    String lowerName = modelName.toLowerCase(Locale.ROOT);
    String lowerFamily = normalizedFamily.toLowerCase(Locale.ROOT);
    if (lowerName.contains(lowerFamily)) {
      return modelName;
    }
    return normalizedFamily + " (" + modelName + ")";
  }

  private String normalizeModelFamily(String modelFamily) {
    if (modelFamily == null || modelFamily.isBlank()) {
      return null;
    }
    String lower = modelFamily.toLowerCase(Locale.ROOT);
    if (lower.contains("xgb")) {
      return "XGBoost";
    }
    if (lower.contains("cat")) {
      return "CatBoost";
    }
    if (lower.contains("lgb")) {
      return "LightGBM";
    }
    if (lower.contains("random")) {
      return "Random Forest";
    }
    return modelFamily;
  }

  private String inferTarget(List<String> features, ModelExperiment experiment) {
    StringBuilder combined = new StringBuilder();
    if (experiment.getExperimentName() != null) {
      combined.append(experiment.getExperimentName()).append(' ');
    }
    if (experiment.getSourcePath() != null) {
      combined.append(experiment.getSourcePath()).append(' ');
    }
    if (features != null && !features.isEmpty()) {
      combined.append(String.join(" ", features));
    }

    String lower = combined.toString().toLowerCase(Locale.ROOT);
    if (lower.contains("tmax")) {
      return "daily maximum temperature";
    }
    if (lower.contains("tmin")) {
      return "daily minimum temperature";
    }
    if (lower.contains("tavg") || lower.contains("tmean")) {
      return "daily mean temperature";
    }
    if (lower.contains("prcp") || lower.contains("precip") || lower.contains("rain")) {
      return "daily precipitation";
    }
    if (lower.contains("gust")) {
      return "wind gust";
    }
    if (lower.contains("wind")) {
      return "wind speed";
    }
    if (lower.contains("dew")) {
      return "dew point";
    }
    if (lower.contains("rh") || lower.contains("humidity")) {
      return "relative humidity";
    }
    if (lower.contains("cloud")) {
      return "cloud cover";
    }
    if (lower.contains("temp") || lower.contains("tmp")) {
      return "temperature";
    }
    return "the target weather variable";
  }

  private FeatureSignals extractFeatureSignals(List<String> features, String sourcePath) {
    Set<String> sources = new LinkedHashSet<>();
    Set<String> derived = new LinkedHashSet<>();

    String pathLower = sourcePath == null ? "" : sourcePath.toLowerCase(Locale.ROOT);
    if (pathLower.contains("mos")) {
      sources.add("MOS aggregates");
    }

    if (features == null) {
      features = List.of();
    }

    for (String feature : features) {
      String lower = feature == null ? "" : feature.toLowerCase(Locale.ROOT);
      if (lower.contains("nbm")) {
        sources.add("NBM guidance");
      }
      if (lower.contains("hrrr")) {
        sources.add("HRRR guidance");
      }
      if (lower.contains("rap")) {
        sources.add("RAP guidance");
      }
      if (lower.contains("gefs")) {
        sources.add("GEFS/ATMOS guidance");
      }
      if (lower.contains("gfs")) {
        sources.add("GFS guidance");
      }
      if (lower.contains("nam")) {
        sources.add("NAM guidance");
      }
      if (lower.contains("mos")) {
        sources.add("MOS aggregates");
      }
      if (lower.contains("obs")) {
        sources.add("historical observations");
      }
      if (lower.contains("blend")) {
        sources.add("blended guidance");
      }

      if (lower.contains("ens")) {
        derived.add("ensemble distribution summaries");
      }
      if (lower.contains("kalman")) {
        derived.add("Kalman-style offset correction");
      }
      if (lower.contains("knn") || lower.contains("analog")) {
        derived.add("analog or nearest-neighbor summaries");
      }
      if (lower.contains("bias") || lower.contains("offset")) {
        derived.add("systematic-offset history features");
      }
      if (lower.contains("corr") || lower.contains("agreement")) {
        derived.add("guidance agreement trackers");
      }
      if (lower.contains("spread") || lower.contains("std")) {
        derived.add("spread-derived uncertainty proxies");
      }
      if (lower.contains("doy") || lower.contains("month") || lower.contains("weekend")
          || lower.contains("sin") || lower.contains("cos")) {
        derived.add("calendar time features");
      }
      if (lower.contains("roll") || lower.contains("rolling")) {
        derived.add("rolling statistics");
      }
      if (lower.contains("z_") || lower.contains("zscore")) {
        derived.add("standardized anomalies");
      }
      if (lower.startsWith("isnan_") || lower.contains("missing")) {
        derived.add("explicit missingness flags");
      }
    }

    return new FeatureSignals(List.copyOf(sources), List.copyOf(derived));
  }

  private String buildModelPhrase(ModelExperiment experiment,
                                  JsonNode primary,
                                  JsonNode metadata,
                                  String key) {
    String descriptor = resolveModelDescriptor(experiment, primary != null ? primary : metadata);
    String normalized = descriptor == null ? "machine-learning model" : descriptor.trim();
    String lower = normalized.toLowerCase(Locale.ROOT);
    if (lower.contains("xgb") || lower.contains("xgboost")) {
      normalized = "XGBoost model";
    } else if (lower.contains("catboost")) {
      normalized = "CatBoost model";
    } else if (lower.contains("lightgbm") || lower.contains("lgb")) {
      normalized = "LightGBM model";
    } else if (lower.contains("random forest")) {
      normalized = "Random Forest model";
    } else if ("machine learning".equalsIgnoreCase(normalized)) {
      normalized = "machine-learning model";
    }

    Integer k = findFirstInt(primary, Set.of("k"));
    if (k == null) {
      JsonNode knnNode = primary == null ? null : primary.path("knn_lite");
      k = readInt(knnNode, "k");
    }
    if (k != null) {
      normalized = normalized + " with a lightweight analog layer";
    }

    return withArticle(normalized);
  }

  private String inferFocusPhrase(ModelExperiment experiment,
                                  JsonNode metadata,
                                  JsonNode primary,
                                  List<String> features,
                                  String key) {
    String groupLabel = inferGroupLabel(experiment.getSourcePath());
    StringBuilder hint = new StringBuilder();
    if (groupLabel != null) {
      hint.append(groupLabel).append(' ');
    }
    if (experiment.getExperimentName() != null) {
      hint.append(experiment.getExperimentName()).append(' ');
    }
    if (experiment.getSourcePath() != null) {
      hint.append(experiment.getSourcePath()).append(' ');
    }
    String described = findFirstText(metadata, Set.of("description", "summary", "goal", "purpose"));
    if (described != null && described.length() < 200) {
      hint.append(described);
    }

    String lower = hint.toString().toLowerCase(Locale.ROOT);
    FocusTheme theme = FocusTheme.from(lower, features);
    long seed = seedFor(key, "focus");
    List<String> options = theme.phrases();
    if (options.isEmpty()) {
      return "compare a candidate feature recipe and training configuration";
    }
    return options.get((int) Math.floorMod(seed, options.size()));
  }

  private String inferGroupLabel(String sourcePath) {
    if (sourcePath == null || sourcePath.isBlank()) {
      return null;
    }
    String normalized = sourcePath.replace("\\", "/");
    String[] parts = normalized.split("/");
    for (int i = parts.length - 2; i >= 0; i--) {
      if ("artifacts".equalsIgnoreCase(parts[i])) {
        if (i + 1 < parts.length) {
          return parts[i + 1];
        }
      }
    }
    if (parts.length >= 2) {
      return parts[parts.length - 2];
    }
    return normalized;
  }

  private List<String> pickSome(List<String> items, String key, String salt, int maxItems) {
    if (items == null || items.isEmpty() || maxItems <= 0) {
      return List.of();
    }
    if (items.size() <= maxItems) {
      return items;
    }
    List<String> copy = new ArrayList<>(items);
    shuffleDeterministic(copy, seedFor(key, salt));
    return copy.subList(0, Math.min(maxItems, copy.size()));
  }

  private List<String> buildSpecificDetails(ModelExperiment experiment,
                                            JsonNode metadata,
                                            JsonNode primary,
                                            List<String> features,
                                            String key) {
    List<String> details = new ArrayList<>();

    Integer featureCount = findFirstInt(primary, Set.of("feature_count", "featureCount"));
    if (featureCount == null && features != null && !features.isEmpty()) {
      featureCount = features.size();
    }
    if (featureCount != null) {
      details.add("uses a " + featureCount + "-feature recipe captured with the run");
    }

    JsonNode knnNode = primary == null ? null : primary.path("knn_lite");
    Integer k = readInt(knnNode, "k");
    if (k != null) {
      Integer lagDays = readInt(knnNode, "label_lag_days", "labelLagDays");
      if (lagDays != null) {
        details.add(String.format(Locale.ROOT,
            "adds kNN neighborhood summaries (k=%d, label lag %d day)", k, lagDays));
      } else {
        details.add(String.format(Locale.ROOT, "adds kNN neighborhood summaries (k=%d)", k));
      }
    }

    JsonNode modelParams = primary == null ? null : primary.path("model").path("params");
    if (modelParams != null && modelParams.isObject()) {
      List<String> params = extractParamHighlights(modelParams, key);
      if (!params.isEmpty()) {
        details.add("highlights key training knobs like " + String.join(", ", params));
      }
    } else {
      JsonNode paramsNode = primary == null ? null : primary.path("params");
      if (paramsNode != null && paramsNode.isObject()) {
        List<String> params = extractParamHighlights(paramsNode, key);
        if (!params.isEmpty()) {
          details.add("varies parameters such as " + String.join(", ", params));
        }
      }
    }

    if (features != null && !features.isEmpty()) {
      boolean hasSeason = hasAnyFeature(features, "sin_", "cos_", "doy", "month", "week_of_year");
      boolean hasOneHot = hasAnyFeature(features, "month_oh_", "dow_oh_", "doy_oh_");
      if (hasSeason && hasOneHot) {
        details.add("includes both harmonic seasonality terms and one-hot calendar indicators");
      } else if (hasSeason) {
        details.add("leans on harmonic seasonality terms to represent the annual cycle");
      } else if (hasOneHot) {
        details.add("uses one-hot calendar indicators to capture seasonal regimes");
      }

      if (hasAnyFeature(features, "ens_raw_", "ensemble")) {
        details.add("adds ensemble distribution summaries (means, quantiles, spread)");
      }
      if (hasAnyFeature(features, "mos_")) {
        details.add("extends the predictor set with MOS-derived aggregates over multiple windows");
      }
      if (hasAnyFeature(features, "roll_", "rolling")) {
        details.add("incorporates rolling observation context and anomaly-style transforms");
      }
      if (hasAnyFeature(features, "isnan_")) {
        details.add("uses explicit missingness flags to make guidance gaps visible to the model");
      }
      if (hasAnyFeature(features, "kalman")) {
        details.add("tests a Kalman-style correction signal as part of the feature set");
      }
      if (hasAnyFeature(features, "bias_") || hasAnyFeature(features, "offset")) {
        details.add("tracks recent systematic offsets between guidance and observations");
      }
      if (hasAnyFeature(features, "corr_") || hasAnyFeature(features, "agreement")) {
        details.add("captures guidance agreement/consistency over rolling windows");
      }
    }

    String groupLabel = inferGroupLabel(experiment.getSourcePath());
    if (groupLabel != null) {
      String lower = groupLabel.toLowerCase(Locale.ROOT);
      if (lower.contains("sweep")) {
        details.add("is one candidate within a structured configuration sweep");
      }
      if (lower.contains("baseline")) {
        details.add("anchors the experiment set with a simple baseline recipe");
      }
      if (lower.contains("mosvars")) {
        details.add("focuses on MOS-variable augmentation on top of the baseline guidance blend");
      }
      if (lower.contains("time_feature")) {
        details.add("is part of a time-feature sweep that compares alternative feature recipes");
      }
    }

    List<String> shuffled = new ArrayList<>(details);
    shuffleDeterministic(shuffled, seedFor(key, "details"));
    return shuffled;
  }

  private String assembleDescription(String key,
                                     String stationId,
                                     String target,
                                     String modelPhrase,
                                     String focus,
                                     List<String> sources,
                                     List<String> techniques,
                                     List<String> details,
                                     String groupLabel) {
    long seed = seedFor(key, "template");
    int template = (int) Math.floorMod(seed, 6);

    String stationUpper = stationId == null ? null : stationId.trim().toUpperCase(Locale.ROOT);
    String stationFor = stationUpper == null || stationUpper.isBlank() ? "for an unspecified station" : "for " + stationUpper;
    String stationAt = stationUpper == null || stationUpper.isBlank() ? "at an unspecified station" : "at " + stationUpper;
    String stationOn = stationUpper == null || stationUpper.isBlank() ? "on an unspecified station" : "on " + stationUpper;

    String targetClause = target == null || target.isBlank() ? "a weather target" : target;
    String sourcesClause = sources == null || sources.isEmpty() ? null : formatOxfordList(sources);
    String techniquesClause = techniques == null || techniques.isEmpty() ? null : formatOxfordList(techniques);
    String detailClause = null;
    if (details != null && !details.isEmpty()) {
      List<String> picked = pickSome(details, key, "detail_sentence_" + template, 2);
      if (!picked.isEmpty()) {
        detailClause = String.join("; ", picked);
      }
    }

    String opener = choose(seed, "opener",
        "This experiment",
        "This run",
        "This configuration",
        "This variant",
        "This training run",
        "The setup");
    String actionVerb = choose(seed, "verb", "predict", "forecast", "estimate", "project");
    String actionVerb3p = choose(seed, "verb3p", "predicts", "forecasts", "estimates", "projects");

    String joined;
    switch (template) {
      case 0 -> {
        joined = String.format(Locale.ROOT,
            "%s uses %s to %s %s %s. %s.",
            opener, modelPhrase, actionVerb, targetClause, stationFor, capitalizeFirst(focus));
        joined = appendClauses(joined, sourcesClause, techniquesClause, detailClause, seed);
      }
      case 1 -> {
        String lead = stationUpper == null ? "For this station" : "For " + stationUpper;
        joined = String.format(Locale.ROOT,
            "%s, this run targets %s using %s. It %s.",
            lead, targetClause, modelPhrase, focus);
        joined = appendClauses(joined, sourcesClause, techniquesClause, detailClause, seed);
      }
      case 2 -> {
        String group = choose(seed, "case2_intro", "This experiment", "This configuration", "This variant", "The run");
        joined = String.format(Locale.ROOT,
            "%s is designed around how to %s %s. It trains %s %s and %s. %s.",
            group, actionVerb, targetClause, modelPhrase, stationAt, focus, buildClause(sourcesClause, techniquesClause, detailClause, seed));
      }
      case 3 -> {
        String intro = choose(seed, "case3_intro", "A stored configuration", "A captured setup", "A saved run profile", "A recorded configuration");
        joined = String.format(Locale.ROOT,
            "%s %s that %s %s with %s. The emphasis is on how it %s. %s.",
            intro, stationOn, actionVerb3p, targetClause, modelPhrase, focus, buildClause(sourcesClause, techniquesClause, detailClause, seed));
      }
      case 4 -> {
        joined = String.format(Locale.ROOT,
            "%s fits %s %s to %s %s. It %s. %s.",
            opener, modelPhrase, stationAt, actionVerb, targetClause, focus,
            buildClause(sourcesClause, techniquesClause, detailClause, seed));
      }
      default -> {
        joined = String.format(Locale.ROOT,
            "%s tests how to %s %s %s by training %s. It %s.",
            opener, actionVerb, targetClause, stationFor, modelPhrase, focus);
        joined = appendClauses(joined, sourcesClause, techniquesClause, detailClause, seed);
      }
    }

    return normalizeWhitespace(joined);
  }

  private String appendClauses(String base,
                               String sourcesClause,
                               String techniquesClause,
                               String detailClause,
                               long seed) {
    List<String> bits = new ArrayList<>();
    if (sourcesClause != null && !sourcesClause.isBlank()) {
      bits.add(choose(seed, "src_phrase", "Inputs draw from", "Predictors come from", "Signals come from") + " " + sourcesClause);
    }
    if (techniquesClause != null && !techniquesClause.isBlank()) {
      bits.add(choose(seed, "tech_phrase", "feature engineering emphasizes", "feature work highlights", "it leans on") + " " + techniquesClause);
    }
    if (detailClause != null && !detailClause.isBlank()) {
      bits.add(capitalizeFirst(detailClause));
    }
    if (bits.isEmpty()) {
      return base;
    }
    String glue = choose(seed, "glue", " ", " ");
    String extra = String.join("; ", bits);
    if (!extra.endsWith(".")) {
      extra = extra + ".";
    }
    return base + glue + extra;
  }

  private String buildClause(String sourcesClause,
                             String techniquesClause,
                             String detailClause,
                             long seed) {
    List<String> bits = new ArrayList<>();
    if (sourcesClause != null && !sourcesClause.isBlank()) {
      bits.add(choose(seed, "src_short", "sources include", "inputs include", "signals include") + " " + sourcesClause);
    }
    if (techniquesClause != null && !techniquesClause.isBlank()) {
      bits.add(choose(seed, "tech_short", "techniques include", "feature work includes", "engineering includes") + " " + techniquesClause);
    }
    if (detailClause != null && !detailClause.isBlank()) {
      bits.add(detailClause);
    }
    if (bits.isEmpty()) {
      return choose(seed, "fallback", "The run metadata captures the configuration details for later reuse");
    }
    return capitalizeFirst(String.join("; ", bits));
  }

  private String enforceWordRange(String description,
                                  int minWords,
                                  int maxWords,
                                  String key,
                                  List<String> details,
                                  String groupLabel,
                                  SplitInfo splitInfo) {
    String text = stripMetricTerms(normalizeWhitespace(description));
    int words = countWords(text);
    int safety = 0;

    while (words < minWords && safety < 6) {
      safety++;
      String extra = buildExtraSentence(key, details, groupLabel, splitInfo, safety);
      if (extra == null || extra.isBlank()) {
        break;
      }
      text = normalizeWhitespace(text + " " + extra);
      words = countWords(text);
    }

    if (words > maxWords) {
      text = trimToWords(text, maxWords);
    }

    if (countWords(text) < minWords) {
      long seed = seedFor(key, "last_resort");
      String extra = choose(seed, "last_resort_sentence",
          "The stored run metadata preserves the feature recipe and key settings for later replication and iteration.",
          "Run metadata keeps the feature recipe and configuration context so the setup can be reproduced cleanly.",
          "The record retains configuration context (feature lists, split windows, and settings when available) for future runs.");
      text = normalizeWhitespace(text + " " + extra);
      if (countWords(text) > maxWords) {
        text = trimToWords(text, maxWords);
      }
    }

    return normalizeWhitespace(text);
  }

  private String buildExtraSentence(String key,
                                    List<String> details,
                                    String groupLabel,
                                    SplitInfo splitInfo,
                                    int iteration) {
    long seed = seedFor(key, "extra_" + iteration);

    if (details != null && !details.isEmpty()) {
      List<String> picked = pickSome(details, key, "extra_details_" + iteration, 2);
      if (!picked.isEmpty()) {
        String intro = choose(seed, "extra_intro",
            "Notably, it",
            "In practice, it",
            "It also",
            "Additionally, it");
        return intro + " " + String.join(" and ", picked) + ".";
      }
    }

    if (splitInfo != null && splitInfo.hasAny()) {
      return choose(seed, "split",
          "It uses a time-based split recorded alongside the run for reproducibility.",
          "A persisted train/validation/test split anchors the run's time windowing.",
          "The run metadata includes explicit time windows for splitting data into fit and holdout periods.");
    }

    return choose(seed, "generic",
        "The stored metadata keeps the feature recipe and configuration context so the run can be replicated later.",
        "Run metadata preserves configuration context (feature lists and split windows when present) for follow-on experiments.",
        "This record retains enough run context to reproduce the feature recipe and iterate on the configuration.");
  }

  private SplitInfo extractSplitInfo(JsonNode metadata) {
    if (metadata == null) {
      return SplitInfo.empty();
    }
    SplitInfo direct = parseSplitNode(metadata);
    if (direct.hasAny()) {
      return direct;
    }
    SplitInfo recursive = findSplitRecursive(metadata);
    return recursive.hasAny() ? recursive : SplitInfo.empty();
  }

  private SplitInfo findSplitRecursive(JsonNode node) {
    if (node == null || node.isMissingNode() || node.isNull()) {
      return SplitInfo.empty();
    }
    SplitInfo parsed = parseSplitNode(node);
    if (parsed.hasAny()) {
      return parsed;
    }
    if (node.isObject()) {
      var fields = node.fields();
      while (fields.hasNext()) {
        var entry = fields.next();
        SplitInfo nested = findSplitRecursive(entry.getValue());
        if (nested.hasAny()) {
          return nested;
        }
      }
    } else if (node.isArray()) {
      for (JsonNode child : node) {
        SplitInfo nested = findSplitRecursive(child);
        if (nested.hasAny()) {
          return nested;
        }
      }
    }
    return SplitInfo.empty();
  }

  private SplitInfo parseSplitNode(JsonNode node) {
    if (node == null || node.isMissingNode() || node.isNull()) {
      return SplitInfo.empty();
    }
    JsonNode train = node.path("train");
    JsonNode validation = node.path("validation");
    if (validation.isMissingNode()) {
      validation = node.path("val");
    }
    JsonNode test = node.path("test");

    String trainStart = readText(train, "start", "train_start", "trainStart");
    String trainEnd = readText(train, "end", "train_end", "trainEnd");
    String valStart = readText(validation, "start", "val_start", "validation_start", "valStart");
    String valEnd = readText(validation, "end", "val_end", "validation_end", "valEnd");
    String testStart = readText(test, "start", "test_start", "testStart");
    String testEnd = readText(test, "end", "test_end", "testEnd");

    Integer trainN = readInt(train, "n", "rows");
    Integer valN = readInt(validation, "n", "rows");
    Integer testN = readInt(test, "n", "rows");

    if (!hasAny(trainStart, trainEnd, valStart, valEnd, testStart, testEnd)) {
      trainStart = readText(node, "train_start", "trainStart");
      trainEnd = readText(node, "train_end", "trainEnd");
      valStart = readText(node, "val_start", "validation_start", "valStart");
      valEnd = readText(node, "val_end", "validation_end", "valEnd");
      testStart = readText(node, "test_start", "testStart");
      testEnd = readText(node, "test_end", "testEnd");
    }

    return new SplitInfo(trainStart, trainEnd, valStart, valEnd, testStart, testEnd,
        trainN, valN, testN);
  }

  private String buildSplitSentence(SplitInfo splitInfo) {
    String trainRange = formatRange(splitInfo.trainStart(), splitInfo.trainEnd());
    String valRange = formatRange(splitInfo.valStart(), splitInfo.valEnd());
    String testRange = formatRange(splitInfo.testStart(), splitInfo.testEnd());
    String counts = String.format(Locale.ROOT, "%s/%s/%s",
        formatCount(splitInfo.trainN()),
        formatCount(splitInfo.valN()),
        formatCount(splitInfo.testN()));

    return String.format(Locale.ROOT,
        "Training, validation, and test windows are %s, %s, and %s with sample counts %s when known.",
        trainRange, valRange, testRange, counts);
  }

  private String formatRange(String start, String end) {
    if (start == null && end == null) {
      return "unspecified";
    }
    if (start == null) {
      return "through " + end;
    }
    if (end == null) {
      return "from " + start;
    }
    return start + " to " + end;
  }

  private String formatCount(Integer value) {
    return value == null ? "NA" : value.toString();
  }

  private List<String> collectFeatureNames(JsonNode metadata, int limit) {
    List<String> features = new ArrayList<>();
    collectFeatureNames(metadata, limit, features);
    return features;
  }

  private void collectFeatureNames(JsonNode node, int limit, List<String> sink) {
    if (node == null || node.isNull() || node.isMissingNode() || sink.size() >= limit) {
      return;
    }
    if (node.isArray()) {
      boolean allText = true;
      for (JsonNode child : node) {
        if (!child.isTextual()) {
          allText = false;
          break;
        }
      }
      if (allText) {
        for (JsonNode child : node) {
          if (sink.size() >= limit) {
            break;
          }
          sink.add(child.asText());
        }
        return;
      }
      for (JsonNode child : node) {
        collectFeatureNames(child, limit, sink);
        if (sink.size() >= limit) {
          return;
        }
      }
      return;
    }
    if (node.isObject()) {
      var fields = node.fields();
      while (fields.hasNext()) {
        var entry = fields.next();
        String key = entry.getKey().toLowerCase(Locale.ROOT);
        JsonNode child = entry.getValue();
        if (key.contains("feature")) {
          collectFeatureNames(child, limit, sink);
        } else {
          collectFeatureNames(child, limit, sink);
        }
        if (sink.size() >= limit) {
          return;
        }
      }
    }
  }

  private String findFirstText(JsonNode node, Set<String> keys) {
    if (node == null || node.isNull() || node.isMissingNode()) {
      return null;
    }
    if (node.isObject()) {
      var fields = node.fields();
      while (fields.hasNext()) {
        var entry = fields.next();
        String key = entry.getKey();
        JsonNode value = entry.getValue();
        if (keys.contains(key) && value.isTextual()) {
          return value.asText();
        }
        String nested = findFirstText(value, keys);
        if (nested != null) {
          return nested;
        }
      }
    } else if (node.isArray()) {
      for (JsonNode child : node) {
        String nested = findFirstText(child, keys);
        if (nested != null) {
          return nested;
        }
      }
    }
    return null;
  }

  private Integer findFirstInt(JsonNode node, Set<String> keys) {
    if (node == null || node.isNull() || node.isMissingNode()) {
      return null;
    }
    if (node.isObject()) {
      var fields = node.fields();
      while (fields.hasNext()) {
        var entry = fields.next();
        String key = entry.getKey();
        JsonNode value = entry.getValue();
        if (keys.contains(key) && value.isNumber()) {
          return value.intValue();
        }
        Integer nested = findFirstInt(value, keys);
        if (nested != null) {
          return nested;
        }
      }
    } else if (node.isArray()) {
      for (JsonNode child : node) {
        Integer nested = findFirstInt(child, keys);
        if (nested != null) {
          return nested;
        }
      }
    }
    return null;
  }

  private String readText(JsonNode node, String... keys) {
    if (node == null || node.isNull() || node.isMissingNode()) {
      return null;
    }
    for (String key : keys) {
      JsonNode value = node.get(key);
      if (value != null && value.isTextual()) {
        return value.asText();
      }
    }
    return null;
  }

  private Integer readInt(JsonNode node, String... keys) {
    if (node == null || node.isNull() || node.isMissingNode()) {
      return null;
    }
    for (String key : keys) {
      JsonNode value = node.get(key);
      if (value != null && value.isNumber()) {
        return value.intValue();
      }
    }
    return null;
  }

  private boolean hasAny(String... values) {
    for (String value : values) {
      if (value != null && !value.isBlank()) {
        return true;
      }
    }
    return false;
  }

  private String nullToFallback(String value, String fallback) {
    if (value == null || value.isBlank()) {
      return fallback;
    }
    return value.trim();
  }

  private void writeNdjsonSnapshot(Path path, List<DescriptionSnapshotRow> rows) {
    try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
      for (DescriptionSnapshotRow row : rows) {
        writer.write(objectMapper.writeValueAsString(row));
        writer.newLine();
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write snapshot: " + path, ex);
    }
  }

  private void writeMarkdownSnapshot(Path path, Instant generatedAt, List<DescriptionSnapshotRow> rows) {
    List<DescriptionSnapshotRow> sorted = new ArrayList<>(rows);
    sorted.sort((a, b) -> {
      String ak = a.experimentKey() == null ? "" : a.experimentKey();
      String bk = b.experimentKey() == null ? "" : b.experimentKey();
      return ak.compareTo(bk);
    });

    try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
      writer.write("# Model Experiment Descriptions");
      writer.newLine();
      writer.write("generated_at_utc: " + generatedAt);
      writer.newLine();
      writer.write("row_count: " + sorted.size());
      writer.newLine();
      writer.newLine();

      for (DescriptionSnapshotRow row : sorted) {
        writer.write("## " + nullToFallback(row.experimentKey(), String.valueOf(row.id())));
        writer.newLine();
        writer.write("- id: " + row.id());
        writer.newLine();
        if (row.stationId() != null) {
          writer.write("- station_id: " + row.stationId());
          writer.newLine();
        }
        if (row.modelFamily() != null || row.modelName() != null) {
          writer.write("- model: " + nullToFallback(row.modelFamily(), "NA") + " / " + nullToFallback(row.modelName(), "NA"));
          writer.newLine();
        }
        if (row.sourcePath() != null) {
          writer.write("- source_path: " + row.sourcePath());
          writer.newLine();
        }
        writer.write("- word_count: " + row.wordCount());
        writer.newLine();
        writer.write("- description_hash: " + row.descriptionHash());
        writer.newLine();
        writer.newLine();
        writer.write(row.descriptionText());
        writer.newLine();
        writer.newLine();
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write markdown snapshot: " + path, ex);
    }
  }

  private Map<Long, String> readDescriptionsFromNdjson(Path path) {
    Map<Long, String> mapping = new HashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        JsonNode node = objectMapper.readTree(line);
        JsonNode idNode = node.get("id");
        JsonNode descNode = node.get("descriptionText");
        if (idNode == null || !idNode.isNumber() || descNode == null || !descNode.isTextual()) {
          continue;
        }
        mapping.put(idNode.asLong(), descNode.asText());
      }
      return mapping;
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read snapshot: " + path, ex);
    }
  }

  private List<String> extractParamHighlights(JsonNode paramsNode, String key) {
    if (paramsNode == null || !paramsNode.isObject()) {
      return List.of();
    }
    List<String> candidates = new ArrayList<>();
    addParamIfPresent(paramsNode, candidates, "max_depth", "maxDepth");
    addParamIfPresent(paramsNode, candidates, "learning_rate", "learningRate");
    addParamIfPresent(paramsNode, candidates, "n_estimators", "nEstimators");
    addParamIfPresent(paramsNode, candidates, "subsample");
    addParamIfPresent(paramsNode, candidates, "colsample_bytree", "colsampleBytree");
    addParamIfPresent(paramsNode, candidates, "min_child_weight", "minChildWeight");
    addParamIfPresent(paramsNode, candidates, "reg_lambda", "lambda", "regLambda");
    addParamIfPresent(paramsNode, candidates, "reg_alpha", "alpha", "regAlpha");
    addParamIfPresent(paramsNode, candidates, "tree_method", "treeMethod");

    if (candidates.isEmpty()) {
      return List.of();
    }
    List<String> shuffled = new ArrayList<>(candidates);
    shuffleDeterministic(shuffled, seedFor(key, "params"));
    return shuffled.subList(0, Math.min(3, shuffled.size()));
  }

  private void addParamIfPresent(JsonNode paramsNode, List<String> sink, String... keys) {
    for (String key : keys) {
      JsonNode value = paramsNode.get(key);
      if (value == null || value.isNull() || value.isMissingNode()) {
        continue;
      }
      if (value.isTextual()) {
        sink.add(key + "=" + value.asText());
        return;
      }
      if (value.isBoolean()) {
        sink.add(key + "=" + value.asBoolean());
        return;
      }
      if (value.isNumber()) {
        double raw = value.asDouble();
        long rounded = Math.round(raw);
        if (Math.abs(raw - rounded) < 1e-9) {
          sink.add(key + "=" + rounded);
        } else {
          sink.add(String.format(Locale.ROOT, "%s=%.4f", key, raw));
        }
        return;
      }
    }
  }

  private boolean hasAnyFeature(List<String> features, String... needles) {
    if (features == null || features.isEmpty()) {
      return false;
    }
    for (String feature : features) {
      String lower = feature == null ? "" : feature.toLowerCase(Locale.ROOT);
      for (String needle : needles) {
        if (needle == null || needle.isBlank()) {
          continue;
        }
        if (lower.contains(needle.toLowerCase(Locale.ROOT))) {
          return true;
        }
      }
    }
    return false;
  }

  private String formatOxfordList(List<String> items) {
    if (items == null || items.isEmpty()) {
      return "";
    }
    if (items.size() == 1) {
      return items.get(0);
    }
    if (items.size() == 2) {
      return items.get(0) + " and " + items.get(1);
    }
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < items.size(); i++) {
      if (i > 0) {
        builder.append(i == items.size() - 1 ? ", and " : ", ");
      }
      builder.append(items.get(i));
    }
    return builder.toString();
  }

  private String normalizeWhitespace(String value) {
    if (value == null) {
      return "";
    }
    return WHITESPACE.matcher(value.trim()).replaceAll(" ");
  }

  private int countWords(String text) {
    String normalized = normalizeWhitespace(text);
    if (normalized.isBlank()) {
      return 0;
    }
    return normalized.split(" ").length;
  }

  private String trimToWords(String text, int maxWords) {
    String normalized = normalizeWhitespace(text);
    if (maxWords <= 0) {
      return "";
    }
    String[] tokens = normalized.split(" ");
    if (tokens.length <= maxWords) {
      return normalized;
    }
    StringBuilder trimmed = new StringBuilder();
    for (int i = 0; i < maxWords; i++) {
      if (i > 0) {
        trimmed.append(' ');
      }
      trimmed.append(tokens[i]);
    }
    String result = trimmed.toString();
    if (!result.endsWith(".")) {
      result = result + ".";
    }
    return result;
  }

  private String stripMetricTerms(String text) {
    if (text == null || text.isBlank()) {
      return "";
    }
    String cleaned = text;
    cleaned = cleaned.replaceAll("(?i)\\bmae\\b", "");
    cleaned = cleaned.replaceAll("(?i)\\brmse\\b", "");
    cleaned = cleaned.replaceAll("(?i)\\bbrier\\b", "");
    cleaned = cleaned.replaceAll("(?i)\\bbias\\b", "offset");
    cleaned = cleaned.replaceAll("(?i)\\bcorr\\b", "agreement");
    return normalizeWhitespace(cleaned);
  }

  private String capitalizeFirst(String value) {
    if (value == null || value.isBlank()) {
      return "";
    }
    String trimmed = value.trim();
    return trimmed.substring(0, 1).toUpperCase(Locale.ROOT) + trimmed.substring(1);
  }

  private String withArticle(String phrase) {
    if (phrase == null || phrase.isBlank()) {
      return "a model";
    }
    String trimmed = phrase.trim();
    String lower = trimmed.toLowerCase(Locale.ROOT);
    if (lower.startsWith("a ") || lower.startsWith("an ")) {
      return trimmed;
    }
    boolean vowel = lower.startsWith("a") || lower.startsWith("e") || lower.startsWith("i")
        || lower.startsWith("o") || lower.startsWith("u");
    String article = vowel ? "an " : "a ";
    if (lower.startsWith("xgboost") || lower.startsWith("xgb")) {
      article = "an ";
    }
    return article + trimmed;
  }

  private long seedFor(String key, String salt) {
    String base = key == null ? "" : key;
    String hex = Hashing.sha256Hex(base + "::" + salt);
    return Long.parseUnsignedLong(hex.substring(0, 16), 16);
  }

  private void shuffleDeterministic(List<String> items, long seed) {
    if (items == null || items.size() <= 1) {
      return;
    }
    Random random = new Random(seed);
    for (int i = items.size() - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      String tmp = items.get(i);
      items.set(i, items.get(j));
      items.set(j, tmp);
    }
  }

  private String choose(long seed, String salt, String... options) {
    if (options == null || options.length == 0) {
      return "";
    }
    long derived = seedFor(Long.toString(seed), salt);
    int idx = (int) Math.floorMod(derived, options.length);
    return options[idx];
  }

  public record DescriptionSnapshot(Path ndjsonPath, Path markdownPath, int rowCount) {
  }

  private Path resolveOutputDir(String outputDir) {
    if (outputDir == null || outputDir.isBlank()) {
      return resolveRepoRoot().resolve("artifacts/experiment_descriptions").toAbsolutePath().normalize();
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

  private record DescriptionSnapshotRow(Long id,
                                        String experimentKey,
                                        String experimentName,
                                        String stationId,
                                        String modelFamily,
                                        String modelName,
                                        String sourcePath,
                                        int wordCount,
                                        String descriptionText,
                                        String descriptionHash) {
  }

  private enum FocusTheme {
    BASELINE(List.of(
        "establishes a clean baseline with a compact guidance-driven feature set",
        "anchors later comparisons by keeping the feature recipe intentionally minimal",
        "sets a reference configuration before introducing heavier feature engineering",
        "keeps the predictor set lean to measure the value of later additions",
        "provides a simple control run with core guidance and light calendar context",
        "serves as a reference point before turning on broader feature blocks")),
    MOSVARS(List.of(
        "expands the baseline guidance blend with MOS-derived aggregate variables",
        "tests whether MOS aggregate blocks add signal beyond the core guidance stack",
        "focuses on MOS-variable augmentation layered on top of the baseline recipe",
        "adds MOS summary blocks and windowed aggregates to enrich the guidance signal",
        "leans on MOS-derived statistics while keeping the base guidance blend intact",
        "treats MOS augmentation as the primary lever for additional signal")),
    SWEEP(List.of(
        "runs a structured sweep over model capacity and regularization settings",
        "systematically varies key hyperparameters to map configuration tradeoffs",
        "compares multiple candidate configurations within the same feature recipe",
        "explores a grid of candidate settings to probe sensitivity to complexity",
        "tries multiple training configurations to identify a stable, repeatable choice",
        "benchmarks several variants under the same split to isolate configuration effects")),
    TIME_FEATURES(List.of(
        "probes time-aware feature engineering such as seasonality, lags, and rolling context",
        "compares alternative calendar and history feature recipes across a controlled setup",
        "tests whether explicit time context stabilizes predictions across seasonal regimes",
        "emphasizes calendar seasonality and rolling context to smooth month-to-month shifts",
        "checks whether time-of-year structure and lagged history improve robustness",
        "compares alternative formulations of seasonality terms and rolling windows")),
    KNN_ANALOG(List.of(
        "injects analog or k-nearest-neighbor summaries to capture similarity to past days",
        "adds nearest-neighbor neighborhood diagnostics as an extra signal layer",
        "uses analog-style features to better represent local regime shifts",
        "summarizes similarity to historical days through neighbor-distance diagnostics",
        "adds neighborhood-based statistics to capture regimes not expressed in raw guidance",
        "explores kNN-style analog features as a complement to guidance inputs")),
    KALMAN(List.of(
        "incorporates Kalman-style correction signals to stabilize guidance inputs",
        "tests state-space correction features layered on top of raw guidance",
        "explores correction-driven features that track systematic offsets over time",
        "uses state-space style corrections to dampen drift in guidance behavior",
        "tests correction signals that adapt over time as guidance characteristics shift",
        "focuses on correction-driven features that stabilize guidance before learning")),
    STACKING(List.of(
        "evaluates stacking/blending where stage-one predictors feed a final learner",
        "tests a two-stage setup that combines base predictions through a meta-model",
        "blends several first-pass predictors and learns a final correction layer",
        "learns a meta-model that combines multiple base signals into a final estimate",
        "tests multi-stage blending to exploit complementary strengths across predictors",
        "uses stage-one outputs as features for a final combining layer")),
    GENERAL(List.of(
        "compares a candidate feature recipe and training configuration",
        "explores a specific model/feature configuration captured in the run metadata",
        "tests a stored configuration aimed at improving generalization for this station",
        "records a distinct model/feature recipe intended to be compared across runs",
        "focuses on a particular combination of guidance inputs and engineered context",
        "captures a candidate recipe that could later be promoted into production"));

    private final List<String> phrases;

    FocusTheme(List<String> phrases) {
      this.phrases = phrases;
    }

    List<String> phrases() {
      return phrases;
    }

    static FocusTheme from(String hintLower, List<String> features) {
      String hint = hintLower == null ? "" : hintLower;
      if (containsAny(hint, "stack", "blend2", "meta_model")) {
        return STACKING;
      }
      if (containsAny(hint, "kalman")) {
        return KALMAN;
      }
      if (containsAny(hint, "knn", "analog")) {
        return KNN_ANALOG;
      }
      if (containsAny(hint, "time_feature", "timefeatures")) {
        return TIME_FEATURES;
      }
      if (containsAny(hint, "sweep", "grid", "results.json", "summary.json")) {
        return SWEEP;
      }
      if (containsAny(hint, "mosvars", "mos_vars", "mos_")) {
        return MOSVARS;
      }
      if (containsAny(hint, "baseline", "minimal")) {
        return BASELINE;
      }
      if (features != null && !features.isEmpty()) {
        boolean hasMos = false;
        boolean hasKnn = false;
        boolean hasKalman = false;
        for (String feature : features) {
          String lower = feature == null ? "" : feature.toLowerCase(Locale.ROOT);
          if (lower.contains("mos_")) {
            hasMos = true;
          }
          if (lower.contains("knn") || lower.contains("analog")) {
            hasKnn = true;
          }
          if (lower.contains("kalman")) {
            hasKalman = true;
          }
        }
        if (hasKalman) {
          return KALMAN;
        }
        if (hasKnn) {
          return KNN_ANALOG;
        }
        if (hasMos) {
          return MOSVARS;
        }
      }
      return GENERAL;
    }

    private static boolean containsAny(String haystack, String... needles) {
      if (haystack == null || haystack.isBlank()) {
        return false;
      }
      for (String needle : needles) {
        if (needle == null || needle.isBlank()) {
          continue;
        }
        if (haystack.contains(needle)) {
          return true;
        }
      }
      return false;
    }
  }

  private record FeatureSignals(List<String> sources, List<String> derived) {
  }

  private record SplitInfo(String trainStart,
                           String trainEnd,
                           String valStart,
                           String valEnd,
                           String testStart,
                           String testEnd,
                           Integer trainN,
                           Integer valN,
                           Integer testN) {
    static SplitInfo empty() {
      return new SplitInfo(null, null, null, null, null, null, null, null, null);
    }

    boolean hasAny() {
      return trainStart != null || trainEnd != null || valStart != null || valEnd != null
          || testStart != null || testEnd != null || trainN != null || valN != null || testN != null;
    }
  }
}
