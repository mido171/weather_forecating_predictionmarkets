package com.predictionmarkets.weather.gribstream;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.common.StandardTimeClimateWindow;
import com.predictionmarkets.weather.executors.GribstreamTaskExecutor;
import com.predictionmarkets.weather.repository.GribstreamForecastValueUpsertRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class GribstreamVariableIngestService {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamVariableIngestService.class);
  private static final Set<String> ENSEMBLE_MODELS = Set.of("gefsatmos");
  private static final int ALIAS_LIMIT = 64;
  private static final int NAME_LIMIT = 32;
  private static final int LEVEL_LIMIT = 64;
  private static final int INFO_LIMIT = 64;
  private static final int TEXT_LIMIT = 128;

  private final GribstreamClient client;
  private final GribstreamProperties gribstreamProperties;
  private final GribstreamVariableIngestProperties ingestProperties;
  private final GribstreamVariableCatalogLoader catalogLoader;
  private final GribstreamVariableWhitelistLoader whitelistLoader;
  private final GribstreamForecastValueUpsertRepository repository;
  private final GribstreamTaskExecutor taskExecutor;
  private final ObjectMapper objectMapper;

  private final Object batchLock = new Object();
  private volatile Map<String, List<VariableBatch>> cachedBatches;
  private volatile Map<String, List<GribstreamVariableWhitelistEntry>> cachedWhitelist;

  public GribstreamVariableIngestService(GribstreamClient client,
                                         GribstreamProperties gribstreamProperties,
                                         GribstreamVariableIngestProperties ingestProperties,
                                         GribstreamVariableCatalogLoader catalogLoader,
                                         GribstreamVariableWhitelistLoader whitelistLoader,
                                         GribstreamForecastValueUpsertRepository repository,
                                         GribstreamTaskExecutor taskExecutor,
                                         ObjectMapper objectMapper) {
    this.client = client;
    this.gribstreamProperties = gribstreamProperties;
    this.ingestProperties = ingestProperties;
    this.catalogLoader = catalogLoader;
    this.whitelistLoader = whitelistLoader;
    this.repository = repository;
    this.taskExecutor = taskExecutor;
    this.objectMapper = objectMapper;
  }

  public int ingestForDate(StationSpec station, LocalDate targetDateLocal, Instant asOfUtc) {
    Objects.requireNonNull(station, "station is required");
    Objects.requireNonNull(targetDateLocal, "targetDateLocal is required");
    Objects.requireNonNull(asOfUtc, "asOfUtc is required");
    Map<String, List<VariableBatch>> modelBatches = loadBatches();
    if (modelBatches.isEmpty()) {
      logger.warn("[GRIBSTREAM-VARS] No variable catalog loaded. Skipping.");
      return 0;
    }
    ZoneId zoneId = ZoneId.of(station.zoneId());
    StandardTimeClimateWindow.UtcRange range =
        StandardTimeClimateWindow.computeUtcRange(zoneId, targetDateLocal);
    List<Callable<Integer>> tasks = new ArrayList<>();
    int minHorizon = Math.max(0, gribstreamProperties.getDefaultMinHorizonHours());
    for (Map.Entry<String, List<VariableBatch>> entry : modelBatches.entrySet()) {
      String modelCode = entry.getKey();
      int maxHorizon = resolveMaxHorizon(modelCode);
      for (VariableBatch batch : entry.getValue()) {
        tasks.add(() -> ingestBatch(station, targetDateLocal, asOfUtc, range,
            modelCode, minHorizon, maxHorizon, batch));
      }
    }
    if (tasks.isEmpty()) {
      return 0;
    }
    List<Integer> results = taskExecutor.invokeAllOrFail(tasks);
    return results.stream().mapToInt(Integer::intValue).sum();
  }

  private int ingestBatch(StationSpec station,
                          LocalDate targetDateLocal,
                          Instant asOfUtc,
                          StandardTimeClimateWindow.UtcRange range,
                          String modelCode,
                          int minHorizon,
                          int maxHorizon,
                          VariableBatch batch) {
    List<Integer> members = resolveMembers(modelCode);
    GribstreamHistoryRequest request = new GribstreamHistoryRequest(
        range.startUtc().toString(),
        range.endUtc().toString(),
        asOfUtc.toString(),
        minHorizon,
        maxHorizon,
        List.of(new GribstreamCoordinate(station.lat(), station.lon(), station.name())),
        batch.variables(),
        members);
    GribstreamRawResponse response;
    try {
      response = client.fetchHistoryRaw(modelCode, request);
    } catch (GribstreamEmptyResponseException ex) {
      logger.warn("[GRIBSTREAM-VARS] model={} station={} date={} status=empty_response",
          modelCode, station.stationId(), targetDateLocal);
      return 0;
    }
    List<GribstreamValueRow> rows = GribstreamGenericResponseParser.parseRows(
        objectMapper,
        response.responseBytes(),
        modelCode,
        response.requestSha256());
    if (rows.isEmpty()) {
      return 0;
    }
    List<GribstreamForecastValueUpsertRepository.UpsertRow> upsertRows = new ArrayList<>();
    for (GribstreamValueRow row : rows) {
      if (row.forecastedAt() == null || row.forecastedTime() == null) {
        continue;
      }
      enforceNoLeakage(row.forecastedAt(), asOfUtc, modelCode);
      Integer member = row.member();
      Map<String, String> values = row.values();
      if (values == null || values.isEmpty()) {
        continue;
      }
      for (Map.Entry<String, String> valueEntry : values.entrySet()) {
        String alias = normalizeAlias(valueEntry.getKey());
        if (alias == null) {
          continue;
        }
        GribstreamVariableSpec spec = batch.aliasToSpec().get(alias);
        if (spec == null) {
          continue;
        }
        String raw = valueEntry.getValue();
        if (raw == null || raw.isBlank()) {
          continue;
        }
        Double valueNum = parseDouble(raw);
        String valueText = valueNum == null ? truncate(raw.trim(), TEXT_LIMIT) : null;
        upsertRows.add(new GribstreamForecastValueUpsertRepository.UpsertRow(
            station.stationId(),
            station.zoneId(),
            modelCode,
            asOfUtc,
            row.forecastedAt(),
            row.forecastedTime(),
            member,
            truncate(spec.name(), NAME_LIMIT),
            truncate(spec.level(), LEVEL_LIMIT),
            truncate(spec.info(), INFO_LIMIT),
            truncate(alias, ALIAS_LIMIT),
            valueNum,
            valueText,
            response.requestJson(),
            response.requestSha256(),
            response.responseSha256(),
            response.retrievedAtUtc(),
            null));
      }
    }
    if (upsertRows.isEmpty()) {
      return 0;
    }
    int[] results = repository.upsertAll(upsertRows);
    int sum = 0;
    for (int result : results) {
      sum += result;
    }
    logger.info("[GRIBSTREAM-VARS] model={} station={} date={} batchSize={} rowsUpserted={}",
        modelCode, station.stationId(), targetDateLocal, batch.variables().size(), sum);
    return sum;
  }

  private Map<String, List<VariableBatch>> loadBatches() {
    Map<String, List<VariableBatch>> cached = cachedBatches;
    if (cached != null) {
      return cached;
    }
    synchronized (batchLock) {
      if (cachedBatches != null) {
        return cachedBatches;
      }
      List<String> models = resolveModels();
      if (models.isEmpty()) {
        cachedBatches = Collections.emptyMap();
        return cachedBatches;
      }
      Map<String, List<GribstreamVariableWhitelistEntry>> whitelist = resolveWhitelist();
      boolean whitelistEnabled = whitelist != null && !whitelist.isEmpty();
      if (ingestProperties.isWhitelistRequired() && !whitelistEnabled) {
        throw new IllegalStateException("gribstream.variable-ingest.whitelist-resource is required");
      }
      Map<String, List<GribstreamVariableSpec>> catalog =
          catalogLoader.loadCatalog(ingestProperties.getCatalogResource(), models);
      Map<String, List<VariableBatch>> batches = new LinkedHashMap<>();
      for (String model : models) {
        List<GribstreamVariableSpec> specs = catalog.get(model);
        if (specs == null || specs.isEmpty()) {
          logger.warn("[GRIBSTREAM-VARS] No catalog entries found for model={}", model);
          continue;
        }
        List<GribstreamVariableSpec> filtered = filterSpecs(model, specs);
        if (whitelistEnabled) {
          List<GribstreamVariableWhitelistEntry> modelWhitelist = whitelist.get(model);
          if (modelWhitelist == null || modelWhitelist.isEmpty()) {
            logger.warn("[GRIBSTREAM-VARS] Whitelist has no entries for model={}; skipping.", model);
            continue;
          }
          filtered = applyWhitelist(model, filtered, modelWhitelist);
          if (filtered.isEmpty()) {
            logger.warn("[GRIBSTREAM-VARS] Whitelist filtered all variables for model={}", model);
            continue;
          }
        }
        if (filtered.isEmpty()) {
          logger.warn("[GRIBSTREAM-VARS] No usable variables for model={}", model);
          continue;
        }
        List<VariableBatch> modelBatches = buildBatches(filtered);
        batches.put(model, modelBatches);
      }
      cachedBatches = batches;
      return cachedBatches;
    }
  }

  private Map<String, List<GribstreamVariableWhitelistEntry>> resolveWhitelist() {
    Map<String, List<GribstreamVariableWhitelistEntry>> cached = cachedWhitelist;
    if (cached != null) {
      return cached;
    }
    Map<String, List<GribstreamVariableWhitelistEntry>> loaded =
        whitelistLoader.loadWhitelist(ingestProperties.getWhitelistResource());
    cachedWhitelist = loaded;
    return loaded;
  }

  private List<GribstreamVariableSpec> applyWhitelist(String model,
                                                      List<GribstreamVariableSpec> specs,
                                                      List<GribstreamVariableWhitelistEntry> whitelistEntries) {
    Map<String, GribstreamVariableSpec> byKey = new HashMap<>();
    for (GribstreamVariableSpec spec : specs) {
      String key = GribstreamVariableWhitelistLoader.normalizeKey(
          spec.name(), spec.level(), spec.info());
      byKey.put(key, spec);
    }
    List<GribstreamVariableSpec> filtered = new ArrayList<>();
    List<GribstreamVariableWhitelistEntry> missing = new ArrayList<>();
    Set<String> seen = new LinkedHashSet<>();
    for (GribstreamVariableWhitelistEntry entry : whitelistEntries) {
      String key = entry.normalizedKey();
      if (!seen.add(key)) {
        continue;
      }
      GribstreamVariableSpec spec = byKey.get(key);
      if (spec == null) {
        missing.add(entry);
        continue;
      }
      filtered.add(spec);
    }
    if (!missing.isEmpty()) {
      int limit = Math.min(10, missing.size());
      StringBuilder preview = new StringBuilder();
      for (int i = 0; i < limit; i++) {
        if (i > 0) {
          preview.append("; ");
        }
        preview.append(missing.get(i).describe());
      }
      logger.warn("[GRIBSTREAM-VARS] Whitelist entries not found model={} missingCount={} sample={}",
          model, missing.size(), preview);
    }
    return filtered;
  }

  private List<String> resolveModels() {
    List<String> configured = ingestProperties.getModels();
    if (configured != null && !configured.isEmpty()) {
      return normalizeModels(configured);
    }
    if (gribstreamProperties.getModels() != null && !gribstreamProperties.getModels().isEmpty()) {
      return normalizeModels(new ArrayList<>(gribstreamProperties.getModels().keySet()));
    }
    return List.of();
  }

  private List<String> normalizeModels(List<String> models) {
    Set<String> normalized = new LinkedHashSet<>();
    for (String model : models) {
      if (model == null || model.isBlank()) {
        continue;
      }
      normalized.add(model.trim().toLowerCase(Locale.ROOT));
    }
    return new ArrayList<>(normalized);
  }

  private List<GribstreamVariableSpec> filterSpecs(String model, List<GribstreamVariableSpec> specs) {
    int minHorizon = Math.max(0, gribstreamProperties.getDefaultMinHorizonHours());
    int maxHorizon = resolveMaxHorizon(model);
    List<GribstreamVariableSpec> filtered = new ArrayList<>();
    for (GribstreamVariableSpec spec : specs) {
      if (spec == null || spec.name() == null || spec.name().isBlank()) {
        continue;
      }
      if (spec.level() == null || spec.level().isBlank()) {
        continue;
      }
      if (!isWithinHorizon(spec, minHorizon, maxHorizon)) {
        continue;
      }
      filtered.add(spec);
    }
    int maxVariables = ingestProperties.getMaxVariablesPerModel();
    if (maxVariables > 0 && filtered.size() > maxVariables) {
      return new ArrayList<>(filtered.subList(0, maxVariables));
    }
    return filtered;
  }

  private boolean isWithinHorizon(GribstreamVariableSpec spec, int minHorizon, int maxHorizon) {
    int specMin = spec.minHorizonHours();
    int specMax = spec.maxHorizonHours();
    if (specMax > 0 && specMax < minHorizon) {
      return false;
    }
    if (specMin > 0 && specMin > maxHorizon) {
      return false;
    }
    return true;
  }

  private List<VariableBatch> buildBatches(List<GribstreamVariableSpec> specs) {
    int batchSize = Math.max(1, ingestProperties.getBatchSize());
    List<GribstreamVariable> variables = new ArrayList<>(specs.size());
    Map<String, GribstreamVariableSpec> aliasToSpec = new LinkedHashMap<>();
    Map<String, Integer> aliasCounts = new HashMap<>();
    for (GribstreamVariableSpec spec : specs) {
      String alias = buildAlias(spec, aliasCounts);
      aliasToSpec.put(alias, spec);
      variables.add(new GribstreamVariable(spec.name(), spec.level(), spec.info(), alias));
    }
    List<VariableBatch> batches = new ArrayList<>();
    for (int i = 0; i < variables.size(); i += batchSize) {
      int end = Math.min(i + batchSize, variables.size());
      List<GribstreamVariable> batchVars = new ArrayList<>(variables.subList(i, end));
      Map<String, GribstreamVariableSpec> batchSpecs = new LinkedHashMap<>();
      for (GribstreamVariable variable : batchVars) {
        batchSpecs.put(variable.alias(), aliasToSpec.get(variable.alias()));
      }
      batches.add(new VariableBatch(
          Collections.unmodifiableList(batchVars),
          Collections.unmodifiableMap(batchSpecs)));
    }
    return batches;
  }

  private String buildAlias(GribstreamVariableSpec spec, Map<String, Integer> aliasCounts) {
    String name = slug(spec.name());
    String level = slug(spec.level());
    String info = slug(spec.info());
    String base = name + "_" + level;
    if (!info.isEmpty()) {
      base = base + "_" + info;
    }
    String hash = Hashing.sha256Hex(spec.name() + "|" + spec.level() + "|" + spec.info())
        .substring(0, 8);
    String candidate = base;
    if (candidate.length() > ALIAS_LIMIT - 9) {
      candidate = candidate.substring(0, ALIAS_LIMIT - 9);
    }
    candidate = candidate + "_" + hash;
    int count = aliasCounts.getOrDefault(candidate, 0);
    aliasCounts.put(candidate, count + 1);
    if (count == 0) {
      return candidate;
    }
    String suffix = "_" + count;
    int maxBase = ALIAS_LIMIT - suffix.length();
    if (candidate.length() > maxBase) {
      candidate = candidate.substring(0, maxBase);
    }
    return candidate + suffix;
  }

  private String slug(String input) {
    if (input == null) {
      return "";
    }
    String raw = input.trim().toLowerCase(Locale.ROOT);
    if (raw.isEmpty()) {
      return "";
    }
    String normalized = raw.replaceAll("[^a-z0-9]+", "_");
    normalized = normalized.replaceAll("^_+", "").replaceAll("_+$", "");
    return normalized;
  }

  private String normalizeAlias(String alias) {
    if (alias == null || alias.isBlank()) {
      return null;
    }
    return alias.trim();
  }

  private int resolveMaxHorizon(String modelCode) {
    GribstreamProperties.ModelProperties model = gribstreamProperties.getModels().get(modelCode);
    if (model == null) {
      throw new IllegalArgumentException("Missing gribstream.models." + modelCode + ".maxHorizonHours");
    }
    int maxHorizon = model.getMaxHorizonHours();
    if (maxHorizon < 1) {
      throw new IllegalArgumentException("maxHorizonHours must be >= 1 for model " + modelCode);
    }
    return maxHorizon;
  }

  private List<Integer> resolveMembers(String modelCode) {
    if (!ENSEMBLE_MODELS.contains(modelCode)) {
      return null;
    }
    List<Integer> members = gribstreamProperties.getGefs().getMembers();
    if (members == null || members.isEmpty()) {
      throw new IllegalArgumentException("gribstream.gefs.members is required for model " + modelCode);
    }
    return members;
  }

  private void enforceNoLeakage(Instant forecastedAt, Instant asOfUtc, String modelCode) {
    if (forecastedAt.isAfter(asOfUtc)) {
      throw new IllegalStateException("Gribstream leakage guard model=" + modelCode
          + " forecastedAtUtc=" + forecastedAt + " asOfUtc=" + asOfUtc);
    }
  }

  private Double parseDouble(String raw) {
    String text = raw.trim();
    if (text.isEmpty()) {
      return null;
    }
    if (text.equalsIgnoreCase("nan") || text.equalsIgnoreCase("inf") || text.equalsIgnoreCase("-inf")) {
      return null;
    }
    try {
      return Double.parseDouble(text);
    } catch (NumberFormatException ex) {
      return null;
    }
  }

  private String truncate(String value, int limit) {
    if (value == null) {
      return "";
    }
    String trimmed = value.trim();
    if (trimmed.length() <= limit) {
      return trimmed;
    }
    return trimmed.substring(0, limit);
  }

  private record VariableBatch(
      List<GribstreamVariable> variables,
      Map<String, GribstreamVariableSpec> aliasToSpec) {
  }
}
