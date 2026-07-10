package com.predictionmarkets.weather.executors.export;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.config.GribstreamCliExportProperties;
import com.predictionmarkets.weather.models.CliDaily;
import com.predictionmarkets.weather.models.CliDailyId;
import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.models.GribstreamMetric;
import com.predictionmarkets.weather.models.MosDailyValue;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.GribstreamDailyFeatureRepository;
import com.predictionmarkets.weather.repository.MosDailyValueRepository;
import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;

public final class GribstreamDailyFeatureCliExportExecutor {
  private static final Logger logger =
      LoggerFactory.getLogger(GribstreamDailyFeatureCliExportExecutor.class);
  private static final LocalTime ASOF_TIME_UTC = LocalTime.of(12, 0);
  private static final String MOS_TMAX_VARIABLE_CODE = "n_x";
  private static final String MOS_MODEL_GFS = "GFS";
  private static final String MOS_MODEL_NAM = "NAM";
  private static final String MOS_GFS_MAX_HEADER = "gfs_n_x_max";
  private static final String MOS_NAM_MAX_HEADER = "nam_n_x_max";
  private static final Feature[] FEATURE_ORDER = {
      Feature.NBM_TMAX_F,
      Feature.HRRR_TMAX_F,
      Feature.RAP_TMAX_F,
      Feature.GEFSATMOSMEAN_TMAX_F,
      Feature.GEFSATMOS_TMP_SPREAD_F
  };
  private static final Map<FeatureKey, Feature> FEATURE_LOOKUP = buildFeatureLookup();

  private GribstreamDailyFeatureCliExportExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamDailyFeatureRepository featureRepository =
          context.getBean(GribstreamDailyFeatureRepository.class);
      CliDailyRepository cliRepository =
          context.getBean(CliDailyRepository.class);
      MosDailyValueRepository mosRepository =
          context.getBean(MosDailyValueRepository.class);
      GribstreamCliExportProperties properties =
          context.getBean(GribstreamCliExportProperties.class);
      int pageSize = properties.getPageSize();
      if (pageSize < 1) {
        throw new IllegalArgumentException("gribstream.cli-export.page-size must be >= 1");
      }
      String stationId = normalizeStationId(properties.getStationId());
      if (stationId == null) {
        throw new IllegalArgumentException("gribstream.cli-export.station-id is required");
      }
      Path outputPath = resolveOutputPath(Paths.get(properties.getOutputPath()), stationId);
      boolean append = properties.isAppend();
      snapshot("Gribstream CLI export starting outputPath=" + outputPath.toAbsolutePath()
          + " pageSize=" + pageSize
          + " append=" + append
          + " stationId=" + stationId);
      long exported = exportRows(
          featureRepository,
          cliRepository,
          mosRepository,
          outputPath,
          pageSize,
          append,
          stationId);
      snapshot("Gribstream CLI export complete rows=" + exported);
    }
  }

  private static long exportRows(GribstreamDailyFeatureRepository featureRepository,
                                 CliDailyRepository cliRepository,
                                 MosDailyValueRepository mosRepository,
                                 Path outputPath,
                                 int pageSize,
                                 boolean append,
                                 String stationId) {
    ensureParentDirectory(outputPath);
    boolean writeHeader = shouldWriteHeader(outputPath, append);
    try (BufferedWriter writer = Files.newBufferedWriter(
        outputPath,
        StandardCharsets.UTF_8,
        append
            ? new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.APPEND}
            : new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING})) {
      if (writeHeader) {
        writer.write(headerLine());
        writer.newLine();
      }
      long exported = writeRows(featureRepository, cliRepository, mosRepository, writer, pageSize,
          stationId);
      writer.flush();
      return exported;
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write Gribstream CLI export to " + outputPath, ex);
    }
  }

  private static long writeRows(GribstreamDailyFeatureRepository featureRepository,
                                CliDailyRepository cliRepository,
                                MosDailyValueRepository mosRepository,
                                BufferedWriter writer,
                                int pageSize,
                                String stationId) throws IOException {
    Sort sort = Sort.by(
        Sort.Order.asc("stationId"),
        Sort.Order.asc("targetDateLocal"),
        Sort.Order.asc("asofUtc"),
        Sort.Order.asc("modelCode"),
        Sort.Order.asc("metric"),
        Sort.Order.asc("id"));
    Map<LocalDate, String> targetCache = new HashMap<>();
    Map<LocalDate, String> gfsCache = new HashMap<>();
    Map<LocalDate, String> namCache = new HashMap<>();
    GroupAccumulator current = null;
    long exported = 0L;
    int pageNumber = 0;
    while (true) {
      Page<GribstreamDailyFeatureEntity> page =
          featureRepository.findByStationId(stationId, PageRequest.of(pageNumber, pageSize, sort));
      for (GribstreamDailyFeatureEntity entity : page) {
        if (!isExpectedAsof(entity)) {
          continue;
        }
        Feature feature = resolveFeature(entity);
        if (feature == null) {
          continue;
        }
        GroupKey key = new GroupKey(entity.getStationId(),
            entity.getTargetDateLocal(),
            entity.getAsofUtc());
        if (current == null) {
          current = new GroupAccumulator(key);
        } else if (!current.matches(key)) {
          if (writeGroup(writer, current, cliRepository, mosRepository, targetCache, gfsCache,
              namCache, stationId)) {
            exported++;
          }
          current = new GroupAccumulator(key);
        }
        current.add(feature, entity.getValueF());
      }
      if (!page.hasNext()) {
        break;
      }
      pageNumber++;
    }
    if (current != null) {
      if (writeGroup(writer, current, cliRepository, mosRepository, targetCache, gfsCache,
          namCache, stationId)) {
        exported++;
      }
    }
    return exported;
  }

  private static boolean writeGroup(BufferedWriter writer,
                                    GroupAccumulator group,
                                    CliDailyRepository cliRepository,
                                    MosDailyValueRepository mosRepository,
                                    Map<LocalDate, String> targetCache,
                                    Map<LocalDate, String> gfsCache,
                                    Map<LocalDate, String> namCache,
                                    String stationId) throws IOException {
    String targetTmax = resolveTargetTmax(cliRepository, targetCache, stationId,
        group.key().targetDateLocal());
    if (targetTmax == null) {
      return false;
    }
    String gfsMax = resolveMosValue(mosRepository, gfsCache, stationId, MOS_MODEL_GFS,
        group.key().targetDateLocal(), group.key().asofUtc());
    String namMax = resolveMosValue(mosRepository, namCache, stationId, MOS_MODEL_NAM,
        group.key().targetDateLocal(), group.key().asofUtc());
    StringBuilder builder = new StringBuilder();
    appendValue(builder, group.key().stationId());
    appendValue(builder, group.key().targetDateLocal().toString());
    appendValue(builder, group.key().asofUtc().toString());
    for (Feature feature : FEATURE_ORDER) {
      appendValue(builder, formatValue(group.value(feature)));
    }
    appendValue(builder, gfsMax);
    appendValue(builder, namMax);
    appendValue(builder, targetTmax);
    writer.write(builder.toString());
    writer.newLine();
    return true;
  }

  private static boolean isExpectedAsof(GribstreamDailyFeatureEntity entity) {
    if (entity == null || entity.getTargetDateLocal() == null || entity.getAsofUtc() == null) {
      return false;
    }
    Instant expected = computeExpectedAsof(entity.getTargetDateLocal());
    return expected.equals(entity.getAsofUtc());
  }

  private static Instant computeExpectedAsof(LocalDate targetDateLocal) {
    return targetDateLocal.minusDays(1).atTime(ASOF_TIME_UTC).toInstant(ZoneOffset.UTC);
  }

  private static String resolveTargetTmax(CliDailyRepository cliRepository,
                                          Map<LocalDate, String> cache,
                                          String stationId,
                                          LocalDate targetDateLocal) {
    if (cache.containsKey(targetDateLocal)) {
      return cache.get(targetDateLocal);
    }
    CliDailyId id = new CliDailyId(stationId, targetDateLocal);
    CliDaily cliDaily = cliRepository.findById(id).orElse(null);
    String result = formatTmax(cliDaily == null ? null : cliDaily.getTmaxF());
    cache.put(targetDateLocal, result);
    return result;
  }

  private static String resolveMosValue(MosDailyValueRepository mosRepository,
                                        Map<LocalDate, String> cache,
                                        String stationId,
                                        String model,
                                        LocalDate targetDateLocal,
                                        Instant asofUtc) {
    if (cache.containsKey(targetDateLocal)) {
      return cache.get(targetDateLocal);
    }
    String value = null;
    Optional<MosDailyValue> row = mosRepository
        .findFirstByStationIdAndModelAndTargetDateLocalAndVariableCodeAndRuntimeUtcLessThanEqualOrderByRuntimeUtcDesc(
            stationId,
            model,
            targetDateLocal,
            MOS_TMAX_VARIABLE_CODE,
            asofUtc);
    if (row.isPresent()) {
      value = formatTmax(row.get().getValueMax());
    }
    cache.put(targetDateLocal, value);
    return value;
  }

  private static String formatTmax(BigDecimal value) {
    return value == null ? null : value.toPlainString();
  }

  private static String headerLine() {
    StringBuilder builder = new StringBuilder();
    appendValue(builder, "station_id");
    appendValue(builder, "target_date_local");
    appendValue(builder, "asof_utc");
    for (Feature feature : FEATURE_ORDER) {
      appendValue(builder, feature.headerName());
    }
    appendValue(builder, MOS_GFS_MAX_HEADER);
    appendValue(builder, MOS_NAM_MAX_HEADER);
    appendValue(builder, "target_tmax_f");
    return builder.toString();
  }

  private static String formatValue(Double value) {
    return value == null ? "" : Double.toString(value);
  }

  private static Feature resolveFeature(GribstreamDailyFeatureEntity entity) {
    if (entity == null || entity.getMetric() == null || entity.getModelCode() == null) {
      return null;
    }
    String model = entity.getModelCode().trim().toLowerCase(Locale.ROOT);
    return FEATURE_LOOKUP.get(new FeatureKey(model, entity.getMetric()));
  }

  private static Map<FeatureKey, Feature> buildFeatureLookup() {
    Map<FeatureKey, Feature> map = new HashMap<>();
    map.put(new FeatureKey("nbm", GribstreamMetric.TMAX_F), Feature.NBM_TMAX_F);
    map.put(new FeatureKey("hrrr", GribstreamMetric.TMAX_F), Feature.HRRR_TMAX_F);
    map.put(new FeatureKey("rap", GribstreamMetric.TMAX_F), Feature.RAP_TMAX_F);
    map.put(new FeatureKey("gefsatmosmean", GribstreamMetric.TMAX_F),
        Feature.GEFSATMOSMEAN_TMAX_F);
    map.put(new FeatureKey("gefsatmos", GribstreamMetric.TMP_SPREAD_F),
        Feature.GEFSATMOS_TMP_SPREAD_F);
    return map;
  }

  private static void appendValue(StringBuilder builder, String value) {
    if (builder.length() > 0) {
      builder.append(',');
    }
    builder.append(escape(value));
  }

  private static String escape(String value) {
    if (value == null) {
      return "";
    }
    boolean needsQuotes = value.indexOf(',') >= 0
        || value.indexOf('"') >= 0
        || value.indexOf('\n') >= 0
        || value.indexOf('\r') >= 0;
    if (!needsQuotes) {
      return value;
    }
    String escaped = value.replace("\"", "\"\"");
    return "\"" + escaped + "\"";
  }

  private static String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      return null;
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private static Path resolveOutputPath(Path basePath, String stationId) {
    if (stationId == null || stationId.isBlank()) {
      return basePath;
    }
    Path fileName = basePath.getFileName();
    if (fileName == null) {
      return basePath;
    }
    String prefix = stationId + "_";
    String fileNameText = fileName.toString();
    String resolvedName = fileNameText.startsWith(prefix) ? fileNameText : prefix + fileNameText;
    Path parent = basePath.getParent();
    return parent == null ? Paths.get(resolvedName) : parent.resolve(resolvedName);
  }

  private static void ensureParentDirectory(Path outputPath) {
    try {
      Path parent = outputPath.getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to create output directory for " + outputPath, ex);
    }
  }

  private static boolean shouldWriteHeader(Path outputPath, boolean append) {
    if (!append) {
      return true;
    }
    try {
      return !(Files.exists(outputPath) && Files.size(outputPath) > 0);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to check output file size for " + outputPath, ex);
    }
  }

  private static void snapshot(String message) {
    String payload = "[GRIBSTREAM-CLI-EXPORT] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private enum Feature {
    NBM_TMAX_F("nbm_tmax_f"),
    HRRR_TMAX_F("hrrr_tmax_f"),
    RAP_TMAX_F("rap_tmax_f"),
    GEFSATMOSMEAN_TMAX_F("gefsatmosmean_tmax_f"),
    GEFSATMOS_TMP_SPREAD_F("gefsatmos_tmp_spread_f");

    private final String headerName;

    Feature(String headerName) {
      this.headerName = headerName;
    }

    public String headerName() {
      return headerName;
    }
  }

  private record FeatureKey(String modelCode, GribstreamMetric metric) {
  }

  private record GroupKey(String stationId, LocalDate targetDateLocal, Instant asofUtc) {
  }

  private static final class GroupAccumulator {
    private final GroupKey key;
    private final EnumMap<Feature, Double> values = new EnumMap<>(Feature.class);

    private GroupAccumulator(GroupKey key) {
      this.key = key;
    }

    public GroupKey key() {
      return key;
    }

    public void add(Feature feature, Double value) {
      if (feature == null || value == null) {
        return;
      }
      values.put(feature, value);
    }

    public Double value(Feature feature) {
      return values.get(feature);
    }

    public boolean matches(GroupKey other) {
      return key.stationId().equals(other.stationId())
          && key.targetDateLocal().equals(other.targetDateLocal())
          && key.asofUtc().equals(other.asofUtc());
    }
  }
}
