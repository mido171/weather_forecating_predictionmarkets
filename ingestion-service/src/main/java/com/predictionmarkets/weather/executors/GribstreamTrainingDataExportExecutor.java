package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.models.CliDaily;
import com.predictionmarkets.weather.models.CliDailyId;
import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.models.GribstreamMetric;
import com.predictionmarkets.weather.models.MosAsofFeature;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.GribstreamDailyFeatureRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureRepository;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDate;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;

public final class GribstreamTrainingDataExportExecutor {
  private static final Logger logger =
      LoggerFactory.getLogger(GribstreamTrainingDataExportExecutor.class);

  private static final Feature[] GRIBSTREAM_FEATURES = {
      Feature.GEFSATMOSMEAN_TMAX_F,
      Feature.RAP_TMAX_F,
      Feature.HRRR_TMAX_F,
      Feature.NBM_TMAX_F,
      Feature.GEFSATMOS_TMP_SPREAD_F
  };

  private static final Feature[] FEATURE_ORDER = {
      Feature.GFS_TMAX_F,
      Feature.NAM_TMAX_F,
      Feature.GEFSATMOSMEAN_TMAX_F,
      Feature.RAP_TMAX_F,
      Feature.HRRR_TMAX_F,
      Feature.NBM_TMAX_F,
      Feature.GEFSATMOS_TMP_SPREAD_F,
      Feature.ACTUAL_TMAX_F
  };

  private static final Map<FeatureKey, Feature> FEATURE_LOOKUP = buildFeatureLookup();

  private GribstreamTrainingDataExportExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamDailyFeatureRepository repository =
          context.getBean(GribstreamDailyFeatureRepository.class);
      MosAsofFeatureRepository mosRepository =
          context.getBean(MosAsofFeatureRepository.class);
      CliDailyRepository cliRepository =
          context.getBean(CliDailyRepository.class);
      GribstreamTrainingDataProperties properties =
          context.getBean(GribstreamTrainingDataProperties.class);
      int pageSize = properties.getPageSize();
      if (pageSize < 1) {
        throw new IllegalArgumentException("gribstream.training-data.page-size must be >= 1");
      }
      Long mosAsofPolicyId = properties.getMosAsofPolicyId();
      if (mosAsofPolicyId == null) {
        throw new IllegalArgumentException("gribstream.training-data.mos-asof-policy-id is required");
      }
      Path outputPath = Paths.get(properties.getOutputPath());
      boolean append = properties.isAppend();
      snapshot("Training data export starting outputPath=" + outputPath.toAbsolutePath()
          + " pageSize=" + pageSize
          + " append=" + append
          + " mosAsofPolicyId=" + mosAsofPolicyId);
      ExportStats stats =
          exportTrainingData(repository, mosRepository, cliRepository, mosAsofPolicyId,
              outputPath, pageSize, append);
      snapshot("Training data export complete rows=" + stats.written()
          + " skippedIncomplete=" + stats.skipped()
          + " groups=" + stats.groups());
    }
  }

  private static ExportStats exportTrainingData(GribstreamDailyFeatureRepository repository,
                                                MosAsofFeatureRepository mosRepository,
                                                CliDailyRepository cliRepository,
                                                Long mosAsofPolicyId,
                                                Path outputPath,
                                                int pageSize,
                                                boolean append) {
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
      return writeRows(repository, mosRepository, cliRepository, mosAsofPolicyId, writer, pageSize);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write training data to " + outputPath, ex);
    }
  }

  private static ExportStats writeRows(GribstreamDailyFeatureRepository repository,
                                       MosAsofFeatureRepository mosRepository,
                                       CliDailyRepository cliRepository,
                                       Long mosAsofPolicyId,
                                       BufferedWriter writer,
                                       int pageSize) throws IOException {
    Sort sort = Sort.by(
        Sort.Order.asc("stationId"),
        Sort.Order.asc("targetDateLocal"),
        Sort.Order.asc("asofUtc"),
        Sort.Order.asc("modelCode"),
        Sort.Order.asc("metric"),
        Sort.Order.asc("id"));
    GroupAccumulator current = null;
    long groups = 0L;
    long written = 0L;
    long skipped = 0L;
    int pageNumber = 0;
    while (true) {
      Page<GribstreamDailyFeatureEntity> page =
          repository.findAll(PageRequest.of(pageNumber, pageSize, sort));
      for (GribstreamDailyFeatureEntity entity : page) {
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
          groups++;
          if (current.hasAllGribstreamFeatures()
              && applyMosAndCliFeatures(current, mosRepository, cliRepository, mosAsofPolicyId)) {
            writeRow(writer, current);
            written++;
          } else {
            skipped++;
          }
          current = new GroupAccumulator(key);
        }
        current.add(feature, entity.getValueF());
      }
      if (!page.hasNext()) {
        break;
      }
      pageNumber++;
      if (groups % 5000 == 0 && groups > 0) {
        snapshot("Training data export progress groups=" + groups
            + " rows=" + written
            + " skipped=" + skipped);
      }
    }
    if (current != null) {
      groups++;
      if (current.hasAllGribstreamFeatures()
          && applyMosAndCliFeatures(current, mosRepository, cliRepository, mosAsofPolicyId)) {
        writeRow(writer, current);
        written++;
      } else {
        skipped++;
      }
    }
    writer.flush();
    return new ExportStats(groups, written, skipped);
  }

  private static void writeRow(BufferedWriter writer, GroupAccumulator group) throws IOException {
    StringBuilder builder = new StringBuilder();
    appendValue(builder, group.key().stationId());
    appendValue(builder, group.key().targetDateLocal().toString());
    appendValue(builder, group.key().asofUtc().toString());
    for (Feature feature : FEATURE_ORDER) {
      Double value = group.value(feature);
      appendValue(builder, value == null ? "" : Double.toString(value));
    }
    writer.write(builder.toString());
    writer.newLine();
  }

  private static void appendValue(StringBuilder builder, String value) {
    if (builder.length() > 0) {
      builder.append(',');
    }
    builder.append(escape(value));
  }

  private static String headerLine() {
    StringBuilder builder = new StringBuilder();
    appendValue(builder, "station_id");
    appendValue(builder, "target_date_local");
    appendValue(builder, "asof_utc");
    for (Feature feature : FEATURE_ORDER) {
      appendValue(builder, feature.headerName());
    }
    return builder.toString();
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

  private static Feature resolveFeature(GribstreamDailyFeatureEntity entity) {
    if (entity == null || entity.getMetric() == null || entity.getModelCode() == null) {
      return null;
    }
    String model = entity.getModelCode().trim().toLowerCase(Locale.ROOT);
    return FEATURE_LOOKUP.get(new FeatureKey(model, entity.getMetric()));
  }

  private static boolean applyMosAndCliFeatures(GroupAccumulator group,
                                                MosAsofFeatureRepository mosRepository,
                                                CliDailyRepository cliRepository,
                                                Long mosAsofPolicyId) {
    MosAsofFeature gfs = resolveMosFeature(mosRepository, group.key(), mosAsofPolicyId, MosModel.GFS);
    if (gfs == null || gfs.getTmaxF() == null) {
      return false;
    }
    MosAsofFeature nam = resolveMosFeature(mosRepository, group.key(), mosAsofPolicyId, MosModel.NAM);
    if (nam == null || nam.getTmaxF() == null) {
      return false;
    }
    CliDaily actual = resolveCliDaily(cliRepository, group.key());
    if (actual == null || actual.getTmaxF() == null) {
      return false;
    }
    group.add(Feature.GFS_TMAX_F, gfs.getTmaxF().doubleValue());
    group.add(Feature.NAM_TMAX_F, nam.getTmaxF().doubleValue());
    group.add(Feature.ACTUAL_TMAX_F, actual.getTmaxF().doubleValue());
    return group.isComplete();
  }

  private static MosAsofFeature resolveMosFeature(MosAsofFeatureRepository mosRepository,
                                                  GroupKey key,
                                                  Long mosAsofPolicyId,
                                                  MosModel model) {
    return mosRepository
        .findByIdStationIdAndIdTargetDateLocalAndIdAsofPolicyIdAndIdModel(
            key.stationId(),
            key.targetDateLocal(),
            mosAsofPolicyId,
            model)
        .filter(feature -> key.asofUtc().equals(feature.getAsofUtc()))
        .orElse(null);
  }

  private static CliDaily resolveCliDaily(CliDailyRepository cliRepository, GroupKey key) {
    CliDailyId id = new CliDailyId(key.stationId(), key.targetDateLocal());
    return cliRepository.findById(id).orElse(null);
  }

  private static Map<FeatureKey, Feature> buildFeatureLookup() {
    Map<FeatureKey, Feature> map = new HashMap<>();
    map.put(new FeatureKey("gefsatmosmean", GribstreamMetric.TMAX_F),
        Feature.GEFSATMOSMEAN_TMAX_F);
    map.put(new FeatureKey("rap", GribstreamMetric.TMAX_F), Feature.RAP_TMAX_F);
    map.put(new FeatureKey("hrrr", GribstreamMetric.TMAX_F), Feature.HRRR_TMAX_F);
    map.put(new FeatureKey("nbm", GribstreamMetric.TMAX_F), Feature.NBM_TMAX_F);
    map.put(new FeatureKey("gefsatmos", GribstreamMetric.TMP_SPREAD_F),
        Feature.GEFSATMOS_TMP_SPREAD_F);
    return map;
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
    String payload = "[GRIBSTREAM-TRAINING-DATA] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private enum Feature {
    GFS_TMAX_F("gfs_tmax_f"),
    NAM_TMAX_F("nam_tmax_f"),
    GEFSATMOSMEAN_TMAX_F("gefsatmosmean_tmax_f"),
    RAP_TMAX_F("rap_tmax_f"),
    HRRR_TMAX_F("hrrr_tmax_f"),
    NBM_TMAX_F("nbm_tmax_f"),
    GEFSATMOS_TMP_SPREAD_F("gefsatmos_tmp_spread_f"),
    ACTUAL_TMAX_F("actual_tmax_f");

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

    public boolean hasAllGribstreamFeatures() {
      for (Feature feature : GRIBSTREAM_FEATURES) {
        if (!values.containsKey(feature)) {
          return false;
        }
      }
      return true;
    }

    public boolean matches(GroupKey other) {
      return key.stationId().equals(other.stationId())
          && key.targetDateLocal().equals(other.targetDateLocal())
          && key.asofUtc().equals(other.asofUtc());
    }

    public boolean isComplete() {
      for (Feature feature : FEATURE_ORDER) {
        if (!values.containsKey(feature)) {
          return false;
        }
      }
      return true;
    }

    public Double value(Feature feature) {
      return values.get(feature);
    }
  }

  private record ExportStats(long groups, long written, long skipped) {
  }
}
