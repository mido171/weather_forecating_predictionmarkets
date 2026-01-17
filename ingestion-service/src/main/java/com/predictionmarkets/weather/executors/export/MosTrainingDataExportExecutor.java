package com.predictionmarkets.weather.executors.export;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.config.MosTrainingDataProperties;
import com.predictionmarkets.weather.config.PipelineProperties;
import com.predictionmarkets.weather.models.CliDaily;
import com.predictionmarkets.weather.models.CliDailyId;
import com.predictionmarkets.weather.models.MosDailyValue;
import com.predictionmarkets.weather.repository.CliDailyRepository;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;

public final class MosTrainingDataExportExecutor {
  private static final Logger logger =
      LoggerFactory.getLogger(MosTrainingDataExportExecutor.class);

  private static final List<String> MODELS = List.of("GFS", "NAM");
  private static final List<String> STAT_SUFFIXES = List.of("min", "max", "mean", "median");

  private MosTrainingDataExportExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      MosDailyValueRepository mosRepository =
          context.getBean(MosDailyValueRepository.class);
      CliDailyRepository cliRepository =
          context.getBean(CliDailyRepository.class);
      MosTrainingDataProperties properties =
          context.getBean(MosTrainingDataProperties.class);
      PipelineProperties pipelineProperties =
          context.getBean(PipelineProperties.class);
      int pageSize = properties.getPageSize();
      if (pageSize < 1) {
        throw new IllegalArgumentException("mos.training-data.page-size must be >= 1");
      }
      String stationFilter = resolveStationFilter(properties, pipelineProperties);
      Path outputPath = resolveOutputPath(Paths.get(properties.getOutputPath()), stationFilter);
      boolean append = properties.isAppend();
      List<String> variableCodes = mosRepository.findDistinctVariableCodesByModelIn(MODELS);
      snapshot("MOS training data export starting outputPath=" + outputPath.toAbsolutePath()
          + " pageSize=" + pageSize
          + " append=" + append
          + " stationId=" + (stationFilter == null ? "ALL" : stationFilter)
          + " models=" + MODELS
          + " variables=" + variableCodes.size());
      ExportStats stats = exportTrainingData(
          mosRepository, cliRepository, variableCodes, outputPath, pageSize, append, stationFilter);
      snapshot("MOS training data export complete rows=" + stats.written()
          + " skippedMissingTarget=" + stats.skipped()
          + " groups=" + stats.groups());
    }
  }

  private static ExportStats exportTrainingData(MosDailyValueRepository mosRepository,
                                                CliDailyRepository cliRepository,
                                                List<String> variableCodes,
                                                Path outputPath,
                                                int pageSize,
                                                boolean append,
                                                String stationFilter) {
    ensureParentDirectory(outputPath);
    boolean writeHeader = shouldWriteHeader(outputPath, append);
    try (BufferedWriter writer = Files.newBufferedWriter(
        outputPath,
        StandardCharsets.UTF_8,
        append
            ? new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.APPEND}
            : new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING})) {
      if (writeHeader) {
        writer.write(headerLine(variableCodes));
        writer.newLine();
      }
      return writeRows(mosRepository, cliRepository, variableCodes, writer, pageSize,
          stationFilter);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write MOS training data to " + outputPath, ex);
    }
  }

  private static ExportStats writeRows(MosDailyValueRepository mosRepository,
                                       CliDailyRepository cliRepository,
                                       List<String> variableCodes,
                                       BufferedWriter writer,
                                       int pageSize,
                                       String stationFilter) throws IOException {
    Sort sort = Sort.by(
        Sort.Order.asc("stationId"),
        Sort.Order.asc("targetDateLocal"),
        Sort.Order.asc("asofUtc"),
        Sort.Order.asc("model"),
        Sort.Order.asc("variableCode"),
        Sort.Order.asc("id"));
    GroupAccumulator current = null;
    long groups = 0L;
    long written = 0L;
    long skipped = 0L;
    int pageNumber = 0;
    while (true) {
      Page<MosDailyValue> page = stationFilter == null
          ? mosRepository.findByModelIn(MODELS, PageRequest.of(pageNumber, pageSize, sort))
          : mosRepository.findByStationIdAndModelIn(
              stationFilter, MODELS, PageRequest.of(pageNumber, pageSize, sort));
      for (MosDailyValue value : page) {
        GroupKey key = new GroupKey(
            value.getStationId(),
            value.getTargetDateLocal(),
            value.getAsofUtc());
        if (current == null) {
          current = new GroupAccumulator(key);
        } else if (!current.matches(key)) {
          groups++;
          if (writeGroup(writer, current, cliRepository, variableCodes)) {
            written++;
          } else {
            skipped++;
          }
          current = new GroupAccumulator(key);
        }
        current.add(value);
      }
      if (!page.hasNext()) {
        break;
      }
      pageNumber++;
      if (groups % 5000 == 0 && groups > 0) {
        snapshot("MOS training data export progress groups=" + groups
            + " rows=" + written
            + " skipped=" + skipped);
      }
    }
    if (current != null) {
      groups++;
      if (writeGroup(writer, current, cliRepository, variableCodes)) {
        written++;
      } else {
        skipped++;
      }
    }
    writer.flush();
    return new ExportStats(groups, written, skipped);
  }

  private static boolean writeGroup(BufferedWriter writer,
                                    GroupAccumulator group,
                                    CliDailyRepository cliRepository,
                                    List<String> variableCodes) throws IOException {
    Double targetTmax = resolveCliDailyTmax(cliRepository, group.key());
    if (targetTmax == null) {
      return false;
    }
    StringBuilder builder = new StringBuilder();
    appendValue(builder, group.key().stationId());
    appendValue(builder, group.key().targetDateLocal().toString());
    appendValue(builder, group.key().asofUtc() == null ? "" : group.key().asofUtc().toString());
    for (String model : MODELS) {
      String normalizedModel = model.toLowerCase(Locale.ROOT);
      for (String variable : variableCodes) {
        VariableStats stats = group.stats(normalizedModel, variable);
        appendValue(builder, toString(stats == null ? null : stats.min()));
        appendValue(builder, toString(stats == null ? null : stats.max()));
        appendValue(builder, toString(stats == null ? null : stats.mean()));
        appendValue(builder, toString(stats == null ? null : stats.median()));
      }
    }
    appendValue(builder, Double.toString(targetTmax));
    writer.write(builder.toString());
    writer.newLine();
    return true;
  }

  private static String headerLine(List<String> variableCodes) {
    StringBuilder builder = new StringBuilder();
    appendValue(builder, "station_id");
    appendValue(builder, "target_date_local");
    appendValue(builder, "asof_utc");
    for (String model : MODELS) {
      String prefix = model.toLowerCase(Locale.ROOT);
      for (String variable : variableCodes) {
        for (String stat : STAT_SUFFIXES) {
          appendValue(builder, prefix + "_" + variable + "_" + stat);
        }
      }
    }
    appendValue(builder, "target_tmax_f");
    return builder.toString();
  }

  private static Double resolveCliDailyTmax(CliDailyRepository cliRepository, GroupKey key) {
    CliDailyId id = new CliDailyId(key.stationId(), key.targetDateLocal());
    CliDaily cliDaily = cliRepository.findById(id).orElse(null);
    if (cliDaily == null || cliDaily.getTmaxF() == null) {
      return null;
    }
    return cliDaily.getTmaxF().doubleValue();
  }

  private static String toString(BigDecimal value) {
    return value == null ? "" : value.toPlainString();
  }

  private static String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      return null;
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }

  private static Path resolveOutputPath(Path basePath, String stationFilter) {
    if (stationFilter == null || stationFilter.isBlank()) {
      return basePath;
    }
    Path fileName = basePath.getFileName();
    if (fileName == null) {
      return basePath;
    }
    String prefix = stationFilter + "_";
    String fileNameText = fileName.toString();
    String resolvedName = fileNameText.startsWith(prefix) ? fileNameText : prefix + fileNameText;
    Path parent = basePath.getParent();
    return parent == null ? Paths.get(resolvedName) : parent.resolve(resolvedName);
  }

  private static String resolveStationFilter(MosTrainingDataProperties properties,
                                             PipelineProperties pipelineProperties) {
    String configured = normalizeStationId(properties.getStationId());
    if (configured != null) {
      return configured;
    }
    if (pipelineProperties == null) {
      return null;
    }
    List<String> pipelineStationIds = parseStationIds(pipelineProperties.getStationIdsToRun());
    if (pipelineStationIds.size() == 1) {
      return pipelineStationIds.get(0);
    }
    return null;
  }

  private static List<String> parseStationIds(String stationIdsToRun) {
    if (stationIdsToRun == null || stationIdsToRun.isBlank()) {
      return List.of();
    }
    Set<String> stationIds = new LinkedHashSet<>();
    for (String token : stationIdsToRun.split(",")) {
      String trimmed = token.trim();
      if (!trimmed.isEmpty()) {
        stationIds.add(trimmed.toUpperCase(Locale.ROOT));
      }
    }
    return new ArrayList<>(stationIds);
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
    String payload = "[MOS-TRAINING-DATA] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private record GroupKey(String stationId, LocalDate targetDateLocal, Instant asofUtc) {
  }

  private record VariableStats(BigDecimal min, BigDecimal max, BigDecimal mean, BigDecimal median) {
  }

  private static final class GroupAccumulator {
    private final GroupKey key;
    private final Map<String, Map<String, VariableStats>> valuesByModel = new HashMap<>();

    private GroupAccumulator(GroupKey key) {
      this.key = key;
    }

    public GroupKey key() {
      return key;
    }

    public void add(MosDailyValue value) {
      if (value == null || value.getModel() == null || value.getVariableCode() == null) {
        return;
      }
      String model = value.getModel().trim().toLowerCase(Locale.ROOT);
      String variable = value.getVariableCode().trim().toLowerCase(Locale.ROOT);
      Map<String, VariableStats> modelMap =
          valuesByModel.computeIfAbsent(model, k -> new LinkedHashMap<>());
      modelMap.put(variable, new VariableStats(
          value.getValueMin(),
          value.getValueMax(),
          value.getValueMean(),
          value.getValueMedian()));
    }

    public VariableStats stats(String model, String variable) {
      Map<String, VariableStats> modelMap = valuesByModel.get(model);
      if (modelMap == null) {
        return null;
      }
      return modelMap.get(variable);
    }

    public boolean matches(GroupKey other) {
      return key.stationId().equals(other.stationId())
          && key.targetDateLocal().equals(other.targetDateLocal())
          && ((key.asofUtc() == null && other.asofUtc() == null)
          || (key.asofUtc() != null && key.asofUtc().equals(other.asofUtc())));
    }
  }

  private record ExportStats(long groups, long written, long skipped) {
  }
}
