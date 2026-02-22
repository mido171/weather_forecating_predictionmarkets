package com.predictionmarkets.weather.executors.export;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.config.MosTrainingDataProperties;
import com.predictionmarkets.weather.gribstream.GribstreamDailyFeatureCsv;
import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.repository.GribstreamDailyFeatureRepository;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
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
  private static final String DEFAULT_STATION_ID = "KMIA";

  private MosTrainingDataExportExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamDailyFeatureRepository repository =
          context.getBean(GribstreamDailyFeatureRepository.class);
      MosTrainingDataProperties properties =
          context.getBean(MosTrainingDataProperties.class);
      int pageSize = properties.getPageSize();
      if (pageSize < 1) {
        throw new IllegalArgumentException("mos.training-data.page-size must be >= 1");
      }
      String stationFilter = normalizeStationId(properties.getStationId());
      if (stationFilter == null) {
        stationFilter = DEFAULT_STATION_ID;
      }
      Path outputPath = resolveOutputPath(Paths.get(properties.getOutputPath()), stationFilter);
      boolean append = properties.isAppend();
      snapshot("Gribstream daily feature export starting outputPath=" + outputPath.toAbsolutePath()
          + " pageSize=" + pageSize
          + " append=" + append
          + " stationId=" + stationFilter);
      long exported = exportDailyFeatures(repository, outputPath, pageSize, append, stationFilter);
      snapshot("Gribstream daily feature export complete rows=" + exported);
    }
  }

  private static long exportDailyFeatures(GribstreamDailyFeatureRepository repository,
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
        writer.write(GribstreamDailyFeatureCsv.headerLine());
        writer.newLine();
      }
      long exported = writeRows(repository, writer, pageSize, stationFilter);
      writer.flush();
      return exported;
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write Gribstream daily features to " + outputPath, ex);
    }
  }

  private static long writeRows(GribstreamDailyFeatureRepository repository,
                                BufferedWriter writer,
                                int pageSize,
                                String stationFilter) throws IOException {
    Sort sort = Sort.by("id").ascending();
    long exported = 0L;
    int pageNumber = 0;
    while (true) {
      Page<GribstreamDailyFeatureEntity> page =
          repository.findByStationId(stationFilter, PageRequest.of(pageNumber, pageSize, sort));
      for (GribstreamDailyFeatureEntity entity : page) {
        writer.write(GribstreamDailyFeatureCsv.toCsvLine(entity));
        writer.newLine();
      }
      exported += page.getNumberOfElements();
      if (!page.hasNext()) {
        break;
      }
      pageNumber++;
    }
    return exported;
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
    String payload = "[GRIBSTREAM-DAILY-FEATURE-EXPORT] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
