package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.config.GribstreamTransferProperties;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;

public final class GribstreamDailyFeatureCsvExportExecutor {
  private static final Logger logger =
      LoggerFactory.getLogger(GribstreamDailyFeatureCsvExportExecutor.class);

  private GribstreamDailyFeatureCsvExportExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamDailyFeatureRepository repository =
          context.getBean(GribstreamDailyFeatureRepository.class);
      GribstreamTransferProperties properties = context.getBean(GribstreamTransferProperties.class);
      GribstreamTransferProperties.Export export = properties.getExport();
      int pageSize = export.getPageSize();
      if (pageSize < 1) {
        throw new IllegalArgumentException("gribstream.transfer.export.page-size must be >= 1");
      }
      Path outputPath = Paths.get(export.getOutputPath());
      snapshot("CSV export starting outputPath=" + outputPath.toAbsolutePath()
          + " pageSize=" + pageSize
          + " includeHeader=" + export.isIncludeHeader());
      long exported = exportToCsv(repository, outputPath, pageSize, export.isIncludeHeader());
      snapshot("CSV export complete rows=" + exported);
    }
  }

  private static long exportToCsv(GribstreamDailyFeatureRepository repository,
                                  Path outputPath,
                                  int pageSize,
                                  boolean includeHeader) {
    try {
      Path parent = outputPath.getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to create output directory for " + outputPath, ex);
    }
    long exported = 0L;
    try (BufferedWriter writer = Files.newBufferedWriter(
        outputPath,
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING)) {
      if (includeHeader) {
        writer.write(GribstreamDailyFeatureCsv.headerLine());
        writer.newLine();
      }
      int pageNumber = 0;
      while (true) {
        Page<GribstreamDailyFeatureEntity> page = repository.findAll(
            PageRequest.of(pageNumber, pageSize, Sort.by("id").ascending()));
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
      writer.flush();
      return exported;
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write CSV export to " + outputPath, ex);
    }
  }

  private static void snapshot(String message) {
    String payload = "[GRIBSTREAM-CSV-EXPORT] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
