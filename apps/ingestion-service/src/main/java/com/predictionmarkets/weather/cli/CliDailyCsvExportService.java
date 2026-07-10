package com.predictionmarkets.weather.cli;

import com.predictionmarkets.weather.iem.IemCliDaily;
import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import org.springframework.stereotype.Service;

@Service
public class CliDailyCsvExportService {
  private static final DateTimeFormatter INSTANT_FORMATTER = DateTimeFormatter.ISO_INSTANT;

  public long exportFetchedDays(String stationId,
                                LocalDate startDate,
                                LocalDate endDate,
                                List<IemCliDaily> days,
                                Path outputPath,
                                boolean includeHeader) {
    String normalizedStation = normalizeStationId(stationId);
    if (startDate == null || endDate == null) {
      throw new IllegalArgumentException("startDate and endDate are required");
    }
    if (endDate.isBefore(startDate)) {
      throw new IllegalArgumentException("endDate must be >= startDate");
    }
    if (outputPath == null) {
      throw new IllegalArgumentException("outputPath is required");
    }

    ensureParentDirectory(outputPath);
    List<IemCliDaily> sorted = days.stream()
        .sorted(Comparator.comparing(IemCliDaily::targetDateLocal))
        .toList();

    try (BufferedWriter writer = Files.newBufferedWriter(
        outputPath,
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING)) {
      if (includeHeader) {
        writer.write(
            "station_id,target_date_local,tmax_f,tmin_f,report_issued_at_utc,truth_source_url");
        writer.newLine();
      }
      long written = 0L;
      for (IemCliDaily day : sorted) {
        if (!normalizedStation.equals(normalizeStationId(day.stationId()))) {
          continue;
        }
        if (day.targetDateLocal().isBefore(startDate) || day.targetDateLocal().isAfter(endDate)) {
          continue;
        }
        writer.write(toCsvLine(day));
        writer.newLine();
        written++;
      }
      writer.flush();
      return written;
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to write CLI settlement CSV to " + outputPath, ex);
    }
  }

  private void ensureParentDirectory(Path outputPath) {
    try {
      Path parent = outputPath.getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to create output directory for " + outputPath, ex);
    }
  }

  private String toCsvLine(IemCliDaily day) {
    return csv(day.stationId())
        + "," + csv(day.targetDateLocal() == null ? null : day.targetDateLocal().toString())
        + "," + csv(decimal(day.tmaxF()))
        + "," + csv(decimal(day.tminF()))
        + "," + csv(formatInstant(day.reportIssuedAtUtc()))
        + "," + csv(day.truthSourceUrl());
  }

  private String decimal(BigDecimal value) {
    return value == null ? null : value.stripTrailingZeros().toPlainString();
  }

  private String formatInstant(Instant value) {
    return value == null ? null : INSTANT_FORMATTER.format(value);
  }

  private String csv(String value) {
    if (value == null) {
      return "";
    }
    String escaped = value.replace("\"", "\"\"");
    if (escaped.contains(",") || escaped.contains("\"") || escaped.contains("\n")
        || escaped.contains("\r")) {
      return "\"" + escaped + "\"";
    }
    return escaped;
  }

  private String normalizeStationId(String stationId) {
    if (stationId == null || stationId.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return stationId.trim().toUpperCase(Locale.ROOT);
  }
}
