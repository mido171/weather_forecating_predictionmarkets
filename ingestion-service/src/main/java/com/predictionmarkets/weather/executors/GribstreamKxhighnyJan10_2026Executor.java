package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.common.TimeSemantics;
import com.predictionmarkets.weather.gribstream.GribstreamAsOfSupplier;
import com.predictionmarkets.weather.gribstream.GribstreamDailyFeatureJob;
import com.predictionmarkets.weather.gribstream.GribstreamDailyOpinionResult;
import com.predictionmarkets.weather.gribstream.GribstreamRunnerProperties;
import com.predictionmarkets.weather.gribstream.StationSpec;
import com.predictionmarkets.weather.models.AsofTimeZone;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

public final class GribstreamKxhighnyJan10_2026Executor {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamKxhighnyJan10_2026Executor.class);

  private GribstreamKxhighnyJan10_2026Executor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamDailyFeatureJob job = context.getBean(GribstreamDailyFeatureJob.class);
      GribstreamRunnerProperties properties = context.getBean(GribstreamRunnerProperties.class);
      StationSpec station = new StationSpec(
          "KNYC",
          "America/New_York",
          40.77898,
          -73.96925,
          "KNYC");
      LocalDate startDateLocal = requireStartDate(properties);
      LocalDate endDateLocal = requireEndDate(properties);
      GribstreamAsOfSupplier asOfSupplier = resolveAsOfSupplier(properties);
      snapshot("Starting Gribstream example station=" + station.stationId()
          + " targetDateLocalRange=" + startDateLocal + ".." + endDateLocal
          + " " + describeAsOf(properties));
      List<GribstreamDailyOpinionResult> results =
          job.runRange(station, startDateLocal, endDateLocal, asOfSupplier);
      if (results.isEmpty()) {
        snapshot("No Gribstream results (already complete per checkpoint).");
        return;
      }
      GribstreamDailyOpinionResult result = results.get(results.size() - 1);
      logSummary(result);
      snapshot("Gribstream example complete.");
    }
  }

  private static void logSummary(GribstreamDailyOpinionResult result) {
    Map<String, Double> sorted = new TreeMap<>(result.tmaxByModelF());
    for (Map.Entry<String, Double> entry : sorted.entrySet()) {
      snapshot("model=" + entry.getKey() + " tmax_f=" + format(entry.getValue()));
    }
    snapshot("model=gefsatmos tmp_spread_f=" + format(result.gefsSpreadF()));
  }

  private static String format(Double value) {
    if (value == null) {
      return "null";
    }
    return String.format(Locale.ROOT, "%.2f", value);
  }

  private static void snapshot(String message) {
    String payload = "[GRIBSTREAM-EXECUTOR] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private static LocalDate requireStartDate(GribstreamRunnerProperties properties) {
    LocalDate start = properties.getStartDateLocal();
    if (start == null) {
      throw new IllegalArgumentException("app.runners.gribstream-example.start-date-local is required");
    }
    return start;
  }

  private static LocalDate requireEndDate(GribstreamRunnerProperties properties) {
    LocalDate end = properties.getEndDateLocal();
    if (end == null) {
      throw new IllegalArgumentException("app.runners.gribstream-example.end-date-local is required");
    }
    return end;
  }

  private static GribstreamAsOfSupplier resolveAsOfSupplier(GribstreamRunnerProperties properties) {
    LocalTime asOfLocalTime = properties.getAsOfLocalTime();
    if (asOfLocalTime != null) {
      AsofTimeZone asOfTimeZone = resolveAsOfTimeZone(properties.getAsOfTimeZone());
      return (station, targetDateLocal) -> {
        ZoneId stationZoneId = ZoneId.of(station.zoneId());
        ZoneId asOfZoneId = asOfTimeZone == AsofTimeZone.UTC ? ZoneOffset.UTC : stationZoneId;
        return TimeSemantics.computeAsOfTimes(
            targetDateLocal,
            asOfLocalTime,
            stationZoneId,
            asOfZoneId).asOfUtc();
      };
    }
    Instant asOfUtc = properties.getAsOfUtc();
    if (asOfUtc == null) {
      throw new IllegalArgumentException(
          "app.runners.gribstream-example.asof-local-time/asof-time-zone or asof-utc is required");
    }
    return (station, targetDateLocal) -> asOfUtc;
  }

  private static AsofTimeZone resolveAsOfTimeZone(AsofTimeZone asOfTimeZone) {
    return asOfTimeZone == null ? AsofTimeZone.LOCAL : asOfTimeZone;
  }

  private static String describeAsOf(GribstreamRunnerProperties properties) {
    if (properties.getAsOfLocalTime() != null) {
      return "asOfLocalTime=" + properties.getAsOfLocalTime()
          + " asOfTimeZone=" + resolveAsOfTimeZone(properties.getAsOfTimeZone());
    }
    return "asOfUtc=" + properties.getAsOfUtc();
  }
}
