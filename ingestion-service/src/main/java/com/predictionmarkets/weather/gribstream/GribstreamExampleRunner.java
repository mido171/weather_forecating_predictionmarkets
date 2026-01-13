package com.predictionmarkets.weather.gribstream;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class GribstreamExampleRunner implements CommandLineRunner {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamExampleRunner.class);

  private final GribstreamDailyFeatureJob job;
  private final GribstreamRunnerProperties properties;

  public GribstreamExampleRunner(GribstreamDailyFeatureJob job,
                                 GribstreamRunnerProperties properties) {
    this.job = job;
    this.properties = properties;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    StationSpec station = new StationSpec(
        "KNYC",
        "America/New_York",
        40.77898,
        -73.96925,
        "KNYC");
    LocalDate startDateLocal = requireStartDate();
    LocalDate endDateLocal = requireEndDate();
    Instant asOfUtc = requireAsOfUtc();
    snapshot("Starting Gribstream example runner station=" + station.stationId()
        + " targetDateLocalRange=" + startDateLocal + ".." + endDateLocal
        + " asOfUtc=" + asOfUtc);
    List<GribstreamDailyOpinionResult> results =
        job.runRange(station, startDateLocal, endDateLocal, asOfUtc);
    if (results.isEmpty()) {
      snapshot("No Gribstream results (already complete per checkpoint).");
      return;
    }
    GribstreamDailyOpinionResult result = results.get(results.size() - 1);
    logSummary(result);
    snapshot("Gribstream example runner complete.");
  }

  private void logSummary(GribstreamDailyOpinionResult result) {
    Map<String, Double> sorted = new TreeMap<>(result.tmaxByModelF());
    for (Map.Entry<String, Double> entry : sorted.entrySet()) {
      snapshot("model=" + entry.getKey() + " tmax_f=" + format(entry.getValue()));
    }
    snapshot("model=gefsatmos tmp_spread_f=" + format(result.gefsSpreadF()));
  }

  private String format(Double value) {
    if (value == null) {
      return "null";
    }
    return String.format(Locale.ROOT, "%.2f", value);
  }

  private void snapshot(String message) {
    String payload = "[GRIBSTREAM-RUNNER] " + message;
    logger.info(payload);
    System.out.println(payload);
  }

  private LocalDate requireStartDate() {
    LocalDate start = properties.getStartDateLocal();
    if (start == null) {
      throw new IllegalArgumentException("app.runners.gribstream-example.start-date-local is required");
    }
    return start;
  }

  private LocalDate requireEndDate() {
    LocalDate end = properties.getEndDateLocal();
    if (end == null) {
      throw new IllegalArgumentException("app.runners.gribstream-example.end-date-local is required");
    }
    return end;
  }

  private Instant requireAsOfUtc() {
    Instant asOfUtc = properties.getAsOfUtc();
    if (asOfUtc == null) {
      throw new IllegalArgumentException("app.runners.gribstream-example.asof-utc is required");
    }
    return asOfUtc;
  }
}
