package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.common.TimeSemantics;
import com.predictionmarkets.weather.executors.PipelineProperties;
import com.predictionmarkets.weather.models.AsofTimeZone;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
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
  private final GribstreamProperties gribstreamProperties;
  private final PipelineProperties pipelineProperties;

  public GribstreamExampleRunner(GribstreamDailyFeatureJob job,
                                 GribstreamRunnerProperties properties,
                                 GribstreamProperties gribstreamProperties,
                                 PipelineProperties pipelineProperties) {
    this.job = job;
    this.properties = properties;
    this.gribstreamProperties = gribstreamProperties;
    this.pipelineProperties = pipelineProperties;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    List<StationSpec> stations = resolveStations();
    List<StationSpec> selectedStations = filterStations(stations);
    LocalDate startDateLocal = requireStartDate();
    LocalDate endDateLocal = requireEndDate();
    GribstreamAsOfSupplier asOfSupplier = resolveAsOfSupplier();
    for (StationSpec station : selectedStations) {
      snapshot("Starting Gribstream example runner station=" + station.stationId()
          + " targetDateLocalRange=" + startDateLocal + ".." + endDateLocal
          + " " + describeAsOf());
      List<GribstreamDailyOpinionResult> results =
          job.runRange(station, startDateLocal, endDateLocal, asOfSupplier);
      if (results.isEmpty()) {
        snapshot("No Gribstream results (already complete per checkpoint).");
        continue;
      }
      GribstreamDailyOpinionResult result = results.get(results.size() - 1);
      logSummary(result);
    }
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

  private List<StationSpec> resolveStations() {
    List<GribstreamProperties.StationProperties> configured = gribstreamProperties.getStations();
    if (configured == null || configured.isEmpty()) {
      throw new IllegalArgumentException("gribstream.stations is required");
    }
    List<StationSpec> stations = new ArrayList<>(configured.size());
    for (GribstreamProperties.StationProperties station : configured) {
      stations.add(new StationSpec(
          station.getStationId(),
          station.getZoneId(),
          station.getLatitude(),
          station.getLongitude(),
          station.getName()));
    }
    return stations;
  }

  private List<StationSpec> filterStations(List<StationSpec> stations) {
    List<String> stationIds = parseStationIds(pipelineProperties.getStationIdsToRun());
    if (stationIds.isEmpty()) {
      return stations;
    }
    Map<String, StationSpec> byId = new HashMap<>();
    for (StationSpec station : stations) {
      byId.put(station.stationId(), station);
    }
    List<StationSpec> selected = new ArrayList<>(stationIds.size());
    for (String stationId : stationIds) {
      StationSpec station = byId.get(stationId);
      if (station == null) {
        throw new IllegalArgumentException(
            "pipeline.station-ids-to-run includes stationId=" + stationId
                + " but gribstream.stations does not define it");
      }
      selected.add(station);
    }
    return selected;
  }

  private static List<String> parseStationIds(String stationIdsToRun) {
    if (stationIdsToRun == null || stationIdsToRun.isBlank()) {
      return List.of();
    }
    Set<String> stationIds = new LinkedHashSet<>();
    for (String token : Arrays.asList(stationIdsToRun.split(","))) {
      String trimmed = token.trim();
      if (!trimmed.isEmpty()) {
        stationIds.add(trimmed.toUpperCase(Locale.ROOT));
      }
    }
    return new ArrayList<>(stationIds);
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

  private GribstreamAsOfSupplier resolveAsOfSupplier() {
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

  private AsofTimeZone resolveAsOfTimeZone(AsofTimeZone asOfTimeZone) {
    return asOfTimeZone == null ? AsofTimeZone.LOCAL : asOfTimeZone;
  }

  private String describeAsOf() {
    if (properties.getAsOfLocalTime() != null) {
      return "asOfLocalTime=" + properties.getAsOfLocalTime()
          + " asOfTimeZone=" + resolveAsOfTimeZone(properties.getAsOfTimeZone());
    }
    return "asOfUtc=" + properties.getAsOfUtc();
  }
}
