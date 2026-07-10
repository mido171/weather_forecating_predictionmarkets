package com.predictionmarkets.weather.executors;

import com.predictionmarkets.weather.IngestionServiceApplication;
import com.predictionmarkets.weather.common.TimeSemantics;
import com.predictionmarkets.weather.config.PipelineProperties;
import com.predictionmarkets.weather.gribstream.GribstreamAsOfSupplier;
import com.predictionmarkets.weather.gribstream.GribstreamExecutorProperties;
import com.predictionmarkets.weather.gribstream.GribstreamProperties;
import com.predictionmarkets.weather.gribstream.GribstreamRunnerProperties;
import com.predictionmarkets.weather.gribstream.GribstreamVariableIngestJob;
import com.predictionmarkets.weather.gribstream.GribstreamVariableIngestProperties;
import com.predictionmarkets.weather.gribstream.StationSpec;
import com.predictionmarkets.weather.models.AsofTimeZone;
import jakarta.annotation.PreDestroy;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.stereotype.Component;

@Component
public class GribstreamTaskExecutor {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamTaskExecutor.class);
  private final ExecutorService executor;

  public GribstreamTaskExecutor(GribstreamExecutorProperties properties) {
    int threadCount = Math.max(1, properties.getThreadCount());
    this.executor = Executors.newFixedThreadPool(threadCount, namedThreadFactory());
  }

  public <T> List<T> invokeAllOrFail(List<Callable<T>> tasks) {
    List<Future<T>> futures = new ArrayList<>(tasks.size());
    for (Callable<T> task : tasks) {
      futures.add(executor.submit(task));
    }
    List<T> results = new ArrayList<>(tasks.size());
    for (Future<T> future : futures) {
      try {
        results.add(future.get());
      } catch (InterruptedException ex) {
        Thread.currentThread().interrupt();
        cancelRemaining(futures);
        throw new IllegalStateException("Gribstream task interrupted", ex);
      } catch (ExecutionException ex) {
        cancelRemaining(futures);
        Throwable cause = ex.getCause() == null ? ex : ex.getCause();
        if (cause instanceof RuntimeException runtimeException) {
          throw runtimeException;
        }
        throw new IllegalStateException("Gribstream task failed", cause);
      }
    }
    return results;
  }

  private void cancelRemaining(List<? extends Future<?>> futures) {
    for (Future<?> future : futures) {
      if (!future.isDone()) {
        future.cancel(true);
      }
    }
  }

  private static ThreadFactory namedThreadFactory() {
    AtomicInteger counter = new AtomicInteger(1);
    return runnable -> {
      Thread thread = new Thread(runnable);
      thread.setName("gribstream-worker-" + counter.getAndIncrement());
      thread.setDaemon(false);
      return thread;
    };
  }

  @PreDestroy
  public void shutdown() {
    executor.shutdown();
    try {
      if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
        executor.shutdownNow();
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      executor.shutdownNow();
    }
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamVariableIngestJob job = context.getBean(GribstreamVariableIngestJob.class);
      GribstreamVariableIngestProperties ingestProperties =
          context.getBean(GribstreamVariableIngestProperties.class);
      GribstreamProperties gribstreamProperties = context.getBean(GribstreamProperties.class);
      GribstreamRunnerProperties runnerProperties = context.getBean(GribstreamRunnerProperties.class);
      PipelineProperties pipelineProperties = context.getBean(PipelineProperties.class);
      if (!ingestProperties.isEnabled()) {
        throw new IllegalArgumentException("gribstream.variable-ingest.enabled must be true");
      }

      List<StationSpec> stations = resolveStations(gribstreamProperties);
      List<StationSpec> selectedStations =
          filterStations(stations, pipelineProperties.getStationIdsToRun());
      LocalDate startDateLocal = requireStartDate(runnerProperties);
      LocalDate endDateLocal = requireEndDate(runnerProperties);
      GribstreamAsOfSupplier asOfSupplier = resolveAsOfSupplier(runnerProperties);
      snapshot("Starting Gribstream variable ingest"
          + " stations=" + selectedStations.size()
          + " dateRange=" + startDateLocal + ".." + endDateLocal
          + " models=" + describeModels(ingestProperties, gribstreamProperties)
          + " batchSize=" + ingestProperties.getBatchSize()
          + " maxVariablesPerModel=" + ingestProperties.getMaxVariablesPerModel()
          + " " + describeAsOf(runnerProperties));

      for (StationSpec station : selectedStations) {
        snapshot("Station " + station.stationId() + " ingest starting");
        int upserted = job.runRange(station, startDateLocal, endDateLocal, asOfSupplier);
        snapshot("Station " + station.stationId() + " ingest complete upserted=" + upserted);
      }
      snapshot("Gribstream variable ingest complete.");
    }
  }

  private static List<StationSpec> resolveStations(GribstreamProperties gribstreamProperties) {
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

  private static List<StationSpec> filterStations(List<StationSpec> stations, String stationIdsToRun) {
    List<String> stationIds = parseStationIds(stationIdsToRun);
    if (stationIds.isEmpty()) {
      return stations;
    }
    Map<String, StationSpec> byId = stations.stream()
        .collect(java.util.stream.Collectors.toMap(StationSpec::stationId, station -> station));
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

  private static String describeModels(GribstreamVariableIngestProperties ingestProperties,
                                       GribstreamProperties gribstreamProperties) {
    List<String> configured = ingestProperties.getModels();
    if (configured != null && !configured.isEmpty()) {
      return configured.toString();
    }
    Map<String, GribstreamProperties.ModelProperties> models = gribstreamProperties.getModels();
    if (models == null || models.isEmpty()) {
      return "[]";
    }
    return models.keySet().toString();
  }

  private static void snapshot(String message) {
    String payload = "[GRIBSTREAM-VARS-EXECUTOR] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
