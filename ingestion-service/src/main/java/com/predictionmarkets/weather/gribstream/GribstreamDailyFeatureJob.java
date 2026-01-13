package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.backfill.CheckpointHeartbeat;
import com.predictionmarkets.weather.backfill.IngestCheckpointService;
import com.predictionmarkets.weather.models.IngestCheckpoint;
import com.predictionmarkets.weather.models.IngestCheckpointStatus;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class GribstreamDailyFeatureJob {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamDailyFeatureJob.class);
  private static final String JOB_NAME = "gribstream_daily_features";

  private final GribstreamDailyTmaxService dailyService;
  private final IngestCheckpointService checkpointService;
  private final StationRegistryRepository stationRegistryRepository;

  public GribstreamDailyFeatureJob(GribstreamDailyTmaxService dailyService,
                                   IngestCheckpointService checkpointService,
                                   StationRegistryRepository stationRegistryRepository) {
    this.dailyService = dailyService;
    this.checkpointService = checkpointService;
    this.stationRegistryRepository = stationRegistryRepository;
  }

  public List<GribstreamDailyOpinionResult> runRange(StationSpec station,
                                                     LocalDate start,
                                                     LocalDate end,
                                                     Instant asOfUtc) {
    Objects.requireNonNull(asOfUtc, "asOfUtc is required");
    return runRange(station, start, end, (ignoredStation, ignoredDate) -> asOfUtc);
  }

  public List<GribstreamDailyOpinionResult> runRange(StationSpec station,
                                                     LocalDate start,
                                                     LocalDate end,
                                                     GribstreamAsOfSupplier asOfSupplier) {
    Objects.requireNonNull(station, "station is required");
    Objects.requireNonNull(start, "start is required");
    Objects.requireNonNull(end, "end is required");
    Objects.requireNonNull(asOfSupplier, "asOfSupplier is required");
    if (end.isBefore(start)) {
      throw new IllegalArgumentException("end must be >= start");
    }
    String stationId = station.stationId();
    boolean checkpointsEnabled = stationRegistryRepository.existsById(stationId);
    if (!checkpointsEnabled) {
      snapshot("job=gribstream_daily_features station=" + stationId
          + " checkpoint=skipped reason=station_registry_missing");
    }
    IngestCheckpoint existing = checkpointsEnabled
        ? checkpointService.findCheckpoint(JOB_NAME, stationId, null).orElse(null)
        : null;
    LocalDate cursorDate = existing != null ? existing.getCursorDate() : null;
    if (cursorDate != null && !cursorDate.isBefore(end)) {
      snapshot("job=gribstream_daily_features station=" + stationId
          + " progress=complete cursorDate=" + cursorDate);
      if (checkpointsEnabled) {
        checkpointService.markComplete(JOB_NAME, stationId, null, cursorDate, null);
      }
      return List.of();
    }
    LocalDate effectiveStart = start;
    if (cursorDate != null && !cursorDate.isBefore(start)) {
      effectiveStart = cursorDate.plusDays(1);
    }
    if (effectiveStart.isAfter(end)) {
      snapshot("job=gribstream_daily_features station=" + stationId
          + " progress=complete cursorDate=" + cursorDate);
      if (checkpointsEnabled) {
        checkpointService.markComplete(JOB_NAME, stationId, null, cursorDate, null);
      }
      return List.of();
    }
    List<GribstreamDailyOpinionResult> results = new ArrayList<>();
    AtomicReference<LocalDate> cursorRef = new AtomicReference<>(cursorDate);
    CheckpointHeartbeat heartbeat = null;
    if (checkpointsEnabled) {
      heartbeat = CheckpointHeartbeat.start(() ->
          checkpointService.markRunning(JOB_NAME, stationId, null, cursorRef.get(), null));
    }
    try {
      if (checkpointsEnabled) {
        checkpointService.markRunning(JOB_NAME, stationId, null, cursorRef.get(), null);
      }
      LocalDate current = effectiveStart;
      int totalDays = (int) (end.toEpochDay() - effectiveStart.toEpochDay() + 1);
      int processed = 0;
      while (!current.isAfter(end)) {
        Instant asOfUtc = asOfSupplier.resolve(station, current);
        if (asOfUtc == null) {
          throw new IllegalArgumentException("asOfUtc is required");
        }
        GribstreamDailyOpinionResult result =
            dailyService.computeAndPersistDailyOpinions(station, current, asOfUtc);
        results.add(result);
        processed++;
        cursorDate = current;
        if (checkpointsEnabled) {
          cursorRef.set(cursorDate);
          checkpointService.markRunning(JOB_NAME, stationId, null, cursorDate, null);
        }
        snapshot("job=gribstream_daily_features station=" + stationId
            + " targetDate=" + current
            + " progress=" + processed + "/" + totalDays
            + " remaining=" + (totalDays - processed)
            + " status=running");
        current = current.plusDays(1);
      }
      cursorDate = end;
      if (checkpointsEnabled) {
        cursorRef.set(cursorDate);
        if (heartbeat != null) {
          heartbeat.close();
        }
        checkpointService.markComplete(JOB_NAME, stationId, null, cursorDate, null);
      }
      snapshot("job=gribstream_daily_features station=" + stationId
          + " progress=complete cursorDate=" + cursorDate);
    } catch (RuntimeException ex) {
      if (checkpointsEnabled) {
        if (heartbeat != null) {
          heartbeat.close();
        }
        checkpointService.markFailed(JOB_NAME, stationId, null, cursorDate, null, ex);
      }
      snapshot("job=gribstream_daily_features station=" + stationId
          + " status=failed");
      throw ex;
    }
    return results;
  }

  public String jobName() {
    return JOB_NAME;
  }

  private void snapshot(String message) {
    String payload = "[GRIBSTREAM] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
