package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.backfill.CheckpointHeartbeat;
import com.predictionmarkets.weather.backfill.IngestCheckpointService;
import com.predictionmarkets.weather.models.IngestCheckpoint;
import com.predictionmarkets.weather.models.IngestCheckpointStatus;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class GribstreamVariableIngestJob {
  private static final Logger logger = LoggerFactory.getLogger(GribstreamVariableIngestJob.class);
  private static final String JOB_NAME = "gribstream_variable_ingest";

  private final GribstreamVariableIngestService ingestService;
  private final IngestCheckpointService checkpointService;
  private final StationRegistryRepository stationRegistryRepository;

  public GribstreamVariableIngestJob(GribstreamVariableIngestService ingestService,
                                     IngestCheckpointService checkpointService,
                                     StationRegistryRepository stationRegistryRepository) {
    this.ingestService = ingestService;
    this.checkpointService = checkpointService;
    this.stationRegistryRepository = stationRegistryRepository;
  }

  public int runRange(StationSpec station,
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
    LocalDate cursorDate = null;
    if (checkpointsEnabled) {
      IngestCheckpoint existing =
          checkpointService.findCheckpoint(JOB_NAME, stationId, null).orElse(null);
      cursorDate = existing != null ? existing.getCursorDate() : null;
    }
    if (cursorDate != null && !cursorDate.isBefore(end)) {
      snapshot("job=gribstream_variable_ingest station=" + stationId
          + " progress=complete cursorDate=" + cursorDate);
      if (checkpointsEnabled) {
        checkpointService.markComplete(JOB_NAME, stationId, null, cursorDate, null);
      }
      return 0;
    }
    LocalDate effectiveStart = start;
    if (cursorDate != null && !cursorDate.isBefore(start)) {
      effectiveStart = cursorDate.plusDays(1);
    }
    if (effectiveStart.isAfter(end)) {
      snapshot("job=gribstream_variable_ingest station=" + stationId
          + " progress=complete cursorDate=" + cursorDate);
      if (checkpointsEnabled) {
        checkpointService.markComplete(JOB_NAME, stationId, null, cursorDate, null);
      }
      return 0;
    }
    AtomicReference<LocalDate> cursorRef = new AtomicReference<>(cursorDate);
    CheckpointHeartbeat heartbeat = null;
    if (checkpointsEnabled) {
      heartbeat = CheckpointHeartbeat.start(() ->
          checkpointService.markRunning(JOB_NAME, stationId, null, cursorRef.get(), null));
    }
    int total = 0;
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
        int upserted = ingestService.ingestForDate(station, current, asOfUtc);
        total += upserted;
        processed++;
        cursorDate = current;
        if (checkpointsEnabled) {
          cursorRef.set(cursorDate);
          checkpointService.markRunning(JOB_NAME, stationId, null, cursorDate, null);
        }
        snapshot("job=gribstream_variable_ingest station=" + stationId
            + " targetDate=" + current
            + " progress=" + processed + "/" + totalDays
            + " remaining=" + (totalDays - processed)
            + " upserted=" + upserted);
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
      snapshot("job=gribstream_variable_ingest station=" + stationId
          + " progress=complete cursorDate=" + cursorDate
          + " totalUpserted=" + total);
      return total;
    } catch (RuntimeException ex) {
      if (checkpointsEnabled) {
        if (heartbeat != null) {
          heartbeat.close();
        }
        checkpointService.markFailed(JOB_NAME, stationId, null, cursorDate, null, ex);
      }
      snapshot("job=gribstream_variable_ingest station=" + stationId + " status=failed");
      throw ex;
    }
  }

  public String jobName() {
    return JOB_NAME;
  }

  private void snapshot(String message) {
    String payload = "[GRIBSTREAM-VARS] " + message;
    logger.info(payload);
    System.out.println(payload);
  }
}
