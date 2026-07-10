package com.predictionmarkets.weather.backfill;

import com.predictionmarkets.weather.models.IngestCheckpoint;
import com.predictionmarkets.weather.models.IngestCheckpointStatus;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import org.springframework.stereotype.Service;

@Service
public class IngestCheckpointService {
  private static final int ERROR_DETAILS_LIMIT = 1024;

  private final IngestCheckpointRepository repository;

  public IngestCheckpointService(IngestCheckpointRepository repository) {
    this.repository = repository;
  }

  public Optional<IngestCheckpoint> findCheckpoint(String jobName, String stationId, MosModel model) {
    String normalizedJobName = normalizeJobName(jobName);
    String normalizedStationId = normalizeStationId(stationId);
    return findExisting(normalizedJobName, normalizedStationId, model);
  }

  public IngestCheckpoint markRunning(String jobName,
                                      String stationId,
                                      MosModel model,
                                      LocalDate cursorDate,
                                      Instant cursorRuntimeUtc) {
    return saveCheckpoint(jobName, stationId, model, IngestCheckpointStatus.RUNNING, cursorDate,
        cursorRuntimeUtc, null);
  }

  public IngestCheckpoint markComplete(String jobName,
                                       String stationId,
                                       MosModel model,
                                       LocalDate cursorDate,
                                       Instant cursorRuntimeUtc) {
    return saveCheckpoint(jobName, stationId, model, IngestCheckpointStatus.COMPLETE, cursorDate,
        cursorRuntimeUtc, null);
  }

  public IngestCheckpoint markFailed(String jobName,
                                     String stationId,
                                     MosModel model,
                                     LocalDate cursorDate,
                                     Instant cursorRuntimeUtc,
                                     Throwable error) {
    return saveCheckpoint(jobName, stationId, model, IngestCheckpointStatus.FAILED, cursorDate,
        cursorRuntimeUtc, formatError(error));
  }

  private IngestCheckpoint saveCheckpoint(String jobName,
                                          String stationId,
                                          MosModel model,
                                          IngestCheckpointStatus status,
                                          LocalDate cursorDate,
                                          Instant cursorRuntimeUtc,
                                          String errorDetails) {
    String normalizedJobName = normalizeJobName(jobName);
    String normalizedStationId = normalizeStationId(stationId);
    Objects.requireNonNull(status, "status is required");

    IngestCheckpoint checkpoint = findExisting(normalizedJobName, normalizedStationId, model)
        .orElseGet(IngestCheckpoint::new);
    checkpoint.setJobName(normalizedJobName);
    checkpoint.setStationId(normalizedStationId);
    checkpoint.setModel(model);
    checkpoint.setCursorDate(cursorDate);
    checkpoint.setCursorRuntimeUtc(cursorRuntimeUtc);
    checkpoint.setStatus(status);
    checkpoint.setErrorDetails(truncate(errorDetails));
    checkpoint.setUpdatedAtUtc(Instant.now());
    return repository.save(checkpoint);
  }

  private Optional<IngestCheckpoint> findExisting(String jobName, String stationId, MosModel model) {
    if (model == null) {
      return repository.findByJobNameAndStationIdAndModelIsNull(jobName, stationId);
    }
    return repository.findByJobNameAndStationIdAndModel(jobName, stationId, model);
  }

  private String formatError(Throwable error) {
    if (error == null) {
      return null;
    }
    String message = error.getMessage();
    String type = error.getClass().getSimpleName();
    if (message == null || message.isBlank()) {
      return type;
    }
    return type + ": " + message;
  }

  private String truncate(String value) {
    if (value == null || value.length() <= ERROR_DETAILS_LIMIT) {
      return value;
    }
    return value.substring(0, ERROR_DETAILS_LIMIT);
  }

  private String normalizeJobName(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("jobName is required");
    }
    return value.trim();
  }

  private String normalizeStationId(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("stationId is required");
    }
    return value.trim().toUpperCase(Locale.ROOT);
  }
}
