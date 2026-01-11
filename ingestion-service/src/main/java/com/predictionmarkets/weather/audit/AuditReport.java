package com.predictionmarkets.weather.audit;

import com.predictionmarkets.weather.models.MosModel;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

public record AuditReport(
    String schemaVersion,
    Instant generatedAtUtc,
    LocalDate startDateLocal,
    LocalDate endDateLocal,
    Long asofPolicyId,
    List<String> stationIds,
    List<MosModel> models,
    Status overallStatus,
    NoLeakageResult noLeakage,
    StationCoverageResult stationCoverage,
    MosAvailabilityResult mosAvailability,
    FeatureCoverageResult featureCoverage,
    AlignmentResult alignment,
    CliRevisionResult cliRevisions) {

  public enum Status {
    PASS,
    FAIL,
    SKIPPED
  }

  public record NoLeakageResult(Status status, long violationCount, List<LeakageSample> samples) {
  }

  public record LeakageSample(String stationId,
                              LocalDate targetDateLocal,
                              MosModel model,
                              Instant asofUtc,
                              Instant chosenRuntimeUtc) {
  }

  public record StationCoverageResult(Status status, List<StationCoverage> stations) {
  }

  public record StationCoverage(String stationId,
                                long expectedDays,
                                long actualDays,
                                long missingDays,
                                List<LocalDate> sampleMissingDates) {
  }

  public record MosAvailabilityResult(Status status, List<MosAvailability> stations) {
  }

  public record MosAvailability(String stationId,
                                MosModel model,
                                long expectedDays,
                                long runtimeDays,
                                long missingRuntimeDays,
                                Instant minRuntimeUtc,
                                Instant maxRuntimeUtc,
                                List<LocalDate> sampleMissingRuntimeDates) {
  }

  public record FeatureCoverageResult(Status status, List<FeatureCoverage> stations) {
  }

  public record FeatureCoverage(String stationId,
                                MosModel model,
                                long expectedRows,
                                long actualRows,
                                long missingRows,
                                long missingTmaxCount,
                                double missingTmaxPercent,
                                List<MissingReasonCount> topMissingReasons) {
  }

  public record MissingReasonCount(String reason, long count) {
  }

  public record AlignmentResult(Status status,
                                int maxForecastDays,
                                long checkedRows,
                                List<AlignmentViolation> samples) {
  }

  public record AlignmentViolation(String stationId,
                                   MosModel model,
                                   LocalDate targetDateLocal,
                                   Instant chosenRuntimeUtc,
                                   LocalDate runtimeLocalDate,
                                   long daysAhead,
                                   String message) {
  }

  public record CliRevisionResult(Status status, String note, Long revisionCount) {
  }
}
