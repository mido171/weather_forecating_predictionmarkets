package com.predictionmarkets.weather.klga.iemmos;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;

public record IemMosChunk(
    String chunkId,
    String jobId,
    IemMosStation station,
    IemMosProduct product,
    String cutoffId,
    Instant windowStartUtc,
    Instant windowEndUtc,
    String requestSha256,
    String requestJson) {

  public LocalDate startDate() {
    return LocalDate.ofInstant(windowStartUtc, java.time.ZoneOffset.UTC);
  }

  public LocalDate endDateInclusive() {
    return LocalDate.ofInstant(windowEndUtc.minusSeconds(1), java.time.ZoneOffset.UTC);
  }

  public long windowDays() {
    return Math.max(1L, Duration.between(windowStartUtc, windowEndUtc).toDays());
  }
}
