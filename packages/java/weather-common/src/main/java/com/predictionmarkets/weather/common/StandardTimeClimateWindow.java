package com.predictionmarkets.weather.common;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.zone.ZoneRules;
import java.util.Objects;

public final class StandardTimeClimateWindow {
  private StandardTimeClimateWindow() {
  }

  public static UtcRange computeUtcRange(ZoneId zone, LocalDate targetDateLocal) {
    Objects.requireNonNull(zone, "zone must not be null");
    Objects.requireNonNull(targetDateLocal, "targetDateLocal must not be null");
    ZoneRules rules = zone.getRules();
    Instant middayLocal = targetDateLocal.atTime(LocalTime.NOON).atZone(zone).toInstant();
    ZoneOffset standardOffset = rules.getStandardOffset(middayLocal);
    OffsetDateTime start = OffsetDateTime.of(targetDateLocal, LocalTime.MIDNIGHT, standardOffset);
    Instant startUtc = start.toInstant();
    Instant endUtc = startUtc.plusSeconds(24 * 60L * 60L);
    return new UtcRange(startUtc, endUtc);
  }

  public record UtcRange(Instant startUtc, Instant endUtc) {
  }
}
