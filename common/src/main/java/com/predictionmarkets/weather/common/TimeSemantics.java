package com.predictionmarkets.weather.common;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Objects;

/**
 * Time semantics helpers for computing as-of instants and CLI standard-time day windows.
 *
 * <p>As-of example (from docs/time-semantics.md): target date 2025-07-10, station
 * America/New_York, asOfLocalTime 23:00 -> asOfLocal 2025-07-09T23:00-04:00,
 * asOfUtc 2025-07-10T03:00Z.</p>
 *
 * <p>Standard-time window example: target date 2025-07-10, standard offset -300 minutes
 * -> windowStartUtc 2025-07-10T05:00Z, windowEndUtc 2025-07-11T05:00Z.</p>
 */
public final class TimeSemantics {
  private TimeSemantics() {
  }

  public static AsOfTimes computeAsOfTimes(LocalDate targetDateLocal,
                                           LocalTime asOfLocalTime,
                                           ZoneId stationZoneId) {
    return computeAsOfTimes(targetDateLocal, asOfLocalTime, stationZoneId, stationZoneId);
  }

  public static AsOfTimes computeAsOfTimes(LocalDate targetDateLocal,
                                           LocalTime asOfLocalTime,
                                           ZoneId stationZoneId,
                                           ZoneId asOfZoneId) {
    Objects.requireNonNull(targetDateLocal, "targetDateLocal must not be null");
    Objects.requireNonNull(asOfLocalTime, "asOfLocalTime must not be null");
    Objects.requireNonNull(stationZoneId, "stationZoneId must not be null");
    ZoneId resolvedAsOfZoneId = asOfZoneId != null ? asOfZoneId : stationZoneId;
    LocalDate asOfDateLocal = targetDateLocal.minusDays(1);
    ZonedDateTime decisionLocalZdt =
        ZonedDateTime.of(asOfDateLocal, asOfLocalTime, resolvedAsOfZoneId);
    Instant asOfUtc = decisionLocalZdt.toInstant();
    ZonedDateTime asOfLocalZdt = ZonedDateTime.ofInstant(asOfUtc, stationZoneId);
    return new AsOfTimes(asOfUtc, asOfLocalZdt);
  }

  public static StandardTimeWindow computeStandardTimeWindow(LocalDate targetDateLocal,
                                                             int standardOffsetMinutes) {
    Objects.requireNonNull(targetDateLocal, "targetDateLocal must not be null");
    ZoneOffset standardOffset = ZoneOffset.ofTotalSeconds(standardOffsetMinutes * 60);
    OffsetDateTime windowStart = OffsetDateTime.of(targetDateLocal, LocalTime.MIDNIGHT, standardOffset);
    OffsetDateTime windowEnd = OffsetDateTime.of(targetDateLocal.plusDays(1), LocalTime.MIDNIGHT, standardOffset);
    return new StandardTimeWindow(windowStart.toInstant(), windowEnd.toInstant());
  }

  public static void assertRuntimeNotAfterAsOf(Instant runtimeUtc, Instant asOfUtc) {
    Objects.requireNonNull(runtimeUtc, "runtimeUtc must not be null");
    Objects.requireNonNull(asOfUtc, "asOfUtc must not be null");
    if (runtimeUtc.isAfter(asOfUtc)) {
      throw new IllegalArgumentException("MOS runtimeUtc must be <= asOfUtc (runtimeUtc="
          + runtimeUtc + ", asOfUtc=" + asOfUtc + ")");
    }
  }

  public record AsOfTimes(Instant asOfUtc, ZonedDateTime asOfLocalZdt) {
  }

  public record StandardTimeWindow(Instant windowStartUtc, Instant windowEndUtc) {
  }
}
