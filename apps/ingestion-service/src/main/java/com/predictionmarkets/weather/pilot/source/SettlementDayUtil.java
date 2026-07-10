package com.predictionmarkets.weather.pilot.source;

import java.time.Instant;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;

public final class SettlementDayUtil {
  private SettlementDayUtil() {
  }

  public static LocalDate localStandardDay(ZoneId zoneId, Instant instant) {
    ZoneOffset standardOffset = zoneId.getRules().getStandardOffset(instant);
    return instant.atOffset(standardOffset).toLocalDate();
  }

  public static Instant startOfLocalStandardDayUtc(ZoneId zoneId, LocalDate localDate) {
    ZoneOffset standardOffset = zoneId.getRules().getStandardOffset(Instant.now());
    return OffsetDateTime.of(localDate, java.time.LocalTime.MIDNIGHT, standardOffset).toInstant();
  }
}
