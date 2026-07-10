package com.predictionmarkets.weather.common;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class StandardTimeClimateWindowTest {

  @Test
  void computesStandardTimeWindowForNyC() {
    ZoneId zone = ZoneId.of("America/New_York");
    LocalDate targetDate = LocalDate.of(2026, 1, 10);

    StandardTimeClimateWindow.UtcRange range =
        StandardTimeClimateWindow.computeUtcRange(zone, targetDate);

    assertEquals(Instant.parse("2026-01-10T05:00:00Z"), range.startUtc());
    assertEquals(Instant.parse("2026-01-11T05:00:00Z"), range.endUtc());
  }
}
