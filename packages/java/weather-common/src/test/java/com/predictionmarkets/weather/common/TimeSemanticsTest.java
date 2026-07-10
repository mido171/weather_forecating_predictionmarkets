package com.predictionmarkets.weather.common;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.List;
import org.junit.jupiter.api.Test;

class TimeSemanticsTest {
  private static final LocalTime AS_OF_LOCAL_TIME = LocalTime.of(23, 0);

  @Test
  void computeAsOfTimes_handlesDstStartAndEndAcrossZones() {
    List<AsOfCase> cases = List.of(
        new AsOfCase("America/New_York", LocalDate.of(2024, 3, 10), Instant.parse("2024-03-10T04:00:00Z")),
        new AsOfCase("America/New_York", LocalDate.of(2024, 11, 3), Instant.parse("2024-11-03T03:00:00Z")),
        new AsOfCase("America/Chicago", LocalDate.of(2024, 3, 10), Instant.parse("2024-03-10T05:00:00Z")),
        new AsOfCase("America/Chicago", LocalDate.of(2024, 11, 3), Instant.parse("2024-11-03T04:00:00Z")),
        new AsOfCase("America/Los_Angeles", LocalDate.of(2024, 3, 10), Instant.parse("2024-03-10T07:00:00Z")),
        new AsOfCase("America/Los_Angeles", LocalDate.of(2024, 11, 3), Instant.parse("2024-11-03T06:00:00Z"))
    );

    for (AsOfCase asOfCase : cases) {
      ZoneId zoneId = ZoneId.of(asOfCase.zoneId());
      TimeSemantics.AsOfTimes result = TimeSemantics.computeAsOfTimes(
          asOfCase.targetDateLocal(), AS_OF_LOCAL_TIME, zoneId);
      assertEquals(asOfCase.expectedAsOfUtc(), result.asOfUtc(),
          () -> "Unexpected asOfUtc for " + asOfCase.zoneId() + " on " + asOfCase.targetDateLocal());
      ZonedDateTime expectedLocal = ZonedDateTime.of(
          asOfCase.targetDateLocal().minusDays(1), AS_OF_LOCAL_TIME, zoneId);
      assertEquals(expectedLocal, result.asOfLocalZdt(),
          () -> "Unexpected asOfLocalZdt for " + asOfCase.zoneId() + " on " + asOfCase.targetDateLocal());
    }
  }

  @Test
  void computeStandardTimeWindow_handlesDstStartAndEndAcrossZones() {
    List<WindowCase> cases = List.of(
        new WindowCase(-300, LocalDate.of(2024, 3, 10),
            Instant.parse("2024-03-10T05:00:00Z"), Instant.parse("2024-03-11T05:00:00Z")),
        new WindowCase(-300, LocalDate.of(2024, 11, 3),
            Instant.parse("2024-11-03T05:00:00Z"), Instant.parse("2024-11-04T05:00:00Z")),
        new WindowCase(-360, LocalDate.of(2024, 3, 10),
            Instant.parse("2024-03-10T06:00:00Z"), Instant.parse("2024-03-11T06:00:00Z")),
        new WindowCase(-360, LocalDate.of(2024, 11, 3),
            Instant.parse("2024-11-03T06:00:00Z"), Instant.parse("2024-11-04T06:00:00Z")),
        new WindowCase(-480, LocalDate.of(2024, 3, 10),
            Instant.parse("2024-03-10T08:00:00Z"), Instant.parse("2024-03-11T08:00:00Z")),
        new WindowCase(-480, LocalDate.of(2024, 11, 3),
            Instant.parse("2024-11-03T08:00:00Z"), Instant.parse("2024-11-04T08:00:00Z"))
    );

    for (WindowCase windowCase : cases) {
      TimeSemantics.StandardTimeWindow result = TimeSemantics.computeStandardTimeWindow(
          windowCase.targetDateLocal(), windowCase.standardOffsetMinutes());
      assertEquals(windowCase.expectedStartUtc(), result.windowStartUtc(),
          () -> "Unexpected windowStartUtc for offset " + windowCase.standardOffsetMinutes()
              + " on " + windowCase.targetDateLocal());
      assertEquals(windowCase.expectedEndUtc(), result.windowEndUtc(),
          () -> "Unexpected windowEndUtc for offset " + windowCase.standardOffsetMinutes()
              + " on " + windowCase.targetDateLocal());
    }
  }

  @Test
  void assertRuntimeNotAfterAsOf_enforcesNoLeakage() {
    Instant asOfUtc = Instant.parse("2024-03-10T04:00:00Z");
    Instant before = Instant.parse("2024-03-10T03:00:00Z");
    Instant after = Instant.parse("2024-03-10T05:00:00Z");

    assertDoesNotThrow(() -> TimeSemantics.assertRuntimeNotAfterAsOf(before, asOfUtc));
    assertDoesNotThrow(() -> TimeSemantics.assertRuntimeNotAfterAsOf(asOfUtc, asOfUtc));

    IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
        () -> TimeSemantics.assertRuntimeNotAfterAsOf(after, asOfUtc));
    assertTrue(ex.getMessage().contains("runtimeUtc"));
    assertTrue(ex.getMessage().contains("asOfUtc"));
  }

  private record AsOfCase(String zoneId, LocalDate targetDateLocal, Instant expectedAsOfUtc) {
  }

  private record WindowCase(int standardOffsetMinutes,
                            LocalDate targetDateLocal,
                            Instant expectedStartUtc,
                            Instant expectedEndUtc) {
  }
}
